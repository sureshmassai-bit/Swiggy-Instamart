"""Instamart Copilot web app.

    python -m webapp.manage init     # once: create users, credentials, Excel record, dumps
    python -m webapp.app             # http://127.0.0.1:5000

Security notes: passwords are PBKDF2-SHA256 hashes; sessions are signed HttpOnly cookies; every POST
needs a per-session CSRF token; every query is parameterised; templates auto-escape; users can only
see their own data; /admin/* is admin-only; login is rate limited per username.
"""
from datetime import datetime, timedelta, timezone
from functools import wraps
import hmac
import os
import re
import secrets
import sqlite3

from werkzeug.middleware.proxy_fix import ProxyFix
from flask import (Flask, abort, flash, g, redirect, render_template, request, send_file, session, url_for)

from webapp import records
from webapp.config import Config
from webapp.db import (BOT_USERNAME, CITIES, ORDER_STATUSES, TICKET_STATUSES, bot_user_id, connect, customer_of,
                       hash_password, now, segment_of, verify_password)
from webapp.support_bot import get_bot, order_dicts
from webapp.triage_bridge import triage_text

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORDER_REF_RE = re.compile(r"^(ORD-\d{6}|WEB-\d{6})$")
DUMMY_HASH = None


def create_app(cfg=None):
    cfg = cfg or Config()
    app = Flask(__name__)
    app.config.update(SECRET_KEY=cfg.secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      MAX_CONTENT_LENGTH=64 * 1024, CFG=cfg)

    if os.environ.get("INSTAMART_PUBLIC_HTTPS") == "1":
        # Behind a TLS-terminating tunnel/proxy such as ngrok: trust ONE proxy hop for scheme/host and send
        # the session cookie only over HTTPS. Leave unset for plain local use.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
        app.config["SESSION_COOKIE_SECURE"] = True

    global DUMMY_HASH
    DUMMY_HASH = hash_password(cfg, "not-a-real-password")      # equalises timing for unknown usernames

    # ------------------------------------------------------------------ plumbing
    def db():
        if "db" not in g:
            g.db = connect(cfg)
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        con = g.pop("db", None)
        if con is not None:
            con.close()

    def commit():
        db().commit()
        try:
            records.refresh(cfg)                                 # Excel record + app SQL dump stay in sync
        except Exception as exc:                                 # a locked spreadsheet must not lose the user's action
            app.logger.warning("record refresh failed: %s", exc)

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(24)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token
    app.jinja_env.filters["tkt"] = lambda n: f"TKT-{int(n):06d}"
    app.jinja_env.filters["web_order"] = lambda n: f"WEB-{int(n):06d}"
    app.jinja_env.filters["inr"] = lambda v: f"₹{float(v):,.2f}"
    app.jinja_env.filters["pretty"] = lambda s: (s or "").replace("_", " ")

    @app.before_request
    def protect():
        if request.method == "POST":
            sent = request.form.get("csrf", "") or request.headers.get("X-CSRF", "")
            if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
                abort(400, "Invalid or missing CSRF token. Reload the page and try again.")
        g.user = None
        uid = session.get("user_id")
        if uid:
            row = db().execute("SELECT * FROM users WHERE user_id = ? AND active = 1", (uid,)).fetchone()
            if row:
                g.user = row
            else:
                session.clear()

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "same-origin"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; form-action 'self'; frame-ancestors 'none'"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def inject():
        return {"me": g.get("user"), "ORDER_STATUSES": ORDER_STATUSES, "TICKET_STATUSES": TICKET_STATUSES}

    def login_required(role=None):
        def deco(fn):
            @wraps(fn)
            def wrapper(*a, **kw):
                if g.user is None:
                    return redirect(url_for("login", next=request.path))
                if role and g.user["role"] != role:
                    abort(403)
                return fn(*a, **kw)
            return wrapper
        return deco

    def home_for(user):
        return url_for("admin_home") if user["role"] == "admin" else url_for("user_home")

    def page_args():
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        return page, (page - 1) * cfg.page_size

    def paged(total, page):
        pages = max(1, -(-total // cfg.page_size))
        return {"page": page, "pages": pages, "total": total}

    # ------------------------------------------------------------------ auth
    @app.route("/")
    def index():
        return redirect(home_for(g.user)) if g.user else redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(home_for(g.user))
        if request.method == "POST":
            username = request.form.get("username", "").strip().lower()[:60]
            password = request.form.get("password", "")[:200]
            since = (datetime.now(timezone.utc) - timedelta(minutes=cfg.lockout_minutes)).strftime("%Y-%m-%d %H:%M:%S")
            fails = db().execute("SELECT COUNT(*) FROM login_failures WHERE username = ? AND ts >= ?", (username, since)).fetchone()[0]
            if fails >= cfg.max_failed_logins:
                abort(429, f"Too many failed attempts. Try again in {cfg.lockout_minutes} minutes.")
            row = db().execute("SELECT * FROM users WHERE username = ? AND active = 1", (username,)).fetchone()
            ok = verify_password(row["password_hash"] if row else DUMMY_HASH, password) and row is not None
            if not ok:
                db().execute("INSERT INTO login_failures VALUES (?, ?)", (username, now()))
                db().commit()
                flash("Invalid username or password.", "error")
                return render_template("login.html"), 401
            db().execute("DELETE FROM login_failures WHERE username = ?", (username,))
            db().commit()
            session.clear()
            session["user_id"] = row["user_id"]
            csrf_token()
            return redirect(home_for(row))
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if g.user:
            return redirect(home_for(g.user))
        if request.method == "POST":
            f = request.form
            username = f.get("username", "").strip().lower()
            full_name, email = f.get("full_name", "").strip(), f.get("email", "").strip()
            city, gender = f.get("city", ""), f.get("gender", "unknown")
            password, confirm = f.get("password", ""), f.get("confirm", "")
            errors = []
            if not USERNAME_RE.match(username):
                errors.append("Username: 3-30 characters, lowercase letters, digits or underscore.")
            if not 2 <= len(full_name) <= 80:
                errors.append("Full name must be 2-80 characters.")
            if not EMAIL_RE.match(email) or len(email) > 120:
                errors.append("Enter a valid email address.")
            if city not in CITIES:
                errors.append("Choose a city from the list.")
            if gender not in ("male", "female", "unknown"):
                errors.append("Choose a gender option.")
            age = None
            if f.get("age", "").strip():
                if f["age"].strip().isdigit() and 10 <= int(f["age"]) <= 100:
                    age = int(f["age"])
                else:
                    errors.append("Age must be a number from 10 to 100.")
            if len(password) < 8 or password.lower() == username:
                errors.append("Password must be at least 8 characters and different from the username.")
            if password != confirm:
                errors.append("Passwords do not match.")
            if db().execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                errors.append("That username is taken.")
            if errors:
                for e in errors:
                    flash(e, "error")
                return render_template("register.html", form=f, cities=CITIES), 400
            n = db().execute("SELECT COUNT(*) FROM web_customers").fetchone()[0] + 1
            cid = f"CW{n:05d}"
            ts = now()
            con = db()
            con.execute("INSERT INTO web_customers VALUES (?,?,?,?,?,?)", (cid, full_name, city, age, gender, ts))
            con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, created_at) "
                        "VALUES (?, 'user', ?, ?, ?, ?, ?)", (username, cid, full_name, email, hash_password(cfg, password), ts))
            commit()
            flash("Account created. Please sign in.", "ok")
            return redirect(url_for("login"))
        return render_template("register.html", form={}, cities=CITIES)

    # ------------------------------------------------------------------ user area
    def products(category=None):
        sql = "SELECT product_id, product_name, category, price_inr FROM core.products"
        args = ()
        if category:
            sql += " WHERE category = ?"
            args = (category,)
        return db().execute(sql + " ORDER BY category, product_name", args).fetchall()

    @app.get("/home")
    @login_required("user")
    def user_home():
        cid = g.user["customer_id"]
        seg, premium = segment_of(db(), cid)
        stats = db().execute(
            "SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN status='completed' OR status='delivered' THEN amount END),0) spent "
            "FROM all_orders WHERE customer_id = ?", (cid,)).fetchone()
        recent = db().execute("SELECT * FROM all_orders WHERE customer_id = ? ORDER BY created_at DESC LIMIT 5", (cid,)).fetchall()
        open_tickets = db().execute("SELECT COUNT(*) FROM tickets WHERE customer_id = ? AND status != 'closed'", (cid,)).fetchone()[0]
        return render_template("user_home.html", customer=customer_of(db(), cid), segment=seg, premium=premium,
                               stats=stats, recent=recent, open_tickets=open_tickets)

    @app.get("/shop")
    @login_required("user")
    def shop():
        cat = request.args.get("category", "")
        cats = [r[0] for r in db().execute("SELECT DISTINCT category FROM core.products ORDER BY category")]
        return render_template("shop.html", items=products(cat if cat in cats else None), cats=cats, current=cat)

    @app.post("/shop/order")
    @login_required("user")
    def place_order():
        pid = request.form.get("product_id", "")
        qty_raw = request.form.get("quantity", "1")
        prod = db().execute("SELECT * FROM core.products WHERE product_id = ?", (pid,)).fetchone()
        if prod is None or not qty_raw.isdigit() or not 1 <= int(qty_raw) <= 20:
            flash("Choose a valid product and a quantity from 1 to 20.", "error")
            return redirect(url_for("shop")), 303
        qty, price = int(qty_raw), float(prod["price_inr"])        # price comes from the database, never the form
        ts = now()
        cur = db().execute(
            "INSERT INTO web_orders (customer_id, user_id, product_id, product_name, quantity, unit_price, amount, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?, 'placed', ?, ?)",
            (g.user["customer_id"], g.user["user_id"], pid, prod["product_name"], qty, price, round(price * qty, 2), ts, ts))
        commit()
        flash(f"Order WEB-{cur.lastrowid:06d} placed: {qty} x {prod['product_name']} = ₹{price * qty:,.2f}.", "ok")
        return redirect(url_for("my_orders"))

    @app.get("/orders")
    @login_required("user")
    def my_orders():
        cid = g.user["customer_id"]
        web = db().execute("SELECT * FROM web_orders WHERE customer_id = ? ORDER BY order_no DESC", (cid,)).fetchall()
        hist = db().execute("SELECT o.order_id, o.product_id, p.product_name, o.quantity, o.order_ts, o.status, o.order_amount "
                            "FROM core.orders o JOIN core.products p USING (product_id) WHERE o.customer_id = ? "
                            "ORDER BY o.order_ts DESC LIMIT 15", (cid,)).fetchall()
        return render_template("orders.html", web=web, hist=hist)

    @app.post("/orders/<int:order_no>/cancel")
    @login_required("user")
    def cancel_order(order_no):
        row = db().execute("SELECT * FROM web_orders WHERE order_no = ? AND user_id = ?", (order_no, g.user["user_id"])).fetchone()
        if row is None:
            abort(404)
        if row["status"] != "placed":
            flash("Only orders that are still 'placed' can be cancelled.", "error")
        else:
            db().execute("UPDATE web_orders SET status='cancelled', updated_at=? WHERE order_no=?", (now(), order_no))
            commit()
            flash(f"Order WEB-{order_no:06d} cancelled.", "ok")
        return redirect(url_for("my_orders"))

    def owns_order_ref(ref):
        cid = g.user["customer_id"]
        return db().execute("SELECT 1 FROM all_orders WHERE order_ref = ? AND customer_id = ?", (ref, cid)).fetchone() is not None

    MAX_CHAT_MESSAGES = 40

    def log_chat(outcome, turns):
        db().execute("INSERT INTO chat_log (user_id, customer_id, turns, outcome, created_at) VALUES (?,?,?,?,?)",
                     (g.user["user_id"], g.user["customer_id"], max(0, min(int(turns), 99)), outcome, now()))

    @app.post("/chat")
    @login_required("user")
    def chat():
        """One bot turn. The conversation lives in the browser (the customer's earlier messages come back as `history`)."""
        data = request.get_json(silent=True) or {}
        text, turns, history = str(data.get("message", "")).strip(), data.get("turns", 0), data.get("history", [])
        if (not 2 <= len(text) <= 1000 or not isinstance(turns, int) or not isinstance(history, list) or len(history) > 20
                or any(not isinstance(h, str) or len(h) > 1000 for h in history)):
            return {"error": "Please write 2-1000 characters."}, 400
        cid = g.user["customer_id"]
        rows = db().execute("SELECT * FROM all_orders WHERE customer_id = ? ORDER BY created_at DESC LIMIT 50", (cid,)).fetchall()
        _, premium = segment_of(db(), cid)
        category = triage_text(text, premium)["category"]                    # PII masked inside before any model call
        res = get_bot().respond(history + [text], category, order_dicts(rows), turns)
        return {"reply": res.reply, "handled": res.handled, "ask": res.ask, "reason": res.reason}

    @app.post("/chat/solved")
    @login_required("user")
    def chat_solved():
        turns = (request.get_json(silent=True) or {}).get("turns", 0)
        log_chat("solved", turns if isinstance(turns, int) else 0)
        commit()
        return {"ok": True}

    @app.post("/chat/ticket")
    @login_required("user")
    def chat_ticket():
        """The bot could not resolve it: raise a ticket that carries the whole conversation for a person."""
        data = request.get_json(silent=True) or {}
        msgs = data.get("messages")
        if not isinstance(msgs, list) or not 1 <= len(msgs) <= MAX_CHAT_MESSAGES:
            return {"error": "Nothing to send."}, 400
        clean = []
        for m in msgs:
            who, text = (m or {}).get("who"), str((m or {}).get("text", "")).strip()[:2000]
            if who not in ("user", "bot") or not text:
                return {"error": "Bad message."}, 400
            clean.append((who, text))
        user_texts = [t for w, t in clean if w == "user"]
        if not user_texts:
            return {"error": "Describe the problem first."}, 400
        last = user_texts[-1]                                             # the latest message is usually the unresolved one
        subject = (last if len(last) >= 5 else "Help request")[:120]
        body = "\n".join(user_texts)[:2000]
        if len(body) < 10:
            body = (body + " (sent from help chat)")
        cid = g.user["customer_id"]
        _, premium = segment_of(db(), cid)
        tri = triage_text(f"{subject}. {body}", premium)
        ts = now()
        cur = db().execute(
            "INSERT INTO tickets (customer_id, user_id, order_ref, subject, body, category, triage_status, route, priority, status, handled_by, bot_replies, created_at, updated_at) "
            "VALUES (?,?,NULL,?,?,?,?,?,?, 'open', 'human', ?, ?, ?)",
            (cid, g.user["user_id"], subject, body, tri["category"], tri["triage_status"], tri["route"], tri["priority"],
             sum(1 for w, _ in clean if w == "bot"), ts, ts))
        bot_id = bot_user_id(db())
        for who, text in clean:
            db().execute("INSERT INTO ticket_messages (ticket_no, author_id, author_role, message, created_at) VALUES (?,?,?,?,?)",
                         (cur.lastrowid, g.user["user_id"] if who == "user" else bot_id, "user" if who == "user" else "admin", text, ts))
        log_chat("ticket", sum(1 for w, _ in clean if w == "user"))
        commit()
        return {"ticket_no": cur.lastrowid, "ref": f"TKT-{cur.lastrowid:06d}", "url": url_for("ticket_detail", ticket_no=cur.lastrowid)}

    @app.get("/tickets")
    @login_required("user")
    def my_tickets():
        rows = db().execute("SELECT * FROM tickets WHERE customer_id = ? ORDER BY ticket_no DESC", (g.user["customer_id"],)).fetchall()
        return render_template("tickets.html", tickets=rows)

    # ------------------------------------------------------------------ tickets (shared)
    def load_ticket(ticket_no):
        t = db().execute("SELECT * FROM tickets WHERE ticket_no = ?", (ticket_no,)).fetchone()
        if t is None or (g.user["role"] != "admin" and t["user_id"] != g.user["user_id"]):
            abort(404)                                               # 404, not 403: do not reveal other users' ticket numbers
        return t

    @app.get("/tickets/<int:ticket_no>")
    @login_required()
    def ticket_detail(ticket_no):
        t = load_ticket(ticket_no)
        msgs = db().execute("SELECT m.*, u.full_name, u.username FROM ticket_messages m JOIN users u ON u.user_id = m.author_id "
                            "WHERE ticket_no = ? ORDER BY msg_no", (ticket_no,)).fetchall()
        cust = customer_of(db(), t["customer_id"]) if g.user["role"] == "admin" else None
        seg = segment_of(db(), t["customer_id"]) if g.user["role"] == "admin" else None
        return render_template("ticket_detail.html", t=t, msgs=msgs, cust=cust, seg=seg, bot_username=BOT_USERNAME)

    @app.post("/tickets/<int:ticket_no>/reply")
    @login_required()
    def ticket_reply(ticket_no):
        t = load_ticket(ticket_no)
        message = request.form.get("message", "").strip()
        if not 2 <= len(message) <= 2000:
            flash("A reply must be 2-2000 characters.", "error")
            return redirect(url_for("ticket_detail", ticket_no=ticket_no))
        if t["status"] == "closed" and g.user["role"] != "admin":
            flash("This ticket is closed. Please raise a new ticket.", "error")
            return redirect(url_for("ticket_detail", ticket_no=ticket_no))
        ts = now()
        db().execute("INSERT INTO ticket_messages (ticket_no, author_id, author_role, message, created_at) VALUES (?,?,?,?,?)",
                     (ticket_no, g.user["user_id"], g.user["role"], message, ts))
        new_status = "answered" if g.user["role"] == "admin" else "open"   # an admin reply answers; a user follow-up reopens
        db().execute("UPDATE tickets SET status = ?, handled_by = 'human', updated_at = ? WHERE ticket_no = ?", (new_status, ts, ticket_no))
        commit()
        flash("Reply sent.", "ok")
        return redirect(url_for("ticket_detail", ticket_no=ticket_no))

    @app.post("/tickets/<int:ticket_no>/status")
    @login_required("admin")
    def ticket_status(ticket_no):
        load_ticket(ticket_no)
        status = request.form.get("status", "")
        if status not in TICKET_STATUSES:
            abort(400, "Unknown status.")
        db().execute("UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_no = ?", (status, now(), ticket_no))
        commit()
        flash(f"Ticket set to {status}.", "ok")
        return redirect(url_for("ticket_detail", ticket_no=ticket_no))

    # ------------------------------------------------------------------ admin
    @app.get("/admin")
    @login_required("admin")
    def admin_home():
        one = lambda sql, *a: db().execute(sql, a).fetchone()[0]
        stats = {
            "customers": one("SELECT COUNT(*) FROM all_customers"),
            "web_customers": one("SELECT COUNT(*) FROM web_customers"),
            "users": one("SELECT COUNT(*) FROM users WHERE role='user'"),
            "open_tickets": one("SELECT COUNT(*) FROM tickets WHERE status='open'"),
            "high_open": one("SELECT COUNT(*) FROM tickets WHERE status='open' AND priority='HIGH'"),
            "web_orders": one("SELECT COUNT(*) FROM web_orders"),
            "web_revenue": one("SELECT COALESCE(SUM(amount),0) FROM web_orders WHERE status != 'cancelled'"),
            "history_orders": one("SELECT COUNT(*) FROM core.orders"),
            "chats_solved": one("SELECT COUNT(*) FROM chat_log WHERE outcome='solved'"),
            "chats_ticketed": one("SELECT COUNT(*) FROM chat_log WHERE outcome='ticket'"),
        }
        queue = db().execute("SELECT t.*, u.username FROM tickets t JOIN users u USING (user_id) WHERE t.status='open' "
                             "ORDER BY (t.priority='HIGH') DESC, t.ticket_no LIMIT 8").fetchall()
        return render_template("admin_home.html", stats=stats, queue=queue)

    @app.get("/admin/customers")
    @login_required("admin")
    def admin_customers():
        q = request.args.get("q", "").strip()[:60]
        city = request.args.get("city", "")
        seg = request.args.get("segment", "")
        page, offset = page_args()
        where, args = ["1=1"], []
        if q:
            where.append("(c.customer_id LIKE ? OR c.full_name LIKE ? OR u.username LIKE ?)")
            args += [f"%{q}%"] * 3
        if city in CITIES:
            where.append("c.city = ?")
            args.append(city)
        segs = [r[0] for r in db().execute("SELECT DISTINCT segment_name FROM seg ORDER BY 1")]
        if seg in segs:
            where.append("s.segment_name = ?")
            args.append(seg)
        base = (" FROM all_customers c LEFT JOIN seg s USING (customer_id) LEFT JOIN users u ON u.customer_id = c.customer_id "
                "LEFT JOIN (SELECT customer_id, COUNT(*) n, SUM(CASE WHEN status='completed' THEN order_amount END) rev "
                "           FROM core.orders GROUP BY customer_id) h ON h.customer_id = c.customer_id "
                "LEFT JOIN (SELECT customer_id, COUNT(*) n FROM web_orders GROUP BY customer_id) w ON w.customer_id = c.customer_id "
                "LEFT JOIN (SELECT customer_id, COUNT(*) n FROM tickets GROUP BY customer_id) t ON t.customer_id = c.customer_id "
                " WHERE " + " AND ".join(where))
        total = db().execute("SELECT COUNT(*)" + base, args).fetchone()[0]
        rows = db().execute("SELECT c.customer_id, c.full_name, c.city, c.age, c.gender, c.source, s.segment_name, s.is_premium, "
                            "u.username, COALESCE(h.n,0) hist_orders, COALESCE(h.rev,0) hist_rev, COALESCE(w.n,0) web_orders, "
                            "COALESCE(t.n,0) tickets" + base + " ORDER BY c.customer_id LIMIT ? OFFSET ?",
                            args + [cfg.page_size, offset]).fetchall()
        return render_template("admin_customers.html", rows=rows, q=q, city=city, seg=seg, segs=segs, cities=CITIES, **paged(total, page))

    @app.get("/admin/customers/<customer_id>")
    @login_required("admin")
    def admin_customer(customer_id):
        cust = customer_of(db(), customer_id)
        if cust is None:
            abort(404)
        seg, premium = segment_of(db(), customer_id)
        user = db().execute("SELECT username, email, active, created_at FROM users WHERE customer_id = ?", (customer_id,)).fetchone()
        summary = db().execute("SELECT COUNT(*) n, SUM(status='completed') done, SUM(status='cancelled') canc, "
                               "COALESCE(SUM(CASE WHEN status='completed' THEN amount END),0) rev FROM all_orders "
                               "WHERE customer_id = ? AND source = 'history'", (customer_id,)).fetchone()
        hist = db().execute("SELECT o.order_id, p.product_name, o.quantity, o.order_ts, o.status, o.order_amount, o.delivery_minutes "
                            "FROM core.orders o JOIN core.products p USING (product_id) WHERE o.customer_id = ? "
                            "ORDER BY o.order_ts DESC LIMIT 20", (customer_id,)).fetchall()
        web = db().execute("SELECT * FROM web_orders WHERE customer_id = ? ORDER BY order_no DESC", (customer_id,)).fetchall()
        tix = db().execute("SELECT * FROM tickets WHERE customer_id = ? ORDER BY ticket_no DESC", (customer_id,)).fetchall()
        return render_template("admin_customer.html", c=cust, seg=seg, premium=premium, user=user, summary=summary,
                               hist=hist, web=web, tix=tix)

    @app.get("/admin/orders")
    @login_required("admin")
    def admin_orders():
        status = request.args.get("status", "")
        page, offset = page_args()
        where, args = ("WHERE status = ?", [status]) if status in ORDER_STATUSES else ("", [])
        total = db().execute(f"SELECT COUNT(*) FROM web_orders {where}", args).fetchone()[0]
        rows = db().execute(f"SELECT w.*, u.username FROM web_orders w JOIN users u USING (user_id) {where} "
                            "ORDER BY order_no DESC LIMIT ? OFFSET ?", args + [cfg.page_size, offset]).fetchall()
        return render_template("admin_orders.html", rows=rows, status=status, **paged(total, page))

    @app.post("/admin/orders/<int:order_no>/status")
    @login_required("admin")
    def admin_order_status(order_no):
        status = request.form.get("status", "")
        if status not in ORDER_STATUSES:
            abort(400, "Unknown status.")
        cur = db().execute("UPDATE web_orders SET status = ?, updated_at = ? WHERE order_no = ?", (status, now(), order_no))
        if cur.rowcount == 0:
            abort(404)
        commit()
        flash(f"Order WEB-{order_no:06d} set to {status.replace('_', ' ')}.", "ok")
        return redirect(request.referrer or url_for("admin_orders"))

    @app.get("/admin/tickets")
    @login_required("admin")
    def admin_tickets():
        status = request.args.get("status", "open")
        page, offset = page_args()
        where, args = ("WHERE t.status = ?", [status]) if status in TICKET_STATUSES else ("", [])
        total = db().execute(f"SELECT COUNT(*) FROM tickets t {where}", args).fetchone()[0]
        rows = db().execute(f"SELECT t.*, u.username FROM tickets t JOIN users u USING (user_id) {where} "
                            "ORDER BY (t.priority='HIGH') DESC, t.ticket_no DESC LIMIT ? OFFSET ?",
                            args + [cfg.page_size, offset]).fetchall()
        return render_template("admin_tickets.html", rows=rows, status=status, **paged(total, page))

    @app.get("/admin/export.xlsx")
    @login_required("admin")
    def admin_export():
        path = records.sync_excel(cfg)
        return send_file(path, as_attachment=True, download_name="instamart_web_records.xlsx")

    from webapp.insights import bp as insights_bp              # Parts A-D analytics, policy Q&A, refund agent
    app.register_blueprint(insights_bp)

    # ------------------------------------------------------------------ errors
    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(429)
    def http_error(err):
        titles = {400: "Bad request", 403: "Not allowed", 404: "Not found", 429: "Slow down"}
        return render_template("error.html", code=err.code, title=titles.get(err.code, "Error"),
                               detail=err.description if err.code in (400, 429) else ""), err.code

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("INSTAMART_PORT", "5000")), debug=False)
