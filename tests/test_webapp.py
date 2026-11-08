"""Web app: authentication, authorisation, orders, tickets (PII masking), admin replies, Excel record, dumps."""
import re
import sqlite3
from pathlib import Path

import openpyxl
import pytest

import mock_llm
from webapp import records
from webapp.app import create_app
from webapp.config import ROOT, Config
from webapp.db import connect, hash_password, now


@pytest.fixture
def env(tmp_path):
    cfg = Config(app_db=tmp_path / "app.db", excel_path=tmp_path / "rec" / "r.xlsx", dump_dir=tmp_path / "dump",
                 credentials_dir=tmp_path / "cred", secret_key_path=tmp_path / "key", pw_iterations=1000)
    con = connect(cfg)
    seg = {r[0]: r[1] for r in con.execute("SELECT customer_id, segment_name FROM seg")}
    premium = next(c for c, s in sorted(seg.items()) if s == "Premium")
    other = next(c for c, s in sorted(seg.items()) if s != "Premium")
    ts = now()
    for name, role, cid in [("admin", "admin", None), ("alice", "user", premium), ("bob", "user", other)]:
        con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, created_at) VALUES (?,?,?,?,?,?,?)",
                    (name, role, cid, name.title(), f"{name}@x.com", hash_password(cfg, f"{name}-Pass-123"), ts))
    con.commit()
    con.close()
    app = create_app(cfg)
    app.config["TESTING"] = True
    return app, cfg, {"alice": premium, "bob": other}


def csrf_of(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def login(app, user):
    c = app.test_client()
    r = c.post("/login", data={"username": user, "password": f"{user}-Pass-123", "csrf": csrf_of(c)})
    assert r.status_code == 302, r.get_data(as_text=True)[:300]
    return c


def post(c, path, page="/home", **data):
    return c.post(path, data={"csrf": csrf_of(c, page), **data})


def chat_ticket(c, *user_msgs, page="/tickets"):
    """What the help widget does when the bot could not resolve the problem: send the whole chat as a ticket."""
    msgs = []
    for m in user_msgs:
        msgs += [{"who": "user", "text": m}, {"who": "bot", "text": "I could not find this in our policies."}]
    return c.post("/chat/ticket", json={"messages": msgs}, headers={"X-CSRF": csrf_of(c, page)})


def test_login_logout_and_redirects(env):
    app, *_ = env
    c = app.test_client()
    assert c.get("/home").status_code == 302 and "/login" in c.get("/home").headers["Location"]
    assert login(app, "admin").get("/").headers["Location"].endswith("/admin")
    u = login(app, "alice")
    assert u.get("/").headers["Location"].endswith("/home") and u.get("/home").status_code == 200
    post(u, "/logout")
    assert u.get("/home").status_code == 302


def test_wrong_password_and_lockout(env):
    app, cfg, _ = env
    c = app.test_client()
    for _ in range(cfg.max_failed_logins):
        assert c.post("/login", data={"username": "alice", "password": "nope", "csrf": csrf_of(c)}).status_code == 401
    r = c.post("/login", data={"username": "alice", "password": "alice-Pass-123", "csrf": csrf_of(c)})
    assert r.status_code == 429                            # locked even with the right password
    assert c.post("/login", data={"username": "ghost", "password": "x", "csrf": csrf_of(c)}).status_code == 401


def test_csrf_is_required(env):
    app, *_ = env
    c = login(app, "alice")
    assert c.post("/shop/order", data={"product_id": "P001", "quantity": "1"}).status_code == 400
    assert c.post("/shop/order", data={"product_id": "P001", "quantity": "1", "csrf": "forged"}).status_code == 400
    assert app.test_client().post("/login", data={"username": "alice", "password": "alice-Pass-123"}).status_code == 400


def test_role_separation(env):
    app, *_ = env
    u = login(app, "alice")
    for path in ["/admin", "/admin/customers", "/admin/orders", "/admin/tickets", "/admin/export.xlsx", "/admin/customers/C0001"]:
        assert u.get(path).status_code == 403, path
    a = login(app, "admin")
    for path in ["/home", "/shop", "/orders"]:
        assert a.get(path).status_code == 403, path        # the admin does not shop or raise tickets
    assert post(a, "/shop/order", "/admin", product_id="P001", quantity="1").status_code == 403
    anon = app.test_client()
    assert anon.get("/admin").status_code == 302


def test_register_creates_customer_but_never_admin(env):
    app, cfg, _ = env
    c = app.test_client()
    base = {"username": "carol", "full_name": "Carol Jones", "email": "carol@x.com", "city": "Pune", "age": "31",
            "gender": "female", "password": "longenough1", "confirm": "longenough1", "role": "admin", "csrf": csrf_of(app.test_client(), "/register")}
    c.get("/register")
    base["csrf"] = csrf_of(c, "/register")
    assert c.post("/register", data=base).status_code == 302
    con = sqlite3.connect(cfg.app_db)
    assert con.execute("SELECT role, customer_id FROM users WHERE username='carol'").fetchone() == ("user", "CW00001")
    for bad in [{"password": "short", "confirm": "short"}, {"username": "Bad Name!"}, {"city": "Atlantis"}, {"age": "7"}, {"email": "nope"}, {"confirm": "different1"}]:
        c2 = app.test_client()
        data = {**base, "username": "dave", "csrf": csrf_of(c2, "/register"), **bad}
        assert c2.post("/register", data=data).status_code == 400, bad
    assert c.post("/register", data={**base, "csrf": csrf_of(c, "/register")}).status_code in (302, 400)   # duplicate or logged-in redirect
    assert con.execute("SELECT COUNT(*) FROM users WHERE username='carol'").fetchone()[0] == 1


def test_order_uses_server_side_price_and_validates(env):
    app, cfg, ids = env
    u = login(app, "alice")
    price = sqlite3.connect(cfg.core_db).execute("SELECT price_inr FROM products WHERE product_id='P001'").fetchone()[0]
    r = post(u, "/shop/order", "/shop", product_id="P001", quantity="3", unit_price="0.01", amount="0.01")   # tampered fields ignored
    assert r.status_code == 302
    row = sqlite3.connect(cfg.app_db).execute("SELECT customer_id, quantity, unit_price, amount, status FROM web_orders").fetchone()
    assert row == (ids["alice"], 3, price, round(price * 3, 2), "placed")
    for bad in [{"product_id": "NOPE", "quantity": "1"}, {"product_id": "P001", "quantity": "0"}, {"product_id": "P001", "quantity": "21"}, {"product_id": "P001", "quantity": "-1"}, {"product_id": "P001", "quantity": "x"}]:
        post(u, "/shop/order", "/shop", **bad)
    assert sqlite3.connect(cfg.app_db).execute("SELECT COUNT(*) FROM web_orders").fetchone()[0] == 1
    assert "WEB-000001" in u.get("/orders").get_data(as_text=True)


def test_order_cancel_rules_and_ownership(env):
    app, cfg, _ = env
    a, b = login(app, "alice"), login(app, "bob")
    post(a, "/shop/order", "/shop", product_id="P001", quantity="1")
    assert post(b, "/orders/1/cancel", "/orders").status_code == 404           # another user's order
    assert post(a, "/orders/1/cancel", "/orders").status_code == 302
    assert sqlite3.connect(cfg.app_db).execute("SELECT status FROM web_orders").fetchone()[0] == "cancelled"
    adm = login(app, "admin")
    post(adm, "/admin/orders/1/status", "/admin/orders", status="packed")
    post(a, "/orders/1/cancel", "/orders")                                      # no longer 'placed'... and not cancellable again
    assert sqlite3.connect(cfg.app_db).execute("SELECT status FROM web_orders").fetchone()[0] == "packed"
    assert post(adm, "/admin/orders/1/status", "/admin/orders", status="bogus").status_code == 400


def test_ticket_masks_pii_before_model_and_routes(env, monkeypatch):
    app, cfg, ids = env
    seen = []
    real = mock_llm.chat
    monkeypatch.setattr(mock_llm, "chat", lambda m, t=None: (seen.append(m), real(m, t))[1])
    u = login(app, "alice")                                                      # alice is Premium
    body = "I was charged twice. Email me at jane.doe@example.com or call 9876543210 about ORD-100048 refund."
    r = chat_ticket(u, body)
    assert r.status_code == 200 and r.get_json()["ref"] == "TKT-000001"
    sent = " ".join(m["content"] for msgs in seen for m in msgs)
    assert seen and "jane.doe@example.com" not in sent and "9876543210" not in sent and "ORD-100048" not in sent
    assert "[EMAIL]" in sent and "[PHONE]" in sent and "[ORDER_ID]" in sent
    t = sqlite3.connect(cfg.app_db).execute("SELECT category, route, priority, status, body FROM tickets").fetchone()
    assert t[3] == "open" and t[4] == body                                       # the customer's own text is kept as written
    assert t[1] in ("premium_desk", "security_team", "human_review")
    for bad in ([], [{"who": "bot", "text": "hi"}], [{"who": "admin", "text": "x"}], [{"who": "user", "text": " "}],
                [{"who": "user", "text": "x"}] * 41, "not a list"):
        assert u.post("/chat/ticket", json={"messages": bad}, headers={"X-CSRF": csrf_of(u, "/tickets")}).status_code == 400
    assert u.post("/chat/ticket", json={"messages": [{"who": "user", "text": "hello there friend"}]}).status_code == 400   # CSRF header required
    assert sqlite3.connect(cfg.app_db).execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


def test_ticket_privacy_between_users_and_admin_reply_flow(env):
    app, cfg, _ = env
    a, b, adm = login(app, "alice"), login(app, "bob"), login(app, "admin")
    chat_ticket(a, "Late delivery today")
    assert b.get("/tickets/1").status_code == 404
    assert post(b, "/tickets/1/reply", "/tickets", message="hello").status_code == 404
    assert "Late delivery today" not in b.get("/tickets").get_data(as_text=True)
    assert "Late delivery today" in adm.get("/admin/tickets").get_data(as_text=True)
    assert a.post("/tickets/1/status", data={"status": "closed", "csrf": csrf_of(a, "/tickets")}).status_code == 403   # users cannot close
    post(adm, "/tickets/1/reply", "/tickets/1", message="Sorry about that, we have credited your account.")
    con = sqlite3.connect(cfg.app_db)
    assert con.execute("SELECT status FROM tickets").fetchone()[0] == "answered"
    assert "credited your account" in a.get("/tickets/1").get_data(as_text=True)
    post(a, "/tickets/1/reply", "/tickets/1", message="Thank you, one more question")
    assert con.execute("SELECT status FROM tickets").fetchone()[0] == "open"      # a follow-up reopens
    post(adm, "/tickets/1/status", "/tickets/1", status="closed")
    assert con.execute("SELECT status FROM tickets").fetchone()[0] == "closed"
    post(a, "/tickets/1/reply", "/tickets/1", message="Can I still write?")
    n = con.execute("SELECT COUNT(*) FROM ticket_messages").fetchone()[0]
    post(a, "/tickets/1/reply", "/tickets/1", message="Still closed?")
    assert con.execute("SELECT COUNT(*) FROM ticket_messages").fetchone()[0] == n  # closed: customer reply refused
    human = con.execute("SELECT COUNT(*) FROM ticket_messages m JOIN users u USING (user_id) WHERE 0").fetchone() if False else None
    assert con.execute("SELECT COUNT(*) FROM ticket_messages m JOIN users u ON u.user_id = m.author_id "
                       "WHERE m.author_role='admin' AND u.username='admin'").fetchone()[0] == 1


def test_admin_sees_all_customers_and_details(env):
    app, cfg, ids = env
    adm = login(app, "admin")
    html = adm.get("/admin/customers").get_data(as_text=True)
    assert "1200 results" in html and "C0001" in html
    assert "1 result" in adm.get(f"/admin/customers?q={ids['alice']}").get_data(as_text=True)
    assert adm.get("/admin/customers?segment=Premium").status_code == 200
    page = adm.get(f"/admin/customers/{ids['alice']}").get_data(as_text=True)
    assert ids["alice"] in page and "Orders placed on this site" in page and "alice" in page
    assert adm.get("/admin/customers/NOPE").status_code == 404
    assert adm.get("/admin/customers?q=%27%20OR%201=1%20--").status_code == 200   # SQL-injection probe is just a search string
    assert "0 results" in adm.get("/admin/customers?q=%27%20OR%201=1%20--").get_data(as_text=True)


def test_xss_is_escaped(env):
    app, *_ = env
    a, adm = login(app, "alice"), login(app, "admin")
    chat_ticket(a, "<script>alert(1)</script> subject", "<img src=x onerror=alert(1)> broken thing")
    for client in (a, adm):
        html = client.get("/tickets/1").get_data(as_text=True)
        assert "<script>alert(1)" not in html and "<img src=x" not in html and "&lt;script&gt;" in html
    assert "default-src 'self'" in a.get("/home").headers["Content-Security-Policy"]


def test_excel_record_and_dump_stay_in_sync_without_password_data(env):
    app, cfg, ids = env
    a = login(app, "alice")
    post(a, "/shop/order", "/shop", product_id="P001", quantity="2")
    chat_ticket(a, "=HYPERLINK(\"http://evil\",\"x\") please help me")
    wb = openpyxl.load_workbook(cfg.excel_path)
    assert wb.sheetnames == ["Users", "WebCustomers", "Orders", "Tickets", "ChatLog", "TicketMessages"]
    users = {r[1]: r for r in wb["Users"].iter_rows(min_row=2, values_only=True)}
    assert set(users) == {"admin", "alice", "bob", "copilot_bot"} and users["alice"][3] == ids["alice"]
    assert "password" not in " ".join(str(c.value).lower() for c in wb["Users"][1])
    assert wb["Orders"].max_row == 2 and wb["Tickets"].max_row == 2
    body = [c.value for c in wb["Tickets"][2]][5]
    assert body.startswith("'=") , "text starting with = must not be stored as a live formula"
    assert not any(str(v).startswith("pbkdf2") for ws in wb for row in ws.iter_rows(values_only=True) for v in row)
    dump = (cfg.dump_dir / "app_dump.sql").read_text()
    assert "INSERT INTO \"web_orders\"" in dump and "alice-Pass-123" not in dump and "pbkdf2:sha256" in dump
    assert a.get("/admin/export.xlsx").status_code == 403
    export = login(app, "admin").get("/admin/export.xlsx")
    assert export.status_code == 200 and export.data[:2] == b"PK"


def test_dump_restore_round_trip(env, tmp_path):
    app, cfg, _ = env
    a = login(app, "alice")
    post(a, "/shop/order", "/shop", product_id="P001", quantity="1")
    restored = Config(app_db=tmp_path / "restored.db", dump_dir=cfg.dump_dir, pw_iterations=1000)
    records.restore_app(restored)
    con = sqlite3.connect(restored.app_db)
    assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 3 and con.execute("SELECT COUNT(*) FROM web_orders").fetchone()[0] == 1
    with pytest.raises(FileExistsError):
        records.restore_app(restored)
    app2 = create_app(restored)
    app2.config["TESTING"] = True
    assert login(app2, "alice").get("/orders").status_code == 200               # restored hashes still verify


def test_core_database_is_read_only_and_untouched(env):
    app, cfg, _ = env
    con = connect(cfg)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM core.orders")
    con.close()
    assert sqlite3.connect(cfg.core_db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309


def test_secrets_ignored_but_dump_and_excel_are_committable():
    import subprocess
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
    for secret in ("credentials/ADMIN_CREDENTIALS.txt", "credentials/USER_CREDENTIALS.txt", "data/app.db", "data/.secret_key"):
        assert ignored(secret), f"{secret} must be git-ignored"
    for shared in ("data/dump/app_dump.sql", "data/dump/instamart_dump.sql", "data/records/instamart_web_records.xlsx"):
        assert not ignored(shared), f"{shared} must be committable"


def test_public_https_mode_sets_secure_cookie(env, monkeypatch):
    app, cfg, _ = env
    c = app.test_client()
    assert "Secure" not in (c.get("/login").headers.get("Set-Cookie") or "")           # local mode: plain HTTP works
    monkeypatch.setenv("INSTAMART_PUBLIC_HTTPS", "1")
    app2 = create_app(cfg)
    app2.config["TESTING"] = True
    c2 = app2.test_client()
    resp = c2.get("/login", headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "demo.ngrok-free.app"},
                  base_url="https://demo.ngrok-free.app")
    cookie = resp.headers.get("Set-Cookie", "")
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie
