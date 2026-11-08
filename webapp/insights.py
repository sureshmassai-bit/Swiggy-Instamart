"""Admin analytics for Parts A-D: business overview, SQL, data quality, segments + lapse model, ticket triage,
policy Q&A and the refund agent.

Every number on these pages is read from the real pipeline outputs (outputs/*.csv|json, data/instamart.db) or
computed live from the read-only Part A database; nothing is hard-coded. If an artifact is missing the page says
which command rebuilds it instead of showing sample data.
"""
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import json
import re
import sqlite3
import uuid

import pandas as pd
from flask import (Blueprint, abort, current_app, flash, g, redirect, render_template, request, send_file, url_for)

from src import data_pipeline, policy_rag, refund_agent as ra, sql_analysis, triage as tr
from webapp.triage_bridge import triage_text

bp = Blueprint("insights", __name__, url_prefix="/admin/insights")

ORDER_ID_RE = re.compile(r"^ORD-\d{6}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REBUILD = "python run_project.py"
_CACHE = {}


# ------------------------------------------------------------------------------ helpers
def cfg():
    return current_app.config["CFG"]


@bp.before_request
def admin_only():
    if g.get("user") is None:
        return redirect(url_for("login", next=request.path))
    if g.user["role"] != "admin":
        abort(403)


def _cached(path, loader):
    """Load a file once per modification time (pages stay fast; a pipeline re-run is picked up automatically)."""
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[0] == str(path)]:
            del _CACHE[k]
        _CACHE[key] = loader(path)
    return _CACHE[key]


def out_json(name):
    return _cached(cfg().outputs_dir / name, lambda p: json.loads(p.read_text(encoding="utf-8")))


def out_csv(name, **kw):
    return _cached(cfg().outputs_dir / name, lambda p: pd.read_csv(p, keep_default_na=False, **kw))


def core():
    """Separate READ-ONLY connection to the Part A database."""
    path = cfg().core_db
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run: python -m src.data_pipeline")
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def bars(items, fmt=lambda v: f"{v:,.0f}", width=230):
    """Pre-compute an SVG horizontal bar chart: [(label, value)] -> rows with bar widths (non-negative scale)."""
    items = [(str(lbl), float(v)) for lbl, v in items]
    top = max([v for _, v in items] + [0]) or 1.0
    return [{"label": lbl, "value": v, "text": fmt(v), "w": max(1.0, v / top * width) if v > 0 else 0.0,
             "y": i * 28} for i, (lbl, v) in enumerate(items)]


def records(df, limit=None):
    if df is None:
        return []
    return (df.head(limit) if limit else df).to_dict("records")


def missing(*names):
    return [n for n in names if not (cfg().outputs_dir / n).exists()]


@bp.app_template_filter("ts")
def fmt_ts(epoch):
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


# ----------------------------------------------------------------------------- overview
ARTIFACTS = [("A", "data/clean/customers.csv"), ("A", "data/clean/products.csv"), ("A", "data/clean/orders.csv"),
             ("A", "data/instamart.db"), ("A", "sql/queries.sql"), ("A", "outputs/audit_report.json"),
             ("A", "outputs/cleaning_report.json"), ("B", "outputs/customer_segments.csv"),
             ("B", "outputs/lapse_model.joblib"), ("B", "outputs/retention_list.csv"),
             ("B", "outputs/feature_importance.csv"), ("C", "outputs/triage_results.csv"),
             ("C", "outputs/triage_budget_demo.csv"), ("D", "outputs/rag_evaluation.csv"),
             ("D", "outputs/agent_runs.jsonl"), ("D", "outputs/refund_ledger.jsonl"), ("D", "state/")]


@bp.get("")
def overview():
    from webapp.config import ROOT
    status = []
    for part, rel in ARTIFACTS:
        p = ROOT / rel
        if rel == "outputs/agent_runs.jsonl":
            p = cfg().run_log_path
        elif rel == "outputs/refund_ledger.jsonl":
            p = cfg().ledger_path
        elif rel == "state/":
            p = cfg().state_dir
        elif rel.startswith("outputs/"):
            p = cfg().outputs_dir / rel.split("/", 1)[1]
        elif rel == "data/instamart.db":
            p = cfg().core_db
        status.append({"part": part, "path": rel, "exists": p.exists(),
                       "modified": p.stat().st_mtime if p.exists() else None})
    try:
        with core() as con:
            one = lambda sql: con.execute(sql).fetchone()
            k = one("SELECT (SELECT COUNT(*) FROM customers) customers, (SELECT COUNT(*) FROM products) products, "
                    "COUNT(*) orders, SUM(status='completed') completed, SUM(status='cancelled') cancelled, "
                    "ROUND(SUM(CASE WHEN status='completed' THEN order_amount END),2) revenue, "
                    "ROUND(AVG(CASE WHEN status='completed' THEN order_amount END),2) aov, "
                    "ROUND(AVG(CASE WHEN status='completed' THEN delivery_minutes END),1) avg_delivery, "
                    "SUM(status='completed' AND delivery_minutes > 45) late FROM orders")
            by_city = con.execute("SELECT c.city, SUM(o.order_amount) rev FROM orders o JOIN customers c USING (customer_id) "
                                  "WHERE o.status='completed' GROUP BY c.city ORDER BY rev DESC").fetchall()
            by_cat = con.execute("SELECT p.category, SUM(o.order_amount) rev FROM orders o JOIN products p USING (product_id) "
                                 "WHERE o.status='completed' GROUP BY p.category ORDER BY rev DESC").fetchall()
            by_month = con.execute("SELECT substr(order_ts,1,7) m, SUM(order_amount) rev FROM orders "
                                   "WHERE status='completed' GROUP BY m ORDER BY m").fetchall()
    except (FileNotFoundError, sqlite3.Error) as exc:
        flash(f"Part A database unavailable: {exc}", "error")
        return render_template("insights/overview.html", k=None, status=status, rebuild=REBUILD)
    inr = lambda v: f"₹{v:,.0f}"
    segs = out_csv("segment_profile.csv")
    tri = out_json("triage_summary.json")
    ledger = ra_ledger()
    return render_template(
        "insights/overview.html", k=k, status=status, rebuild=REBUILD,
        city_chart=bars([(r["city"], r["rev"]) for r in by_city], inr),
        cat_chart=bars([(r["category"], r["rev"]) for r in by_cat], inr),
        month_chart=bars([(r["m"], r["rev"]) for r in by_month], inr),
        seg_chart=bars([(r["segment_name"], r["size"]) for r in records(segs)]) if segs is not None else None,
        triage=tri["few_shot"] if tri else None, lapse=out_json("lapse_metrics.json"),
        refunds={"issued": sum(e["outcome"] == "issued" for e in ledger),
                 "issued_inr": sum(e["refund_inr"] for e in ledger if e["outcome"] == "issued"),
                 "escalated": sum(e["outcome"] == "escalated" for e in ledger),
                 "rejected": sum(e["outcome"] == "rejected" for e in ledger)})


# ---------------------------------------------------------------------------------- SQL
@bp.get("/sql")
def sql():
    text = sql_analysis.SQL_FILE.read_text(encoding="utf-8")
    titles = dict(re.findall(r"^-- (Q\d+):\s*(.*)$", text, flags=re.M))
    results, error = [], None
    try:
        with core() as con:
            for name, query in sql_analysis.load_queries().items():
                df = pd.read_sql_query(query, con)                     # live, against the read-only database
                results.append({"name": name, "title": titles.get(name, ""), "sql": query, "rows": len(df),
                                "columns": list(df.columns), "preview": df.head(10).round(2).to_dict("records")})
    except (FileNotFoundError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        error = str(exc)
    eda = cfg().outputs_dir / "eda_notes.md"
    return render_template("insights/sql.html", results=results, error=error,
                           eda=eda.read_text(encoding="utf-8") if eda.exists() else None)


# ------------------------------------------------------------------------- data quality
@bp.get("/data-quality")
def data_quality():
    audit, cleaning = out_json("audit_report.json"), out_json("cleaning_report.json")
    try:
        checkpoints = [{"name": n, "ok": ok, "detail": d} for n, ok, d in data_pipeline.run_checkpoints()]
        cp_error = None
    except Exception as exc:                                    # missing files etc.: say so, never fake a pass
        checkpoints, cp_error = [], f"{type(exc).__name__}: {exc}"
    missing_rows = []
    if audit:
        for table in ("customers", "products", "orders", "tickets"):
            for col, n in audit[table]["missing_count"].items():
                if n:
                    missing_rows.append({"table": table, "column": col, "missing": n,
                                         "pct": audit[table]["missing_pct"][col]})
    return render_template("insights/data_quality.html", audit=audit, cleaning=cleaning, checkpoints=checkpoints,
                           cp_error=cp_error, missing_rows=missing_rows, rebuild=REBUILD,
                           missing_chart=bars([(f"{r['table']}.{r['column']}", r["missing"]) for r in missing_rows]))


# ------------------------------------------------------------------------ segments / lapse
@bp.get("/segments")
def segments():
    diag, profile = out_csv("kmeans_diagnostics.csv"), out_csv("segment_profile.csv")
    lapse, imp = out_json("lapse_metrics.json"), out_csv("feature_importance.csv")
    retention, seg = out_csv("retention_list.csv"), out_csv("customer_segments.csv")
    name = request.args.get("segment", "")
    names = sorted(seg.segment_name.unique()) if seg is not None else []
    members = []
    if seg is not None and name in names:
        members = records(seg[seg.segment_name == name].sort_values("monetary", ascending=False), 50)
    return render_template(
        "insights/segments.html", diag=records(diag), profile=records(profile), lapse=lapse,
        importance=records(imp), retention=records(retention), names=names, name=name, members=members,
        member_total=int((seg.segment_name == name).sum()) if seg is not None and name else 0,
        sil_chart=bars([(f"K={int(r['k'])}", r["silhouette"]) for r in records(diag)], lambda v: f"{v:.3f}"),
        inertia_chart=bars([(f"K={int(r['k'])}", r["inertia"]) for r in records(diag)], lambda v: f"{v:,.1f}"),
        imp_chart=bars([(r["feature"], r["permutation_importance_mean"]) for r in records(imp)], lambda v: f"{v:+.4f}"),
        has_pca=(cfg().outputs_dir / "pca_segments.png").exists(),
        missing=missing("customer_segments.csv", "lapse_metrics.json", "feature_importance.csv"), rebuild=REBUILD)


@bp.get("/pca.png")
def pca_png():
    path = cfg().outputs_dir / "pca_segments.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


# ------------------------------------------------------------------------------- triage
TRIAGE_FILTERS = ("route", "status", "category", "segment_name")


@bp.get("/triage")
def triage():
    df, summary = out_csv("triage_results.csv"), out_json("triage_summary.json")
    if df is None or summary is None:
        return render_template("insights/triage.html", df=None, missing=missing("triage_results.csv", "triage_summary.json"),
                               rebuild=REBUILD)
    none = "(none)"                                  # filter value for an empty cell (e.g. unclassified tickets)
    options = {f: sorted((x or none) for x in df[f].astype(str).unique()) for f in TRIAGE_FILTERS}
    chosen = {f: request.args.get(f, "") for f in TRIAGE_FILTERS}
    view = df
    for f in TRIAGE_FILTERS:
        v = chosen[f]
        if v and v in options[f]:
            view = view[view[f].astype(str) == ("" if v == none else v)]
        else:
            chosen[f] = ""
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    size = cfg().page_size
    pages = max(1, -(-len(view) // size))
    page = min(page, pages)
    few = summary["few_shot"]
    return render_template(
        "insights/triage.html", df=df, rows=records(view.iloc[(page - 1) * size: page * size]), total=len(view),
        page=page, pages=pages, options=options, chosen=chosen, few=few, zero=summary["zero_shot"],
        demo=summary.get("budget_demo_1000_tokens"), budget_csv=(cfg().outputs_dir / "triage_budget_demo.csv").exists(),
        status_chart=bars(sorted(few["status_counts"].items(), key=lambda kv: -kv[1])),
        route_chart=bars(sorted(few["route_counts"].items(), key=lambda kv: -kv[1])),
        cat_chart=bars(df[df.category != ""].category.value_counts().items()),
        trial=None, missing=[], rebuild=REBUILD)


@bp.post("/triage/try")
def triage_try():
    text = request.form.get("text", "")
    premium = request.form.get("premium") == "1"
    if len(text) > 4000:
        flash("Ticket text must be at most 4000 characters.", "error")
        return redirect(url_for("insights.triage"))
    masked, counts = tr.redact(text)
    result = triage_text(text, premium)                      # PII masked inside before any model call
    flash(f"Triage: status {result['triage_status']}, category {result['category'] or '—'}, "
          f"route {result['route']} ({result['priority']}). Model saw: “{masked.strip()[:300] or '(nothing: not sent)'}” "
          f"[masked {counts['email']} email, {counts['phone']} phone, {counts['order_id']} order id]", "ok")
    return redirect(url_for("insights.triage"))


# ------------------------------------------------------------------------------- policy
@bp.route("/policy", methods=["GET", "POST"])
def policy():
    question, result, masked_counts = "", None, None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not 3 <= len(question) <= 300:
            flash("Ask a question of 3-300 characters.", "error")
        else:
            safe, masked_counts = tr.redact(question)          # no customer PII goes into the prompt
            try:
                result = policy_rag.answer(safe)
            except Exception as exc:                            # surface, do not invent an answer
                current_app.logger.exception("policy answer failed")
                flash(f"Policy assistant failed: {type(exc).__name__}. Check data/policies/.", "error")
    ev, summ = out_csv("rag_evaluation.csv"), out_json("rag_summary.json")
    return render_template("insights/policy.html", question=question, result=result, masked=masked_counts,
                           evaluation=records(ev), summary=summ, fallback=policy_rag.FALLBACK,
                           examples=[r["question"] for r in records(ev)][:6])


# ------------------------------------------------------------------------- refund agent
@contextmanager
def agent_paths():
    """Run the Part D agent against this app's configured database, ledger, log and checkpoint folder."""
    c = cfg()
    with ra.RUN_LOCK:
        old_db = ra.DB
        ra.DB = c.core_db
        try:
            with ra.isolated(c.ledger_path, c.state_dir, c.run_log_path):
                yield
        finally:
            ra.DB = old_db


def ra_ledger():
    with agent_paths():
        return ra.read_ledger()


def _int_field(name, lo, hi, default):
    raw = request.form.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or not lo <= int(raw) <= hi:
        raise ValueError(f"{name.replace('_', ' ')} must be a whole number from {lo} to {hi}")
    return int(raw)


@bp.get("/refunds")
def refunds():
    with agent_paths():
        ledger = list(reversed(ra.read_ledger()))
        runs = list(reversed(ra.read_run_log()))
        ckpts = ra.list_checkpoints()
    return render_template("insights/refunds.html", ledger=ledger[:100], runs=runs[:100], ckpts=ckpts[:50],
                           resumable=[c for c in ckpts if c["status"] != "done"], limits=ra,
                           totals={"issued": sum(e["refund_inr"] for e in ledger if e["outcome"] == "issued"),
                                   "n_issued": sum(e["outcome"] == "issued" for e in ledger),
                                   "n_escalated": sum(e["outcome"] == "escalated" for e in ledger),
                                   "n_rejected": sum(e["outcome"] == "rejected" for e in ledger)},
                           scenarios=records(out_csv("agent_scenarios.csv")))


@bp.post("/refunds/run")
def refunds_run():
    order_id = request.form.get("order_id", "").strip().upper()
    try:
        if not ORDER_ID_RE.match(order_id):
            raise ValueError("Order id must look like ORD-123456")
        max_iter = _int_field("max_iterations", 1, ra.MAX_ITERATIONS, ra.MAX_ITERATIONS)
        tokens = _int_field("token_limit", 50, ra.TOKEN_LIMIT, ra.TOKEN_LIMIT)
        seconds = _int_field("time_limit_seconds", 0, ra.TIME_LIMIT_SECONDS, ra.TIME_LIMIT_SECONDS)
        crash = _int_field("crash_after", 0, 3, 0) or None
    except ValueError as exc:
        flash(str(exc) + ".", "error")
        return redirect(url_for("insights.refunds"))
    message = f"Customer reports order {order_id} arrived very late and asks for a refund."
    run_id = f"web-{uuid.uuid4().hex[:12]}"
    with agent_paths():
        try:
            res = ra.run_agent(message, scenario_id="web", run_id=run_id, max_iterations=max_iter,
                               token_limit=tokens, time_limit_seconds=seconds, crash_after=crash)
        except ra.SimulatedCrash as exc:
            flash(f"Run {run_id} stopped by the simulated crash ({exc}). Its checkpoint is saved; resume it below.", "error")
            return redirect(url_for("insights.refund_run", run_id=run_id))
        except Exception as exc:
            current_app.logger.exception("refund agent failed")
            flash(f"Refund agent failed: {type(exc).__name__}: {exc}", "error")
            return redirect(url_for("insights.refunds"))
    flash(f"Run {run_id}: {res['status']}" + (f" · ₹{res['refund_inr']:,.2f}" if res["refund_inr"] else ""), "ok")
    return redirect(url_for("insights.refund_run", run_id=run_id))


def _checkpoint_or_404(run_id):
    if not RUN_ID_RE.match(run_id):
        abort(404)
    with agent_paths():
        try:
            return ra.load_checkpoint(run_id)
        except (OSError, json.JSONDecodeError):
            abort(404)


@bp.get("/refunds/run/<run_id>")
def refund_run(run_id):
    st = _checkpoint_or_404(run_id)
    with agent_paths():
        log = [e for e in ra.read_run_log() if e["run_id"] == run_id]
        ledger = [e for e in ra.read_ledger() if e.get("run_id") == run_id]
    return render_template("insights/refund_run.html", st=st, res=st.get("result"), log=log, ledger=ledger)


@bp.post("/refunds/resume/<run_id>")
def refund_resume(run_id):
    _checkpoint_or_404(run_id)
    with agent_paths():
        try:
            res = ra.resume_agent(run_id)
        except Exception as exc:
            current_app.logger.exception("resume failed")
            flash(f"Resume failed: {type(exc).__name__}: {exc}", "error")
            return redirect(url_for("insights.refund_run", run_id=run_id))
    flash(f"Resumed {run_id}: {res['status']}", "ok")
    return redirect(url_for("insights.refund_run", run_id=run_id))
