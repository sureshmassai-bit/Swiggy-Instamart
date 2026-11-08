"""Admin analytics pages (Parts A-D in the web app), the agent run log, feature importance and the budget demo file."""
import json
import re

import pandas as pd
import pytest

import mock_llm
from src import triage as tr
from webapp.app import create_app
from webapp.config import ROOT, Config
from webapp.db import connect, hash_password, now

PAGES = ["/admin/insights", "/admin/insights/sql", "/admin/insights/data-quality", "/admin/insights/segments",
         "/admin/insights/segments?segment=Premium", "/admin/insights/triage", "/admin/insights/triage?route=security_team",
         "/admin/insights/policy", "/admin/insights/refunds", "/admin/insights/pca.png"]


@pytest.fixture
def env(tmp_path):
    """Analytics read the real outputs; anything the agent writes (ledger, log, checkpoints) goes to tmp_path."""
    cfg = Config(app_db=tmp_path / "app.db", excel_path=tmp_path / "r.xlsx", dump_dir=tmp_path / "dump",
                 credentials_dir=tmp_path / "cred", secret_key_path=tmp_path / "key", pw_iterations=1000,
                 ledger_path=tmp_path / "ledger.jsonl", run_log_path=tmp_path / "runs.jsonl", state_dir=tmp_path / "state")
    con = connect(cfg)
    cid = con.execute("SELECT customer_id FROM seg LIMIT 1").fetchone()[0]
    for name, role, c in [("admin", "admin", None), ("bob", "user", cid)]:
        con.execute("INSERT INTO users (username, role, customer_id, full_name, email, password_hash, created_at) VALUES (?,?,?,?,?,?,?)",
                    (name, role, c, name.title(), f"{name}@x.com", hash_password(cfg, f"{name}-Pass-123"), now()))
    con.commit()
    con.close()
    app = create_app(cfg)
    app.config["TESTING"] = True
    return app, cfg, tmp_path


def csrf_of(c, path):
    return re.search(r'name="csrf" value="([^"]+)"', c.get(path).get_data(as_text=True)).group(1)


def login(app, user):
    c = app.test_client()
    assert c.post("/login", data={"username": user, "password": f"{user}-Pass-123", "csrf": csrf_of(c, "/login")}).status_code == 302
    return c


def test_analytics_are_admin_only(env):
    app, *_ = env
    anon = app.test_client()
    assert anon.get("/admin/insights").status_code == 302
    bob = login(app, "bob")
    for p in PAGES:
        assert bob.get(p).status_code == 403, p
    assert bob.post("/admin/insights/refunds/run", data={"csrf": csrf_of(bob, "/home"), "order_id": "ORD-100048"}).status_code == 403


def test_every_page_renders_real_numbers(env):
    app, *_ = env
    c = login(app, "admin")
    for p in PAGES:
        assert c.get(p).status_code == 200, p
    home = c.get("/admin/insights").get_data(as_text=True)
    assert "12,309" in home and "₹2,998,668.80" in home                  # Part A checkpoint values, read live from SQLite
    dq = c.get("/admin/insights/data-quality").get_data(as_text=True)
    assert "ALL 12 CHECKPOINTS PASSED" in dq
    sql = c.get("/admin/insights/sql").get_data(as_text=True)
    assert all(f"<b>Q{i}</b>" in sql for i in range(1, 8)) and sql.count("<details") == 7
    seg = c.get("/admin/insights/segments").get_data(as_text=True)
    lapse = json.loads((ROOT / "outputs/lapse_metrics.json").read_text())
    assert f"{lapse['models'][lapse['selected_model']]['f1']:.3f}" in seg
    tri = c.get("/admin/insights/triage?route=security_team").get_data(as_text=True)
    n = (pd.read_csv(ROOT / "outputs/triage_results.csv").route == "security_team").sum()
    assert f"Results ({n})" in tri


def test_policy_question_is_grounded_and_unsupported_falls_back(env):
    app, *_ = env
    c = login(app, "admin")
    tok = csrf_of(c, "/admin/insights/policy")
    ok = c.post("/admin/insights/policy", data={"csrf": tok, "question": "How many working days does a refund take?"}).get_data(as_text=True)
    assert "5 working days" in ok and "grounded" in ok and "refund_policy.txt" in ok
    no = c.post("/admin/insights/policy", data={"csrf": tok, "question": "Who won the cricket world cup final?"}).get_data(as_text=True)
    assert "I could not find that information" in no
    bad = c.post("/admin/insights/policy", data={"csrf": tok, "question": "x"}).get_data(as_text=True)
    assert "3-300 characters" in bad


def test_policy_question_pii_is_masked_before_the_model(env, monkeypatch):
    app, *_ = env
    seen = []
    real = mock_llm.chat
    monkeypatch.setattr(mock_llm, "chat", lambda m, t=None: seen.append(m) or real(m, t))
    c = login(app, "admin")
    c.post("/admin/insights/policy", data={"csrf": csrf_of(c, "/admin/insights/policy"),
                                           "question": "Refund for ORD-100048, mail a@b.com or 9876543210?"})
    assert seen and all(not p.search(msg["content"]) for msgs in seen for msg in msgs for _, p, _ in tr.PATTERNS)


def test_refund_run_crash_resume_pays_once(env):
    app, cfg, tmp = env
    c = login(app, "admin")
    tok = csrf_of(c, "/admin/insights/refunds")
    r = c.post("/admin/insights/refunds/run", data={"csrf": tok, "order_id": "ord-100048", "crash_after": "2"})
    run_id = r.headers["Location"].rsplit("/", 1)[1]
    assert run_id.startswith("web-") and not cfg.ledger_path.exists()      # crashed before any money moved
    page = c.get(f"/admin/insights/refunds/run/{run_id}").get_data(as_text=True)
    assert "Resume from checkpoint" in page
    done = c.post(f"/admin/insights/refunds/resume/{run_id}", data={"csrf": tok}, follow_redirects=True).get_data(as_text=True)
    assert "REFUND" in done
    c.post("/admin/insights/refunds/run", data={"csrf": tok, "order_id": "ORD-100048"})     # same order again
    ledger = [json.loads(l) for l in cfg.ledger_path.read_text().splitlines()]
    assert [e["outcome"] for e in ledger] == ["issued"] and ledger[0]["refund_inr"] == 128.25
    log = [json.loads(l) for l in cfg.run_log_path.read_text().splitlines()]
    assert [e["event"] for e in log] == ["crashed", "resumed_finished", "finished"]
    assert (tmp / "state" / f"{run_id}.json").exists()
    assert not (ROOT / "state" / f"{run_id}.json").exists()               # the real state folder is untouched


@pytest.mark.parametrize("form,msg", [({"order_id": "100048"}, "ORD-123456"),
                                      ({"order_id": "ORD-100048", "token_limit": "999999"}, "token limit"),
                                      ({"order_id": "ORD-100048", "max_iterations": "abc"}, "max iterations")])
def test_refund_form_validation(env, form, msg):
    app, cfg, _ = env
    c = login(app, "admin")
    html = c.post("/admin/insights/refunds/run", data={"csrf": csrf_of(c, "/admin/insights/refunds"), **form},
                  follow_redirects=True).get_data(as_text=True)
    assert msg in html and not cfg.ledger_path.exists()


def test_refund_limits_from_the_form(env):
    app, cfg, _ = env
    c = login(app, "admin")
    tok = csrf_of(c, "/admin/insights/refunds")
    for form, status in [({"token_limit": "200"}, "FALLBACK_TOKEN_LIMIT"), ({"time_limit_seconds": "0"}, "FALLBACK_TIMEOUT"),
                         ({"max_iterations": "2"}, "FALLBACK_MAX_ITERATIONS")]:
        html = c.post("/admin/insights/refunds/run", data={"csrf": tok, "order_id": "ORD-100048", **form},
                      follow_redirects=True).get_data(as_text=True)
        assert status in html
    assert not cfg.ledger_path.exists()


def test_unknown_or_malicious_run_ids_404(env):
    app, *_ = env
    c = login(app, "admin")
    assert c.get("/admin/insights/refunds/run/does-not-exist").status_code == 404
    assert c.get("/admin/insights/refunds/run/..%2F..%2Fsecret").status_code == 404


def test_triage_trial_masks_pii(env):
    app, *_ = env
    c = login(app, "admin")
    html = c.post("/admin/insights/triage/try", data={"csrf": csrf_of(c, "/admin/insights/triage"),
                                                      "text": "Unauthorized transaction on ORD-100001, mail a@b.com"},
                  follow_redirects=True).get_data(as_text=True)
    assert "security_team" in html and "[ORDER_ID]" in html and "[EMAIL]" in html


# ------------------------------------------------------------------ pipeline outputs added in this revision
def test_agent_run_log_written_for_every_scenario():
    log = [json.loads(l) for l in (ROOT / "outputs/agent_runs.jsonl").read_text().splitlines()]
    finished = {e["scenario_id"]: e for e in log if e["event"] == "finished"}
    assert sorted(finished) == [f"S{i}" for i in range(1, 8)]
    assert finished["S1"]["status"] == "REFUND" and finished["S1"]["refund_inr"] == 128.25
    assert all("user_message" not in e and "messages" not in e for e in log)    # no free text in the log


def test_budget_demo_file():
    demo = pd.read_csv(ROOT / "outputs/triage_budget_demo.csv", keep_default_na=False)
    assert len(demo) == 164 and (demo.status == "budget_exhausted").sum() > 0
    assert demo.tokens_used.sum() <= tr.BUDGET_DEMO_TOKENS
    assert (demo.loc[demo.status == "budget_exhausted", "route"] == "human_review").all()


def test_feature_importance_file():
    imp = pd.read_csv(ROOT / "outputs/feature_importance.csv")
    from src.customer_models import FEATURES
    assert sorted(imp.feature) == sorted(FEATURES)
    assert abs(imp.model_importance_share.sum() - 1) < 1e-3
