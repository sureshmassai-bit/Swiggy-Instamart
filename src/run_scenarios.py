"""Part D: run the seven refund scenarios and the limit / crash-recovery demonstrations.

Run from the repository root (needs data/instamart.db from Part A):
    python -m src.run_scenarios

Scenario orders are derived from the cleaned order table. Choosing the lowest order_id that
matches each description is this project's assumption, because the original brief was not
available in the workspace; the expected amounts (128.25, 135.00, 1999.00) are what identify them.
"""
from pathlib import Path
import json
import shutil
import sqlite3

import pandas as pd

from src import refund_agent as ra

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEMO_DIR = OUT / "agent_demo"

# scenario -> (expected status, expected refund_inr, expected tool sequence)
SEQ_REFUND = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
EXPECTED = {
    "S1": ("REFUND", 128.25, SEQ_REFUND),
    "S2": ("REFUND", 135.00, SEQ_REFUND),
    "S3": ("ESCALATE", 1999.00, ["get_order", "search_policy", "calculate_refund", "escalate_to_human"]),
    "S4": ("DENIED", None, ["get_order", "search_policy"]),
    "S5": ("DENIED", 0.0, ["get_order", "search_policy", "calculate_refund"]),
    "S6": ("NOT_FOUND", None, ["get_order"]),
    "S7": ("FALLBACK_MAX_ITERATIONS", None, ["search_policy"] * 6),
}


def build_scenarios():
    """Return {scenario_id: (description, user_message)} using orders found in the database."""
    with sqlite3.connect(ra.DB) as con:
        orders = pd.read_sql_query("SELECT * FROM orders ORDER BY order_id", con)
    late = orders[(orders.status == "completed") & (orders.delivery_minutes > ra.LATE_MINUTES)]

    def first(df, what):
        if df.empty:
            raise LookupError(f"no order found for scenario: {what}")
        return df.iloc[0]

    s1 = first(late[late.order_amount.round(2) == 128.25], "late completed order of INR 128.25")
    s2 = first(late[late.order_amount.round(2) == 135.00], "late completed order of INR 135.00")
    s3 = first(late[late.order_amount.round(2) == 1999.00], "late completed order of INR 1999.00")
    s4 = first(orders[orders.status == "cancelled"], "cancelled order")
    s5 = first(orders[(orders.status == "completed") & (orders.delivery_minutes <= ra.LATE_MINUTES)], "on-time completed order")
    s7 = first(late, "any late completed order")
    ask = lambda oid: f"Customer reports order {oid} arrived very late and asks for a refund."
    return {
        "S1": ("late order INR 128.25, auto refund", ask(s1.order_id)),
        "S2": ("late order INR 135.00, auto refund", ask(s2.order_id)),
        "S3": ("late order INR 1999.00, above limit -> human", ask(s3.order_id)),
        "S4": ("cancelled order -> denied", ask(s4.order_id)),
        "S5": ("completed, delivered on time -> denied", ask(s5.order_id)),
        "S6": ("order id does not exist", ask("ORD-999999")),
        "S7": ("agent loops -> max iterations", ask(s7.order_id) + " [LOOP]"),
    }


def run_scenarios():
    scenarios = build_scenarios()
    rows = []
    for sid, (desc, message) in scenarios.items():
        res = ra.run_agent(message, scenario_id=sid, run_id=f"{sid}-main")
        exp_status, exp_amount, exp_seq = EXPECTED[sid]
        ok = (res["status"] == exp_status and res["tools_executed"] == exp_seq
              and (exp_amount is None or res["refund_inr"] == exp_amount))
        rows.append({"scenario": sid, "description": desc, "expected_status": exp_status, "status": res["status"],
                     "expected_refund_inr": exp_amount, "refund_inr": res["refund_inr"], "llm_calls": res["llm_calls"],
                     "tokens_used": res["tokens_used"], "tools": " > ".join(res["tools_executed"]), "match": ok})
    return pd.DataFrame(rows), scenarios


def run_demos(s1_message):
    """Limit and crash/recovery demos. They use a private ledger/state so they never touch the main audit trail."""
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    demo = {}
    with ra.isolated(DEMO_DIR / "ledger.jsonl", DEMO_DIR / "state"):
        tok = ra.run_agent(s1_message, "S1-token-limit", token_limit=200, run_id="demo-token")
        demo["token_limit_200"] = {"status": tok["status"], "tokens_used": tok["tokens_used"], "tools": tok["tools_executed"],
                                   "refunds_issued": len([e for e in ra._read_ledger() if e["outcome"] == "issued"])}
        to = ra.run_agent(s1_message, "S1-timeout", time_limit_seconds=0, run_id="demo-timeout")
        demo["timeout_0"] = {"status": to["status"], "llm_calls": to["llm_calls"], "tools": to["tools_executed"]}

        # Crash after the 2nd tool call (checkpoint saved), then resume.
        try:
            ra.run_agent(s1_message, "S1-crash", run_id="demo-crash", crash_after=2)
            raise AssertionError("the simulated crash did not happen")
        except ra.SimulatedCrash as exc:
            crashed = {"error": str(exc), "checkpoint_tools": ra.load_checkpoint("demo-crash")["tools_executed"],
                       "refunds_before_resume": len(ra._read_ledger())}
        res = ra.resume_agent("demo-crash")
        again = ra.resume_agent("demo-crash")             # resuming a finished run must be a no-op
        issues = [e for e in ra._read_ledger() if e["tool"] == "issue_refund" and e["outcome"] == "issued"]
        log = ra.read_run_log()
        demo["crash_resume_run_log"] = [{k: e[k] for k in ("event", "status", "tools_executed", "resumed")}
                                        for e in log if e["run_id"] == "demo-crash"]
        demo["crash_resume"] = {**crashed, "status": res["status"], "refund_inr": res["refund_inr"],
                                "tools_executed_after_resume": res["tools_executed"], "llm_calls_total": res["llm_calls"],
                                "issue_refund_ledger_entries": len(issues), "second_resume_same_result": again == res,
                                "ledger_entries_after_second_resume": len(ra._read_ledger())}

        # Harder case: crash AFTER the refund was paid but BEFORE the checkpoint. The retried call must not pay twice.
        with ra.isolated(DEMO_DIR / "ledger2.jsonl", DEMO_DIR / "state2"):
            try:
                ra.run_agent(s1_message, "S1-crash2", run_id="demo-crash2", crash_after=4, crash_stage="before_checkpoint")
            except ra.SimulatedCrash:
                pass
            res2 = ra.resume_agent("demo-crash2")
            pays = [e for e in ra._read_ledger() if e["tool"] == "issue_refund" and e["outcome"] == "issued"]
            demo["crash_after_payment_before_checkpoint"] = {"status": res2["status"], "refund_inr": res2["refund_inr"],
                                                             "issue_refund_ledger_entries": len(pays),
                                                             "last_observation": res2["steps"][-1]["observation"]}
    return demo


def main():
    OUT.mkdir(exist_ok=True)
    for path in (ra.LEDGER, ra.RUN_LOG):     # regenerate the audit trail and run log from scratch (reproducible)
        if path.exists():
            path.unlink()
    for ckpt in ra.STATE.glob("S*-main.json") if ra.STATE.exists() else []:
        ckpt.unlink()
    df, scenarios = run_scenarios()
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    print(df.drop(columns="description").to_string(index=False))
    df.to_csv(OUT / "agent_scenarios.csv", index=False)
    s1_calls = int(df.loc[df.scenario == "S1", "llm_calls"].iloc[0])
    demo = run_demos(scenarios["S1"][1])
    demo["s1_model_calls"] = s1_calls
    print("\nS1 model calls:", s1_calls, "(expected 5)")
    print(json.dumps(demo, indent=2))
    (OUT / "agent_demos.json").write_text(json.dumps(demo, indent=2, default=str), encoding="utf-8")

    cr = demo["crash_resume"]
    checks = {
        "all seven scenarios match": bool(df.match.all()),
        "S1 uses exactly 5 model calls": s1_calls == 5,
        "token limit 200 -> FALLBACK_TOKEN_LIMIT with no refund": demo["token_limit_200"]["status"] == "FALLBACK_TOKEN_LIMIT"
                                                                  and demo["token_limit_200"]["refunds_issued"] == 0,
        "timeout 0 -> FALLBACK_TIMEOUT before any model call": demo["timeout_0"]["status"] == "FALLBACK_TIMEOUT"
                                                              and demo["timeout_0"]["llm_calls"] == 0,
        "crash after tool 2 left only 2 tools done and no refund": cr["checkpoint_tools"] == SEQ_REFUND[:2] and cr["refunds_before_resume"] == 0,
        "resumed S1 tool sequence is exactly get/search/calculate/issue": cr["tools_executed_after_resume"] == SEQ_REFUND,
        "resumed S1 refund is INR 128.25": cr["status"] == "REFUND" and cr["refund_inr"] == 128.25,
        "exactly one issue_refund ledger entry after resume": cr["issue_refund_ledger_entries"] == 1,
        "resuming a finished run adds no ledger entries": cr["second_resume_same_result"] and cr["ledger_entries_after_second_resume"] == 1,
        "crash between payment and checkpoint cannot double pay": demo["crash_after_payment_before_checkpoint"]["issue_refund_ledger_entries"] == 1,
    }
    log = ra.read_run_log()
    checks["agent_runs.jsonl has one 'finished' entry per scenario"] = sorted(
        e["scenario_id"] for e in log if e["event"] == "finished") == sorted(EXPECTED)
    print("\nVerification:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return all(checks.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
