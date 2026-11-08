"""Part D (agent): refund authorisation, idempotency, scenarios, limits and crash recovery."""
import json
import re
from pathlib import Path

import pytest

import mock_llm
from src import run_scenarios as rs

SEQ = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
S = rs.build_scenarios()
ORDER = lambda sid: re.search(r"ORD-\d{6}", S[sid][1]).group(0)


def test_tool_map_has_the_five_tools(agent_sandbox):
    assert set(agent_sandbox.TOOL_MAP) == {"get_order", "search_policy", "calculate_refund", "issue_refund", "escalate_to_human"}
    assert {t["name"] for t in agent_sandbox.TOOLS_SPEC} == set(agent_sandbox.TOOL_MAP)


def test_calculate_refund_policy(agent_sandbox):
    calc = agent_sandbox.calculate_refund
    assert calc(128.25, 65) == {"refund_inr": 128.25}
    assert calc(100, 45) == {"refund_inr": 0.0}              # exactly 45 minutes is NOT late
    assert calc(100, 46) == {"refund_inr": 100.0}
    assert calc(100, None) == {"refund_inr": 0.0}
    with pytest.raises(ValueError):
        calc(-5, 60)


@pytest.mark.parametrize("sid,tool,amount", [
    ("S4", "issue_refund", 100.0),         # cancelled order
    ("S5", "issue_refund", 100.0),         # completed but on time
    ("S1", "issue_refund", 100.0),         # late, but amount differs from what is due
    ("S3", "issue_refund", 1999.0),        # above INR 1000 needs a human
    ("S1", "escalate_to_human", 128.25),   # within auto limit: no need to escalate
    ("S6", "issue_refund", 50.0),          # unknown order
])
def test_unauthorised_refunds_are_blocked_and_audited(agent_sandbox, sid, tool, amount):
    with pytest.raises(agent_sandbox.RefundNotAuthorized):
        getattr(agent_sandbox, tool)(ORDER(sid), amount)
    entries = agent_sandbox._read_ledger()
    assert [e["outcome"] for e in entries] == ["rejected"] and entries[0]["reason"]


def test_issue_refund_is_idempotent(agent_sandbox):
    first = agent_sandbox.issue_refund(ORDER("S1"), 128.25)
    again = agent_sandbox.issue_refund(ORDER("S1"), 128.25)
    assert first == {"refund_inr": 128.25, "status": "issued"}
    assert again["duplicate_prevented"] and again["refund_inr"] == 128.25
    assert [e["outcome"] for e in agent_sandbox._read_ledger()] == ["issued"]
    agent_sandbox.escalate_to_human(ORDER("S3"), 1999.0)
    agent_sandbox.escalate_to_human(ORDER("S3"), 1999.0)
    assert sum(e["tool"] == "escalate_to_human" for e in agent_sandbox._read_ledger()) == 1


def test_seven_scenarios(agent_sandbox):
    df, _ = rs.run_scenarios()
    assert df.match.all(), df.to_string()
    got = df.set_index("scenario")
    assert got.loc["S1", "refund_inr"] == 128.25 and got.loc["S2", "refund_inr"] == 135.0
    assert got.loc["S3", "status"] == "ESCALATE" and got.loc["S3", "refund_inr"] == 1999.0
    assert got.loc["S4", "status"] == got.loc["S5", "status"] == "DENIED"
    assert got.loc["S6", "status"] == "NOT_FOUND" and got.loc["S7", "status"] == "FALLBACK_MAX_ITERATIONS"
    assert got.loc["S1", "llm_calls"] == 5
    paid = [e["order_id"] for e in agent_sandbox._read_ledger() if e["outcome"] == "issued"]
    assert paid == [ORDER("S1"), ORDER("S2")], "only S1 and S2 may pay; denied/escalated/looping runs must not"


def test_token_limit_and_timeout(agent_sandbox):
    tok = agent_sandbox.run_agent(S["S1"][1], run_id="tok", token_limit=200)
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and "issue_refund" not in tok["tools_executed"]
    to = agent_sandbox.run_agent(S["S1"][1], run_id="to", time_limit_seconds=0)
    assert to["status"] == "FALLBACK_TIMEOUT" and to["llm_calls"] == 0
    assert agent_sandbox._read_ledger() == []


def test_crash_after_second_tool_then_resume(agent_sandbox):
    ra = agent_sandbox
    with pytest.raises(ra.SimulatedCrash):
        ra.run_agent(S["S1"][1], "S1", run_id="r1", crash_after=2)
    ck = ra.load_checkpoint("r1")
    assert ck["status"] == "in_progress" and ck["tools_executed"] == SEQ[:2] and ra._read_ledger() == []
    res = ra.resume_agent("r1")
    assert res["status"] == "REFUND" and res["refund_inr"] == 128.25
    assert res["tools_executed"] == SEQ and res["llm_calls"] == 5
    assert len([e for e in ra._read_ledger() if e["outcome"] == "issued"]) == 1
    assert ra.resume_agent("r1") == res and len(ra._read_ledger()) == 1          # no-op on a finished run


def test_crash_after_payment_before_checkpoint_cannot_double_pay(agent_sandbox):
    ra = agent_sandbox
    with pytest.raises(ra.SimulatedCrash):
        ra.run_agent(S["S1"][1], "S1", run_id="r2", crash_after=4, crash_stage="before_checkpoint")
    assert len(ra._read_ledger()) == 1                                           # money moved...
    assert ra.load_checkpoint("r2")["tools_executed"] == SEQ[:3]                 # ...but the checkpoint missed it
    res = ra.resume_agent("r2")
    assert res["status"] == "REFUND" and res["refund_inr"] == 128.25
    assert res["steps"][-1]["observation"]["duplicate_prevented"] is True
    assert len([e for e in ra._read_ledger() if e["outcome"] == "issued"]) == 1


def test_checkpoint_write_is_atomic(agent_sandbox):
    ra = agent_sandbox
    ra.run_agent(S["S1"][1], run_id="atom")
    assert not list(ra.STATE.glob("*.tmp"))
    assert json.loads((ra.STATE / "atom.json").read_text())["status"] == "done"


def test_model_cannot_trigger_unauthorised_refund(agent_sandbox, monkeypatch):
    """Even if the (mock) model misbehaves and asks to refund a cancelled order, the tool refuses."""
    ra = agent_sandbox
    oid = ORDER("S4")

    def rogue(messages, tools=None):
        return mock_llm._resp(None, [{"id": "c1", "name": "issue_refund", "arguments": {"order_id": oid, "refund_inr": 500.0}}],
                              "tool_calls", messages)
    monkeypatch.setattr(mock_llm, "chat", rogue)
    res = ra.run_agent(f"refund {oid}", run_id="rogue")
    assert res["status"] == "BLOCKED_UNAUTHORIZED"
    assert not [e for e in ra._read_ledger() if e["outcome"] == "issued"]


def test_unknown_tool_and_bad_arguments_fall_back(agent_sandbox, monkeypatch):
    for call in [{"id": "c", "name": "wire_money", "arguments": {}},
                 {"id": "c", "name": "get_order", "arguments": {"wrong": 1}}]:
        monkeypatch.setattr(mock_llm, "chat", lambda m, t=None, call=call: mock_llm._resp(None, [call], "tool_calls", m))
        assert agent_sandbox.run_agent("x", run_id="bad")["status"] == "FALLBACK_TOOL_ERROR"


def test_react_explanation_within_250_words():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- react-start -->(.*?)<!-- react-end -->", text, re.S)
    assert m, "README must contain the ReAct vs plan-execute section between the markers"
    assert 0 < len(m.group(1).split()) <= 250
