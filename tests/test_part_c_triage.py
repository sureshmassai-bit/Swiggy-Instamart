"""Part C: PII masking, guardrails, retries, budgets and routing."""
import re

import pandas as pd
import pytest

import mock_llm
from src import triage as tr

NO_SLEEP = lambda s: None


@pytest.fixture
def capture(monkeypatch):
    """Record every message list actually handed to the model."""
    prompts, real = [], mock_llm.chat

    def spy(messages, tools=None):
        prompts.append(messages)
        return real(messages, tools)
    monkeypatch.setattr(mock_llm, "chat", spy)
    return prompts


def test_redaction_masks_all_three_pii_types():
    text = "Charged twice for ORD-123456. Email jo.hn+x@example.com or call 9876543210."
    safe, counts = tr.redact(text)
    assert safe == "Charged twice for [ORDER_ID]. Email [EMAIL] or call [PHONE]."
    assert counts == {"email": 1, "phone": 1, "order_id": 1}
    # An email containing digits must be consumed as an email, not leave a phone-like fragment.
    assert tr.redact("write to user9876543210@example.com")[0] == "write to [EMAIL]"


def test_assert_no_pii_rejects_any_message():
    tr.assert_no_pii([{"role": "system", "content": tr.FEW}, {"role": "user", "content": "clean text"}])
    for bad in ["mail a@b.com", "call 9876543210", "ORD-123456"]:
        with pytest.raises(AssertionError):
            tr.assert_no_pii([{"role": "user", "content": bad}])
        with pytest.raises(AssertionError):
            tr.assert_no_pii([{"role": "system", "content": bad}, {"role": "user", "content": "ok"}])


def test_no_pii_reaches_the_model_and_guarded_tickets_make_no_call(capture):
    result, summary = tr.run_pipeline(tr.FEW, sleep=NO_SLEEP)
    raw = pd.read_csv(tr.RAW / "tickets.csv", keep_default_na=False)
    assert len(capture) == summary["model_calls"] == 175           # 156 tickets + 19 retried attempts
    for messages in capture:
        assert len(messages) == 2 and "CLASSIFY" in messages[0]["content"].upper()
        for m in messages:
            for _, pat, _ in tr.PATTERNS:
                assert not pat.search(m["content"])
    sent = " ".join(m["content"] for msgs in capture for m in msgs)
    for text in raw.text:                                          # no raw email/phone/order id appears anywhere
        for _, pat, _ in tr.PATTERNS:
            for hit in pat.findall(text):
                assert hit not in sent
    guarded = result[result.status.isin(["skipped_empty", "flagged_long"])]
    assert len(guarded) == 8 and (guarded.attempts == 0).all() and guarded.human_handoff.all()
    assert not any(not m["content"].strip() for msgs in capture for m in msgs[1:])


def test_reference_counts_and_retries():
    result, s = tr.run_pipeline(tr.FEW, sleep=NO_SLEEP)
    assert len(result) == 164
    assert s["status_counts"]["classified_json"] == 135
    assert s["status_counts"]["classified_keyword"] == 12
    assert s["status_counts"]["flagged_invalid_output"] == 9
    assert s["transient_failures"] == 19 and s["retries"] == 19 and s["unrecovered_failures"] == 0
    assert s["tokens"] > 0 and s["estimated_cost_usd"] < s["cost_budget_usd"]
    tr.validate_results(result)


def test_few_shot_beats_zero_shot():
    _, few = tr.run_pipeline(tr.FEW, sleep=NO_SLEEP)
    _, zero = tr.run_pipeline(tr.ZERO, sleep=NO_SLEEP)
    assert few["accuracy_of_sent"] > zero["accuracy_of_sent"]


def test_backoff_grows_exponentially_with_jitter():
    waits = []
    client = tr.LLMClient(sleep=waits.append)
    calls = {"n": 0}
    real = mock_llm.chat

    def flaky(messages, tools=None):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise mock_llm.TransientLLMError("503")
        return real(messages, tools)
    mock_llm.chat, saved = flaky, mock_llm.chat
    try:
        mock_llm.reset()
        _, attempt = client.classify([{"role": "system", "content": tr.ZERO}, {"role": "user", "content": "app crash"}])
    finally:
        mock_llm.chat = saved
    assert attempt == 3 and len(waits) == 3
    for i, w in enumerate(waits):
        assert tr.BASE_DELAY * 2 ** i * 0.9 <= w <= tr.BASE_DELAY * 2 ** i * 1.15


def test_unrecoverable_failure_goes_to_human(monkeypatch):
    monkeypatch.setattr(mock_llm, "chat", lambda m, t=None: (_ for _ in ()).throw(mock_llm.TransientLLMError("down")))
    result, s = tr.run_pipeline(tr.FEW, sleep=NO_SLEEP)
    assert (result.status == "llm_failed").sum() == 156 and s["unrecovered_failures"] == 156
    assert set(result.route) == {"human_review"}


def test_budget_stops_calls_and_hands_off():
    result, s = tr.run_pipeline(tr.FEW, token_budget=1000, sleep=NO_SLEEP)
    assert s["tokens"] <= 1000 and (result.status == "budget_exhausted").sum() > 100
    ex = result[result.status == "budget_exhausted"]
    assert (ex.attempts == 0).all() and set(ex.route) == {"human_review"}
    _, cheap = tr.run_pipeline(tr.FEW, cost_budget_usd=0.0001, sleep=NO_SLEEP)       # cost cap works independently
    assert cheap["estimated_cost_usd"] <= 0.0001


@pytest.mark.parametrize("status,cat,premium,route,prio", [
    ("classified_json", "fraud", True, "security_team", "HIGH"),        # fraud overrides premium
    ("classified_json", "billing", True, "premium_desk", "HIGH"),
    ("classified_keyword", "delivery", False, "support_queue", "NORMAL"),
    ("flagged_invalid_output", "", True, "human_review", "NORMAL"),     # guard outcomes always go to a human
    ("skipped_empty", "", False, "human_review", "NORMAL"),
    ("budget_exhausted", "", True, "human_review", "NORMAL"),
])
def test_routing_rules(status, cat, premium, route, prio):
    assert tr.route_ticket(status, cat, premium) == (route, prio)


def test_parse_category_validation():
    assert tr.parse_category('{"category":"billing","reason":"x"}') == ("billing", "classified_json")
    assert tr.parse_category('{"category":"shipping","reason":"x"}') == ("", "flagged_invalid_output")
    assert tr.parse_category("looks like a fraud problem") == ("fraud", "classified_keyword")
    assert tr.parse_category("billing or delivery?") == ("", "flagged_invalid_output")     # ambiguous -> human
    assert tr.parse_category("123") == ("", "flagged_invalid_output")                      # JSON but not an object


def test_missing_segments_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Part B"):
        tr.run_pipeline(sleep=NO_SLEEP)


def test_saved_triage_results():
    df = pd.read_csv(tr.OUT / "triage_results.csv", keep_default_na=False)
    assert len(df) == 164 and df.ticket_id.is_unique
    assert not df.apply(lambda r: re.search(r"@|ORD-\d{6}", str(r.handoff_reason)), axis=1).any()
    assert {"status", "category", "priority", "route", "human_handoff", "tokens_used", "attempts"} <= set(df.columns)
