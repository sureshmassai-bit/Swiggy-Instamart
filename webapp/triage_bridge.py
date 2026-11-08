"""Classify a web-submitted ticket with the Part C pipeline: PII masked BEFORE the (mock) model sees it."""
import mock_llm
from src import triage as tr


def triage_text(text, is_premium):
    """Returns dict(category, triage_status, route, priority). Never raises; failures go to human review."""
    text = (text or "").strip()
    category = ""
    if not text:
        status = "skipped_empty"
    elif mock_llm.est_tokens(text) > tr.MAX_INPUT_TOKENS:
        status = "flagged_long"                                   # too long: not sent to the model
    else:
        safe, _ = tr.redact(text)
        messages = [{"role": "system", "content": tr.FEW}, {"role": "user", "content": safe}]
        try:
            response, _ = tr.LLMClient(sleep=lambda s: None).classify(messages)   # asserts no PII, retries transient errors
            category, status = tr.parse_category(response.choices[0].message.content or "")
        except mock_llm.TransientLLMError:
            status = "llm_failed"
    route, priority = tr.route_ticket(status, category, bool(is_premium))
    return {"category": category, "triage_status": status, "route": route, "priority": priority}
