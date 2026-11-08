"""Part C: privacy-first support-ticket triage with retries, budgets, validation and routing.

Run from the repository root (needs outputs/customer_segments.csv from Part B):
    python -m src.triage
"""
from pathlib import Path
import json
import random
import re
import time

import pandas as pd

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_INPUT_TOKENS = 400          # longer tickets are never sent to the model
MAX_RETRIES = 3                 # 1 attempt + 3 retries
BASE_DELAY = 0.05               # seconds; doubled each retry, with +-jitter
DEFAULT_TOKEN_BUDGET = 100_000
# Illustrative price (the mock is free). Assumption: USD 0.50 per 1M tokens.
PRICE_PER_1K_TOKENS_USD = 0.0005
DEFAULT_COST_BUDGET_USD = 0.05
COMPLETION_HEADROOM = 60        # tokens reserved per call when checking the budget up front
BUDGET_DEMO_TOKENS = 1000       # deliberately tiny budget for the enforcement demo (outputs/triage_budget_demo.csv)

# Ordered: email first, so the digits in an address are not mistaken for a phone number.
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO = ("CLASSIFY. Allowed categories: billing, delivery, technical, fraud. "
        "Return JSON only in the form {\"category\":\"...\",\"reason\":\"...\"}.")
FEW = ZERO + """
EXAMPLES:
Text: Unauthorized transaction, I suspect fraud. -> {"category":"fraud","reason":"unauthorized transaction"}
Text: My order arrived late. -> {"category":"delivery","reason":"late delivery"}
Text: App crashes at login. -> {"category":"technical","reason":"app crash"}
"""

GUARD_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "llm_failed"}
HANDOFF_REASON = {"skipped_empty": "empty ticket: nothing to classify",
                  "flagged_long": "ticket exceeds input token limit; not sent to model",
                  "flagged_invalid_output": "model output not a valid category",
                  "budget_exhausted": "token/cost budget exhausted",
                  "llm_failed": "model unavailable after retries"}
VALID_ROUTES = {"human_review", "security_team", "premium_desk", "support_queue"}


# ----------------------------------------------------------------------------- privacy
def redact(text):
    """Mask email -> phone -> order id. Returns (masked_text, {pattern: count})."""
    counts = {}
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts


def assert_no_pii(messages):
    """The required guard: PII must be absent from EVERY message (system and user) sent to the model."""
    for m in messages:
        for name, pat, _ in PATTERNS:
            assert not pat.search(m["content"]), f"PII ({name}) found in a prompt; refusing to call the model"


# ---------------------------------------------------------------------------- reliability
class LLMClient:
    """Wraps mock_llm.chat with the PII assertion, retries (exponential backoff + jitter) and counters."""

    def __init__(self, sleep=time.sleep, seed=42, verbose=False):
        self.sleep, self.rng, self.verbose = sleep, random.Random(seed), verbose
        self.stats = {"model_calls": 0, "retries": 0, "transient_failures": 0, "unrecovered_failures": 0,
                      "tokens": 0, "backoff_seconds": 0.0}

    def classify(self, messages, ticket_id=""):
        assert_no_pii(messages)                                   # checked before EVERY call, incl. retries
        for attempt in range(MAX_RETRIES + 1):
            self.stats["model_calls"] += 1
            try:
                response = mock_llm.chat(messages)
                self.stats["tokens"] += response.usage.total_tokens
                return response, attempt
            except mock_llm.TransientLLMError:
                self.stats["transient_failures"] += 1
                if attempt == MAX_RETRIES:
                    self.stats["unrecovered_failures"] += 1
                    raise
                self.stats["retries"] += 1
                wait = BASE_DELAY * (2 ** attempt) * self.rng.uniform(0.9, 1.15)   # backoff + jitter
                self.stats["backoff_seconds"] += wait
                if self.verbose:
                    print(f"retry ticket={ticket_id} attempt={attempt + 1} wait={wait:.3f}s")
                self.sleep(wait)


def parse_category(content):
    """Return (category, status). JSON first; otherwise accept exactly one allowed word (re.findall)."""
    try:
        obj = json.loads(content)
        cat = obj.get("category") if isinstance(obj, dict) else None
        return (cat, "classified_json") if cat in CATEGORIES else ("", "flagged_invalid_output")
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", content.lower()) if w in CATEGORIES}
        return (found.pop(), "classified_keyword") if len(found) == 1 else ("", "flagged_invalid_output")


# -------------------------------------------------------------------------------- routing
def route_ticket(status, category, is_premium):
    """Routing rules. Returns (route, priority).
    1. any guardrail/invalid/budget outcome -> human_review   2. fraud -> security_team (HIGH)
    3. premium customer -> premium_desk (HIGH)                4. otherwise -> support_queue
    """
    if status in GUARD_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def load_segments():
    path = OUT / "customer_segments.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Part B first: python -m src.customer_models")
    return pd.read_csv(path).set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")


# ------------------------------------------------------------------------------ pipeline
def run_pipeline(prompt=FEW, token_budget=DEFAULT_TOKEN_BUDGET, cost_budget_usd=DEFAULT_COST_BUDGET_USD,
                 save_path=None, sleep=time.sleep, verbose=False):
    """Triage all tickets. Returns (results_df, summary). Writes CSV only when save_path is given."""
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    segments = load_segments()
    mock_llm.reset()
    client = LLMClient(sleep=sleep, verbose=verbose)
    redaction_totals = {name: 0 for name, _, _ in PATTERNS}
    rows = []

    for t in tickets.itertuples():
        text = str(t.text)
        status, category, tokens, attempts = "", "", 0, 0
        pii = {name: 0 for name, _, _ in PATTERNS}
        if not text.strip():
            status = "skipped_empty"                                  # no model call
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"                                   # no model call
        else:
            safe, pii = redact(text)
            for k, v in pii.items():
                redaction_totals[k] += v
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": safe}]
            projected = client.stats["tokens"] + sum(mock_llm.est_tokens(m["content"]) for m in messages) + COMPLETION_HEADROOM
            if projected > token_budget or projected / 1000 * PRICE_PER_1K_TOKENS_USD > cost_budget_usd:
                status = "budget_exhausted"                           # no model call
            else:
                try:
                    before = client.stats["tokens"]
                    response, attempt = client.classify(messages, t.ticket_id)
                    attempts = attempt + 1
                    tokens = client.stats["tokens"] - before
                    category, status = parse_category(response.choices[0].message.content or "")
                except mock_llm.TransientLLMError:
                    status = "llm_failed"
        seg = segments.get(t.customer_id, {"segment_name": "unsegmented", "is_premium": False})
        route, priority = route_ticket(status, category, bool(seg["is_premium"]))
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status, "category": category,
                     "segment_name": seg["segment_name"], "is_premium": bool(seg["is_premium"]),
                     "priority": priority, "route": route, "human_handoff": route == "human_review",
                     "handoff_reason": HANDOFF_REASON.get(status, ""), "attempts": attempts,
                     "tokens_used": tokens, "pii_email": pii["email"], "pii_phone": pii["phone"],
                     "pii_order_id": pii["order_id"], "prompt_variant": "few_shot" if prompt == FEW else "zero_shot"})

    result = pd.DataFrame(rows)
    sent = result[result.attempts > 0].merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
    correct = int((sent.category == sent.true_category).sum())
    classified = sent[sent.category != ""]
    stats = dict(client.stats)
    summary = {"prompt_variant": result.prompt_variant.iloc[0], "tickets": len(result),
               "status_counts": result.status.value_counts().to_dict(),
               "route_counts": result.route.value_counts().to_dict(),
               "tickets_sent_to_model": len(sent), "correct": correct,
               "accuracy_of_sent": round(correct / len(sent), 4) if len(sent) else None,
               "accuracy_of_classified": round(correct / len(classified), 4) if len(classified) else None,
               **stats, "backoff_seconds": round(stats["backoff_seconds"], 3),
               "estimated_cost_usd": round(stats["tokens"] / 1000 * PRICE_PER_1K_TOKENS_USD, 6),
               "token_budget": token_budget, "cost_budget_usd": cost_budget_usd,
               "redactions": redaction_totals, "human_handoffs": int(result.human_handoff.sum())}
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        result.to_csv(save_path, index=False)
    return result, summary


def validate_results(result):
    """Rule validation: re-derive routing from first principles and check structural guarantees."""
    assert len(result) == 164, f"expected 164 rows, got {len(result)}"
    assert result.ticket_id.is_unique
    assert set(result.route) <= VALID_ROUTES
    assert set(result.loc[result.category != "", "category"]) <= CATEGORIES
    for r in result.itertuples():
        assert (r.route, r.priority) == route_ticket(r.status, r.category, r.is_premium), f"bad routing {r.ticket_id}"
        assert r.human_handoff == (r.route == "human_review")
        if r.status in {"skipped_empty", "flagged_long", "budget_exhausted"}:
            assert r.attempts == 0 and r.tokens_used == 0, "guarded ticket must not reach the model"
    return True


def main():
    OUT.mkdir(exist_ok=True)
    few, few_s = run_pipeline(FEW, save_path=OUT / "triage_results.csv")
    zero, zero_s = run_pipeline(ZERO, save_path=OUT / "triage_zero_shot.csv")
    validate_results(few)
    validate_results(zero)
    # Budget enforcement demo: a deliberately tiny token budget must stop calls and hand off to humans.
    demo, demo_s = run_pipeline(FEW, token_budget=BUDGET_DEMO_TOKENS, save_path=OUT / "triage_budget_demo.csv")
    validate_results(demo)
    assert demo_s["status_counts"].get("budget_exhausted", 0) > 0, "budget demo did not stop any model call"
    assert demo_s["tokens"] <= BUDGET_DEMO_TOKENS, "budget demo spent more than its token budget"
    (OUT / "triage_summary.json").write_text(json.dumps({"few_shot": few_s, "zero_shot": zero_s, "budget_demo_1000_tokens": demo_s},
                                                        indent=2, default=int), encoding="utf-8")
    for label, s in [("few-shot", few_s), ("zero-shot", zero_s), ("budget demo (1000-token cap)", demo_s)]:
        print(f"\n{label}: {json.dumps(s, default=int)}")
    print("\nRows in triage_results.csv:", len(few), "| rule validation: PASS")


if __name__ == "__main__":
    main()
