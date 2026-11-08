"""Part D: refund tools and a bounded, checkpointed, resumable ReAct loop.

Safety design
* issue_refund / escalate_to_human re-check the order in SQLite themselves; they never trust the
  arguments the model supplied. An unauthorised call raises and is written to the ledger as `rejected`.
* Both side-effecting tools are idempotent per order: the ledger is consulted first, so repeating a
  call (for example after a crash and resume) can never pay twice.
* After every executed tool the loop writes an atomic checkpoint. `resume_agent` continues from it.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import time
import uuid

import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
LEDGER = OUT / "refund_ledger.jsonl"      # audit trail; module globals so tests/demos can redirect them
STATE = ROOT / "state"                    # checkpoints (git-ignored)

LATE_MINUTES = 45                         # refund policy: delivered in MORE than 45 minutes -> full refund
AUTO_REFUND_LIMIT_INR = 1000              # above this a human lead must approve
MAX_ITERATIONS, TOKEN_LIMIT, TIME_LIMIT_SECONDS = 6, 6000, 10
AGENT_SYSTEM = "AGENT: Use tools to verify orders and follow refund policy."


class RefundNotAuthorized(Exception):
    """Raised when a refund/escalation is not allowed by policy and order state."""


class SimulatedCrash(RuntimeError):
    """Raised on purpose by the crash/recovery demonstration."""


_rag = None


def _policy_index():
    global _rag
    _rag = _rag or PolicyRAG()
    return _rag


# ----------------------------------------------------------------------------- ledger
def _read_ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_ledger(entry):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _existing(tool, order_id):
    return next((e for e in _read_ledger() if e["tool"] == tool and e["order_id"] == order_id
                 and e["outcome"] in ("issued", "escalated")), None)


@contextmanager
def isolated(ledger, state):
    """Temporarily redirect the ledger and checkpoint directory (used by demos and tests)."""
    global LEDGER, STATE
    old = LEDGER, STATE
    LEDGER, STATE = Path(ledger), Path(state)
    try:
        yield
    finally:
        LEDGER, STATE = old


# ------------------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",
                          (order_id,)).fetchone()
    return {"error": "order not found"} if not row else dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    return {"results": _policy_index().retrieve(query, 3)}


def _refund_due(order_amount, delivery_minutes):
    """Policy: a completed order delivered in more than 45 minutes gets the full amount, otherwise 0."""
    if order_amount is None or float(order_amount) < 0:
        raise ValueError("order_amount must be a non-negative number")
    late = delivery_minutes is not None and float(delivery_minutes) > LATE_MINUTES
    return round(float(order_amount), 2) if late else 0.0


def calculate_refund(order_amount, delivery_minutes):
    return {"refund_inr": _refund_due(order_amount, delivery_minutes)}


def _authorize(order_id, refund_inr, escalation):
    """Re-derive eligibility from the database. Raises RefundNotAuthorized with the reason."""
    order = get_order(order_id)
    if "error" in order:
        raise RefundNotAuthorized("order not found")
    if order["status"] != "completed":
        raise RefundNotAuthorized(f"order status is {order['status']}, only completed orders can be refunded")
    due = _refund_due(order["order_amount"], order["delivery_minutes"])
    if due <= 0:
        raise RefundNotAuthorized("delivery was within 45 minutes; no refund due")
    if abs(float(refund_inr) - due) > 0.005:
        raise RefundNotAuthorized(f"requested INR {refund_inr} does not match the amount due INR {due}")
    if escalation and due <= AUTO_REFUND_LIMIT_INR:
        raise RefundNotAuthorized("refund is within the automatic limit; no human approval needed")
    if not escalation and due > AUTO_REFUND_LIMIT_INR:
        raise RefundNotAuthorized("refund above INR 1000 needs human approval; use escalate_to_human")
    return due


def _guarded(tool, outcome, order_id, refund_inr, run_id):
    escalation = tool == "escalate_to_human"
    prior = _existing(tool, order_id)
    if prior:                                           # idempotent: report, never repeat the side effect
        return {"refund_inr": prior["refund_inr"], "status": "already_" + outcome, "duplicate_prevented": True}
    try:
        due = _authorize(order_id, refund_inr, escalation)
    except RefundNotAuthorized as exc:
        _append_ledger({"run_id": run_id, "tool": tool, "order_id": order_id, "refund_inr": refund_inr,
                        "outcome": "rejected", "reason": str(exc)})
        raise
    _append_ledger({"run_id": run_id, "tool": tool, "order_id": order_id, "refund_inr": due, "outcome": outcome})
    return {"refund_inr": due, "status": outcome}


_CURRENT_RUN = {"id": None}


def issue_refund(order_id, refund_inr):
    return _guarded("issue_refund", "issued", order_id, refund_inr, _CURRENT_RUN["id"])


def escalate_to_human(order_id, refund_inr):
    return _guarded("escalate_to_human", "escalated", order_id, refund_inr, _CURRENT_RUN["id"])


TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


def _spec(name, description, **props):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
                           "required": list(props)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id.", order_id="string"),
    _spec("search_policy", "Search the policy documents.", query="string"),
    _spec("calculate_refund", "Refund due under the late-delivery policy.", order_amount="number", delivery_minutes="number"),
    _spec("issue_refund", "Issue an automatic refund (<= INR 1000, completed and late orders only).", order_id="string", refund_inr="number"),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human lead.", order_id="string", refund_inr="number"),
]


# ------------------------------------------------------------------------ checkpoints
def _ckpt_path(run_id):
    return STATE / f"{run_id}.json"


def _save_checkpoint(st):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = _ckpt_path(st["run_id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(st, default=str), encoding="utf-8")
    os.replace(tmp, _ckpt_path(st["run_id"]))          # atomic: a crash never leaves a half-written file


def load_checkpoint(run_id):
    return json.loads(_ckpt_path(run_id).read_text(encoding="utf-8"))


def _result(st, status, final):
    refund = next((s["observation"].get("refund_inr") for s in reversed(st["steps"])
                   if isinstance(s["observation"], dict) and "refund_inr" in s["observation"]), None)
    res = {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final,
           "refund_inr": refund, "steps": st["steps"], "tools_executed": list(st["tools_executed"]),
           "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "elapsed_seconds": round(st["elapsed"], 3)}
    st["status"], st["result"] = "done", res
    _save_checkpoint(st)
    return res


# ---------------------------------------------------------------------------- the loop
def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after, crash_stage):
    _CURRENT_RUN["id"] = st["run_id"]
    t0, elapsed0 = time.monotonic(), st["elapsed"]
    while st["iteration"] < max_iterations:
        st["elapsed"] = elapsed0 + time.monotonic() - t0
        if st["elapsed"] >= time_limit_seconds:
            return _result(st, "FALLBACK_TIMEOUT", "FALLBACK_TIMEOUT: hand off to a human agent.")
        response = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["iteration"] += 1
        st["tokens_used"] += response.usage.total_tokens
        if st["tokens_used"] > token_limit:
            return _result(st, "FALLBACK_TOKEN_LIMIT", "FALLBACK_TOKEN_LIMIT: hand off to a human agent.")
        msg = response.choices[0].message
        if not msg.tool_calls:                                   # final answer
            content = msg.content or ""
            return _result(st, content.split(":", 1)[0] or "UNKNOWN", content)
        tc = msg.tool_calls[0]
        call = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        try:
            if tc.name not in TOOL_MAP:
                raise KeyError(f"unknown tool {tc.name}")
            observation = TOOL_MAP[tc.name](**tc.arguments)      # side effect happens here
        except RefundNotAuthorized as exc:
            st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": {"error": str(exc)}})
            return _result(st, "BLOCKED_UNAUTHORIZED", f"BLOCKED_UNAUTHORIZED: {exc}. Handing off to a human agent.")
        except (KeyError, TypeError, ValueError) as exc:
            st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": {"error": str(exc)}})
            return _result(st, "FALLBACK_TOOL_ERROR", f"FALLBACK_TOOL_ERROR: {exc}. Handing off to a human agent.")
        st["tools_executed"].append(tc.name)
        if crash_after == len(st["tools_executed"]) and crash_stage == "before_checkpoint":
            raise SimulatedCrash("crash after the side effect but BEFORE the checkpoint was written")
        st["messages"] += [{"role": "assistant", "content": None, "tool_calls": [call]},
                           {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(observation)}]
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        _save_checkpoint(st)
        if crash_after == len(st["tools_executed"]) and crash_stage == "after_checkpoint":
            raise SimulatedCrash(f"crash after tool #{crash_after} (checkpoint saved)")
    return _result(st, "FALLBACK_MAX_ITERATIONS", "FALLBACK_MAX_ITERATIONS: hand off to a human agent.")


def run_agent(user_message, scenario_id="demo", max_iterations=MAX_ITERATIONS, token_limit=TOKEN_LIMIT,
              time_limit_seconds=TIME_LIMIT_SECONDS, run_id=None, crash_after=None, crash_stage="after_checkpoint"):
    run_id = run_id or str(uuid.uuid4())
    st = {"run_id": run_id, "scenario_id": scenario_id, "status": "in_progress", "user_message": user_message,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "iteration": 0, "llm_calls": 0, "tools_executed": [], "steps": [], "elapsed": 0.0,
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    _save_checkpoint(st)
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after, crash_stage)


def resume_agent(run_id, crash_after=None, crash_stage="after_checkpoint"):
    """Continue an interrupted run from its checkpoint. A finished run just returns its stored result."""
    st = load_checkpoint(run_id)
    if st["status"] == "done":
        return st["result"]
    lim = st["limits"]
    return _loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"], crash_after, crash_stage)
