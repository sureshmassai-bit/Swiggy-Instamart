"""First-line support bot for the help chat. Fully offline: no model, no network, no API key.

It holds a short conversation instead of passing every message to a person:
  * it looks up the customer's OWN orders and asks which one the problem is about (listing them),
  * it explains the order's status against the delivery / refund / payment policy (with the real numbers),
  * it answers general questions with verbatim policy sentences (sentence-level TF-IDF retrieval),
  * it asks a clarifying question when a message is too vague to act on,
  * it hands over to a person only when it cannot fix the problem itself (it cannot issue refunds, speed up a rider
    or approve compensation), when the customer asks for a person or says it is still not solved, for fraud,
    or after MAX_BOT_REPLIES replies (the escalation policy: hand over within 6 steps, with a short summary).
Every hand-over message states what the agent will be told. The bot reads masked text for retrieval, and the order
facts it quotes come from the database rows passed in, never from the customer's message.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import triage as tr

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"
BOT_USERNAME = "copilot_bot"
BOT_NAME = "Instamart Copilot Assistant"
MIN_SCORE = 0.30          # policy questions score >= 0.36, vague complaints <= 0.25 (see README)
MAX_BOT_REPLIES = 6       # escalation policy: a person must take over within 6 steps
PROMISE_MIN, RAIN_MIN, LATE_REFUND_MIN = 30, 60, 45
WANTS_HUMAN = re.compile(r"\b(human|agent|person|manager|supervisor|call me|call back|speak to|talk to)\b", re.I)
ORDER_REF = re.compile(r"\b(?:ORD-\d{6}|WEB-\d{6})\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
UNHAPPY = re.compile(r"\b(still|not (solved|resolved|working|helpful|received|refunded|returned|found|got)|didn'?t help|doesn'?t help|"
                     r"useless|no use|worst|nothing (happened|changed))\b", re.I)
INTENTS = [   # order matters: first match wins
    ("rider", re.compile(r"\b(rude|misbehav\w*|abus\w*|shout\w*|impolite|harass\w*)\b", re.I)),
    ("item", re.compile(r"\b(missing|wrong|damaged|expired|rotten|spoil\w*|broken|leak\w*|stale|short|incomplete|torn|open(ed)?)\b.*\b(item|items|product|products|food|milk|bread|egg|eggs|vegetable|vegetables|fruit|fruits|curd|rice|oil|snacks|packet|pack|order)\b|\b(item|items|product|products|food|milk|bread|egg|eggs|vegetable|vegetables|fruit|fruits|curd|rice|oil|snacks|packet|pack|order)\b.*\b(missing|wrong|damaged|expired|rotten|spoil\w*|broken|leak\w*|stale|short|incomplete|torn|open(ed)?)\b", re.I)),
    ("payment_double", re.compile(r"\b(charged twice|double (charge|charged|payment)|two charges|paid twice|charged two)\b", re.I)),
    ("payment_failed", re.compile(r"\b(deduct\w*|debited|payment (failed|issue|problem)|money (cut|gone|taken)|amount (cut|deducted))\b", re.I)),
    ("refund", re.compile(r"\brefund\w*|money back\b", re.I)),
    ("cancel", re.compile(r"\bcancel\w*\b", re.I)),
    ("delivery", re.compile(r"(not (getting|received|delivered|arrived|come|coming)|haven'?t (got|received)|didn'?t (get|receive|arrive)|never (came|arrived|received)|"
                            r"where('?s| is) my|track|\blate\b|delay\w*|still waiting|yet to|when will|not yet|(is|was) not delivered)", re.I)),
]
PROBLEM = re.compile(r"\b(my (order|delivery|item|items|payment|food|money|refund)|i (was|am|got|ordered|paid|received)|i'?m|i'?ve|didn'?t|haven'?t|not getting|"
                     r"never|order (is|was|has)|it (is|was|has))\b", re.I)
QUESTION = re.compile(r"^\W*(what|how|when|is|are|can|does|do|why|who|which|will|after|on|where)\b", re.I)
SECTION_TITLES = {"refund_policy.txt": "Refunds", "delivery_policy.txt": "Delivery", "payment_policy.txt": "Payments",
                  "privacy_policy.txt": "Privacy", "escalation_policy.txt": "Support"}
IN_PROGRESS = {"placed": "placed and waiting to be packed", "packed": "packed and waiting for a rider",
               "out_for_delivery": "out for delivery with a rider"}
DONE = {"completed", "delivered"}


@dataclass
class BotResult:
    handled: bool            # True: the bot is still helping (answer or question); False: a person has to take over
    reply: str               # text to post in the chat
    reason: str              # answered / ask / fraud / asked_for_human / low_confidence / step_limit / needs_agent / still_unresolved
    confidence: float = 0.0
    sources: tuple = ()
    ask: bool = False        # True when the bot is waiting for the customer's answer (no "Solved?" buttons)


def _pretty(s):
    return str(s).replace("_", " ")


def _inr(x):
    return f"₹{float(x):,.2f}"


def _age_minutes(order, now):
    """Minutes since a WEB order was placed (history rows have no live age)."""
    try:
        if order.get("source") != "web":
            return None
        placed = datetime.strptime(str(order["created_at"])[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0, int((now - placed).total_seconds() // 60))
    except (ValueError, KeyError, TypeError):
        return None


def _line(order):
    return f"{order['order_ref']} · {_inr(order['amount'])} · {_pretty(order['status'])} · {str(order['created_at'])[:10]}"


class SupportBot:
    def __init__(self, policy_dir=POLICY_DIR):
        self.sentences, self.sources = [], []
        for path in sorted(Path(policy_dir).glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            for s in SENTENCE.split(" ".join(text.split())):
                if len(s.split()) >= 5 and not s.endswith("Policy."):
                    self.sentences.append(s.strip())
                    self.sources.append(path.name)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.sentences)

    def retrieve(self, question, k=2):
        scores = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(float(scores[i]), self.sentences[i], self.sources[i]) for i in order]

    # ------------------------------------------------------------------ the conversation
    def respond(self, texts, category="", orders=(), prior_bot_replies=0, force_human=False, now=None):
        """`texts`: everything the customer has written in this chat, oldest first (a plain string also works).
        `orders`: the customer's own orders as dicts (order_ref, status, amount, created_at, delivery_minutes, source),
        newest first. Returns what to say and whether a person has to take over."""
        texts = [texts] if isinstance(texts, str) else [t for t in texts if t and t.strip()]
        texts = texts or [""]
        latest = texts[-1].strip()
        orders, now = list(orders), now or datetime.now(timezone.utc)
        if category == "fraud":
            return BotResult(False, "This looks like a possible fraud or unauthorised transaction, so I have passed it to our security team "
                             "right away. A person will contact you within 2 hours. Please do not share OTPs or card details with anyone "
                             "in the meantime.", "fraud")
        if force_human or WANTS_HUMAN.search(latest):
            return self._handover("Understood. ", texts, orders, "asked_for_human")
        if prior_bot_replies >= MAX_BOT_REPLIES:
            return self._handover("I have not been able to settle this within a few steps. ", texts, orders, "step_limit")
        if prior_bot_replies >= 1 and UNHAPPY.search(latest):
            return self._handover("I'm sorry that did not fix it. ", texts, orders, "still_unresolved")

        intent = None if (QUESTION.match(latest) and not PROBLEM.search(latest)) else self._intent(texts)   # a plain policy question is answered from the policy
        if intent:
            return self._order_flow(intent, texts, orders, prior_bot_replies, now)
        return self._policy_answer(texts, prior_bot_replies, orders)

    def _intent(self, texts):
        latest = texts[-1]
        for name, rx in INTENTS:
            if rx.search(latest):
                return name
        short_reply = len(latest.split()) <= 4 or ORDER_REF.search(latest)
        if short_reply:                                   # an answer to my question ("1", "the latest", an order id): carry the topic
            for earlier in reversed(texts[:-1]):
                for name, rx in INTENTS:
                    if rx.search(earlier):
                        return name
        return None

    def _resolve_order(self, texts, orders, intent):
        by_ref = {o["order_ref"].upper(): o for o in orders}
        for t in reversed(texts):                          # an order id typed anywhere in the chat
            for ref in ORDER_REF.findall(t):
                if ref.upper() in by_ref:
                    return by_ref[ref.upper()], False
        latest = texts[-1].strip()
        m = re.fullmatch(r"\W*#?\s*([1-4])\W*", latest)
        if m and int(m.group(1)) <= len(orders):
            return orders[int(m.group(1)) - 1], False
        if orders and re.search(r"\b(first|latest|last|recent|newest|this one|that one|my order)\b", latest, re.I):
            return orders[0], False
        if intent in ("delivery", "cancel"):               # exactly one live order: that is surely the one
            live = [o for o in orders if o["status"] in IN_PROGRESS]
            if len(live) == 1:
                return live[0], True
        return None, False

    def _ask_which(self, intent, orders, prior):
        if not orders:
            if prior >= 1:
                return None
            return BotResult(True, "I can't see any orders on your account yet. Please type the order id (for example WEB-000001) and tell me "
                             "what went wrong, and I'll check it for you.", "ask", ask=True)
        lines = [f"{i}) {_line(o)}" for i, o in enumerate(orders[:4], 1)]
        return BotResult(True, "I'm sorry about that, let's sort it out. Which order is this about? Reply with its number or type the order id:\n"
                         + "\n".join(lines), "ask", ask=True)

    def _order_flow(self, intent, texts, orders, prior, now):
        if intent == "payment_failed":
            return BotResult(True, "If a payment fails but money is deducted, the amount is returned automatically within 48 hours. "
                             "Please wait for that window; if it has already passed, reply “still not refunded” and I'll pass it to a billing agent "
                             "with the details.  [Payments]", "answered")
        order, assumed = self._resolve_order(texts, orders, intent)
        if intent == "payment_double":
            if not order:
                ask = self._ask_which(intent, orders, prior)
                if ask and ask.ask:
                    ask.reply = ask.reply.replace("Which order is this about?", "Which order were you charged twice for?")
                    return ask
                return self._handover("A double charge needs a billing ticket with the order id. ", texts, orders, "needs_agent")
            return self._handover(f"Thanks. I have the order: {_line(order)}. Payment policy says a double charge goes to billing with the order id, "
                                  f"and I can't reverse a charge myself. ", texts, orders, "needs_agent", order=order)
        if not order:
            ask = self._ask_which(intent, orders, prior)
            return ask if ask else self._handover("I could not find that order. ", texts, orders, "needs_agent")
        head = f"I found your order {_line(order)}. "
        st, minutes, age = order["status"], order.get("delivery_minutes"), _age_minutes(order, now)

        if st == "cancelled":
            return BotResult(True, head + "It was cancelled, so you were never charged and no refund applies.  [Refunds]", "answered")

        if intent == "cancel":
            if st == "placed":
                return BotResult(True, head + "It's still “placed”, so you can cancel it yourself: open My orders and press Cancel next to it. "
                                 "A cancelled order is never charged.  [Refunds]", "answered")
            if st in IN_PROGRESS:
                return self._handover(head + f"It is already {IN_PROGRESS[st]}, and the app can only cancel orders that are still “placed”. ",
                                      texts, orders, "needs_agent", order=order)
            return BotResult(True, head + "It has already been delivered, so it can't be cancelled. If something was wrong with it, tell me what.  "
                             "[Refunds]", "answered")

        if intent in ("rider", "item"):
            what = "the rider's behaviour" if intent == "rider" else "the problem with the items"
            return self._handover(head + f"Thank you for telling me about {what}; I'm sorry. I can't approve compensation or a replacement myself. ",
                                  texts, orders, "needs_agent", order=order)

        if intent == "refund":
            if st in IN_PROGRESS:
                return BotResult(True, head + "Refunds apply only to completed orders, and this one is not delivered yet. If it is taking too long, "
                                 "tell me and I'll look at the delivery.  [Refunds]", "answered")
            if minutes is not None and int(minutes) > LATE_REFUND_MIN:
                approval = ("up to ₹1000 is approved automatically by the support system" if float(order["amount"]) <= 1000
                            else "above ₹1000 needs a support lead's approval")
                return self._handover(head + f"It was delivered in {int(minutes)} minutes, which is more than {LATE_REFUND_MIN}, so you are entitled to a full "
                                      f"refund of {_inr(order['amount'])} (a refund {approval}; it is credited to the original payment method within "
                                      "5 working days, and must be requested within 7 days of delivery). I can't release money myself. ",
                                      texts, orders, "needs_agent", order=order)
            if minutes is not None:
                return BotResult(True, head + f"It was delivered in {int(minutes)} minutes ({LATE_REFUND_MIN} or less), so no late-delivery refund is paid. "
                                 "If something else was wrong with it, tell me what.  [Refunds]", "answered")
            return BotResult(True, head + "A refund can be requested only for completed orders, within 7 days of delivery. Late deliveries "
                             f"(more than {LATE_REFUND_MIN} minutes) get a full refund. Tell me what went wrong with this order and I'll take it from there.  "
                             "[Refunds]", "answered", ask=True)

        # delivery problems
        if st in IN_PROGRESS:
            where = f"It is {IN_PROGRESS[st]}."
            if age is None:
                return BotResult(True, head + where + " Delivery is promised within 30 minutes of confirmation, up to 60 in heavy rain, and the rider "
                                 "will call you twice, so please keep your phone nearby.  [Delivery]", "answered")
            if age <= PROMISE_MIN:
                return BotResult(True, head + where + f" It was placed {age} minute(s) ago, which is inside our {PROMISE_MIN}-minute delivery promise, so "
                                 "it should reach you soon. The rider will try to call you up to twice, so please keep your phone nearby.  [Delivery]", "answered")
            if age <= RAIN_MIN:
                return BotResult(True, head + where + f" It was placed {age} minutes ago, past the {PROMISE_MIN}-minute promise, but in heavy rain delivery can "
                                 f"take up to {RAIN_MIN} minutes, so please allow about {RAIN_MIN - age} more minutes. If it hasn't arrived by then, reply "
                                 "“still not received” and I'll pass it to an agent.  [Delivery]", "answered")
            return self._handover(head + where + f" It was placed {age} minutes ago, beyond even the {RAIN_MIN}-minute rain limit, and I'm sorry. "
                                  "I can't speed up a rider myself"
                                  + (", but you can also cancel it from My orders while it is “placed”" if st == "placed" else "") + ". ",
                                  texts, orders, "needs_agent", order=order)
        late = (f" Our records show it was delivered in {int(minutes)} minutes"
                + (f", which is more than {LATE_REFUND_MIN}, so a full refund applies." if int(minutes) > LATE_REFUND_MIN else ".")) if minutes is not None else ""
        return BotResult(True, head + "It is marked as delivered." + late + " Please check with anyone at home, the security desk or neighbours. "
                         "If it truly did not reach you, reply “still not received” and I'll pass it to an agent right away.  [Delivery]", "answered", ask=True)

    def _policy_answer(self, texts, prior, orders):
        text = " ".join(texts[-2:])
        hits = [h for h in self.retrieve(text, k=3) if h[0] >= MIN_SCORE][:2]
        if not hits:
            if prior == 0:
                return BotResult(True, "I want to get this right. Is it about an order (a late, missing or wrong delivery), a payment, a refund, or "
                                 "your account? A few more details help me fix it.", "ask", ask=True)
            return self._handover("I could not find this in our policies. ", texts, orders, "low_confidence")
        lines = ["Here is what I found:"] + [f"• {s}  [{SECTION_TITLES.get(src, src)}]" for _, s, src in hits]
        lines.append("Did this solve it? If not, press “Not solved – create a ticket” and a support agent will take over.")
        return BotResult(True, "\n".join(lines), "answered", hits[0][0], tuple(sorted({src for _, _, src in hits})))

    def _handover(self, lead, texts, orders, reason, order=None):
        order = order or self._resolve_order(texts, orders, "delivery")[0]
        told = " / ".join(t.strip() for t in texts[-2:])[:160]
        summary = f"order {order['order_ref']} ({_pretty(order['status'])}, {_inr(order['amount'])}); " if order else ""
        return BotResult(False, f"{lead}I'm passing this to a support agent with a short summary: {summary}you told me “{told}”. "
                         "Press “Create a ticket” below and an agent will reply on the ticket, so you won't have to repeat anything.", reason)


@lru_cache(maxsize=1)
def get_bot():
    return SupportBot()


def mask_for_bot(text):
    safe, _ = tr.redact(text or "")
    return safe


def order_dicts(rows):
    """Database rows (all_orders) -> plain dicts for the bot."""
    return [{"order_ref": r["order_ref"], "status": r["status"], "amount": r["amount"], "created_at": r["created_at"],
             "delivery_minutes": r["delivery_minutes"], "source": r["source"]} for r in rows]
