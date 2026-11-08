"""Support bot: answers first from the policy files, hands over to a person only when it cannot resolve the ticket."""
import sqlite3

import pytest

from webapp.support_bot import MAX_BOT_REPLIES, SupportBot, mask_for_bot
from tests.test_webapp import csrf_of, env, login, post   # noqa: F401  (fixtures and helpers shared with the web tests)


@pytest.fixture(scope="module")
def bot():
    return SupportBot()


def test_bot_answers_policy_questions_with_policy_sentences(bot):
    cases = {"What is the delivery fee on small orders?": "INR 25",
             "How many days do I have to raise a refund request?": "7 days",
             "Is cash on delivery available for a large order?": "INR 2500",
             "Money was deducted but my payment failed, when do I get it back?": "48 hours"}
    for q, fact in cases.items():
        r = bot.respond(q)
        assert r.handled and r.reason == "answered" and fact in r.reply, (q, r.reply)
    # grounded: every bullet is a verbatim policy sentence
    r = bot.respond("What is the delivery fee on small orders?")
    bullets = [l[2:].rsplit("  [", 1)[0] for l in r.reply.splitlines() if l.startswith("• ")]
    assert bullets and all(b in bot.sentences for b in bullets)


def test_bot_hands_over_when_it_cannot_help(bot):
    r = bot.respond("What is the favourite colour of the company mascot?")
    assert r.handled and r.ask and r.reason == "ask"                       # first it asks what the problem is about
    assert bot.respond("What is the favourite colour of the company mascot?", prior_bot_replies=1).reason == "low_confidence"
    assert bot.respond("Who won the cricket world cup final?", prior_bot_replies=1).handled is False
    assert bot.respond("someone stole my card", category="fraud").reason == "fraud"
    assert bot.respond("I want to speak to a manager about my refund").reason == "asked_for_human"
    assert bot.respond("What is the delivery fee?", prior_bot_replies=MAX_BOT_REPLIES).reason == "step_limit"
    assert bot.respond("What is the delivery fee?", force_human=True).handled is False
    assert MAX_BOT_REPLIES == 6                                            # escalation policy: a person within 6 steps


def test_bot_only_sees_masked_text():
    m = mask_for_bot("mail jane@example.com phone 9876543210 order ORD-100048")
    assert "jane@example.com" not in m and "9876543210" not in m and "ORD-100048" not in m


def chat(c, text, turns=0, page="/home"):
    return c.post("/chat", json={"message": text, "turns": turns}, headers={"X-CSRF": csrf_of(c, page)})


def test_chat_answers_from_policy_and_needs_login_and_csrf(env):
    app, cfg, ids = env
    a = login(app, "alice")
    r = chat(a, "What is the delivery fee on small orders?")
    j = r.get_json()
    assert r.status_code == 200 and j["handled"] and "INR 25" in j["reply"] and j["reason"] == "answered"
    assert app.test_client().post("/chat", json={"message": "hello there"}).status_code in (400, 302, 403)
    assert a.post("/chat", json={"message": "hello there"}).status_code == 400            # no CSRF header
    assert chat(a, "x").status_code == 400 and chat(a, "y" * 1001).status_code == 400
    assert login(app, "admin").post("/chat", json={"message": "hello there"}, headers={"X-CSRF": "x"}).status_code in (400, 403)
    assert sqlite3.connect(cfg.app_db).execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0   # chatting never raises a ticket


def test_chat_hands_over_unknown_fraud_and_step_limit(env):
    app, cfg, ids = env
    a = login(app, "alice")
    assert a.post("/chat", json={"message": "What is the favourite colour of the company mascot?", "turns": 1}, headers={"X-CSRF": csrf_of(a, "/home")}).get_json()["handled"] is False
    assert chat(a, "Someone made an unauthorized fraud payment on my card").get_json()["reason"] == "fraud"
    assert chat(a, "I want to talk to a manager").get_json()["reason"] == "asked_for_human"
    assert chat(a, "What is the delivery fee on small orders?", turns=MAX_BOT_REPLIES).get_json()["reason"] == "step_limit"


def test_chat_works_the_problem_with_the_customers_own_orders(env):
    app, cfg, ids = env
    a, b = login(app, "alice"), login(app, "bob")
    post(a, "/shop/order", "/shop", product_id="P001", quantity="1")
    hist = ["im not getting my delivery"]
    j = chat(a, hist[0]).get_json()
    assert j["handled"] and j["reason"] == "answered" and "WEB-000001" in j["reply"] and "30-minute" in j["reply"]   # one live order: assumed
    theirs = chat(b, "Where is my order WEB-000001?").get_json()["reply"]
    assert "WEB-000001" not in theirs.split("Which order")[0] or "Which order" in theirs                        # bob cannot see alice's order
    assert "WEB-000001" not in theirs
    r = a.post("/chat", json={"message": "still not received", "turns": 1, "history": hist}, headers={"X-CSRF": csrf_of(a, "/home")}).get_json()
    assert r["reason"] == "still_unresolved" and r["handled"] is False and "WEB-000001" in r["reply"]            # summary names the order


def test_chat_validates_history(env):
    app, cfg, ids = env
    a = login(app, "alice")
    h = {"X-CSRF": csrf_of(a, "/home")}
    assert a.post("/chat", json={"message": "hello there", "history": "x"}, headers=h).status_code == 400
    assert a.post("/chat", json={"message": "hello there", "history": ["a"] * 21}, headers=h).status_code == 400
    assert a.post("/chat", json={"message": "hello there", "history": [1]}, headers=h).status_code == 400


def test_not_solved_creates_a_ticket_with_the_whole_chat_for_the_admin(env):
    app, cfg, ids = env
    a, adm = login(app, "alice"), login(app, "admin")
    msgs = [{"who": "user", "text": "My payment failed but money is deducted"}, {"who": "bot", "text": "Refund in 48 hours [Payments]"},
            {"who": "user", "text": "It has been 5 days already"}, {"who": "bot", "text": "I have passed this to an agent."}]
    r = a.post("/chat/ticket", json={"messages": msgs}, headers={"X-CSRF": csrf_of(a, "/home")})
    assert r.status_code == 200 and r.get_json()["ref"] == "TKT-000001"
    con = sqlite3.connect(cfg.app_db)
    assert con.execute("SELECT status, handled_by, subject FROM tickets").fetchone() == ("open", "human", "It has been 5 days already")
    assert con.execute("SELECT COUNT(*) FROM ticket_messages").fetchone()[0] == 4
    assert con.execute("SELECT outcome FROM chat_log").fetchone()[0] == "ticket"
    page = adm.get("/tickets/1").get_data(as_text=True)
    assert "5 days already" in page and "Instamart Copilot Assistant (bot)" in page
    assert "5 days already" in adm.get("/admin/tickets").get_data(as_text=True)        # in the admin queue, waiting for a person
    post(adm, "/tickets/1/reply", "/tickets/1", message="Sorry, we have refunded you.")
    assert con.execute("SELECT status FROM tickets").fetchone()[0] == "answered"
    assert "refunded you" in a.get("/tickets/1").get_data(as_text=True)


def test_solved_chat_is_logged_and_no_ticket_form_remains(env):
    app, cfg, ids = env
    a = login(app, "alice")
    assert a.post("/chat/solved", json={"turns": 2}, headers={"X-CSRF": csrf_of(a, "/home")}).status_code == 200
    assert sqlite3.connect(cfg.app_db).execute("SELECT outcome, turns FROM chat_log").fetchone() == ("solved", 2)
    page = a.get("/tickets").get_data(as_text=True)
    assert 'name="subject"' not in page and "chat-fab" in page                             # tickets come only from the help chat
    assert "chat-fab" not in login(app, "admin").get_data(as_text=True) if False else "chat-fab" not in login(app, "admin").get("/admin").get_data(as_text=True)
    assert a.post("/tickets", data={"csrf": csrf_of(a, "/tickets"), "subject": "direct ticket", "body": "x" * 20}).status_code == 405


def test_bot_account_cannot_log_in(env):
    app, cfg, ids = env
    a = login(app, "alice")
    chat(a, "What is the delivery fee?")
    a.post("/chat/ticket", json={"messages": [{"who": "user", "text": "something is wrong with my order"}]}, headers={"X-CSRF": csrf_of(a, "/home")})
    c = app.test_client()
    r = c.post("/login", data={"username": "copilot_bot", "password": "!disabled", "csrf": csrf_of(c)})
    assert r.status_code != 302


def test_login_survives_an_app_restart(env):
    """The session key and users are stored on disk, so after the server restarts the customer is still signed in."""
    from webapp.app import create_app
    app, cfg, ids = env
    a = login(app, "alice")
    cookie = a.get_cookie("session").value
    restarted = create_app(cfg)                                # a brand-new process would build the app the same way
    b = restarted.test_client()
    b.set_cookie("session", cookie)
    assert b.get("/home").status_code == 200                   # still signed in, no new login needed
    assert b.post("/chat", json={"message": "What is the delivery fee?"}, headers={"X-CSRF": csrf_of(b, "/home")}).status_code == 200


# ---------------------------------------------------------------- conversation logic with fixed orders and a fixed clock
from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)


def _o(ref, status, amount, minutes=None, age=None, source="web"):
    created = (NOW - timedelta(minutes=age)).strftime("%Y-%m-%d %H:%M:%S") if age is not None else "2025-09-02 13:33:13"
    return dict(order_ref=ref, status=status, amount=amount, created_at=created, delivery_minutes=minutes, source=source)


HIST = [_o("ORD-100009", "completed", 646.70, 20, source="history"), _o("ORD-100005", "completed", 1300.0, 61, source="history")]


def say(bot, texts, orders, prior=0):
    return bot.respond(texts, "", orders, prior, now=NOW)


def test_delivery_with_one_live_order_checks_its_age(bot):
    live = lambda age: [_o("WEB-000003", "out_for_delivery", 350, age=age)] + HIST
    r = say(bot, "im not getting my delivery", live(20))
    assert r.handled and "inside our 30-minute" in r.reply and "WEB-000003" in r.reply
    r = say(bot, "im not getting my delivery", live(45))
    assert r.handled and "up to 60" in r.reply and "15 more minutes" in r.reply
    r = say(bot, "im not getting my delivery", live(75))
    assert not r.handled and r.reason == "needs_agent" and "75 minutes" in r.reply          # only now a person is needed


def test_bot_asks_which_order_then_resolves_it(bot):
    r = say(bot, "im not getting my delivery", HIST)
    assert r.ask and "1) ORD-100009" in r.reply and "2) ORD-100005" in r.reply
    r = say(bot, ["im not getting my delivery", "2"], HIST, prior=1)
    assert r.handled and "ORD-100005" in r.reply and "more than 45" in r.reply
    r = say(bot, ["im not getting my delivery", "ORD-100009"], HIST, prior=1)
    assert "ORD-100009" in r.reply and "more than 45" not in r.reply
    r = say(bot, "im not getting my delivery", [])
    assert r.ask and "can't see any orders" in r.reply
    assert say(bot, ["im not getting my delivery", "no idea"], [], prior=1).reason == "needs_agent"


def test_refund_cancel_payment_and_complaint_paths(bot):
    r = say(bot, "please refund ORD-100005", HIST)                           # late and above 1000: entitled, but a person must approve
    assert not r.handled and "full refund of ₹1,300.00" in r.reply and "support lead" in r.reply
    r = say(bot, "please refund ORD-100009", HIST)
    assert r.handled and "no late-delivery refund" in r.reply
    placed = [_o("WEB-000004", "placed", 120, age=2)] + HIST
    assert "press Cancel" in say(bot, "cancel my order", placed).reply
    packed = [_o("WEB-000004", "packed", 120, age=2)] + HIST
    assert say(bot, "cancel my order", packed).reason == "needs_agent"
    assert "48 hours" in say(bot, "My payment failed but money was deducted", HIST).reply
    r = say(bot, ["I was charged twice", "ORD-100009"], HIST, prior=1)
    assert not r.handled and "goes to billing" in r.reply
    assert say(bot, "my rider was rude", HIST).ask                           # asks which order first
    r = say(bot, ["my rider was rude", "ORD-100009"], HIST, prior=1)
    assert not r.handled and "ORD-100009" in r.reply and "compensation" in r.reply
    r = say(bot, "the milk in ORD-100009 was expired", HIST)
    assert not r.handled and "ORD-100009" in r.reply
    cancelled = [_o("ORD-100001", "cancelled", 99, source="history")]
    assert "never charged" in say(bot, "refund ORD-100001", cancelled).reply


def test_policy_questions_are_not_mistaken_for_order_problems(bot):
    for q in ["What is the delivery fee on small orders?", "How many days do I have to raise a refund request?",
              "Can wallet balance be refunded as cash?", "When are delivery addresses deleted?"]:
        r = say(bot, q, HIST)
        assert r.reason == "answered" and not r.ask and "Which order" not in r.reply, q
