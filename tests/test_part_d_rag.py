"""Part D (retrieval): chunking, TF-IDF retrieval, grounded answers and fallback."""
import pandas as pd
import pytest

import mock_llm
from src import policy_rag as pr


@pytest.fixture(scope="module")
def rag():
    return pr.PolicyRAG()


def test_chunking_matches_reference_counts(rag):
    assert rag.chunk_counts() == pr.EXPECTED_CHUNKS and len(rag.chunks) == 15
    for c in rag.chunks:
        assert len(c["text"].split()) <= 40


def test_chunk_overlap_is_ten_words():
    words = [f"w{i}" for i in range(100)]
    ch = pr.chunk_words(words, 40, 10)
    assert [len(c.split()) for c in ch] == [40, 40, 40, 10] or all(len(c.split()) <= 40 for c in ch)
    assert ch[0].split()[-10:] == ch[1].split()[:10]
    assert " ".join(ch[-1].split()[-1:]) == "w99"                      # the tail is never dropped


def test_all_supported_questions_grounded(rag):
    ev = pr.evaluate(rag)
    s = pr.summarize(ev)
    assert s["expected_doc_in_top3"] == 12 and s["expected_doc_top1"] == 12
    assert s["expected_fact_found"] == 12 and s["false_fallbacks_on_supported"] == 0


def test_unsupported_questions_trigger_fallback(rag):
    ev = pr.evaluate(rag)
    unsupported = ev[ev.expected_doc == "NONE"]
    assert len(unsupported) == 2 and unsupported.fallback.all()
    assert all(a.startswith("I could not find that information in the policy documents.") for a in unsupported.answer)
    out = pr.answer("What is the favourite colour of the company mascot?", rag)
    assert out["is_fallback"]


def test_answer_uses_required_two_message_format(monkeypatch, rag):
    seen = []
    real = mock_llm.chat
    monkeypatch.setattr(mock_llm, "chat", lambda m, t=None: (seen.append(m), real(m, t))[1])
    out = pr.answer("Who must approve a refund above INR 1000?", rag)
    assert len(seen) == 1 and len(seen[0]) == 2
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert "POLICY ASSISTANT" in seen[0][0]["content"]
    assert seen[0][1]["content"].startswith("CONTEXT:") and "QUESTION:" in seen[0][1]["content"]
    assert "human support lead" in out["answer"] and out["top_source"] == "refund_policy.txt"


def test_answer_text_comes_from_retrieved_context(rag):
    """Grounding: the answer sentence must be assembled from the retrieved chunks' words."""
    out = pr.answer("What is the delivery fee on small orders?", rag)
    context_words = set(" ".join(h["text"] for h in out["retrieved"]).lower().split())
    assert set(out["answer"].lower().split()) <= context_words | {"."}


def test_retrieval_is_cosine_ranked(rag):
    hits = rag.retrieve("refund above INR 1000 human approval", 3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True) and 0 <= scores[-1] <= scores[0] <= 1.0001


def test_zero_overlap_comparison_shows_a_failure():
    cmp, failures = pr.compare_overlap()
    assert cmp["overlap_10"]["expected_fact_found"] == 12
    assert cmp["overlap_0"]["chunks"] < cmp["overlap_10"]["chunks"]
    assert cmp["overlap_0"]["expected_fact_found"] < 12 and len(failures) >= 1
    assert isinstance(failures, pd.DataFrame)
