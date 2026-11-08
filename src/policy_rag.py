"""Part D (retrieval): chunk policy files, TF-IDF + cosine retrieval and grounded answers.

Run from the repository root:
    python -m src.policy_rag
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parents[1]
POL = ROOT / "data" / "policies"
OUT = ROOT / "outputs"

CHUNK_WORDS, CHUNK_OVERLAP = 40, 10
EXPECTED_CHUNKS = {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
                   "payment_policy.txt": 3, "privacy_policy.txt": 2}   # total 15
FALLBACK = "I could not find that information in the policy documents."
SYSTEM_PROMPT = ("POLICY ASSISTANT: answer only from the supplied context. "
                 f"If it is not there, say: {FALLBACK}")


def chunk_words(words, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Word windows of `size` words advancing by size-overlap; the last window ends at the final word."""
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


class PolicyRAG:
    def __init__(self, chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
        self.chunk_size, self.overlap, self.chunks = chunk_size, overlap, []
        for path in sorted(POL.glob("*.txt")):
            for text in chunk_words(path.read_text(encoding="utf-8").split(), chunk_size, overlap):
                self.chunks.append({"source": path.name, "text": text})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def chunk_counts(self):
        return pd.Series([c["source"] for c in self.chunks]).value_counts().sort_index().to_dict()

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        idx = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in idx]

    def answer(self, question, k=3):
        """Exactly two messages: a POLICY ASSISTANT system message, then CONTEXT + QUESTION."""
        hits = self.retrieve(question, k)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
        text = response.choices[0].message.content
        return {"answer": text, "is_fallback": text.startswith("I could not find"),
                "sources": list(dict.fromkeys(h["source"] for h in hits)), "top_source": hits[0]["source"],
                "retrieved": hits}


_DEFAULT = None


def answer(question, rag=None, k=3):
    """Module-level entry point: retrieve with TF-IDF/cosine, then ask the (mock) policy assistant."""
    global _DEFAULT
    if rag is None:
        _DEFAULT = _DEFAULT or PolicyRAG()
        rag = _DEFAULT
    return rag.answer(question, k)


def evaluate(rag):
    """Evaluate the 14 questions in data/raw/policy_eval.csv (12 answerable, 2 unsupported)."""
    df = pd.read_csv(ROOT / "data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for r in df.itertuples():
        ans = rag.answer(r.question)
        supported = r.expected_doc != "NONE"
        rows.append({"question": r.question, "expected_doc": r.expected_doc,
                     "retrieved_doc": r.expected_doc in ans["sources"] if supported else False,
                     "top1_doc": ans["top_source"] == r.expected_doc if supported else False,
                     "fact_found": r.expected_fact in ans["answer"] if supported else False,
                     "fallback": ans["is_fallback"], "answer": ans["answer"]})
    return pd.DataFrame(rows)


def summarize(ev):
    sup, uns = ev[ev.expected_doc != "NONE"], ev[ev.expected_doc == "NONE"]
    return {"questions_supported": len(sup), "expected_doc_in_top3": int(sup.retrieved_doc.sum()),
            "expected_doc_top1": int(sup.top1_doc.sum()), "expected_fact_found": int(sup.fact_found.sum()),
            "unsupported_questions": len(uns), "fallback_triggered": int(uns.fallback.sum()),
            "false_fallbacks_on_supported": int(sup.fallback.sum())}


def compare_overlap():
    """Compare overlap=10 against overlap=0 (same 40-word chunks). Returns (summary, failures_df)."""
    out, failures = {}, []
    for overlap in (CHUNK_OVERLAP, 0):
        rag = PolicyRAG(CHUNK_WORDS, overlap)
        ev = evaluate(rag)
        out[f"overlap_{overlap}"] = {"chunks": len(rag.chunks), "per_doc": rag.chunk_counts(), **summarize(ev)}
        if overlap == 0:
            sup = ev[ev.expected_doc != "NONE"]
            failures = sup[~(sup.retrieved_doc & sup.fact_found)][["question", "expected_doc", "retrieved_doc", "fact_found", "answer"]]
    return out, pd.DataFrame(failures)


def main():
    rag = PolicyRAG()
    counts = rag.chunk_counts()
    print("Chunks per document:", counts, "| total:", len(rag.chunks))
    assert counts == EXPECTED_CHUNKS and len(rag.chunks) == 15
    ev = evaluate(rag)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(ev[["question", "expected_doc", "retrieved_doc", "top1_doc", "fact_found", "fallback"]].to_string(index=False))
    s = summarize(ev)
    print("\nRAG evaluation:", json.dumps(s))
    cmp, failures = compare_overlap()
    print("\nOverlap comparison:", json.dumps(cmp, indent=1))
    print(f"\nZero-overlap retrieval/answer failures: {len(failures)}")
    if len(failures):
        print(failures[["question", "expected_doc", "retrieved_doc", "fact_found"]].to_string(index=False))
    OUT.mkdir(exist_ok=True)
    ev.to_csv(OUT / "rag_evaluation.csv", index=False)
    failures.to_csv(OUT / "rag_zero_overlap_failures.csv", index=False)
    (OUT / "rag_summary.json").write_text(json.dumps({"chunks": counts, "evaluation": s, "overlap_comparison": cmp}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
