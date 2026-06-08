"""
Task 6 - Lexical Search Module using BM25.

BM25 combines term frequency, inverse document frequency, and document length
normalization. It is a strong keyword-search baseline for legal/news corpora.
"""

import json
import math
from collections import Counter

from src.task4_chunking_indexing import INDEX_FILE, run_pipeline, tokenize

K1 = 1.5
B = 0.75


def load_corpus() -> list[dict]:
    """Load chunks from the local index created in Task 4."""
    if not INDEX_FILE.exists():
        run_pipeline()

    corpus = []
    if not INDEX_FILE.exists():
        return corpus

    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                corpus.append(
                    {
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {}),
                    }
                )
    return corpus


def build_bm25_index(corpus: list[dict]) -> dict:
    """
    Build a BM25 index from corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_docs = [tokenize(doc["content"]) for doc in corpus]
    doc_count = len(tokenized_docs)
    avg_doc_len = (
        sum(len(tokens) for tokens in tokenized_docs) / doc_count if doc_count else 0.0
    )

    term_frequencies = [Counter(tokens) for tokens in tokenized_docs]
    document_frequency = Counter()
    for tokens in tokenized_docs:
        for token in set(tokens):
            document_frequency[token] += 1

    return {
        "corpus": corpus,
        "tokenized_docs": tokenized_docs,
        "term_frequencies": term_frequencies,
        "document_frequency": document_frequency,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
    }


def score_document(query_tokens: list[str], doc_index: int, bm25: dict) -> float:
    """Compute BM25 score for one document."""
    score = 0.0
    doc_tokens = bm25["tokenized_docs"][doc_index]
    doc_len = len(doc_tokens)
    if not doc_len or not bm25["avg_doc_len"]:
        return 0.0

    tf = bm25["term_frequencies"][doc_index]
    doc_count = bm25["doc_count"]
    document_frequency = bm25["document_frequency"]

    for token in query_tokens:
        freq = tf.get(token, 0)
        if not freq:
            continue

        df = document_frequency[token]
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        numerator = freq * (K1 + 1)
        denominator = freq + K1 * (1 - B + B * doc_len / bm25["avg_doc_len"])
        score += idf * numerator / denominator

    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search by exact keyword relevance using BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    corpus = load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    query_tokens = tokenize(query)
    results = []

    for index, doc in enumerate(corpus):
        score = score_document(query_tokens, index, bm25)
        results.append(
            {
                "content": doc["content"],
                "score": float(score),
                "metadata": doc.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
