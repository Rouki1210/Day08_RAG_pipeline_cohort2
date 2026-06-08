"""
Task 10 - Generation with citations.

This file implements an extractive RAG answerer that works without an LLM API
key. If you later add an OpenAI/Gemini call, keep the same context formatting
and citation discipline.
"""

import re

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from src.task4_chunking_indexing import tokenize
from src.task9_retrieval_pipeline import retrieve

load_dotenv()

# top_k=5 is enough evidence for short RAG answers while keeping the context
# compact. top_p=0.9 and temperature=0.3 are good defaults for factual LLM RAG:
# enough natural wording, but limited randomness.
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a citation in brackets.
If the context is insufficient, say: I cannot verify this information."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Input by score: [1, 2, 3, 4, 5]
    Output:         [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[index] for index in range(0, len(chunks), 2)]
    back = [chunks[index] for index in range(len(chunks) - 1, 0, -2)]
    return front + back


def source_label(chunk: dict, fallback_index: int) -> str:
    """Build a compact citation label from chunk metadata."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or f"Source {fallback_index}"
    doc_type = metadata.get("type", "unknown")
    return f"{source}, {doc_type}"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks as citation-ready context.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        label = source_label(chunk, index)
        score = float(chunk.get("score", 0.0))
        context_parts.append(
            f"[Document {index} | Source: {label} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def sentence_relevance(query: str, sentence: str) -> float:
    """Score a sentence by query-token overlap."""
    query_tokens = set(tokenize(query))
    sentence_tokens = set(tokenize(sentence))
    if not query_tokens or not sentence_tokens:
        return 0.0
    return len(query_tokens & sentence_tokens) / len(query_tokens)


def extract_answer(query: str, chunks: list[dict], max_sentences: int = 4) -> str:
    """Create a concise extractive answer with citations."""
    candidates = []
    for index, chunk in enumerate(chunks, 1):
        label = source_label(chunk, index)
        for sentence in split_sentences(chunk.get("content", "")):
            score = sentence_relevance(query, sentence)
            if score > 0:
                candidates.append((score, sentence, label))

    if not candidates:
        return "I cannot verify this information"

    candidates.sort(key=lambda item: item[0], reverse=True)
    used = set()
    cited_sentences = []
    for _, sentence, label in candidates:
        normalized = sentence.lower()
        if normalized in used:
            continue
        used.add(normalized)
        cited_sentences.append(f"{sentence} [{label}]")
        if len(cited_sentences) >= max_sentences:
            break

    return " ".join(cited_sentences)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = extract_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "context": context,
        "generation_config": {
            "top_k": top_k,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
        },
    }


if __name__ == "__main__":
    result = generate_with_citation("Nhung nghe si nao lien quan toi ma tuy?")
    print(result["answer"])
