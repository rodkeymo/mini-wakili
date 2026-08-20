"""Tools used by agent nodes."""

from __future__ import annotations

from typing import Any, Dict, List

from rag.retriever import hybrid_search


def search_legal_corpus(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return hybrid_search(query, top_k=top_k)


def refine_query(question: str, hits: List[Dict[str, Any]]) -> str:
    """Lightweight refine for a second retrieval round."""
    if not hits:
        return question
    meta = hits[0].get("metadata") or {}
    act = meta.get("act") or meta.get("source") or ""
    if act:
        return f"{question} (focus on {act})"
    # Fall back: append a short snippet keyword
    text = (hits[0].get("text") or "")[:80]
    return f"{question} {text}".strip()