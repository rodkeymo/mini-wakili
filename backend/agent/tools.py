"""Tools available to the agent."""

from __future__ import annotations

from typing import Any, Dict, List

from rag.retriever import hybrid_search


def search_legal_corpus(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieve relevant passages from the legal corpus.
    Returns list of {id, text, metadata, score}.
    """
    return hybrid_search(query, top_k=top_k)


def refine_query(original: str, previous_results: List[Dict[str, Any]]) -> str:
    """
    Very lightweight query refinement for the agentic re-try.
    In production this would be an LLM call; here we keep it deterministic
    and transparent for the take-home.
    """
    if not previous_results:
        return f"{original} (statute OR case law OR limitation OR contract)"
    # Add a couple of high-signal terms from the best hit if available
    top_meta = previous_results[0].get("metadata", {})
    extra = top_meta.get("act") or top_meta.get("title") or ""
    if extra:
        return f"{original} {extra}"
    return original
