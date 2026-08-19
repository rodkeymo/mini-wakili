"""Basic retrieval smoke tests (require prior ingest)."""

import pytest
from pathlib import Path

CHROMA = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


@pytest.mark.skipif(not CHROMA.exists(), reason="Run python -m rag.ingest first")
def test_hybrid_search_returns_hits():
    from rag.retriever import hybrid_search

    hits = hybrid_search("limitation period contract", top_k=3)
    assert isinstance(hits, list)
    if hits:
        assert "id" in hits[0]
        assert "text" in hits[0]
        assert "score" in hits[0]
