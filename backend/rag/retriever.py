"""Retrieval over the local Chroma store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "wakili_legal"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. Run `python -m rag.ingest` first."
        )
    ef = embedding_functions.DefaultEmbeddingFunction()
    _client = PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    return _collection


def hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}") from e

    hits: List[Dict[str, Any]] = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return hits

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, doc_id in enumerate(ids):
        dist = float(dists[i]) if dists else 1.0
        score = max(0.0, 1.0 - dist)
        hits.append(
            {
                "id": doc_id,
                "text": docs[i] or "",
                "metadata": metas[i] or {},
                "score": score,
            }
        )

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits