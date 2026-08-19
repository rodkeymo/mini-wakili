"""
Ingest sample legal corpus into a local Chroma vector store.
Run once:  python -m rag.ingest
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "legal_corpus"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "wakili_legal"


def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """Simple character-based chunker with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        # First line can be a title / metadata hint
        lines = raw.strip().split("\n", 1)
        title = lines[0].strip().lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else raw

        for i, chunk in enumerate(chunk_text(body)):
            doc_id = f"{path.stem}_c{i}"
            docs.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "source_file": path.name,
                    "title": title,
                    "chunk_index": i,
                },
            })
    return docs


def ingest() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add .txt files first.")
        return

    # Use a lightweight default embedding function so the project runs
    # without an external API key. Swap for OpenAI / Voyage later.
    ef = embedding_functions.DefaultEmbeddingFunction()

    client = PersistentClient(path=str(CHROMA_DIR))
    # Recreate collection for clean re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )

    print(f"Ingested {len(docs)} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
