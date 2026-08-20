"""Ingest statute .txt files into Chroma."""

from __future__ import annotations

import re
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "legal_corpus"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "wakili_legal"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def _chunk_text(text: str, source: str) -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                {
                    "id": f"{source}_c{idx}",
                    "text": piece,
                    "metadata": {"source": source, "act": source, "chunk": idx},
                }
            )
            idx += 1
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def ingest() -> int:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_docs: list[dict] = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        all_docs.extend(_chunk_text(raw, path.stem))

    if not all_docs:
        print(f"No .txt files in {CORPUS_DIR}")
        return 0

    batch = 100
    for i in range(0, len(all_docs), batch):
        part = all_docs[i : i + batch]
        collection.add(
            ids=[d["id"] for d in part],
            documents=[d["text"] for d in part],
            metadatas=[d["metadata"] for d in part],
        )

    print(f"Ingested {len(all_docs)} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
    return len(all_docs)


if __name__ == "__main__":
    ingest()