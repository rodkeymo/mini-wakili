"""
Simple offline retrieval quality check.
Usage (after ingest):
  cd backend && python ../scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from rag.retriever import hybrid_search

# Tiny golden set for the sample corpus
GOLDEN = [
    {
        "question": "limitation period for contract claims",
        "must_contain": "six years",
    },
    {
        "question": "data subject rights under Data Protection Act",
        "must_contain": "access their personal data",
    },
    {
        "question": "director duty to promote success of company",
        "must_contain": "promote the success",
    },
]


def main():
    print("Mini-Wakili retrieval smoke evaluation\n")
    hits_total = 0
    for g in GOLDEN:
        results = hybrid_search(g["question"], top_k=3)
        text_blob = " ".join(r.get("text", "") for r in results).lower()
        found = g["must_contain"].lower() in text_blob
        hits_total += int(found)
        status = "PASS" if found else "FAIL"
        print(f"[{status}] {g['question'][:50]}…")
        if results:
            print(f"       top score={results[0]['score']:.3f} id={results[0]['id']}")
    print(f"\n{hits_total}/{len(GOLDEN)} golden questions retrieved relevant text.")


if __name__ == "__main__":
    main()
