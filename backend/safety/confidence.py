"""Confidence scoring and refusal gates."""

from __future__ import annotations

from typing import Any, Dict, List

MIN_TOP_SCORE = 0.35
MIN_AVG_TOP3 = 0.28
MIN_CONFIDENCE = 0.32


def compute_confidence(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    scores = [float(h.get("score") or 0.0) for h in hits]
    top = scores[0]
    top3 = scores[:3]
    avg3 = sum(top3) / len(top3)
    return max(0.0, min(1.0, 0.6 * top + 0.4 * avg3))


def should_refuse(hits: List[Dict[str, Any]], conf: float | None = None) -> bool:
    if not hits:
        return True
    top = float(hits[0].get("score") or 0.0)
    scores = [float(h.get("score") or 0.0) for h in hits[:3]]
    avg3 = sum(scores) / len(scores) if scores else 0.0
    c = conf if conf is not None else compute_confidence(hits)
    if top < MIN_TOP_SCORE:
        return True
    if avg3 < MIN_AVG_TOP3:
        return True
    if c < MIN_CONFIDENCE:
        return True
    return False