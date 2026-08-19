"""Confidence scoring and refusal logic."""

from __future__ import annotations

from typing import Any, Dict, List

# Tunable thresholds for the take-home
MIN_TOP_SCORE = 0.35
MIN_AVG_TOP3 = 0.30


def compute_confidence(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    scores = [float(h.get("score", 0.0)) for h in hits]
    top = scores[0]
    avg_top3 = sum(scores[:3]) / min(3, len(scores))
    # Weighted combination
    return 0.6 * top + 0.4 * avg_top3


def should_refuse(hits: List[Dict[str, Any]], confidence: float) -> bool:
    if not hits:
        return True
    top = float(hits[0].get("score", 0.0))
    if top < MIN_TOP_SCORE:
        return True
    if confidence < MIN_AVG_TOP3:
        return True
    return False
