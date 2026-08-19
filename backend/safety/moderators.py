"""Simple input / output moderation for the take-home."""

from __future__ import annotations

import re
from typing import List, Tuple

# Patterns that should never enter a legal research agent
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?system\s+prompt",
    r"you\s+are\s+now\s+dan",
    r"jailbreak",
]

# Output that looks like a definitive legal opinion without human review
OVERCONFIDENT_PATTERNS = [
    r"this\s+is\s+(a\s+)?(final|binding)\s+(legal\s+)?opinion",
    r"you\s+should\s+definitely\s+sue",
    r"the\s+contract\s+is\s+definitely\s+void",
]


def moderate_input(text: str) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    Blocks obvious prompt-injection style inputs.
    """
    lower = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lower):
            return False, "Input rejected by safety filter (possible prompt injection)."

    if len(text.strip()) < 3:
        return False, "Query too short."

    return True, ""


def moderate_output(answer: str, citations: List[dict]) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    - Rejects empty answers that claim success
    - Flags clearly over-confident absolute legal conclusions
    """
    if not answer or not answer.strip():
        return False, "Empty answer blocked."

    lower = answer.lower()
    for pat in OVERCONFIDENT_PATTERNS:
        if re.search(pat, lower):
            return False, (
                "Answer rejected: appears to present a definitive legal conclusion. "
                "Wakili only prepares drafts for lawyer review."
            )

    # Soft check: if we have citations they should be present in the text
    # (already handled more strictly in the verify node)
    return True, ""
