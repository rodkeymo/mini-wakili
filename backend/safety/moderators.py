"""Input and output safety moderators."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|"
    r"system\s+prompt|jailbreak|dan\s+mode|"
    r"reveal\s+your\s+(system|hidden)\s+prompt)",
    re.I,
)

_FINAL_OPINION = re.compile(
    r"\b(this\s+is\s+(final|binding)\s+legal\s+advice|"
    r"you\s+must\s+definitely|"
    r"guaranteed\s+to\s+win\s+in\s+court)\b",
    re.I,
)


def moderate_input(text: str) -> Tuple[bool, str]:
    q = (text or "").strip()
    if len(q) < 3:
        return False, "Please enter a clearer legal research question."
    if _INJECTION.search(q):
        return False, "Request blocked by input safety checks."
    return True, ""


def moderate_output(answer: str, citations: List[Dict[str, Any]]) -> Tuple[bool, str]:
    a = (answer or "").strip()
    if not a:
        return False, "Empty model output blocked."
    if _FINAL_OPINION.search(a):
        return False, "Output blocked: language resembles a final legal opinion."
    return True, ""