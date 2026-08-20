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

_GREETING = re.compile(
    r"^\s*("
    r"hi|hello|hey|howdy|good\s*(morning|afternoon|evening)|"
    r"how\s+are\s+you|what'?s\s+up|who\s+are\s+you|"
    r"thanks|thank\s+you|ok|okay|bye|goodbye|"
    r"help|what\s+can\s+you\s+do"
    r")[\s!?.]*$",
    re.I,
)

_GREETING_PREFIX = (
    r"(?:"
    r"(?:hi|hello|hey|howdy|good\s*(?:morning|afternoon|evening))"
    r"[\s,!?.]*\s*"
    r")?"
)

_INTRO_NAME = (
    r"(?:"
    r"(?:my\s+name\s+is|i'?m|i\s+am|call\s+me|this\s+is)"
    r"\s+"
    r"(?![Tt]he\b|[Aa]n?\b)"
    r"[A-Za-z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*){0,2}"
    r")"
)

_INTRO_ONLY = re.compile(
    r"^\s*" + _GREETING_PREFIX + _INTRO_NAME + r"[\s!?.]*$",
    re.I,
)

_INTRO_BLACKLIST = re.compile(
    r"\b(charged|need|needs|wanted|want|looking|mentioned|sued|accused|"
    r"arrested|contract|document|lease|deed|director|fiduciary|rights|"
    r"obligations|advice|help|question|requirement)\b",
    re.I,
)

_NICE_PLEASED = re.compile(
    r"^\s*"
    r"(?:nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you|"
    r"great\s+to\s+meet\s+you|good\s+to\s+meet\s+you)"
    r"[\s!?.]*$",
    re.I,
)


def is_greeting_or_smalltalk(text: str) -> bool:
    q = (text or "").strip()
    if len(q) > 120:
        return False
    if _GREETING.match(q):
        return True
    if _NICE_PLEASED.match(q):
        return True
    if _INTRO_ONLY.match(q) and not _INTRO_BLACKLIST.search(q):
        return True
    return False


def moderate_input(text: str) -> Tuple[bool, str]:
    q = (text or "").strip()
    if len(q) < 1:
        return False, "Please enter a question."
    if _INJECTION.search(q):
        return False, "Request blocked by input safety checks."
    if len(q) < 3 and not is_greeting_or_smalltalk(q):
        return False, "Please enter a clearer legal research question."
    return True, ""


def moderate_output(answer: str, citations: List[Dict[str, Any]]) -> Tuple[bool, str]:
    a = (answer or "").strip()
    if not a:
        return False, "Empty model output blocked."
    if _FINAL_OPINION.search(a):
        return False, "Output blocked: language resembles a final legal opinion."
    return True, ""