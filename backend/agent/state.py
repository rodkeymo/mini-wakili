"""Shared agent state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    refined_question: Optional[str]
    retrieved: List[Dict[str, Any]]
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    confidence: float
    status: str
    message: Optional[str]
    attempt: int
    moderated_input: bool
    moderated_output: bool
    suggested_topics: List[str]
    matter_id: str
    session_id: str
    memory_text: str
    skip_rag: bool