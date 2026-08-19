"""Shared state for the LangGraph agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Citation(TypedDict):
    source_id: str
    passage: str
    score: float


class AgentState(TypedDict, total=False):
    question: str
    refined_question: Optional[str]
    retrieved: List[Dict[str, Any]]
    answer: Optional[str]
    citations: List[Citation]
    confidence: float
    status: str          # ok | low_confidence | refused | error
    message: Optional[str]
    attempt: int         # for re-query loop
    moderated_input: bool
    moderated_output: bool
