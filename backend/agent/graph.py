"""Agent control loop for Mini-Wakili."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.nodes import (
    node_moderate_input,
    node_retrieve,
    node_decide,
    node_generate,
    node_verify_and_moderate,
)
from agent.state import AgentState


def run_agent(question: str) -> Dict[str, Any]:
    state: AgentState = {
        "question": question,
        "refined_question": None,
        "retrieved": [],
        "answer": None,
        "citations": [],
        "confidence": 0.0,
        "status": "ok",
        "message": None,
        "attempt": 0,
        "moderated_input": False,
        "moderated_output": False,
        "suggested_topics": [],
    }
    steps: List[Dict[str, Any]] = []

    try:
        state = node_moderate_input(state)
        steps.append(
            {"id": "moderate", "label": "Checking input safety", "status": "done"}
        )
        if state.get("status") == "refused":
            return _final(state, steps)

        for round_i in range(2):
            label = (
                "Searching the legal corpus"
                if round_i == 0
                else "Searching again with refined query"
            )
            steps.append(
                {
                    "id": f"retrieve_{round_i}",
                    "label": label,
                    "status": "done",
                    "detail": state.get("refined_question") or state["question"],
                }
            )
            state = node_retrieve(state)
            state = node_decide(state)
            conf = float(state.get("confidence") or 0.0)

            if state.get("status") == "retry":
                steps.append(
                    {
                        "id": f"decide_{round_i}",
                        "label": "Confidence low — deciding to search again",
                        "status": "done",
                        "detail": f"confidence={conf:.2f}",
                    }
                )
                continue

            steps.append(
                {
                    "id": f"decide_{round_i}",
                    "label": "Evidence strong enough — proceeding to answer",
                    "status": "done",
                    "detail": f"confidence={conf:.2f}",
                }
            )
            break

        steps.append(
            {"id": "generate", "label": "Drafting grounded answer", "status": "done"}
        )
        state = node_generate(state)

        steps.append(
            {
                "id": "verify",
                "label": "Verifying citations & output safety",
                "status": "done",
            }
        )
        state = node_verify_and_moderate(state)
        return _final(state, steps)

    except Exception as e:
        steps.append({"id": "error", "label": str(e), "status": "error"})
        return {
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "status": "error",
            "message": str(e),
            "suggested_topics": [],
            "meta": {"attempts": state.get("attempt", 0), "steps": steps},
        }


def _final(state: AgentState, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = state.get("status") or "ok"
    citations = state.get("citations") or []
    if status != "ok":
        citations = []

    return {
        "answer": state.get("answer"),
        "citations": citations,
        "confidence": float(state.get("confidence") or 0.0),
        "status": status,
        "message": state.get("message"),
        "suggested_topics": state.get("suggested_topics") or [],
        "meta": {
            "attempts": state.get("attempt", 0),
            "steps": steps,
        },
    }