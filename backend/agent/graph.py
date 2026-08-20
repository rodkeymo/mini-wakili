"""Agent control loop with optional step streaming."""

from __future__ import annotations

from typing import Any, Dict, Generator, List

from agent.nodes import (
    node_moderate_input,
    node_retrieve,
    node_decide,
    node_generate,
    node_verify_and_moderate,
)
from agent.state import AgentState


def _step(id_: str, label: str, status: str = "done", **extra: Any) -> Dict[str, Any]:
    return {"id": id_, "label": label, "status": status, **extra}


def run_agent_events(
    question: str,
) -> Generator[Dict[str, Any], None, None]:
    """Yields `{"type": "step", "step": {...}}` events, then one final
    `{"type": "result", "result": {...}}`.
    """
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
        "skip_rag": False,
    }
    steps: List[Dict[str, Any]] = []

    def emit(step: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        steps.append(step)
        yield {"type": "step", "step": step}

    try:
        yield from emit(_step("moderate", "Checking input safety", "running"))
        state = node_moderate_input(state)
        # replace last entry with done (keep same id)
        steps[-1] = _step("moderate", "Checking input safety", "done")
        yield {"type": "step", "step": steps[-1]}

        if state.get("status") == "refused" or state.get("skip_rag"):
            if state.get("skip_rag"):
                yield from emit(
                    _step("smalltalk", "Greeting — skipped retrieval", "done")
                )
            yield {"type": "result", "result": _final(state, steps)}
            return

        for round_i in range(2):
            label = (
                "Searching the legal corpus"
                if round_i == 0
                else "Searching again with refined query"
            )
            detail = state.get("refined_question") or state["question"]

            yield from emit(
                _step(f"retrieve_{round_i}", label, "running", detail=detail)
            )
            state = node_retrieve(state)
            steps[-1] = _step(f"retrieve_{round_i}", label, "done", detail=detail)
            yield {"type": "step", "step": steps[-1]}

            yield from emit(
                _step(f"decide_{round_i}", "Evaluating retrieval confidence", "running")
            )
            state = node_decide(state)
            conf = float(state.get("confidence") or 0.0)

            if state.get("status") == "retry":
                steps[-1] = _step(
                    f"decide_{round_i}",
                    "Confidence low — deciding to search again",
                    "done",
                    detail=f"confidence={conf:.2f}",
                )
                yield {"type": "step", "step": steps[-1]}
                continue

            steps[-1] = _step(
                f"decide_{round_i}",
                "Evidence strong enough — proceeding to answer",
                "done",
                detail=f"confidence={conf:.2f}",
            )
            yield {"type": "step", "step": steps[-1]}
            break

        yield from emit(_step("generate", "Drafting grounded answer", "running"))
        state = node_generate(state)
        steps[-1] = _step("generate", "Drafting grounded answer", "done")
        yield {"type": "step", "step": steps[-1]}

        yield from emit(
            _step("verify", "Verifying citations & output safety", "running")
        )
        state = node_verify_and_moderate(state)
        steps[-1] = _step("verify", "Verifying citations & output safety", "done")
        yield {"type": "step", "step": steps[-1]}

        yield {"type": "result", "result": _final(state, steps)}

    except Exception as e:
        yield from emit(_step("error", str(e), "error"))
        yield {
            "type": "result",
            "result": {
                "answer": None,
                "citations": [],
                "confidence": 0.0,
                "status": "error",
                "message": str(e),
                "suggested_topics": [],
                "meta": {"attempts": state.get("attempt", 0), "steps": steps},
            },
        }


def run_agent(question: str) -> Dict[str, Any]:
    """Non-streaming: consume events, return final result only."""
    result: Dict[str, Any] | None = None
    for ev in run_agent_events(question):
        if ev["type"] == "result":
            result = ev["result"]
    return result or {
        "answer": None,
        "citations": [],
        "confidence": 0.0,
        "status": "error",
        "message": "No result",
        "suggested_topics": [],
        "meta": {"attempts": 0, "steps": []},
    }


def _final(state: AgentState, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = state.get("status") or "ok"
    citations = state.get("citations") or []
    answer = state.get("answer")

    if state.get("skip_rag") or status != "ok" or not answer:
        citations = []
    if status != "ok":
        citations = []

    return {
        "answer": answer,
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
