"""LangGraph-style node functions for Mini-Wakili (with NVIDIA LLM support)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from agent.state import AgentState
from agent.tools import search_legal_corpus, refine_query
from safety.moderators import moderate_input, moderate_output
from safety.confidence import compute_confidence, should_refuse

# Minimum similarity to proceed without re-query
RETRIEVAL_THRESHOLD = 0.42
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """You are Wakili, a legal research assistant for a Kenyan bank.
Answer ONLY using the passages provided below.
- Every factual statement must be followed by a citation in the form [source_id].
- If the passages do not contain enough information, reply exactly:
  "I cannot answer from the available approved sources."
- Never invent sections, cases, or dates.
- Never present an uncited claim as law.
- This is a draft for lawyer review, not a final legal opinion.
"""


# ---------------------------------------------------------------------------
# LLM helper (NVIDIA first → OpenAI → deterministic fallback)
# ---------------------------------------------------------------------------

def _get_chat_llm():
    """
    Returns a LangChain Chat model, or None if no API key is configured.
    Prefers NVIDIA (OpenAI-compatible endpoint), then OpenAI.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
                api_key=nvidia_key,
                base_url=os.getenv(
                    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
                ),
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception:
            pass

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception:
            pass

    return None


def _simple_grounded_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback when no LLM API key is available.
    Keeps the take-home runnable offline.
    """
    if not hits:
        return "I cannot answer from the available approved sources."

    top = hits[0]
    sid = top.get("id", "src_1")
    text = (top.get("text") or "").strip()
    snippet = text[:350] + ("…" if len(text) > 350 else "")

    return (
        f"Based on the available sources, the following is relevant to your question "
        f"(\"{question}\"):\n\n"
        f"{snippet} [{sid}]\n\n"
        f"Please review the cited passage and consult a qualified lawyer for advice "
        f"specific to your matter. This is not a final legal opinion."
    )


def _llm_grounded_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    Call NVIDIA / OpenAI with a strict grounding prompt.
    Falls back to the deterministic answerer on any failure.
    """
    llm = _get_chat_llm()
    if llm is None:
        return _simple_grounded_answer(question, hits)

    context_parts = []
    for i, h in enumerate(hits[:5], 1):
        sid = h.get("id", f"src_{i}")
        context_parts.append(f"[{sid}]\n{h.get('text', '')}")
    context = "\n\n".join(context_parts)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {question}"
            ),
        ]
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        return (text or "").strip() or _simple_grounded_answer(question, hits)
    except Exception:
        # Network / auth / rate-limit → degrade gracefully
        return _simple_grounded_answer(question, hits)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_moderate_input(state: AgentState) -> AgentState:
    ok, reason = moderate_input(state["question"])
    if not ok:
        return {
            **state,
            "status": "refused",
            "message": reason,
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "moderated_input": False,
        }
    return {**state, "moderated_input": True, "attempt": state.get("attempt", 0)}


def node_retrieve(state: AgentState) -> AgentState:
    q = state.get("refined_question") or state["question"]
    hits = search_legal_corpus(q, top_k=6)
    return {**state, "retrieved": hits}


def node_decide(state: AgentState) -> AgentState:
    """
    Agentic decision: if retrieval is weak and attempts remain,
    refine the query and signal another retrieval round.
    """
    hits = state.get("retrieved") or []
    attempt = state.get("attempt", 0)
    conf = compute_confidence(hits)

    if conf < RETRIEVAL_THRESHOLD and attempt + 1 < MAX_ATTEMPTS:
        refined = refine_query(state["question"], hits)
        return {
            **state,
            "refined_question": refined,
            "attempt": attempt + 1,
            "confidence": conf,
            "status": "retry",
        }

    return {**state, "confidence": conf, "status": "ready"}


def node_generate(state: AgentState) -> AgentState:
    hits = state.get("retrieved") or []
    conf = state.get("confidence", 0.0)

    if should_refuse(hits, conf):
        return {
            **state,
            "answer": None,
            "citations": [],
            "status": "low_confidence",
            "message": (
                "The corpus does not contain sufficiently relevant material "
                "to answer safely. Please rephrase or consult a qualified lawyer."
            ),
        }

    # Prefer NVIDIA / OpenAI; fall back to deterministic grounded answer
    answer = _llm_grounded_answer(state["question"], hits)

    citations = [
        {
            "source_id": h.get("id", f"src_{i}"),
            "passage": (h.get("text") or "")[:400],
            "score": float(h.get("score", 0.0)),
        }
        for i, h in enumerate(hits[:5], 1)
    ]

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "status": "ok",
        "message": None,
    }


def node_verify_and_moderate(state: AgentState) -> AgentState:
    if state.get("status") in ("low_confidence", "refused", "error"):
        return state

    answer = state.get("answer") or ""
    citations = state.get("citations") or []

    # Citation faithfulness: every [source_id] in the answer must exist
    cited_ids = set(re.findall(r"\[([^\]]+)\]", answer))
    available = {c["source_id"] for c in citations}
    missing = cited_ids - available
    if missing:
        for mid in missing:
            answer = answer.replace(f"[{mid}]", "")
        answer = re.sub(r"\s{2,}", " ", answer).strip()

    ok, reason = moderate_output(answer, citations)
    if not ok:
        return {
            **state,
            "answer": None,
            "citations": [],
            "status": "refused",
            "message": reason,
            "moderated_output": False,
        }

    return {
        **state,
        "answer": answer,
        "status": "ok",
        "moderated_output": True,
    }