"""LangGraph-style node functions for Mini-Wakili (NVIDIA + refusal topics)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from agent.state import AgentState
from agent.tools import search_legal_corpus, refine_query
from safety.moderators import moderate_input, moderate_output, is_greeting_or_smalltalk
from safety.confidence import compute_confidence, should_refuse
from rag.topics import list_corpus_topics

RETRIEVAL_THRESHOLD = 0.42
MAX_ATTEMPTS = 2
REFUSAL_PHRASE = "I cannot answer from the available approved sources."

GREETING_REPLY = (
    "Hello — I'm Mini-Wakili, a legal research assistant for approved Kenyan statute sources. "
    "Ask a question about the Acts in the corpus (for example banking licences, capital markets, "
    "or foreign investment protection). I only answer from those sources and always cite them."
)

SYSTEM_PROMPT = """You are Wakili, a legal research assistant for a NCBA bank.
Answer ONLY using the passages provided below.
- Every factual statement must be followed by a citation in the form [source_id].
- If the passages do not contain enough information, reply exactly:
  "I cannot answer from the available approved sources."
- Never invent sections, cases, or dates.
- Never present an uncited claim as law.
- This is a draft for lawyer review, not a final legal opinion.
"""


def _get_chat_llm():
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
    if not hits:
        return REFUSAL_PHRASE
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

        memory = ""  # optional: state memory injected by graph if you add it
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
        return _simple_grounded_answer(question, hits)


def _is_refusal_answer(answer: str | None) -> bool:
    if not answer:
        return True
    a = answer.strip().lower()
    return (
        a == REFUSAL_PHRASE.lower()
        or "cannot answer from the available approved sources" in a
    )


def _refusal_state(state: AgentState, message: str) -> AgentState:
    return {
        **state,
        "answer": None,
        "citations": [],
        "status": "low_confidence",
        "message": message,
        "suggested_topics": [],
    }


def node_moderate_input(state: AgentState) -> AgentState:
    q = state["question"]
    ok, reason = moderate_input(q)
    if not ok:
        return {
            **state,
            "status": "refused",
            "message": reason,
            "answer": None,
            "citations": [],
            "confidence": 0.0,
            "moderated_input": False,
            "suggested_topics": [],
        }

    if is_greeting_or_smalltalk(q):
        return {
            **state,
            "status": "ok",
            "answer": GREETING_REPLY,
            "citations": [],
            "confidence": 1.0,
            "message": None,
            "suggested_topics": list_corpus_topics(8),
            "moderated_input": True,
            "skip_rag": True,
        }

    return {**state, "moderated_input": True, "attempt": state.get("attempt", 0), "skip_rag": False}


def node_retrieve(state: AgentState) -> AgentState:
    q = state.get("refined_question") or state["question"]
    hits = search_legal_corpus(q, top_k=5)
    return {**state, "retrieved": hits}


def node_decide(state: AgentState) -> AgentState:
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
        return _refusal_state(
            state,
            "The corpus does not contain sufficiently relevant material "
            "to answer safely. Please rephrase your question.",
        )

    answer = _llm_grounded_answer(state["question"], hits)

    if _is_refusal_answer(answer):
        return _refusal_state(state, REFUSAL_PHRASE)

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
        "suggested_topics": [],
    }


def node_verify_and_moderate(state: AgentState) -> AgentState:
    if state.get("status") in ("low_confidence", "refused", "error"):
        return {
            **state,
            "citations": [],
            "suggested_topics": [],
        }

    answer = state.get("answer") or ""
    citations = state.get("citations") or []

    if _is_refusal_answer(answer):
        return _refusal_state(state, REFUSAL_PHRASE)

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
            "suggested_topics": [],
        }

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "status": "ok",
        "moderated_output": True,
        "suggested_topics": [],
    }