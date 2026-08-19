"""LLM client — NVIDIA first, then OpenAI, then deterministic fallback."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_chat_llm():
    """
    Returns a LangChain Chat model, or None if no key is configured
    (in which case the deterministic grounded answerer is used).
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
            api_key=nvidia_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            temperature=0.1,          # low for legal grounding
            max_tokens=1024,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            temperature=0.1,
            max_tokens=1024,
        )

    return None   # fall back to deterministic answerer