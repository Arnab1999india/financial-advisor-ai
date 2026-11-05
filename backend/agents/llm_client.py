"""Resilient LLM invocation wrapper."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from config import settings
from resilience.external import ProviderUnavailableError, call_external


def invoke_llm(llm, messages: list[BaseMessage], *, node: str):
    def _call():
        return llm.invoke(messages)

    try:
        return call_external("gemini", _call, timeout_seconds=settings.llm_timeout_seconds)
    except ProviderUnavailableError as exc:
        raise RuntimeError("LLM service is busy. Please retry shortly.") from exc
