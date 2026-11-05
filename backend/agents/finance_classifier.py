"""Gemini finance vs non-finance classifier — first domain gate after injection."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.domain_classifier import (
    _parse_json_object,
    classify_domain,
    load_domain_config,
    off_topic_message,
)
from agents.llm_client import invoke_llm
from agents.state import AgentState
from config import settings
from logger import logger
from telemetry import make_token_event

_CLASSIFIER_SYSTEM = (
    "You are a binary domain classifier for a finance-only assistant. "
    "Decide whether the user question is about finance. "
    "Finance includes: markets, investing, stocks, funds, ETFs (exchange-traded funds), "
    "bonds, taxes, banking, insurance, loans, credit, financial planning, retirement, "
    "trading, technical analysis (moving averages, RSI, MACD), ROI, valuation, "
    "and personal money management. "
    "Mark is_finance false for geography, recipes, coding, entertainment, "
    "politics, medical advice, shopping product returns, sports, and general trivia. "
    "Short greetings (hi, hello, thanks) may be treated as is_finance true with "
    "is_greeting true so the finance assistant can reply politely. "
    "Return ONLY compact JSON: "
    '{"is_finance": boolean, "is_greeting": boolean, "confidence": 0-1, "reason": "short"}.'
)


def _demo_mode() -> bool:
    key = (settings.gemini_api_key or "").strip()
    return not key or key == "your-gemini-api-key-here"


def _unauthorized_payload(*, reason: str, token_calls: list | None = None) -> dict:
    payload: dict = {
        "domain_status": "unauthorized",
        "finance_query": None,
        "answer": off_topic_message(),
        "confidence": 0.0,
        "model_used": "finance_classifier",
        "source_questions": [],
        "classifier_reason": reason,
    }
    if token_calls:
        payload["token_calls"] = token_calls
    return payload


def _authorized_payload(
    query: str,
    *,
    intent: str | None = None,
    reason: str = "",
    token_calls: list | None = None,
) -> dict:
    payload: dict = {
        "domain_status": "authorized",
        "finance_query": query,
        "classifier_reason": reason,
        "model_used": None,
    }
    if intent:
        payload["intent"] = intent
    if token_calls:
        payload["token_calls"] = token_calls
    return payload


def finance_classifier_node(state: AgentState) -> dict:
    """LLM-first finance vs non-finance gate. Non-finance stops the pipeline."""
    query = (state.get("query") or "").strip()
    if not query:
        return _unauthorized_payload(reason="empty_query")

    demo = _demo_mode()
    if demo:
        decision = classify_domain(query, demo_mode=True)
        logger.info(
            "Finance classifier (no Gemini key): fallback rules status=%s layer=%s",
            decision.status,
            decision.layer,
        )
        if decision.status == "authorized":
            return _authorized_payload(
                decision.finance_query or query,
                intent=decision.intent,
                reason=f"rules_fallback:{decision.reason}",
            )
        return _unauthorized_payload(reason=f"rules_fallback:{decision.reason}")

    from agents.nodes import _get_classifier_llm

    try:
        response = invoke_llm(
            _get_classifier_llm(),
            [
                SystemMessage(content=_CLASSIFIER_SYSTEM),
                HumanMessage(content=f'User message: """{query[:1500]}"""'),
            ],
            node="finance_classifier",
        )
        raw = (getattr(response, "content", None) or "").strip()
        parsed = _parse_json_object(raw)
        is_finance = bool(parsed.get("is_finance"))
        is_greeting = bool(parsed.get("is_greeting"))
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason") or "gemini_classifier")
        min_conf = float(load_domain_config().get("llm_min_confidence", 0.55))
        token_event = make_token_event("finance_classifier", settings.cheap_model, response)
        token_calls = [token_event]

        logger.info(
            "Finance classifier Gemini is_finance=%s greeting=%s conf=%.2f reason=%s",
            is_finance,
            is_greeting,
            confidence,
            reason[:120],
        )
        if is_greeting and confidence >= min_conf:
            return _authorized_payload(
                query,
                intent="chitchat",
                reason=reason,
                token_calls=token_calls,
            )
        if is_finance and confidence >= min_conf:
            return _authorized_payload(query, reason=reason, token_calls=token_calls)
        return _unauthorized_payload(reason=reason, token_calls=token_calls)
    except Exception as exc:
        logger.error(
            "Finance classifier Gemini failed (%s) — falling back to rules.",
            type(exc).__name__,
        )
        decision = classify_domain(query, demo_mode=True)
        if decision.status == "authorized":
            return _authorized_payload(
                decision.finance_query or query,
                intent=decision.intent,
                reason=f"rules_after_llm_error:{decision.reason}",
            )
        return _unauthorized_payload(reason=f"llm_error_then_rules:{decision.reason}")


def dispatch_finance_classifier(state: AgentState) -> str:
    return state.get("domain_status") or "unauthorized"
