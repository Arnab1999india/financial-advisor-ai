"""Post-generation structured output validation with one repair attempt."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.domain_classifier import maybe_append_investment_disclaimer
from agents.state import AgentState
from config import settings
from logger import logger
from telemetry import make_token_event
from validation.output_validator import (
    OutputValidationResult,
    strip_untracked_sentences,
    validate_structured_output,
)
from validation.structured_output import (
    RELIABLE_ANSWER_FALLBACK,
    StructuredAnswer,
    parse_structured_answer,
)

_SKIP_MODELS = {
    "fallback",
    "static",
    "injection_guard",
    "grounding_gate",
    "domain_guardrail",
    "gatekeeper",
    "yfinance",
    "market_data_error",
}


def _skip_validation(state: AgentState) -> bool:
    intent = state.get("intent")
    model = (state.get("model_used") or "").strip()
    if intent in {"chitchat", "trade_sim"}:
        return True
    if model in _SKIP_MODELS:
        return True
    if state.get("retrieval_source") == "market_data":
        return True
    return False


def _used_market_data(state: AgentState) -> bool:
    return state.get("retrieval_source") == "market_data" or state.get("intent") == "market_data"


def _repair_prompt(structured: StructuredAnswer, reasons: list[str], state: AgentState) -> str:
    return (
        "Fix the JSON answer so it complies with grounding rules.\n"
        f"Validation errors: {', '.join(reasons)}\n"
        f"Evidence context excerpt:\n{(state.get('context') or '')[:2000]}\n"
        f"Previous JSON:\n{structured.model_dump_json()}\n"
        "Return ONLY corrected JSON with the same schema."
    )


def output_validate_node(state: AgentState) -> dict:
    if _skip_validation(state):
        return {}

    raw = (state.get("structured_raw") or state.get("answer") or "").strip()
    if not raw:
        return {
            "answer": RELIABLE_ANSWER_FALLBACK,
            "model_used": "output_validate",
            "confidence": 0.0,
        }

    docs = state.get("retrieved_docs") or []
    context = state.get("context") or ""
    used_market = _used_market_data(state)
    retried = bool(state.get("output_validate_retried"))

    try:
        structured = parse_structured_answer(raw)
    except Exception as exc:
        logger.info("Structured parse failed (%s); attempting repair.", exc)
        structured = StructuredAnswer(
            answer=raw,
            citations=[],
            confidence="low",
            used_market_data=used_market,
        )

    result = validate_structured_output(
        structured,
        docs=docs,
        context=context,
        used_market_data=used_market,
    )

    token_calls: list[dict] = []
    if not result.ok and not retried:
        repaired, repair_event = _attempt_repair(state, structured, result.reasons)
        if repair_event:
            token_calls.append(repair_event)
        if repaired is not None:
            structured = repaired
            result = validate_structured_output(
                structured,
                docs=docs,
                context=context,
                used_market_data=used_market,
            )
            retried = True

    if not result.ok:
        logger.warning("Output validation failed: %s", ",".join(result.reasons))
        payload = {
            "answer": RELIABLE_ANSWER_FALLBACK,
            "model_used": "output_validate",
            "confidence": 0.0,
            "output_validate_retried": retried,
        }
        if token_calls:
            payload["token_calls"] = token_calls
        return payload

    answer = strip_untracked_sentences(structured.answer, docs, context)
    query = (state.get("finance_query") or state.get("query") or "").strip()
    answer = maybe_append_investment_disclaimer(answer, query)
    conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
    payload = {
        "answer": answer,
        "confidence": conf_map.get(structured.confidence, 0.7),
        "model_used": state.get("model_used"),
        "structured_citations": [c.model_dump() for c in structured.citations],
        "output_validate_retried": retried,
    }
    if token_calls:
        payload["token_calls"] = token_calls
    return payload


def _attempt_repair(
    state: AgentState,
    structured: StructuredAnswer,
    reasons: list[str],
) -> tuple[StructuredAnswer | None, dict | None]:
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key or api_key == "your-gemini-api-key-here":
        return None, None
    try:
        from agents.nodes import _get_cheap_llm

        from agents.llm_client import invoke_llm

        response = invoke_llm(
            _get_cheap_llm(),
            [
                SystemMessage(content="You repair JSON finance answers to satisfy validators."),
                HumanMessage(content=_repair_prompt(structured, reasons, state)),
            ],
            node="output_validate_repair",
        )
        token_event = make_token_event("output_validate_repair", settings.cheap_model, response)
        parsed = parse_structured_answer(response.content or "")
        return parsed, token_event
    except Exception as exc:
        logger.warning("Output repair failed: %s", exc)
        return None, None
