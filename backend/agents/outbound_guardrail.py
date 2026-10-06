"""
Outbound advice guardrail
=========================
Inspects the generated answer *after* the LLM node.  If the text contains
direct, personalised financial advice (e.g. "You should buy stock X"), the
node intercepts it, rewrites it into educational phrasing, and appends the
statutory risk disclaimer.

Detection is regex-based so it runs even in demo/fallback mode.  When a
Gemini key is present, Flash is used to rewrite the full answer; otherwise a
deterministic phrase substitution is applied.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from agents.state import AgentState
from config import settings
from logger import logger
from telemetry import make_token_event

STATUTORY_RISK_DISCLAIMER = (
    "\n\n---\n"
    "**Statutory risk disclaimer:** This material is for general educational "
    "purposes only. It does not constitute investment advice, a personal "
    "recommendation, or an offer or solicitation to buy or sell any security "
    "or financial instrument. Investing involves risk, including the possible "
    "loss of principal. Past performance is not a reliable indicator of future "
    "results. You should consider your own objectives, financial situation, "
    "and needs, and consult a licensed financial adviser, before making any "
    "investment decision. The operator of this service is not acting as a "
    "registered investment adviser."
)

_DISCLAIMER_MARKER = "does not constitute investment advice"

_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\byou should (?:not )?(?:buy|sell|short|long|invest|hold|dump|purchase|acquire|exit)\b",
        r"\byou ought to (?:buy|sell|invest|short|hold)\b",
        r"\byou need to (?:buy|sell|invest|short|hold)\b",
        r"\byou must (?:buy|sell|invest|hold)\b",
        r"\bi recommend (?:that you )?(?:buy|sell|invest|short|holding|purchasing)\b",
        r"\bmy (?:advice|recommendation) (?:is|would be) to (?:buy|sell|invest)\b",
        r"\b(?:i would|i'd) (?:buy|sell|invest in|go long|go short)\b",
        r"\bgo (?:all in|long|short)(?:\s+on|\s+into)?\b",
        r"\bthis is a strong (?:buy|sell)\b",
        r"\bbuy (?:the dip|now|immediately|today)\b",
        r"\bsell (?:now|immediately|everything|all your)\b",
        r"\bput (?:all |your )?(?:your )?money (?:into|in)\b",
        r"\ballocate \d+\s*% of (?:your )?(?:portfolio|money|savings|income)\b",
        r"\bdon't (?:wait|hesitate),? (?:buy|sell|invest)\b",
        r"\byou (?:should|need to) (?:go )?(?:all[- ]in|double down)\b",
        r"\bguarantee(?:d)? (?:returns?|profit|income)\b",
    )
)

_PHRASE_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\byou should buy\b", re.IGNORECASE), "some investors research whether to buy"),
    (re.compile(r"\byou should sell\b", re.IGNORECASE), "some investors research whether to sell"),
    (re.compile(r"\byou should invest\b", re.IGNORECASE), "some investors research whether to invest"),
    (re.compile(r"\byou should hold\b", re.IGNORECASE), "holding is one approach some investors study"),
    (re.compile(r"\byou should short\b", re.IGNORECASE), "shorting is a strategy some investors study"),
    (re.compile(r"\byou need to (buy|sell|invest)\b", re.IGNORECASE), r"some investors study whether to \1"),
    (
        re.compile(r"\byou must (buy|sell|invest)\b", re.IGNORECASE),
        r"whether to \1 is a personal decision that depends on individual circumstances",
    ),
    (
        re.compile(r"\bi recommend (?:that you )?(buy|sell|invest(?: in)?|holding)\b", re.IGNORECASE),
        r"one educational topic is \1",
    ),
    (
        re.compile(r"\bthis is a strong (buy|sell)\b", re.IGNORECASE),
        r"analyst commentary sometimes uses a 'strong \1' rating, which is not a personal instruction",
    ),
    (re.compile(r"\bbuy (?:now|immediately|today)\b", re.IGNORECASE), "timing a purchase is a personal decision"),
    (re.compile(r"\bsell (?:now|immediately)\b", re.IGNORECASE), "timing a sale is a personal decision"),
    (
        re.compile(r"\bgo all in\b", re.IGNORECASE),
        "concentrating an entire portfolio is a high-risk approach discussed in education only",
    ),
    (
        re.compile(r"\bguarantee(?:d)? (?:returns?|profit|income)\b", re.IGNORECASE),
        "no return or profit is guaranteed",
    ),
)

_REWRITE_SYSTEM_PROMPT = (
    "You are a compliance editor for a financial-education chatbot. "
    "Rewrite the assistant reply so it contains NO personalised buy, sell, "
    "hold, or allocate instructions. Convert directives into general "
    "educational discussion. Preserve factual content and named securities "
    "as examples only. Do not add new numbers, tickers, or promises. "
    "Do not append a disclaimer — that is added separately. "
    "Output ONLY the rewritten reply."
)

_EDUCATIONAL_PREFIX = "Educational framing (not a personal recommendation):\n\n"


def contains_direct_advice(text: str) -> bool:
    """Return True if *text* contains personalised / directive investment advice."""
    if not (text or "").strip():
        return False
    return any(pattern.search(text) for pattern in _ADVICE_PATTERNS)


def _disclaimer_already_present(text: str) -> bool:
    return _DISCLAIMER_MARKER in (text or "").lower()


def append_statutory_disclaimer(text: str) -> str:
    """Append the statutory disclaimer unless it is already present."""
    body = (text or "").rstrip()
    if _disclaimer_already_present(body):
        return body
    return body + STATUTORY_RISK_DISCLAIMER


def amend_to_educational(text: str) -> str:
    """Deterministic rewrite of directive phrases into educational language."""
    amended = text or ""
    for pattern, replacement in _PHRASE_SUBSTITUTIONS:
        amended = pattern.sub(replacement, amended)
    if not amended.lower().startswith("educational framing"):
        amended = _EDUCATIONAL_PREFIX + amended
    return amended


def _rewrite_with_flash(answer: str) -> tuple[str, dict | None]:
    """Ask Flash to rewrite *answer*. Returns (text, token_event_or_none)."""
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key or api_key == "your-gemini-api-key-here":
        return amend_to_educational(answer), None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        llm = ChatGoogleGenerativeAI(
            model=settings.cheap_model,
            temperature=0.0,
            google_api_key=api_key,
        )
        response = llm.invoke(
            [
                SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=f"Rewrite this assistant reply:\n\n{answer}"),
            ]
        )
        rewritten = (response.content or "").strip()
        event = make_token_event("outbound_guardrail", settings.cheap_model, response)
        if not rewritten:
            return amend_to_educational(answer), event
        if contains_direct_advice(rewritten):
            logger.info("Outbound guardrail: Flash rewrite still directive — using deterministic amend.")
            return amend_to_educational(rewritten), event
        if not rewritten.lower().startswith("educational"):
            rewritten = _EDUCATIONAL_PREFIX + rewritten
        return rewritten, event
    except Exception as exc:
        logger.warning("Outbound guardrail Flash rewrite failed (%s); using deterministic amend.", exc)
        return amend_to_educational(answer), None


def outbound_guardrail_node(state: AgentState) -> dict:
    """Post-generation compliance node.

    If the current answer contains direct financial advice, intercept it,
    rewrite to educational phrasing, append the statutory disclaimer, and
    replace the last AI message so downstream memory stores the amended text.
    """
    answer = (state.get("answer") or "").strip()
    if not answer or not contains_direct_advice(answer):
        return {"guardrail_amended": False}

    logger.info("Outbound guardrail: direct financial advice detected — intercepting answer.")
    rewritten, token_event = _rewrite_with_flash(answer)
    amended = append_statutory_disclaimer(rewritten)

    payload: dict = {
        "answer": amended,
        "guardrail_amended": True,
        "messages": [AIMessage(content=amended)],
    }

    last_ai = next(
        (m for m in reversed(state.get("messages") or []) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai is not None and getattr(last_ai, "id", None):
        payload["messages"] = [
            RemoveMessage(id=last_ai.id),
            AIMessage(content=amended),
        ]

    if token_event:
        payload["token_calls"] = [token_event]
    return payload
