"""Step 1 — detect finance/non-finance ambiguous wording and ask for clarification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agents.domain_classifier import load_domain_config
from agents.state import AgentState
from config import settings
from logger import logger


@dataclass(frozen=True)
class AmbiguityResult:
    term: str
    message: str


def _compile_list(key: str) -> tuple[re.Pattern[str], ...]:
    cfg = load_domain_config()
    return tuple(re.compile(p, re.IGNORECASE) for p in cfg.get(key, []))


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _term_present(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def _non_finance_context_lock(text: str) -> bool:
    if _matches_any(_compile_list("off_topic_deny_patterns"), text):
        return True
    return _matches_any(_compile_list("ambiguity_non_finance_lock_patterns"), text)


def _finance_context_lock(text: str) -> bool:
    return _matches_any(_compile_list("ambiguity_finance_lock_patterns"), text)


def assess_ambiguity(query: str) -> Optional[AmbiguityResult]:
    """Return clarification prompt when a homograph term lacks disambiguating context."""
    if not settings.three_step_ambiguity_check:
        return None

    raw = (query or "").strip()
    if not raw:
        return None

    if _matches_any(_compile_list("ambiguity_skip_patterns"), raw):
        return None

    cfg = load_domain_config()
    terms: dict[str, str] = dict(cfg.get("ambiguous_term_clarifications") or {})
    if not terms:
        return None

    if _non_finance_context_lock(raw) and not _finance_context_lock(raw):
        return None

    for term, message in terms.items():
        if not _term_present(term, raw):
            continue

        fin_key = f"ambiguity_finance_lock_{term}"
        non_key = f"ambiguity_non_finance_lock_{term}"
        if _matches_any(_compile_list(fin_key), raw):
            continue
        if _matches_any(_compile_list(non_key), raw):
            continue
        if _finance_context_lock(raw):
            continue

        return AmbiguityResult(term=term, message=message.strip())

    return None


def ambiguity_check_node(state: AgentState) -> dict:
    query = state.get("query") or ""
    hit = assess_ambiguity(query)
    if hit is None:
        return {"ambiguity_blocked": False}

    logger.info("Ambiguity check: clarification required for term=%s", hit.term)
    return {
        "ambiguity_blocked": True,
        "answer": hit.message,
        "confidence": 0.0,
        "model_used": "ambiguity_check",
        "source_questions": [],
    }


def dispatch_ambiguity(state: AgentState) -> str:
    return "clarify" if state.get("ambiguity_blocked") else "continue"
