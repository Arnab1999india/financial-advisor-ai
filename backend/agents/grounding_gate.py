"""
Pre-generation grounding gate — refuse LLM calls when evidence is absent.
"""

from __future__ import annotations

from agents.state import AgentState
from config import settings
from logger import logger
from rag.grounding_policy import insufficient_evidence_message


def _kb_docs_only(state: AgentState) -> bool:
    docs = state.get("retrieved_docs") or []
    if not docs:
        return False
    for doc in docs:
        source_type = (doc.get("source_type") or "faiss").lower()
        if source_type in {"web", "market_data"}:
            return False
    return True


def has_grounding_evidence(state: AgentState) -> bool:
    if settings.strict_two_step_kb_only:
        return _kb_docs_only(state) and (state.get("retrieval_grade") == "relevant")
    docs = state.get("retrieved_docs") or []
    if docs:
        return True
    context = (state.get("context") or "").strip()
    return bool(context)


def assess_evidence_strength(state: AgentState) -> str:
    """Return ``full``, ``partial``, or ``none``."""
    docs = state.get("retrieved_docs") or []
    if not docs:
        return "none"
    confidences = [float(d.get("confidence") or 0.0) for d in docs]
    top = max(confidences) if confidences else 0.0
    if top >= settings.grounding_full_confidence:
        return "full"
    if top >= settings.retrieval_min_confidence:
        return "partial"
    return "none"


def grounding_gate_node(state: AgentState) -> dict:
    """Evaluate whether generation is allowed."""
    strength = assess_evidence_strength(state)
    if strength == "none" and not has_grounding_evidence(state):
        logger.info("Grounding gate: no evidence — will refuse generation.")
        return {"grounding_status": "none"}
    logger.info("Grounding gate: evidence strength=%s", strength)
    return {"grounding_status": strength}


def insufficient_evidence_node(state: AgentState) -> dict:
    """Fixed response when documents and tools provide no usable evidence."""
    return {
        "answer": insufficient_evidence_message(),
        "confidence": 0.0,
        "source_questions": [],
        "model_used": "grounding_gate",
        "grounding_status": "none",
    }


def dispatch_grounding(state: AgentState) -> str:
    if settings.strict_two_step_kb_only:
        if state.get("retrieval_grade") != "relevant":
            return "insufficient_evidence"
        if not _kb_docs_only(state):
            return "insufficient_evidence"
    status = state.get("grounding_status") or "full"
    if status == "none":
        return "insufficient_evidence"
    if settings.strict_two_step_kb_only and status == "partial":
        return "insufficient_evidence"
    return "generate"
