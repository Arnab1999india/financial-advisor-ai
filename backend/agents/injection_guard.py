"""Graph-node injection check (defense in depth after domain gate)."""

from __future__ import annotations

from agents.state import AgentState
from logger import logger
from security.abuse_tracker import abuse_tracker
from security.injection_scan import is_borderline_query, scan_user_query
from security.policy import injection_refusal_message
from security.pii import mask_pii


def injection_guard_node(state: AgentState) -> dict:
    query = state.get("query") or ""
    subject = (state.get("user_id") or state.get("session_id") or "anonymous").strip()
    borderline = is_borderline_query(query)
    finding = scan_user_query(query, borderline=borderline)
    if not finding.blocked:
        return {"injection_blocked": False}

    reason = finding.reason_code or "QUERY_OVERRIDE"
    abuse_tracker.record(subject, reason)
    logger.warning(
        "Injection guard blocked query reason=%s pattern=%s preview=%s",
        reason,
        finding.pattern_id,
        mask_pii(query)[:120],
    )
    return {
        "injection_blocked": True,
        "injection_reason": reason,
        "answer": injection_refusal_message(),
        "confidence": 0.0,
        "model_used": "injection_guard",
        "source_questions": [],
    }


def dispatch_injection(state: AgentState) -> str:
    return "blocked" if state.get("injection_blocked") else "continue"
