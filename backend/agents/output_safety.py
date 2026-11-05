"""Post-generation output safety (link exfiltration blocking)."""

from __future__ import annotations

from agents.state import AgentState
from logger import logger
from security.output_filter import filter_untrusted_links


def output_safety_node(state: AgentState) -> dict:
    answer = state.get("answer") or ""
    filtered = filter_untrusted_links(answer)
    if filtered != answer:
        logger.info("Output safety: removed or blocked non-allow-listed links.")
    return {"answer": filtered}
