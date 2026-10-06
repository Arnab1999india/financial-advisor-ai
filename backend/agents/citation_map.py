"""
Citation map node
=================
Runs after generation (and the outbound guardrail) so mappings reflect the
final answer the user sees.
"""

from __future__ import annotations

from agents.state import AgentState
from logger import logger
from rag.citations import map_answer_to_chunks, source_chunks_payload


def citation_map_node(state: AgentState) -> dict:
    """Attach sentence-level source mappings and normalised source chunks."""
    answer = state.get("answer") or ""
    docs = state.get("retrieved_docs") or []
    chunks = source_chunks_payload(docs)
    mappings = map_answer_to_chunks(answer, docs)
    logger.info(
        "Citation map: %d sentence mapping(s) across %d chunk(s).",
        len(mappings),
        len(chunks),
    )
    return {
        "source_chunks": chunks,
        "sentence_mappings": mappings,
    }
