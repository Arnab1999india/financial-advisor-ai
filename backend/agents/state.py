"""
Agent State — LangGraph TypedDict
==================================
Defines the shared state object that flows through every node in the
LangGraph StateGraph.  Each node receives a copy and returns a (partial)
updated dict that is merged back into the shared state.
"""

from __future__ import annotations

from typing import Annotated, List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def add_token_calls(
    existing: Optional[List[Dict[str, Any]]],
    new: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Concatenate per-node LLM usage events across the graph."""
    return list(existing or []) + list(new or [])


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes.

    Fields
    ------
    messages:
        Full conversation message history.  The ``add_messages`` reducer
        handles both appending new messages and removing old ones via
        ``RemoveMessage`` (used by ``summarize_history_node``).
    query:
        The raw user query string (set at entry, never mutated).
    user_id:
        Optional caller identity.  When present, enables the Cognitive Memory
        System: ``memory_retrieve_node`` fetches past episodes for this user,
        and ``memory_save_node`` persists the current Q&A turn after generation.
        Anonymous requests (None) skip both memory nodes.
    intent:
        Classified intent: ``"financial_qa"``, ``"market_data"``,
        ``"chitchat"``, or ``"trade_sim"``.
        Set by the router node (or pre-set by gatekeeper's chitchat shortcut).
    domain_status:
        Set by ``gatekeeper_node``: ``"authorized"`` if the query is
        finance-related, ``"unauthorized"`` otherwise.
    summary:
        Rolling plain-text summary of older conversation turns.  Written by
        ``summarize_history_node`` when the message list exceeds the configured
        threshold.  Passed to the final generation node to minimise token costs.
    long_term_context:
        Formatted string of past Q&A episodes retrieved from the user's
        CognitiveMemoryStore.  Injected into the Master System Prompt so the
        LLM can reference past preferences and decisions.  Empty string for
        anonymous or first-time users.
    retrieved_docs:
        List of dicts returned by the RAG retrieval node, each with keys
        ``question``, ``answer``, ``category``.
    context:
        Formatted context string built from retrieved_docs, passed to Gemini.
    answer:
        Final answer string to be returned to the caller.
    confidence:
        Optional float [0, 1].  Populated by fallback paths only (Gemini
        does not return a calibrated confidence score).
    source_questions:
        Questions from the retrieved docs — surfaced to the frontend for
        transparency.
    error:
        Non-fatal error message captured during execution (e.g. LLM timeout).
        When set, the agent uses the fallback answer instead.
    token_calls:
        LLM usage events recorded by nodes that invoke Gemini.  The list
        reducer concatenates events from summarise, gatekeeper, and generate
        so FastAPI can roll them into one request-level telemetry payload.
    guardrail_amended:
        True when ``outbound_guardrail_node`` intercepted a directive answer
        and rewrote it into educational phrasing plus the statutory disclaimer.
    retrieval_query:
        Search string used against FAISS / web.  Starts as the user query and
        is replaced by ``query_rewrite_node`` when the grader finds documents
        irrelevant.
    rewrite_attempts:
        How many times the query has already been rewritten this turn.
        Caps the retrieve → grade → rewrite loop.
    retrieval_grade:
        ``"relevant"`` or ``"not_relevant"`` as set by ``grade_documents_node``.
    retrieval_source:
        ``"faiss"`` or ``"web"`` — which corpus supplied ``context``.
    source_chunks:
        Normalised retrieved snippets (confidence, timestamps, optional URL).
    sentence_mappings:
        Sentence-level links from the final answer onto ``source_chunks``.
    approval_required:
        True when the graph must pause for a human approval card.
    approval_reason:
        ``"high_probability_setup"`` or ``"user_requested_simulation"``.
    pending_action:
        Proposed paper-trade / alert parameters shown on the approval card.
    approval_decision:
        Resume payload after the user approves, edits, or rejects.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    user_id: Optional[str]
    intent: Optional[str]
    domain_status: Optional[str]
    summary: Optional[str]
    long_term_context: Optional[str]
    retrieved_docs: Optional[List[dict]]
    context: Optional[str]
    answer: Optional[str]
    confidence: Optional[float]
    source_questions: Optional[List[str]]
    error: Optional[str]
    model_used: Optional[str]
    token_calls: Annotated[List[Dict[str, Any]], add_token_calls]
    guardrail_amended: Optional[bool]
    retrieval_query: Optional[str]
    rewrite_attempts: Optional[int]
    retrieval_grade: Optional[str]
    retrieval_source: Optional[str]
    source_chunks: Optional[List[dict]]
    sentence_mappings: Optional[List[dict]]
    approval_required: Optional[bool]
    approval_reason: Optional[str]
    pending_action: Optional[dict]
    approval_decision: Optional[dict]
