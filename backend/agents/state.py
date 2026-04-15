"""
Agent State — LangGraph TypedDict
==================================
Defines the shared state object that flows through every node in the
LangGraph StateGraph.  Each node receives a copy and returns a (partial)
updated dict that is merged back into the shared state.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Mutable state passed between LangGraph nodes.

    Fields
    ------
    messages:
        Full conversation message history.  The ``Annotated[…, operator.add]``
        annotation tells LangGraph to *append* new messages rather than
        overwrite the list.
    query:
        The raw user query string (set at entry, never mutated).
    intent:
        Classified intent: ``"financial_qa"`` or ``"chitchat"``.
        Set by the router node.
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
    """

    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    intent: Optional[str]
    retrieved_docs: Optional[List[dict]]
    context: Optional[str]
    answer: Optional[str]
    confidence: Optional[float]
    source_questions: Optional[List[str]]
    error: Optional[str]
