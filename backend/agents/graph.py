"""
LangGraph StateGraph — Financial Bot Agent
==========================================
Defines the directed graph that orchestrates query handling:

    START
      │
      ▼
  [router]  ──── intent="chitchat" ────► [chitchat] ──► END
      │
      │ intent="financial_qa"
      ▼
  [rag_retrieve]
      │
      ▼
  [gemini_generate]
      │
      ▼
     END

All nodes are pure functions defined in ``agents.nodes``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    chitchat_node,
    gemini_generate_node,
    rag_retrieve_node,
    router_node,
)
from agents.state import AgentState
from logger import logger


# ---------------------------------------------------------------------------
# Conditional edge resolver
# ---------------------------------------------------------------------------

def _dispatch_intent(state: AgentState) -> str:
    """Return the next node name based on classified intent."""
    return state.get("intent", "financial_qa")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph():
    """Construct and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph agent ready for ``.invoke()`` / ``.ainvoke()``.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("gemini_generate", gemini_generate_node)
    graph.add_node("chitchat", chitchat_node)

    # ── Entry ──────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # ── Conditional dispatch after routing ────────────────────────────────
    graph.add_conditional_edges(
        "router",
        _dispatch_intent,
        {
            "financial_qa": "rag_retrieve",
            "chitchat": "chitchat",
        },
    )

    # ── Linear edges ──────────────────────────────────────────────────────
    graph.add_edge("rag_retrieve", "gemini_generate")
    graph.add_edge("gemini_generate", END)
    graph.add_edge("chitchat", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py
# ---------------------------------------------------------------------------
financial_agent = build_agent_graph()
