"""
LangGraph StateGraph — Financial Bot Agent
==========================================
Defines the directed graph that orchestrates query handling:

    START → summarize_history → memory_retrieve → gatekeeper
         unauthorized → out_of_scope → END
         authorized → router
              chitchat → chitchat → END
              financial_qa → rag_retrieve → grade_documents
                   relevant → gemini_generate
                   not_relevant + rewrites left → query_rewrite → rag_retrieve
                   not_relevant + budget spent → web_search → gemini_generate
              gemini_generate → outbound_guardrail → citation_map → setup_gate
                   approval required → human_approval → execute_simulation → memory_save → END
                   otherwise → memory_save → END
              trade_sim → human_approval → execute_simulation → memory_save → END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.human_approval import (
    dispatch_approval,
    execute_simulation_node,
    human_approval_node,
    setup_gate_node,
)
from agents.nodes import (
    chitchat_node,
    gatekeeper_node,
    gemini_generate_node,
    memory_retrieve_node,
    memory_save_node,
    market_data_node,
    out_of_scope_node,
    rag_retrieve_node,
    router_node,
    summarize_history_node,
)
from agents.citation_map import citation_map_node
from agents.outbound_guardrail import outbound_guardrail_node
from agents.retrieval_loop import (
    dispatch_retrieval_grade,
    grade_documents_node,
    query_rewrite_node,
    web_search_node,
)
from agents.state import AgentState
from logger import logger


def _dispatch_domain(state: AgentState) -> str:
    """Route based on gatekeeper verdict."""
    return state.get("domain_status", "authorized")


def _dispatch_intent(state: AgentState) -> str:
    """Return the next node name based on classified intent."""
    return state.get("intent", "financial_qa")


def build_agent_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("summarize_history", summarize_history_node)
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("router", router_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("gemini_generate", gemini_generate_node)
    graph.add_node("outbound_guardrail", outbound_guardrail_node)
    graph.add_node("citation_map", citation_map_node)
    graph.add_node("setup_gate", setup_gate_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("execute_simulation", execute_simulation_node)
    graph.add_node("memory_save", memory_save_node)
    graph.add_node("chitchat", chitchat_node)

    graph.add_edge(START, "summarize_history")
    graph.add_edge("summarize_history", "memory_retrieve")
    graph.add_edge("memory_retrieve", "gatekeeper")

    graph.add_conditional_edges(
        "gatekeeper",
        _dispatch_domain,
        {
            "authorized": "router",
            "unauthorized": "out_of_scope",
        },
    )
    graph.add_edge("out_of_scope", END)

    graph.add_conditional_edges(
        "router",
        _dispatch_intent,
        {
            "financial_qa": "rag_retrieve",
            "market_data": "market_data",
            "chitchat": "chitchat",
            "trade_sim": "human_approval",
        },
    )
    graph.add_edge("market_data", "citation_map")

    graph.add_edge("rag_retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        dispatch_retrieval_grade,
        {
            "generate": "gemini_generate",
            "rewrite": "query_rewrite",
            "web_search": "web_search",
        },
    )
    graph.add_edge("query_rewrite", "rag_retrieve")
    graph.add_edge("web_search", "gemini_generate")
    graph.add_edge("gemini_generate", "outbound_guardrail")
    graph.add_edge("outbound_guardrail", "citation_map")
    graph.add_edge("citation_map", "setup_gate")
    graph.add_conditional_edges(
        "setup_gate",
        dispatch_approval,
        {
            "human_approval": "human_approval",
            "memory_save": "memory_save",
        },
    )
    graph.add_edge("human_approval", "execute_simulation")
    graph.add_edge("execute_simulation", "memory_save")
    graph.add_edge("memory_save", END)
    graph.add_edge("chitchat", END)

    compiled = graph.compile(checkpointer=MemorySaver())
    logger.info("LangGraph agent compiled successfully (HITL checkpointer enabled).")
    return compiled


financial_agent = build_agent_graph()
