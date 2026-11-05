"""
LangGraph StateGraph — Financial Bot Agent
==========================================
Defines the directed graph that orchestrates query handling:

    START → injection_guard → finance_classifier (Gemini finance vs non-finance)
         unauthorized → out_of_scope → END
         authorized → ambiguity_check → gatekeeper
         unauthorized → out_of_scope → END
         authorized → summarize_history → memory_retrieve → router
              chitchat → chitchat → END
              financial_qa → rag_retrieve → grade_documents
                   relevant → grounding_gate → gemini_generate | insufficient_evidence
                   not_relevant + rewrites left → query_rewrite → rag_retrieve
                   not_relevant + budget spent → web_search → grounding_gate
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
from agents.ambiguity_check import ambiguity_check_node, dispatch_ambiguity
from agents.finance_classifier import dispatch_finance_classifier, finance_classifier_node
from agents.injection_guard import dispatch_injection, injection_guard_node
from agents.output_safety import output_safety_node
from agents.output_validate import output_validate_node
from agents.nodes import (
    chitchat_node,
    gatekeeper_node,
    gemini_generate_node,
    memory_retrieve_node,
    memory_save_node,
    dispatch_market_data,
    market_data_node,
    out_of_scope_node,
    rag_retrieve_node,
    router_node,
    summarize_history_node,
)
from agents.citation_map import citation_map_node
from agents.outbound_guardrail import outbound_guardrail_node
from agents.grounding_gate import (
    dispatch_grounding,
    grounding_gate_node,
    insufficient_evidence_node,
)
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
    graph.add_node("injection_guard", injection_guard_node)
    graph.add_node("finance_classifier", finance_classifier_node)
    graph.add_node("ambiguity_check", ambiguity_check_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("router", router_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("grounding_gate", grounding_gate_node)
    graph.add_node("insufficient_evidence", insufficient_evidence_node)
    graph.add_node("gemini_generate", gemini_generate_node)
    graph.add_node("output_validate", output_validate_node)
    graph.add_node("outbound_guardrail", outbound_guardrail_node)
    graph.add_node("output_safety", output_safety_node)
    graph.add_node("citation_map", citation_map_node)
    graph.add_node("setup_gate", setup_gate_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("execute_simulation", execute_simulation_node)
    graph.add_node("memory_save", memory_save_node)
    graph.add_node("chitchat", chitchat_node)

    graph.add_edge(START, "injection_guard")
    graph.add_conditional_edges(
        "injection_guard",
        dispatch_injection,
        {
            "continue": "finance_classifier",
            "blocked": END,
        },
    )
    graph.add_conditional_edges(
        "finance_classifier",
        dispatch_finance_classifier,
        {
            "authorized": "ambiguity_check",
            "unauthorized": "out_of_scope",
        },
    )
    graph.add_conditional_edges(
        "ambiguity_check",
        dispatch_ambiguity,
        {
            "continue": "gatekeeper",
            "clarify": END,
        },
    )

    graph.add_conditional_edges(
        "gatekeeper",
        _dispatch_domain,
        {
            "authorized": "summarize_history",
            "unauthorized": "out_of_scope",
        },
    )
    graph.add_edge("out_of_scope", END)
    graph.add_edge("summarize_history", "memory_retrieve")
    graph.add_edge("memory_retrieve", "router")

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
    graph.add_conditional_edges(
        "market_data",
        dispatch_market_data,
        {
            "complete": "output_safety",
            "document_fallback": "rag_retrieve",
        },
    )

    graph.add_edge("rag_retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        dispatch_retrieval_grade,
        {
            "generate": "grounding_gate",
            "rewrite": "query_rewrite",
            "web_search": "web_search",
            "kb_missing": "insufficient_evidence",
        },
    )
    graph.add_edge("query_rewrite", "rag_retrieve")
    graph.add_edge("web_search", "grounding_gate")
    graph.add_conditional_edges(
        "grounding_gate",
        dispatch_grounding,
        {
            "generate": "gemini_generate",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    graph.add_edge("insufficient_evidence", "citation_map")
    graph.add_edge("gemini_generate", "output_validate")
    graph.add_edge("output_validate", "outbound_guardrail")
    graph.add_edge("outbound_guardrail", "output_safety")
    graph.add_edge("output_safety", "citation_map")
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
