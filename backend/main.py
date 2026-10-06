"""
Financial Bot API — v2
=======================
FastAPI application wired to the LangGraph agent (Gemini + RAG + Cognitive Memory).

Startup sequence
----------------
1. Load Q&A knowledge base from ``sample_qa.json``.
2. Build HuggingFace embeddings + FAISS index via ``FinancialRetriever``.
3. Compile the LangGraph ``financial_agent`` (router → RAG → Gemini → memory_save).

All subsequent ``POST /chat`` requests flow through the compiled graph.
Pass ``user_id`` in the request body to enable the Cognitive Memory System.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agents.graph import financial_agent
from analysis import load_data_from_file, run_analysis
from config import settings
from logger import logger
from memory.cognitive_store import memory_store
from models import (
    AnalysisResponse,
    ApprovalCard,
    ApprovalResumeRequest,
    HealthResponse,
    MemoryClearResponse,
    MemoryInfoResponse,
    QueryRequest,
    QueryResponse,
    TelemetrySnapshot,
    TokenUsageSummary,
)
from rag.retriever import retriever as financial_retriever
from telemetry import tracker as token_tracker

# ---------------------------------------------------------------------------
# Response cache — only for fully anonymous stateless requests.
# Requests with user_id are always personalised and must bypass the cache so
# that two different users asking the same question get different answers.
# ---------------------------------------------------------------------------
_response_cache: dict[str, QueryResponse] = {}

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=(
        "Financial Bot API — powered by Google Gemini, LangGraph, and RAG. "
        "Supports financial Q&A and algorithmic trading signal generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Load knowledge base and initialise the RAG retriever."""
    logger.info("=== Financial Bot API v2 starting up ===")

    data_path = Path(__file__).parent / settings.data_file
    logger.info("Loading knowledge base: %s", data_path)

    with open(data_path, "r", encoding="utf-8") as fh:
        qa_data = json.load(fh)

    financial_retriever.initialize(qa_data)
    logger.info("RAG retriever ready with %d Q&A pairs.", len(qa_data))
    logger.info("LLM: %s | Embedding: %s", settings.gemini_model, settings.embedding_model)
    logger.info("=== Startup complete ===")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root() -> HealthResponse:
    """Root health-check."""
    return HealthResponse(status="healthy")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Detailed health status — reports 'initializing' if the RAG index is not yet ready."""
    status = "healthy" if financial_retriever.is_ready else "initializing"
    return HealthResponse(status=status)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_payload(thread_id: str, result: dict | None) -> dict | None:
    """Read the paused interrupt value from invoke output or checkpoint state."""
    if isinstance(result, dict):
        items = result.get("__interrupt__")
        if items:
            first = items[0]
            value = getattr(first, "value", None)
            if value is None and isinstance(first, dict):
                value = first.get("value") or first
            if isinstance(value, dict):
                return value
    snapshot = financial_agent.get_state(_thread_config(thread_id))
    interrupts: list = []
    for task in getattr(snapshot, "tasks", None) or []:
        interrupts.extend(list(getattr(task, "interrupts", None) or []))
    extra = getattr(snapshot, "interrupts", None)
    if extra:
        interrupts.extend(list(extra))
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"proposal": value}


def _query_response_from_state(
    result: dict,
    *,
    thread_id: str,
    token_usage: TokenUsageSummary,
    interrupt_value: dict | None = None,
) -> QueryResponse:
    if interrupt_value:
        proposal = interrupt_value.get("proposal") or result.get("pending_action") or {}
        draft = interrupt_value.get("draft_answer") or result.get("answer") or ""
        reason = interrupt_value.get("reason") or result.get("approval_reason") or "pending_approval"
        answer = (
            draft
            or "This run is paused pending your approval of the simulated trade or alert below."
        )
        return QueryResponse(
            answer=answer,
            confidence=result.get("confidence"),
            source_questions=result.get("source_questions"),
            intent=result.get("intent") or "trade_sim",
            model_used=result.get("model_used"),
            memory_used=bool((result.get("long_term_context") or "").strip()),
            guardrail_amended=bool(result.get("guardrail_amended")),
            retrieval_source=result.get("retrieval_source"),
            source_chunks=result.get("source_chunks") or [],
            sentence_mappings=result.get("sentence_mappings") or [],
            status="awaiting_approval",
            thread_id=thread_id,
            approval_card=ApprovalCard(
                reason=str(reason),
                draft_answer=draft,
                proposal=proposal if isinstance(proposal, dict) else {},
            ),
            token_usage=token_usage,
        )

    return QueryResponse(
        answer=result.get("answer") or "I was unable to generate a response.",
        confidence=result.get("confidence"),
        source_questions=result.get("source_questions"),
        intent=result.get("intent"),
        model_used=result.get("model_used"),
        memory_used=bool((result.get("long_term_context") or "").strip()),
        guardrail_amended=bool(result.get("guardrail_amended")),
        retrieval_source=result.get("retrieval_source"),
        source_chunks=result.get("source_chunks") or [],
        sentence_mappings=result.get("sentence_mappings") or [],
        status="complete",
        thread_id=thread_id,
        token_usage=token_usage,
    )


@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat(request: QueryRequest) -> QueryResponse:
    """Process a financial question through the LangGraph agent.

    High-probability setups and trade/alert simulations pause until
    ``POST /chat/resume`` receives an explicit approval or rejection.
    """
    logger.info("Incoming query: %s | user_id: %s", request.query, request.user_id or "anonymous")

    if not financial_retriever.is_ready:
        raise HTTPException(status_code=503, detail="Service initializing. Please retry shortly.")

    is_cacheable = not request.session_id and not request.user_id
    cache_key = request.query.strip().lower()
    if is_cacheable and cache_key in _response_cache:
        logger.info("Response cache hit — skipping pipeline.")
        token_tracker.record_request([], cache_hit=True)
        return _response_cache[cache_key]

    thread_id = request.session_id or str(uuid.uuid4())

    try:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "user_id": request.user_id,
            "intent": None,
            "long_term_context": None,
            "retrieved_docs": None,
            "context": None,
            "answer": None,
            "confidence": None,
            "source_questions": None,
            "error": None,
            "model_used": None,
            "token_calls": [],
            "guardrail_amended": False,
            "retrieval_query": request.query,
            "rewrite_attempts": 0,
            "retrieval_grade": None,
            "retrieval_source": None,
            "source_chunks": [],
            "sentence_mappings": [],
            "approval_required": False,
            "approval_reason": None,
            "pending_action": None,
            "approval_decision": None,
        }

        result = await financial_agent.ainvoke(
            initial_state,
            config=_thread_config(thread_id),
        )
        if not isinstance(result, dict):
            result = {}
        interrupt_value = _interrupt_payload(thread_id, result)
        values = result
        if interrupt_value:
            snapshot = financial_agent.get_state(_thread_config(thread_id))
            values = dict(snapshot.values or {})
        token_usage = TokenUsageSummary.model_validate(
            token_tracker.record_request(values.get("token_calls") or [])
        )
        response = _query_response_from_state(
            values,
            thread_id=thread_id,
            token_usage=token_usage,
            interrupt_value=interrupt_value,
        )

        if is_cacheable and response.status == "complete":
            if len(_response_cache) >= settings.response_cache_size:
                _response_cache.pop(next(iter(_response_cache)))
            _response_cache[cache_key] = response

        return response

    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/resume", response_model=QueryResponse, tags=["Chat"])
async def chat_resume(request: ApprovalResumeRequest) -> QueryResponse:
    """Resume a paused graph after the user approves, edits, or rejects the card."""
    if not financial_retriever.is_ready:
        raise HTTPException(status_code=503, detail="Service initializing. Please retry shortly.")

    thread_id = (request.thread_id or "").strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")

    snapshot = financial_agent.get_state(_thread_config(thread_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="No paused graph found for this thread_id.")

    approved = request.decision.strip().lower() in {"approve", "approved", "edit", "edit_approve"}
    try:
        result = await financial_agent.ainvoke(
            Command(
                resume={
                    "approved": approved,
                    "decision": "approve" if approved else "reject",
                    "params": request.params or {},
                    "notes": request.notes or "",
                }
            ),
            config=_thread_config(thread_id),
        )
        if not isinstance(result, dict):
            result = dict(snapshot.values or {})
        token_usage = TokenUsageSummary.model_validate(
            token_tracker.record_request(result.get("token_calls") or [])
        )
        leftover = _interrupt_payload(thread_id, result)
        return _query_response_from_state(
            result,
            thread_id=thread_id,
            token_usage=token_usage,
            interrupt_value=leftover,
        )
    except Exception as exc:
        logger.error("Chat resume error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Trading analysis
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="CSV, XLS, XLSX, or PDF with OHLC data"),
    strategy: str = Form(..., description="Trading strategy key"),
    data_type: str = Form(..., description="Data type hint (e.g. 'swing', 'intraday')"),
) -> AnalysisResponse:
    """Upload historical OHLC data and receive buy/sell signals."""
    logger.info("Analysis request — strategy=%s  data_type=%s", strategy, data_type)

    try:
        df = load_data_from_file(file)
        result = run_analysis(df, strategy, data_type)
        logger.info("Analysis complete — strategy=%s", strategy)
        return AnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

@app.get("/info", tags=["Info"])
async def get_info() -> dict:
    """Return runtime system information."""
    snap = token_tracker.snapshot()
    return {
        "version": "2.0.0",
        "app_name": settings.app_name,
        "rag_ready": financial_retriever.is_ready,
        "total_qa_pairs": len(financial_retriever.qa_data),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.gemini_model,
        "data_file": settings.data_file,
        "max_rag_results": settings.max_results,
        "memory_top_k": settings.memory_top_k,
        "memory_max_episodes_per_user": settings.memory_max_episodes_per_user,
        "known_memory_users": len(memory_store.known_users),
        "token_telemetry": {
            "chat_requests": snap["chat_requests"],
            "llm_calls": snap["llm_calls"],
            "total_tokens": snap["total_tokens"],
            "estimated_cost_usd": snap["estimated_cost_usd"],
        },
    }


# ---------------------------------------------------------------------------
# Cognitive Memory Management
# ---------------------------------------------------------------------------

@app.get("/memory/{user_id}", response_model=MemoryInfoResponse, tags=["Memory"])
async def get_memory_info(user_id: str) -> MemoryInfoResponse:
    """Return the number of stored memory episodes for a user."""
    count = memory_store.episode_count(user_id)
    return MemoryInfoResponse(
        user_id=user_id,
        episode_count=count,
        message=f"User '{user_id}' has {count} stored memory episode(s).",
    )


@app.delete("/memory/{user_id}", response_model=MemoryClearResponse, tags=["Memory"])
async def clear_memory(user_id: str) -> MemoryClearResponse:
    """Delete all stored memory episodes for a user."""
    cleared = memory_store.clear_user_memory(user_id)
    logger.info("Memory cleared via API for user %s (%d episodes).", user_id, cleared)
    return MemoryClearResponse(
        user_id=user_id,
        episodes_cleared=cleared,
        message=f"Successfully cleared {cleared} episode(s) for user '{user_id}'.",
    )


# ---------------------------------------------------------------------------
# Live Token Telemetry
# ---------------------------------------------------------------------------

@app.get("/telemetry", response_model=TelemetrySnapshot, tags=["Telemetry"])
async def get_telemetry() -> TelemetrySnapshot:
    """Process-wide LLM token usage since API startup (or last reset)."""
    return TelemetrySnapshot.model_validate(token_tracker.snapshot())


@app.post("/telemetry/reset", response_model=TelemetrySnapshot, tags=["Telemetry"])
async def reset_telemetry() -> TelemetrySnapshot:
    """Clear process-wide token counters. Does not affect Streamlit session totals."""
    logger.info("Token telemetry counters reset.")
    return TelemetrySnapshot.model_validate(token_tracker.reset())
