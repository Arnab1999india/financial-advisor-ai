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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agents.graph import financial_agent
from analysis import load_data_from_bytes, run_analysis
from security.pii import mask_pii
from validation.uploads import UploadValidationError, validate_upload
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
from cache.response_cache import get_exact, get_semantic, put_exact, put_semantic
from chat_pipeline import (
    chat_preflight,
    stream_cached_response,
    stream_graph_response,
    try_cache_hit,
)
from logging_structured import StageTimer
from rag.retriever import retriever as financial_retriever
from rate_limit.limiter import rate_limiter
from telemetry.cost_tracker import cost_tracker
from telemetry import tracker as token_tracker

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


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return safe 400 responses without leaking validation internals."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request.", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Consistent JSON errors without stack traces or secrets."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("Unhandled error request_id=%s type=%s", request_id, type(exc).__name__)
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
            headers=getattr(exc, "headers", None) or {"X-Request-ID": request_id},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


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

    try:
        financial_retriever.initialize(qa_data)
        logger.info("RAG retriever ready with %d Q&A pairs.", len(qa_data))
    except Exception as exc:
        financial_retriever.init_error = "Knowledge base failed to load."
        logger.error("FAISS initialisation failed: %s", type(exc).__name__)
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
    if financial_retriever.init_error:
        return HealthResponse(status="degraded")
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
async def chat(body: QueryRequest, request: Request) -> QueryResponse:
    """Process a financial question through the LangGraph agent.

    High-probability setups and trade/alert simulations pause until
    ``POST /chat/resume`` receives an explicit approval or rejection.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "Incoming query (masked): %s | user_id: %s | request_id: %s",
        mask_pii(body.query),
        body.user_id or "anonymous",
        request_id,
    )

    rl_decision = rate_limiter.check_chat(ip=client_ip, user_id=body.user_id)
    if not rl_decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait and try again.",
            headers={
                "Retry-After": str(max(rl_decision.retry_after_seconds, 1)),
                "X-Request-ID": request_id,
            },
        )

    inflight = rate_limiter.acquire_inflight(body.user_id)
    if not inflight.allowed:
        rate_limiter.release_inflight(body.user_id)
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent requests. Please retry shortly.",
            headers={
                "Retry-After": str(max(inflight.retry_after_seconds, 1)),
                "X-Request-ID": request_id,
            },
        )

    if financial_retriever.init_error:
        rate_limiter.release_inflight(body.user_id)
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is unavailable. Please retry later.",
            headers={"X-Request-ID": request_id},
        )
    if not financial_retriever.is_ready:
        rate_limiter.release_inflight(body.user_id)
        raise HTTPException(
            status_code=503,
            detail="Service initializing. Please retry shortly.",
            headers={"X-Request-ID": request_id},
        )

    is_cacheable = not body.session_id and not body.user_id
    cache_key = f"{financial_retriever.index_epoch}:{body.query.strip().lower()}"
    timer = StageTimer(request_id=request_id, stage="chat", user_id=body.user_id)
    if is_cacheable:
        cached = get_exact(cache_key)
        if cached is not None:
            timer.finish(cache_hit=True, extra={"cache_type": "exact"})
            token_tracker.record_request([], cache_hit=True)
            return cached
        if financial_retriever.is_ready:
            try:
                query_vec = financial_retriever.embed_query_cached(body.query)
                semantic_hit = get_semantic(query_vec)
                if semantic_hit is not None:
                    timer.finish(cache_hit=True, extra={"cache_type": "semantic"})
                    token_tracker.record_request([], cache_hit=True)
                    return semantic_hit
            except Exception:
                pass

    thread_id = body.session_id or str(uuid.uuid4())

    try:
        initial_state = {
            "messages": [HumanMessage(content=mask_pii(body.query))],
            "query": mask_pii(body.query),
            "finance_query": None,
            "user_id": body.user_id,
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
            "retrieval_query": mask_pii(body.query),
            "injection_blocked": False,
            "injection_reason": None,
            "ambiguity_blocked": False,
            "structured_raw": None,
            "structured_citations": None,
            "output_validate_retried": False,
            "rewrite_attempts": 0,
            "retrieval_grade": None,
            "retrieval_source": None,
            "source_chunks": [],
            "sentence_mappings": [],
            "approval_required": False,
            "approval_reason": None,
            "pending_action": None,
            "approval_decision": None,
            "grounding_status": None,
            "market_data_status": None,
            "market_unavailable_note": None,
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
        cost_tracker.record(body.user_id, token_usage.estimated_cost_usd)
        timer.finish(
            cache_hit=False,
            tokens_in=token_usage.input_tokens,
            tokens_out=token_usage.output_tokens,
            guardrail=values.get("injection_reason") or values.get("guardrail_amended"),
            extra={"model_used": values.get("model_used"), "intent": values.get("intent")},
        )
        token_decision = rate_limiter.check_daily_tokens(body.user_id, token_usage.total_tokens)
        if not token_decision.allowed:
            rate_limiter.release_inflight(body.user_id)
            raise HTTPException(
                status_code=429,
                detail="Daily token budget exceeded for this user.",
                headers={
                    "Retry-After": str(max(token_decision.retry_after_seconds, 1)),
                    "X-Request-ID": request_id,
                },
            )

        response = _query_response_from_state(
            values,
            thread_id=thread_id,
            token_usage=token_usage,
            interrupt_value=interrupt_value,
        )

        if is_cacheable and response.status == "complete":
            ttl = (
                settings.document_answer_cache_ttl_seconds
                if (values.get("retrieval_source") or "").startswith("faiss")
                else settings.exact_cache_ttl_seconds
            )
            put_exact(cache_key, response, ttl)
            if financial_retriever.is_ready:
                try:
                    query_vec = financial_retriever.embed_query_cached(body.query)
                    put_semantic(
                        query_vec,
                        body.query,
                        response,
                        settings.semantic_cache_ttl_seconds,
                    )
                except Exception:
                    pass

        rate_limiter.release_inflight(body.user_id)
        return response

    except Exception as exc:
        rate_limiter.release_inflight(body.user_id)
        logger.error("Chat endpoint error: %s request_id=%s", type(exc).__name__, request_id)
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
            headers={"X-Request-ID": request_id},
        ) from None


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    """Stream progress and answer tokens via Server-Sent Events (SSE).

    Events: ``status`` (progress), ``token`` (answer text chunks), ``done`` (full QueryResponse JSON), ``error``.
    """
    request_id, _, is_cacheable, cache_key, timer = chat_preflight(body, request)
    thread_id = body.session_id or str(uuid.uuid4())
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Request-ID": request_id,
    }

    cached = try_cache_hit(body, is_cacheable=is_cacheable, cache_key=cache_key, timer=timer)
    if cached is not None:
        return StreamingResponse(stream_cached_response(cached), media_type="text/event-stream", headers=headers)

    return StreamingResponse(
        stream_graph_response(
            body,
            thread_id=thread_id,
            is_cacheable=is_cacheable,
            cache_key=cache_key,
            timer=timer,
            thread_config_fn=_thread_config,
            query_response_from_state_fn=_query_response_from_state,
            interrupt_payload_fn=_interrupt_payload,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/chat/resume", response_model=QueryResponse, tags=["Chat"])
async def chat_resume(request: ApprovalResumeRequest) -> QueryResponse:
    """Resume a paused graph after the user approves, edits, or rejects the card."""
    if financial_retriever.init_error:
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable. Please retry later.")
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
    request: Request,
    file: UploadFile = File(..., description="CSV, XLS, XLSX, or PDF with OHLC data"),
    strategy: str = Form(..., description="Trading strategy key"),
    data_type: str = Form(..., description="Data type hint (e.g. 'swing', 'intraday')"),
    symbol: str | None = Form(None, description="Instrument symbol (metadata for the run)"),
    exchange: str | None = Form(None, description="Exchange id, e.g. NSE or BSE"),
    data_interval: str | None = Form(None, description="Expected candle interval label"),
) -> AnalysisResponse:
    """Upload historical OHLC data and receive buy/sell signals."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"
    upload_decision = rate_limiter.check_upload(client_ip)
    if not upload_decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many upload requests. Please wait and try again.",
            headers={"Retry-After": str(max(upload_decision.retry_after_seconds, 1)), "X-Request-ID": request_id},
        )

    logger.info(
        "Analysis request — strategy=%s data_type=%s symbol=%s exchange=%s interval=%s request_id=%s",
        strategy,
        data_type,
        symbol,
        exchange,
        data_interval,
        request_id,
    )

    try:
        ext, raw = validate_upload(
            file,
            max_bytes=settings.upload_max_bytes,
            max_pdf_pages=settings.upload_max_pdf_pages,
        )
        df = load_data_from_bytes(ext, raw)
        result = run_analysis(df, strategy, data_type)
        if symbol and symbol.strip():
            result["symbol"] = symbol.strip().upper()
        if exchange and exchange.strip():
            result["exchange"] = exchange.strip().upper()
        if data_interval and data_interval.strip():
            result["data_interval"] = data_interval.strip()
        logger.info("Analysis complete — strategy=%s", strategy)
        return AnalysisResponse(**result)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
        "cost_alerts": cost_tracker.snapshot(),
        "rag_init_error": financial_retriever.init_error,
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
