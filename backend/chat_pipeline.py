"""Shared chat preparation, graph execution, and SSE streaming helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import HumanMessage

from agents.graph import financial_agent
from cache.response_cache import get_exact, get_semantic, put_exact, put_semantic
from config import settings
from logging_structured import StageTimer
from models import QueryRequest, QueryResponse, TokenUsageSummary
from rag.retriever import retriever as financial_retriever
from rate_limit.limiter import rate_limiter
from security.pii import mask_pii
from telemetry.cost_tracker import cost_tracker
from telemetry import tracker as token_tracker

# User-visible progress labels keyed by LangGraph node name.
NODE_STATUS: dict[str, str] = {
    "injection_guard": "Running safety checks…",
    "finance_classifier": "Checking if this is a finance question…",
    "ambiguity_check": "Checking whether your question is clear…",
    "gatekeeper": "Checking your question is finance-related…",
    "out_of_scope": "Preparing response…",
    "summarize_history": "Reviewing conversation…",
    "memory_retrieve": "Loading your preferences…",
    "router": "Understanding what you need…",
    "chitchat": "Preparing a reply…",
    "market_data": "Fetching live market data…",
    "rag_retrieve": "Searching the knowledge base…",
    "grade_documents": "Evaluating sources…",
    "query_rewrite": "Refining search…",
    "web_search": "Searching trusted web sources…",
    "grounding_gate": "Checking evidence…",
    "insufficient_evidence": "Preparing response…",
    "gemini_generate": "Writing your answer…",
    "output_validate": "Verifying the answer…",
    "outbound_guardrail": "Running compliance checks…",
    "output_safety": "Filtering links…",
    "citation_map": "Attaching citations…",
    "setup_gate": "Checking approval rules…",
    "human_approval": "Waiting for approval…",
    "execute_simulation": "Running simulation…",
    "memory_save": "Saving to memory…",
}


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


async def _word_stream(text: str, *, chunk_words: int = 3) -> AsyncIterator[str]:
    words = (text or "").split()
    if not words:
        return
    for i in range(0, len(words), chunk_words):
        piece = " ".join(words[i : i + chunk_words])
        if i + chunk_words < len(words):
            piece += " "
        yield sse_event({"type": "token", "text": piece})
        await asyncio.sleep(0)


def build_initial_state(body: QueryRequest) -> dict[str, Any]:
    masked = mask_pii(body.query)
    return {
        "messages": [HumanMessage(content=masked)],
        "query": masked,
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
        "retrieval_query": masked,
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


def chat_preflight(
    body: QueryRequest,
    request: Request,
) -> tuple[str, str, bool, str, StageTimer]:
    """Rate limits, KB readiness, cache lookup. Returns (request_id, thread_id, is_cacheable, cache_key, timer)."""
    request_id = getattr(request.state, "request_id", "")
    client_ip = request.client.host if request.client else "unknown"

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
    return request_id, body.session_id or "", is_cacheable, cache_key, timer


def try_cache_hit(
    body: QueryRequest,
    *,
    is_cacheable: bool,
    cache_key: str,
    timer: StageTimer,
) -> QueryResponse | None:
    if not is_cacheable:
        return None
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
    return None


async def stream_cached_response(response: QueryResponse) -> AsyncIterator[str]:
    yield sse_event({"type": "status", "message": "Found a cached answer…"})
    async for line in _word_stream(response.answer):
        yield line
    yield sse_event({"type": "done", "response": response.model_dump()})


async def stream_graph_response(
    body: QueryRequest,
    *,
    thread_id: str,
    is_cacheable: bool,
    cache_key: str,
    timer: StageTimer,
    thread_config_fn,
    query_response_from_state_fn,
    interrupt_payload_fn,
) -> AsyncIterator[str]:
    config = thread_config_fn(thread_id)
    values: dict[str, Any] = {}
    interrupt_value: dict | None = None

    try:
        yield sse_event({"type": "status", "message": "Starting…"})
        async for chunk in financial_agent.astream(
            build_initial_state(body),
            config=config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                values.update(update)
                label = NODE_STATUS.get(node_name)
                if label:
                    yield sse_event({"type": "status", "message": label, "node": node_name})
        interrupt_value = interrupt_payload_fn(thread_id, values)
        if interrupt_value:
            snapshot = financial_agent.get_state(config)
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
            yield sse_event(
                {
                    "type": "error",
                    "message": "Daily token budget exceeded for this user.",
                    "retry_after": max(token_decision.retry_after_seconds, 1),
                }
            )
            return

        response = query_response_from_state_fn(
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

        final_answer = response.answer or ""
        async for line in _word_stream(final_answer):
            yield line

        yield sse_event({"type": "done", "response": response.model_dump()})
        rate_limiter.release_inflight(body.user_id)
    except Exception:
        rate_limiter.release_inflight(body.user_id)
        yield sse_event({"type": "error", "message": "Internal server error."})
