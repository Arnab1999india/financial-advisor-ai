"""
Financial Bot API — v2
=======================
FastAPI application wired to the LangGraph agent (Gemini + RAG).

Startup sequence
----------------
1. Load Q&A knowledge base from ``sample_qa.json``.
2. Build HuggingFace embeddings + FAISS index via ``FinancialRetriever``.
3. Compile the LangGraph ``financial_agent`` (router → RAG → Gemini).

All subsequent ``/chat`` requests flow through the compiled graph.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from agents.graph import financial_agent
from analysis import load_data_from_file, run_analysis
from config import settings
from logger import logger
from models import AnalysisResponse, HealthResponse, QueryRequest, QueryResponse
from rag.retriever import retriever as financial_retriever

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

@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat(request: QueryRequest) -> QueryResponse:
    """Process a financial question through the LangGraph agent.

    Flow: router → (rag_retrieve → gemini_generate) | chitchat
    """
    logger.info("Incoming query: %s", request.query)

    if not financial_retriever.is_ready:
        raise HTTPException(status_code=503, detail="Service initializing. Please retry shortly.")

    try:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "intent": None,
            "retrieved_docs": None,
            "context": None,
            "answer": None,
            "confidence": None,
            "source_questions": None,
            "error": None,
        }

        result = await financial_agent.ainvoke(initial_state)

        return QueryResponse(
            answer=result.get("answer") or "I was unable to generate a response.",
            confidence=result.get("confidence"),
            source_questions=result.get("source_questions"),
            intent=result.get("intent"),
            model_used=result.get("model_used"),
        )

    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
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
    """Upload historical OHLC data and receive buy/sell signals.

    Supported strategies: ``swing_trading``, ``scalping_trading``,
    ``44_moving_average``, ``trend_trading``, ``range_trading``.
    """
    logger.info("Analysis request — strategy=%s  data_type=%s", strategy, data_type)

    try:
        df = load_data_from_file(file)
        result = run_analysis(df, strategy, data_type)
        logger.info("Analysis complete — strategy=%s", strategy)
        return result
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
    return {
        "version": "2.0.0",
        "app_name": settings.app_name,
        "rag_ready": financial_retriever.is_ready,
        "total_qa_pairs": len(financial_retriever.qa_data),
        "embedding_model": settings.embedding_model,
        "llm_model": settings.gemini_model,
        "data_file": settings.data_file,
        "max_rag_results": settings.max_results,
    }
