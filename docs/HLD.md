# High-Level Design (HLD)

## 1. Purpose

Financial Bot is a domain-specific AI assistant that provides:

1. **Conversational Financial Q&A** — answers questions about stocks, investments, trading strategies, macroeconomics, and financial concepts using Retrieval-Augmented Generation (RAG) backed by Google Gemini.
2. **Algorithmic Trading Signal Generation** — accepts uploaded OHLC (Open, High, Low, Close) data files and returns buy/sell signals computed by technical analysis strategies.

---

## 2. Goals & Non-Goals

### Goals
- Provide accurate, grounded answers using a curated knowledge base (RAG prevents hallucination).
- Support swappable LLM backends via LangChain's unified interface (currently Gemini; OpenAI-compatible tool patterns retained).
- Produce deterministic, auditable trading signals from standard technical indicators.
- Be deployable locally (demo mode, no API key) or in production (Gemini key configured).
- Expose clean REST API consumed by any frontend.

### Non-Goals
- Real-time market data feeds or live trading execution.
- User authentication / multi-tenancy (out of scope for v2).
- Persistent conversation history across sessions (stateless per-request).

---

## 3. System Context

```
┌──────────────────────────────────────────────────────────────┐
│                        External Users                        │
│           (Browser / Streamlit Dashboard / API clients)      │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTPS / HTTP
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Financial Bot API                         │
│                   (FastAPI + LangGraph)                      │
└──────┬───────────────────────────────────────┬───────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────────┐                 ┌──────────────────────┐
│  Google Gemini  │                 │  Local FAISS Index   │
│  (External LLM) │                 │  (Knowledge Base)    │
└─────────────────┘                 └──────────────────────┘
```

---

## 4. Key Components

| Component | Responsibility |
|-----------|----------------|
| **FastAPI Application** (`main.py`) | HTTP routing, request validation, startup lifecycle |
| **LangGraph Agent** (`agents/`) | Orchestrates intent routing → RAG → LLM generation |
| **RAG Retriever** (`rag/retriever.py`) | Embeds Q&A knowledge base, performs MMR vector search |
| **Gemini LLM** (`agents/nodes.py`) | Generates context-grounded natural language answers |
| **LangChain Tools** (`tools/`) | Reusable @tool functions for knowledge search and strategy listing |
| **Analysis Engine** (`analysis.py`, `strategies.py`) | Computes OHLC-based technical trading signals |
| **Configuration** (`config.py`) | Centralised settings via environment variables |
| **Data Models** (`models.py`) | Pydantic request/response schemas |

---

## 5. Data Flow

### 5.1 Chat (Financial Q&A)

```
Client POST /chat {query}
  → FastAPI validates QueryRequest
  → LangGraph agent.ainvoke(initial_state)
      → router_node   : classify intent
      → rag_retrieve_node : FAISS MMR search (top-k Q&A docs)
      → gemini_generate_node : build prompt + call Gemini API
  → FastAPI returns QueryResponse {answer, confidence, source_questions, intent, model_used}
```

### 5.2 Trading Analysis

```
Client POST /analyze (multipart: file + strategy + data_type)
  → FastAPI validates form
  → load_data_from_file : parse CSV/XLS/PDF → clean DataFrame
  → run_analysis : apply selected strategy → compute signals
  → FastAPI returns AnalysisResponse {strategy, buy_signals, sell_signals}
```

---

## 6. Quality Attributes

| Attribute | Approach |
|-----------|----------|
| **Reliability** | Graceful fallback: if Gemini is unavailable, best-match Q&A is returned |
| **Accuracy** | RAG grounds answers in curated knowledge; Gemini supplements |
| **Performance** | FAISS in-memory index; async FastAPI endpoints; `gemini-1.5-flash` (low latency) |
| **Extensibility** | LangGraph nodes are independently replaceable; new strategies plug into `STRATEGIES` dict |
| **Observability** | Structured logging at every node; `/health` and `/info` endpoints |
| **Security** | API key stored only in `.env`; CORS configured; input validated by Pydantic |

---

## 7. Deployment Overview

| Environment | Configuration |
|-------------|--------------|
| **Local / Demo** | `GEMINI_API_KEY` unset → fallback mode (no external calls) |
| **Development** | `GEMINI_API_KEY` set, `DEBUG=true`, `uvicorn --reload` |
| **Production** | `GEMINI_API_KEY` set, `DEBUG=false`, Gunicorn + Uvicorn workers |

See `docs/DEPLOYMENT.md` for full deployment instructions.
