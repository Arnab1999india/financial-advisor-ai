# Financial Bot v2

AI-powered financial assistant using **Google Gemini**, **LangGraph**, and **RAG** (Retrieval-Augmented Generation).

## Features

- **Financial Q&A** — answers grounded in a curated knowledge base via FAISS vector search + Gemini
- **Trading Signal Generation** — upload OHLC CSV/Excel/PDF to receive buy/sell signals from 5 technical strategies
- **LangGraph Agent** — intent routing, RAG retrieval, and LLM generation as a compiled state graph
- **Fallback Mode** — works without a Gemini key using best-match retrieval

## Project Structure

```
financial_bot/
├── backend/
│   ├── agents/              # LangGraph agent (state, nodes, graph)
│   ├── rag/                 # FAISS retriever with MMR search
│   ├── tools/               # LangChain @tool functions
│   ├── tests/               # pytest suite
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings (Pydantic)
│   ├── models.py            # Request/response schemas
│   ├── analysis.py          # File ingestion + data cleaning
│   ├── strategies.py        # Technical indicator strategies
│   ├── sample_qa.json       # Financial Q&A knowledge base (20 pairs)
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Chat UI
│   └── app.py               # Streamlit dashboard
└── docs/
    ├── HLD.md               # High-Level Design
    ├── LLD.md               # Low-Level Design
    ├── API_CONTRACTS.md     # REST API documentation
    ├── ARCHITECTURE.md      # System architecture + diagrams
    └── DEPLOYMENT.md        # Setup and deployment guide
```

## Quick Start

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set GEMINI_API_KEY (https://aistudio.google.com/app/apikey)
uvicorn main:app --reload --port 8000
```

API: `http://localhost:8000` | Swagger UI: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| GET    | `/health`  | Health check |
| GET    | `/info`    | System info (model, RAG status) |
| POST   | `/chat`    | Financial Q&A via LangGraph agent |
| POST   | `/analyze` | Trading signal generation |

See [`docs/API_CONTRACTS.md`](docs/API_CONTRACTS.md) for full request/response schemas.

## LangGraph Workflow

```
query → [router] → financial_qa → [rag_retrieve] → [gemini_generate] → answer
                 → chitchat    → [chitchat_node] → answer
```

## Trading Strategies

| Strategy | Indicator |
|----------|-----------|
| `swing_trading` | SMA 20/50 crossover |
| `scalping_trading` | RSI (< 30 buy, > 70 sell) |
| `44_moving_average` | Price vs SMA-44 |
| `trend_trading` | MACD crossover |
| `range_trading` | Bollinger Bands |

## Security, resilience & cost (Phases 1–8)

### Architecture (request path)

```
Client → FastAPI (/chat)
  → rate limit + validation
  → exact / semantic cache (anonymous queries only)
  → LangGraph:
        injection_guard → gatekeeper (domain)
        → summarize / memory / router
        → RAG loop (FAISS+BM25) | market (yfinance) | web fallback (allow-list)
        → grounding_gate → gemini_generate (JSON)
        → output_validate → outbound_guardrail → output_safety → memory_save
  → structured JSON response + X-Request-ID
```

### Module layout

| Area | Path |
|------|------|
| Domain | `domain_guardrails.json`, `agents/domain_classifier.py` |
| Retrieval / grounding | `rag/`, `grounding_policy.json`, `agents/grounding_gate.py` |
| Security / injection | `security_policy.json`, `security/`, `agents/injection_guard.py` |
| Validation | `validation/`, `models.py` |
| Rate limits | `rate_limit/limiter.py` |
| Cache | `cache/` |
| Resilience | `resilience/`, `agents/llm_client.py` |
| Thresholds | `config.py` + JSON policies above |

### Config tuning (common knobs)

| Setting | Purpose |
|---------|---------|
| `retrieval_min_confidence` / `grounding_full_confidence` | When to treat evidence as partial vs sufficient |
| `semantic_cache_similarity_threshold` | Semantic cache hit strictness (default 0.92) |
| `rate_limit_rpm_per_ip` / `rate_limit_max_inflight_per_user` | Abuse protection |
| `cost_alert_*_usd_daily` | Log warnings when daily spend exceeds budget |
| `llm_max_output_tokens_*` | Cap generation cost |

Environment variables mirror `config.py` field names (see `.env.example`).

### Testing & evaluation

```bash
cd backend
pytest tests/ -q
python evaluation/run_eval.py
```

- **pytest:** domain, grounding, injection, validation, output, rate limits, errors, cache/cost paths (`tests/test_eval_*.py`).
- **eval set:** `backend/evaluation/eval_set.json` (40+ cases).
- **metrics script:** `backend/evaluation/run_eval.py` reports refusal accuracy, groundedness, citation checks; pair with `GET /telemetry` for live token and cache stats.

### Known limitations

- Web and market data depend on **DuckDuckGo/Tavily** and **yfinance** (no commercial SLA).
- Caches and rate limits are **in-process** unless `rate_limit_redis_url` is set.
- **Streaming** `/chat` is not implemented; responses are JSON batch.
- Borderline domain queries may invoke **Flash** when a Gemini key is configured.

See also [`docs/AUDIT_PHASE0.md`](docs/AUDIT_PHASE0.md) and [`docs/SYSTEM_PROMPTS.md`](docs/SYSTEM_PROMPTS.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/AUDIT_PHASE0.md`](docs/AUDIT_PHASE0.md) | Phase 0 audit (FAISS/web/market plain answers) |
| [`docs/SYSTEM_PROMPTS.md`](docs/SYSTEM_PROMPTS.md) | Canonical system prompt locations |
| [`docs/HLD.md`](docs/HLD.md) | System goals, components, data flow, quality attributes |
| [`docs/LLD.md`](docs/LLD.md) | Module design, class specs, error handling strategy |
| [`docs/API_CONTRACTS.md`](docs/API_CONTRACTS.md) | All endpoints, schemas, error codes, cURL examples |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | ASCII architecture diagrams, tech stack table |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Local dev, Docker, Gunicorn, env vars reference |
| [`docs/TESTING_AND_RAG_ACCURACY.md`](docs/TESTING_AND_RAG_ACCURACY.md) | Manual test cases, RAG accuracy protocol, token-efficiency checks, and current test findings |
