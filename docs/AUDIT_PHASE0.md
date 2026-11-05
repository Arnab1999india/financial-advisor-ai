# Phase 0 — Security & Architecture Audit

**Project:** Financial Bot v2 (`backend/` FastAPI + LangGraph, Streamlit frontend)  
**Date:** 2026-11-02 (baseline before hardening phases 1–8)

## Executive summary

The original bot was a capable demo: Gemini + in-memory FAISS over `sample_qa.json`, with optional fallback when no API key was set. It lacked domain enforcement, strong grounding, injection defenses, structured output validation, rate limits, resilience wrappers, and cost controls. Phases 1–7 addressed these; Phase 8 adds automated evaluation.

---

## Plain answers (retrieval & tools)

| Question | Answer |
|----------|--------|
| **Is retrieval FAISS-only?** | **No.** Production retrieval is **hybrid FAISS + BM25** (`rag/retriever.py`), with optional **web fallback** after failed retrieval/rewrite (`agents/retrieval_loop.py`, `tools/web_search.py`). Embeddings: HuggingFace `all-MiniLM-L6-v2`. |
| **Is web search implemented?** | **Yes**, as a **fallback** path only. Tavily when `TAVILY_API_KEY` is set; otherwise DuckDuckGo. Results are **allow-list filtered** (`grounding_policy.json`). Wrapped with timeouts/retries (`resilience/external.py`). |
| **Is live market data implemented?** | **Yes**, for quote-style queries via **yfinance** (`tools/market_data.py`), with short TTL cache and graceful fallback to document Q&A when the provider fails. No paid market-data API is configured by default. |

---

## Stack inventory

- **API:** FastAPI (`main.py`)
- **Orchestration:** LangGraph (`agents/graph.py`)
- **LLM:** Google Gemini via LangChain (`langchain_google_genai`)
- **RAG:** LangChain FAISS + custom BM25 + chunk metadata
- **Memory:** In-process cognitive store (`memory/cognitive_store.py`)
- **Frontend:** Streamlit (`frontend/app.py`)

---

## Initial gaps (pre-hardening)

1. **Domain:** No gatekeeper; general financial knowledge allowed in prompts.
2. **Grounding:** Weak master prompt; no hard refuse when evidence missing.
3. **Injection:** No query/ingestion scanning; KB could contain hostile text.
4. **Validation:** Permissive request bodies; limited upload checks.
5. **Output:** Free-form answers; no citation/number tracing.
6. **Rate limits:** None on `/chat` or uploads.
7. **Errors:** Unhandled exceptions could leak details; no circuit breakers.
8. **Cost:** Summarization/LLM before cheap filters; no response cache tiers.
9. **Logging:** Full queries in logs; no structured request correlation.

---

## Post-hardening module map

| Concern | Location |
|---------|----------|
| Domain guardrails | `domain_guardrails.json`, `agents/domain_classifier.py`, `gatekeeper_node` |
| Grounding | `grounding_policy.json`, `rag/retriever.py`, `agents/grounding_gate.py` |
| Security / injection | `security_policy.json`, `security/*`, `agents/injection_guard.py` |
| Input validation | `validation/*`, Pydantic models in `models.py` |
| Output validation | `validation/output_validator.py`, `agents/output_validate.py` |
| Rate limiting | `rate_limit/limiter.py` |
| Cache | `cache/*` |
| Resilience | `resilience/*`, `agents/llm_client.py` |
| Telemetry / cost | `telemetry/*`, `logging_structured.py` |
| **Thresholds** | Primarily `config.py` + JSON policy files above |

---

## Residual risks

- **yfinance / DuckDuckGo** are best-effort, not SLA-backed market/news sources.
- **In-memory** rate limits and caches do not sync across multiple API replicas without Redis (`rate_limit_redis_url`).
- **Embedding + FAISS** load at startup; large KBs need operational planning (memory, reindex downtime).
- **LLM borderline domain** calls still possible when API key is set (layer 3 classifier).
- **Streamlit** may not send `user_id`; memory and per-user limits depend on client integration.

---

## Recommendations (for your decisions)

1. **Market data:** Keep yfinance for demos; for production consider NSE/BSE vendor APIs, Polygon, Alpha Vantage, etc. (you choose provider + licensing).
2. **Web search:** Keep allow-listed fallback or disable web entirely for strict compliance environments.
3. **Redis:** Enable for multi-instance deployments.
4. **Secrets:** Keep `GEMINI_API_KEY` and optional `TAVILY_API_KEY` in environment only.
