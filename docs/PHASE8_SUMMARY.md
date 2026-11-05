# Final summary — hardening Phases 0–8

## What changed

1. **Domain gatekeeper** — rules, embedding anchors, optional Flash for borderline; mixed-query finance extraction; fixed off-topic copy.
2. **Grounding** — hybrid FAISS+BM25, confidence gates, insufficient-evidence path, strict master prompt, citation formatting.
3. **Injection & input security** — query/ingestion scanning, quarantined KB chunks, delimiter wrapping, PII masking, upload validation.
4. **Output safety** — structured JSON, citation/number/leakage validation, outbound advice guardrail, link allow-list.
5. **Rate limiting** — sliding windows, inflight caps, 429 + `Retry-After`, optional Redis hook.
6. **Resilience & cost** — timeouts, retries, circuit breakers, multi-tier cache, model routing, cost alerts, structured logging.
7. **Phase 8** — expanded pytest suite, `evaluation/eval_set.json`, `evaluation/run_eval.py`, audit and prompt docs, README hardening section.

## Deliverables checklist

| # | Item | Location |
|---|------|----------|
| 1 | Phase 0 audit | [`docs/AUDIT_PHASE0.md`](AUDIT_PHASE0.md) |
| 2 | Modular code | `agents/`, `rag/`, `security/`, `validation/`, `rate_limit/`, `cache/`, `resilience/`, `telemetry/`, `config.py` |
| 3 | System prompts | [`docs/SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md) → `agents/nodes.py` et al. |
| 4 | Tests + eval | `backend/tests/test_eval_*.py`, `backend/evaluation/*` |
| 5 | README section | Root [`README.md`](../README.md) — Security, resilience & cost |
| 6 | This summary | `docs/PHASE8_SUMMARY.md` |

## Decisions for you

1. **Market data provider** — Today: **yfinance** (free, unofficial). For production, pick a licensed feed (NSE/BSE vendor, Polygon, Refinitiv, etc.).
2. **Web search** — Keep Tavily/DDG fallback with allow-list, or disable web edges in `agents/graph.py` for air-gapped deployments.
3. **Redis** — Set `rate_limit_redis_url` when running more than one API instance.
4. **Gemini models** — Confirm `cheap_model` / `power_model` names match your Google AI Studio availability and budget.

## Risks still open

- LLM layer-3 domain calls on ambiguous queries (cost + occasional misclassification).
- Semantic cache false positives if threshold is lowered too far.
- yfinance/DDG outages → degraded answers unless you add paid providers.
- Cognitive memory is not encrypted at rest (in-process FAISS per user).

## Defaults assumed (no blocking questions)

- Evaluation runs **offline** without a Gemini key (`demo_mode=True` for domain).
- Eval set size: **40 cases** in `eval_set.json` (expand toward 50 by adding your proprietary KB Q&A).
- HTTP concurrency burst tests use **mocked** rate limiter where needed to keep CI fast.
