import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache.response_cache import get_exact, put_exact, set_index_epoch
from models import QueryResponse
from rate_limit.limiter import RateLimitDecision
from telemetry import tracker as token_tracker


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


def test_chat_rejects_empty_query(client):
    response = client.post("/chat", json={"query": ""})
    assert response.status_code in (400, 422)


def test_chat_rejects_oversized_query(client):
    from config import settings

    long_q = "a" * (settings.query_max_length + 10)
    response = client.post("/chat", json={"query": long_q})
    assert response.status_code in (400, 422)


def test_chat_rejects_extreme_repetition(client):
    repeated = ("What is NAV? " * 25).strip()
    response = client.post("/chat", json={"query": repeated})
    assert response.status_code in (400, 422)


def test_chat_rejects_extra_json_fields(client):
    response = client.post("/chat", json={"query": "What is NAV?", "evil": True})
    assert response.status_code in (400, 422)


def test_chat_429_includes_retry_after(monkeypatch, client):
    import main

    monkeypatch.setattr(
        main.rate_limiter,
        "check_chat",
        lambda **_: RateLimitDecision(False, retry_after_seconds=30, limit_hit="rpm"),
    )
    response = client.post("/chat", json={"query": "What is an ETF?"})
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "30"
    assert "request_id" in response.json() or response.headers.get("X-Request-ID")


def test_health_degraded_when_faiss_init_error(monkeypatch, client):
    import main

    monkeypatch.setattr(main.financial_retriever, "init_error", "Knowledge base failed to load.")
    response = client.get("/health")
    assert response.json()["status"] == "degraded"


def test_chat_503_when_kb_unavailable(monkeypatch, client):
    import main

    monkeypatch.setattr(main.financial_retriever, "init_error", "Knowledge base failed to load.")
    response = client.post("/chat", json={"query": "What is an ETF?"})
    assert response.status_code == 503
    assert "request_id" in response.json() or response.headers.get("X-Request-ID")


def test_global_error_json_shape(client):
    import main

    if not main.financial_retriever.is_ready:
        pytest.skip("RAG not initialized in this test environment")
    with patch.object(main.rate_limiter, "check_chat", return_value=RateLimitDecision(True)), patch.object(
        main.rate_limiter, "acquire_inflight", return_value=RateLimitDecision(True)
    ), patch.object(main.financial_agent, "ainvoke", side_effect=RuntimeError("boom")):
        response = client.post("/chat", json={"query": "What is an ETF?"})
    assert response.status_code == 500
    body = response.json()
    assert body.get("detail")
    assert body.get("request_id") or response.headers.get("X-Request-ID")


def test_market_provider_circuit_open():
    from resilience.circuit_breaker import get_breaker
    from config import settings

    breaker = get_breaker("test-market-eval")
    for _ in range(settings.circuit_breaker_failure_threshold):
        breaker.record_failure()
    assert breaker.allow() is False


def test_external_retry_on_429_message():
    from resilience.retry import retryable_llm_error

    assert retryable_llm_error(Exception("HTTP 429 Too Many Requests")) is True
    assert retryable_llm_error(ValueError("400 bad request")) is False


def test_llm_timeout_simulation():
    import time
    from config import settings
    from resilience.external import call_external

    def slow():
        time.sleep(2)
        return "ok"

    with pytest.raises(TimeoutError):
        call_external("gemini-eval", slow, timeout_seconds=0.05, retry=False)


def test_exact_cache_hit_skips_pipeline(client):
    import main

    if not main.financial_retriever.is_ready:
        pytest.skip("RAG not initialized in this test environment")
    cache_key = f"{main.financial_retriever.index_epoch}:what is nav?"
    cached = QueryResponse(
        answer="Cached NAV answer.",
        confidence=1.0,
        sources=[],
        status="complete",
    )
    put_exact(cache_key, cached, ttl_seconds=60)
    before = token_tracker.snapshot()["cache_hits"]
    with patch.object(main.rate_limiter, "check_chat", return_value=RateLimitDecision(True)), patch.object(
        main.rate_limiter, "acquire_inflight", return_value=RateLimitDecision(True)
    ), patch.object(main.rate_limiter, "release_inflight"), patch.object(
        main.financial_agent, "ainvoke", new_callable=AsyncMock
    ) as mock_invoke:
        response = client.post("/chat", json={"query": "What is NAV?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Cached NAV answer."
    mock_invoke.assert_not_called()
    after = token_tracker.snapshot()["cache_hits"]
    assert after >= before + 1
    set_index_epoch(main.financial_retriever.index_epoch)


def test_off_topic_no_llm_in_demo_gatekeeper():
    from agents.domain_classifier import classify_domain

    decision = classify_domain("Give me a pasta recipe", demo_mode=True)
    assert decision.status == "unauthorized"
    assert not decision.token_calls
