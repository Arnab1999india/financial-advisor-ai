import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache.market_cache import get_quote, put_quote
from cache.response_cache import get_exact, put_exact, set_index_epoch
from config import settings
from resilience.circuit_breaker import get_breaker
from resilience.retry import retryable_llm_error


def test_exact_cache_respects_ttl():
    set_index_epoch("epoch-1")
    put_exact("k", {"answer": "x"}, ttl_seconds=1)
    assert get_exact("k") is not None
    time.sleep(1.1)
    assert get_exact("k") is None


def test_circuit_opens_after_failures():
    breaker = get_breaker("test-provider")
    for _ in range(settings.circuit_breaker_failure_threshold):
        breaker.record_failure()
    assert breaker.allow() is False


def test_retryable_llm_error_classification():
    assert retryable_llm_error(TimeoutError("timeout")) is True
    assert retryable_llm_error(ValueError("400 invalid request")) is False


def test_market_cache_roundtrip():
    put_quote("TCS.NS", {"ok": True, "price": 100.0})
    assert get_quote("TCS.NS")["price"] == 100.0
