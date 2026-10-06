import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace

from telemetry import TokenTelemetryTracker, extract_usage, estimate_cost_usd, make_token_event, summarise_calls


class _Usage:
    def __init__(self, input_tokens, output_tokens, total_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


def test_extract_usage_from_langchain_object():
    response = SimpleNamespace(usage_metadata=_Usage(12, 8, 20), response_metadata={})
    assert extract_usage(response) == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }


def test_extract_usage_from_gemini_response_metadata():
    response = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 40,
                "total_token_count": 140,
            }
        },
    )
    assert extract_usage(response) == {
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
    }


def test_flash_cost_is_cheaper_than_pro():
    flash = estimate_cost_usd("gemini-1.5-flash", 1_000_000, 1_000_000)
    pro = estimate_cost_usd("gemini-1.5-pro", 1_000_000, 1_000_000)
    assert flash < pro
    assert flash == 0.075 + 0.30
    assert pro == 1.25 + 5.00


def test_summarise_and_process_tracker_does_not_double_count_cache_hits():
    tracker = TokenTelemetryTracker()
    event = make_token_event(
        "gemini_generate",
        "gemini-1.5-flash",
        SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
    )
    first = tracker.record_request([event])
    assert first["total_tokens"] == 15
    assert tracker.snapshot()["total_tokens"] == 15

    tracker.record_request([event], cache_hit=True)
    snap = tracker.snapshot()
    assert snap["chat_requests"] == 2
    assert snap["cache_hits"] == 1
    assert snap["total_tokens"] == 15
    assert snap["llm_calls"] == 1


def test_telemetry_endpoint_returns_snapshot():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.get("/telemetry")
    assert response.status_code == 200
    body = response.json()
    assert "total_tokens" in body
    assert "chat_requests" in body
    assert "recent_calls" in body


def test_summarise_calls_groups_by_node():
    events = [
        {
            "node": "gatekeeper",
            "model": "gemini-1.5-flash",
            "input_tokens": 4,
            "output_tokens": 1,
            "total_tokens": 5,
            "estimated_cost_usd": 0.001,
        },
        {
            "node": "gemini_generate",
            "model": "gemini-1.5-pro",
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "estimated_cost_usd": 0.002,
        },
    ]
    summary = summarise_calls(events)
    assert summary["calls"] == 2
    assert summary["total_tokens"] == 35
    assert summary["by_node"]["gatekeeper"]["total_tokens"] == 5
    assert summary["by_node"]["gemini_generate"]["total_tokens"] == 30
