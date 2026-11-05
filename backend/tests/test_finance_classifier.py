import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.domain_classifier import off_topic_message
from agents.finance_classifier import (
    dispatch_finance_classifier,
    finance_classifier_node,
)


def test_dispatch_routes_on_domain_status():
    assert dispatch_finance_classifier({"domain_status": "authorized"}) == "authorized"
    assert dispatch_finance_classifier({"domain_status": "unauthorized"}) == "unauthorized"
    assert dispatch_finance_classifier({}) == "unauthorized"


def test_gemini_non_finance_returns_guardrail(monkeypatch):
    monkeypatch.setattr(
        "agents.finance_classifier._demo_mode",
        lambda: False,
    )

    class FakeLlm:
        pass

    monkeypatch.setattr(
        "agents.nodes._get_classifier_llm",
        lambda: FakeLlm(),
    )

    fake_response = SimpleNamespace(
        content='{"is_finance": false, "is_greeting": false, "confidence": 0.95, "reason": "geography"}',
        usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        response_metadata={},
    )
    monkeypatch.setattr(
        "agents.finance_classifier.invoke_llm",
        lambda *args, **kwargs: fake_response,
    )

    out = finance_classifier_node({"query": "What is the capital of Japan?"})
    assert out["domain_status"] == "unauthorized"
    assert out["answer"] == off_topic_message()
    assert out["model_used"] == "finance_classifier"


def test_gemini_finance_continues(monkeypatch):
    monkeypatch.setattr("agents.finance_classifier._demo_mode", lambda: False)
    monkeypatch.setattr("agents.nodes._get_classifier_llm", lambda: object())
    fake_response = SimpleNamespace(
        content='{"is_finance": true, "is_greeting": false, "confidence": 0.9, "reason": "etf"}',
        usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        response_metadata={},
    )
    monkeypatch.setattr(
        "agents.finance_classifier.invoke_llm",
        lambda *args, **kwargs: fake_response,
    )
    out = finance_classifier_node({"query": "What is an ETF?"})
    assert out["domain_status"] == "authorized"
    assert out.get("answer") is None
    assert out.get("finance_query") == "What is an ETF?"
