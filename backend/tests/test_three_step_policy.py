import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ambiguity_check import assess_ambiguity
from agents.domain_classifier import classify_domain, load_domain_config, off_topic_message
from rag.grounding_policy import insufficient_evidence_message


def _reload_config():
    load_domain_config.cache_clear()


def test_example1_ambiguous_return():
    _reload_config()
    hit = assess_ambiguity("What is the best way to get a good return?")
    assert hit is not None
    assert "ROI" in hit.message or "investment" in hit.message.lower()


def test_example2_non_financial_return_bypasses_step1():
    _reload_config()
    assert assess_ambiguity("How do I return this broken laptop to Amazon?") is None
    decision = classify_domain("How do I return this broken laptop to Amazon?", demo_mode=True)
    assert decision.status == "unauthorized"


def test_example2_exact_off_topic_message():
    _reload_config()
    msg = off_topic_message()
    assert msg == (
        "I can only help with finance-related questions such as markets, investing, "
        "taxes, banking, and financial planning."
    )


def test_example3_financial_sp500_no_step1():
    _reload_config()
    q = "What is the historical ROI of the S&P 500 since 1950?"
    assert assess_ambiguity(q) is None
    assert classify_domain(q, demo_mode=True).status == "authorized"


def test_44_moving_average_heuristic_grader_not_relevant_without_kb():
    _reload_config()
    from agents.retrieval_loop import heuristic_docs_answer_query

    docs = [
        {
            "question": "What are Bollinger Bands?",
            "answer": "Bollinger Bands use a 20-period simple moving average.",
        }
    ]
    assert heuristic_docs_answer_query("44 moving average", docs) is False


def test_44_moving_average_passes_step2_not_off_topic():
    _reload_config()
    from agents.domain_classifier import classify_domain

    decision = classify_domain("44 moving average", demo_mode=True)
    assert decision.status == "authorized"
    assert decision.layer in {"rules_allow", "embedding"}


def test_step3_kb_message_exact():
    assert insufficient_evidence_message() == (
        "Sufficient information is unavailable in the knowledge base."
    )
