import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.domain_classifier import classify_domain, off_topic_message
from agents.grounding_gate import dispatch_grounding, insufficient_evidence_node
from agents.retrieval_loop import dispatch_retrieval_grade
from config import settings
from rag.grounding_policy import insufficient_evidence_message


def test_step1_off_topic_exact_copy():
    msg = off_topic_message()
    assert msg == (
        "I can only help with finance-related questions such as markets, investing, "
        "taxes, banking, and financial planning."
    )
    decision = classify_domain("What is the capital of Japan?", demo_mode=True)
    assert decision.status == "unauthorized"


def test_step2_kb_missing_exact_copy():
    msg = insufficient_evidence_message()
    assert msg == "Sufficient information is unavailable in the knowledge base."


def test_step2_insufficient_node_uses_kb_message():
    out = insufficient_evidence_node({})
    assert out["answer"] == insufficient_evidence_message()


def test_strict_mode_skips_web_after_rewrite():
    settings.strict_two_step_kb_only = True
    settings.max_query_rewrites = 1
    route = dispatch_retrieval_grade(
        {"retrieval_grade": "not_relevant", "rewrite_attempts": 1}
    )
    assert route == "kb_missing"


def test_strict_grounding_refuses_without_relevant_grade():
    settings.strict_two_step_kb_only = True
    state = {
        "retrieval_grade": "not_relevant",
        "retrieved_docs": [{"source_type": "faiss", "confidence": 0.9}],
        "grounding_status": "full",
    }
    assert dispatch_grounding(state) == "insufficient_evidence"
