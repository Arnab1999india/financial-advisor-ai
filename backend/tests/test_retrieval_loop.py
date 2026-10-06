import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.retrieval_loop import dispatch_retrieval_grade, heuristic_docs_answer_query


def test_heuristic_accepts_overlapping_docs():
    docs = [
        {
            "question": "What is a dividend?",
            "answer": "A dividend is a distribution of company profits to shareholders.",
        }
    ]
    assert heuristic_docs_answer_query("What is a dividend?", docs) is True


def test_heuristic_rejects_unrelated_docs():
    docs = [
        {
            "question": "What is a candlestick chart?",
            "answer": "A candlestick shows open high low close for a session.",
        }
    ]
    assert heuristic_docs_answer_query("Explain the 2008 credit default swap crisis", docs) is False


def test_heuristic_rejects_empty_docs():
    assert heuristic_docs_answer_query("What is NAV?", []) is False


def test_dispatch_generate_when_relevant():
    assert dispatch_retrieval_grade({"retrieval_grade": "relevant", "rewrite_attempts": 0}) == "generate"


def test_dispatch_rewrite_when_budget_remains():
    assert dispatch_retrieval_grade({"retrieval_grade": "not_relevant", "rewrite_attempts": 0}) == "rewrite"


def test_dispatch_web_when_rewrite_budget_spent():
    assert dispatch_retrieval_grade({"retrieval_grade": "not_relevant", "rewrite_attempts": 1}) == "web_search"
