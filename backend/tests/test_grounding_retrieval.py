import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.grounding_gate import assess_evidence_strength, dispatch_grounding, insufficient_evidence_node
from rag.bm25_index import BM25Index
from rag.context_format import format_grounded_context, source_citation_label
from rag.grounding_policy import insufficient_evidence_message, url_on_allowlist
from rag.retriever import FinancialRetriever
from tools.web_search import _filter_allowlist


def test_bm25_finds_exact_ticker_token():
    idx = BM25Index(["ETF SPY tracks the index", "Bond coupon payments"])
    hits = idx.search("What is SPY ETF?", k=2)
    assert hits
    assert hits[0][0] == 0


def test_retriever_applies_confidence_threshold():
    retriever = FinancialRetriever()
    data_path = os.path.join(os.path.dirname(__file__), "..", "sample_qa.json")
    with open(data_path, encoding="utf-8") as fh:
        qa = json.load(fh)
    retriever.initialize(qa)
    hits = retriever.retrieve("What is an ETF exchange traded fund?", k=5)
    assert hits
    assert all(h["confidence"] >= 0.38 for h in hits)
    assert len(hits) <= 4
    assert hits[0].get("source_file")
    assert hits[0].get("page") is not None
    assert hits[0].get("document_date")


def test_format_context_includes_source_citation():
    doc = {
        "source_file": "sample_qa.json",
        "page": 2,
        "document_date": "2024-01-01",
        "section": "basics",
        "question": "Q?",
        "answer": "A.",
        "source_type": "faiss",
    }
    label = source_citation_label(doc)
    assert "sample_qa.json" in label
    assert "page 2" in label
    body = format_grounded_context([doc])
    assert label in body


def test_web_allowlist_filters_unknown_domains():
    hits = _filter_allowlist(
        [
            {"title": "Good", "url": "https://www.rbi.org.in/press", "snippet": "x"},
            {"title": "Bad", "url": "https://random-blog.example.com/x", "snippet": "y"},
        ]
    )
    assert len(hits) == 1
    assert url_on_allowlist("https://nseindia.com/market-data")


def test_insufficient_evidence_node_uses_policy_message():
    out = insufficient_evidence_node({"retrieved_docs": []})
    assert out["answer"] == insufficient_evidence_message()
    assert out["model_used"] == "grounding_gate"


def test_dispatch_routes_none_to_insufficient():
    assert dispatch_grounding({"grounding_status": "none"}) == "insufficient_evidence"
    partial_state = {
        "grounding_status": "partial",
        "retrieval_grade": "relevant",
        "retrieved_docs": [{"source_type": "faiss", "confidence": 0.41}],
    }
    assert dispatch_grounding(partial_state) == "insufficient_evidence"
    full_state = {
        "grounding_status": "full",
        "retrieval_grade": "relevant",
        "retrieved_docs": [{"source_type": "faiss", "confidence": 0.9}],
    }
    assert dispatch_grounding(full_state) == "generate"


def test_assess_evidence_partial_vs_full():
    partial = assess_evidence_strength({"retrieved_docs": [{"confidence": 0.41}]})
    full = assess_evidence_strength({"retrieved_docs": [{"confidence": 0.9}]})
    none = assess_evidence_strength({"retrieved_docs": []})
    assert partial == "partial"
    assert full == "full"
    assert none == "none"
