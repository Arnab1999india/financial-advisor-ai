import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.grounding_gate import assess_evidence_strength, insufficient_evidence_node
from rag.retriever import FinancialRetriever
from validation.output_validator import validate_structured_output
from validation.structured_output import StructuredAnswer, StructuredCitation


def _retriever() -> FinancialRetriever:
    path = os.path.join(os.path.dirname(__file__), "..", "sample_qa.json")
    with open(path, encoding="utf-8") as fh:
        qa = json.load(fh)
    r = FinancialRetriever()
    r.initialize(qa)
    return r


def test_known_kb_question_retrieves_etf():
    hits = _retriever().retrieve("What is an ETF exchange traded fund?", k=4)
    assert hits
    assert any("etf" in (h.get("answer") or "").lower() for h in hits)


def test_unknown_topic_low_or_no_evidence():
    hits = _retriever().retrieve("ZYX-9999 proprietary capital metric 2099", k=4)
    strength = assess_evidence_strength({"retrieved_docs": hits})
    top = max((h.get("confidence") or 0.0) for h in hits) if hits else 0.0
    assert strength in ("none", "partial") or top < 0.52


def test_insufficient_evidence_refusal_copy():
    out = insufficient_evidence_node({"retrieved_docs": []})
    assert out["model_used"] == "grounding_gate"
    assert out["answer"]


def test_exact_number_from_source_passes_validation():
    docs = [
        {
            "source_file": "sample_qa.json",
            "page": 4,
            "answer": "For example, $1,000 at 10% annual compound interest becomes $2,594 after 10 years.",
            "question": "What is compound interest?",
        }
    ]
    structured = StructuredAnswer(
        answer="At 10% compound interest, $1,000 becomes $2,594 after 10 years.",
        citations=[StructuredCitation(source="sample_qa.json", page=4)],
        confidence="high",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=docs, context="", used_market_data=False)
    assert result.ok


def test_wrong_number_fails_validation():
    docs = [
        {
            "source_file": "sample_qa.json",
            "page": 4,
            "answer": "For example, $1,000 at 10% becomes $2,594 after 10 years.",
        }
    ]
    structured = StructuredAnswer(
        answer="The amount grows to $9,999 after 10 years.",
        citations=[StructuredCitation(source="sample_qa.json", page=4)],
        confidence="high",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=docs, context="", used_market_data=False)
    assert not result.ok
    assert any("untracked_number" in r for r in result.reasons)
