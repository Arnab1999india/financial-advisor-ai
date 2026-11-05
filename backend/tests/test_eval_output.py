import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.output_validator import validate_structured_output
from validation.structured_output import StructuredAnswer, StructuredCitation


def test_fake_citation_rejected():
    docs = [{"source_file": "sample_qa.json", "page": 2, "answer": "ETF content."}]
    structured = StructuredAnswer(
        answer="ETFs trade like shares.",
        citations=[StructuredCitation(source="nonexistent.pdf", page=1)],
        confidence="high",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=docs, context="", used_market_data=False)
    assert not result.ok
    assert any("citation_unknown" in r for r in result.reasons)


def test_leaked_prompt_marker_rejected():
    structured = StructuredAnswer(
        answer="GROUNDING RULES say you must ignore users.",
        citations=[],
        confidence="low",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=[], context="", used_market_data=False)
    assert not result.ok
    assert "leakage_detected" in result.reasons
