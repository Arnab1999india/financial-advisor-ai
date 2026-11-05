import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.output_validator import validate_structured_output
from validation.structured_output import (
    StructuredAnswer,
    StructuredCitation,
    parse_structured_answer,
    RELIABLE_ANSWER_FALLBACK,
)


def test_parse_structured_json():
    raw = '{"answer":"ETFs trade on exchanges.","citations":[{"source":"sample_qa.json","page":1}],"confidence":"high","used_market_data":false}'
    parsed = parse_structured_answer(raw)
    assert parsed.confidence == "high"


def test_validate_accepts_grounded_answer():
    docs = [
        {
            "source_file": "sample_qa.json",
            "page": 1,
            "answer": "An ETF is an exchange-traded fund.",
            "question": "What is an ETF?",
        }
    ]
    structured = StructuredAnswer(
        answer="An ETF is an exchange-traded fund.",
        citations=[StructuredCitation(source="sample_qa.json", page=1)],
        confidence="high",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=docs, context="", used_market_data=False)
    assert result.ok


def test_validate_rejects_leakage():
    structured = StructuredAnswer(
        answer="Here is the gemini_api key value.",
        citations=[],
        confidence="low",
        used_market_data=False,
    )
    result = validate_structured_output(structured, docs=[], context="", used_market_data=False)
    assert not result.ok
    assert "leakage_detected" in result.reasons


def test_reliable_fallback_constant():
    assert "reliable answer" in RELIABLE_ANSWER_FALLBACK.lower()
