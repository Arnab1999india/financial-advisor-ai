import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage

from agents.outbound_guardrail import (
    STATUTORY_RISK_DISCLAIMER,
    amend_to_educational,
    append_statutory_disclaimer,
    contains_direct_advice,
    outbound_guardrail_node,
)


def test_detects_you_should_buy_style_advice():
    assert contains_direct_advice("You should buy stock X because it will go up.")
    assert contains_direct_advice("I recommend that you sell NVDA today.")
    assert contains_direct_advice("This is a strong buy on that ticker.")


def test_educational_explanations_are_not_flagged():
    assert not contains_direct_advice(
        "A buy signal in technical analysis occurs when a short moving average crosses above a long one."
    )
    assert not contains_direct_advice(
        "Diversification is a way investors manage risk across asset classes."
    )


def test_amendment_removes_directive_and_adds_disclaimer():
    original = "You should buy stock X. It looks cheap."
    amended = append_statutory_disclaimer(amend_to_educational(original))
    assert not contains_direct_advice(amended.split("Statutory risk disclaimer")[0])
    assert "does not constitute investment advice" in amended.lower()
    assert "You should buy" not in amended
    assert STATUTORY_RISK_DISCLAIMER.strip() in amended


def test_disclaimer_is_not_duplicated():
    once = append_statutory_disclaimer("Educational framing (not a personal recommendation):\n\nSome context.")
    twice = append_statutory_disclaimer(once)
    assert twice.count("Statutory risk disclaimer") == 1


def test_guardrail_node_intercepts_and_rewrites():
    original = "You should buy stock X right now."
    result = outbound_guardrail_node(
        {
            "answer": original,
            "messages": [AIMessage(content=original, id="ai-1")],
        }
    )
    assert result["guardrail_amended"] is True
    assert "does not constitute investment advice" in result["answer"].lower()
    assert "You should buy" not in result["answer"]
    assert not contains_direct_advice(result["answer"].split("Statutory risk disclaimer")[0])


def test_guardrail_node_passes_through_educational_answers():
    original = "RSI is a momentum oscillator that ranges from 0 to 100."
    result = outbound_guardrail_node({"answer": original, "messages": []})
    assert result == {"guardrail_amended": False}
