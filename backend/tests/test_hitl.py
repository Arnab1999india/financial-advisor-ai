import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.human_approval import (
    build_action_proposal,
    dispatch_approval,
    extract_symbol,
    is_high_probability_setup,
    is_simulation_request,
)


def test_detects_paper_trade_request():
    assert is_simulation_request("Please simulate a paper trade: buy 5 AAPL") is True


def test_detects_alert_request():
    assert is_simulation_request("Set a price alert on NVDA when it breaks out") is True


def test_plain_definition_is_not_a_simulation_request():
    assert is_simulation_request("What is a buy signal in technical analysis?") is False


def test_extracts_ticker_from_query():
    assert extract_symbol("Paper trade $MSFT with a tight stop") == "MSFT"


def test_high_probability_setup_requires_actionable_language():
    state = {
        "query": "Is this a high-probability buy setup in AAPL?",
        "answer": "This looks like a high-probability buy setup given the golden-cross confluence.",
        "confidence": 0.88,
        "retrieved_docs": [{"confidence": 0.9, "answer": "AAPL breakout with confluence."}],
    }
    assert is_high_probability_setup(state) is True


def test_educational_buy_definition_is_not_gated():
    state = {
        "query": "What is a buy signal?",
        "answer": "A buy signal in technical analysis occurs when a short moving average crosses above a long one.",
        "confidence": 0.91,
        "retrieved_docs": [{"confidence": 0.91, "answer": "A buy signal is a technical indication."}],
    }
    assert is_high_probability_setup(state) is False


def test_proposal_defaults_and_gate_dispatch():
    state = {
        "query": "simulate a paper trade buy 10 TSLA",
        "answer": "",
        "confidence": 0.4,
    }
    proposal = build_action_proposal(state)
    assert proposal["symbol"] == "TSLA"
    assert proposal["side"] == "BUY"
    assert proposal["quantity"] == 10.0
    gated = {"approval_required": True}
    assert dispatch_approval(gated) == "human_approval"
    assert dispatch_approval({"approval_required": False}) == "memory_save"
