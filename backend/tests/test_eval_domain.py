import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.domain_classifier import classify_domain, extract_finance_query


def test_hinglish_mutual_fund_allowed():
    decision = classify_domain("Mutual fund kya hota hai?", demo_mode=True)
    assert decision.status == "authorized"


def test_hinglish_sip_fd_allowed():
    decision = classify_domain("SIP aur FD mein kya farak hai?", demo_mode=True)
    assert decision.status == "authorized"


def test_borderline_nav_allowed():
    decision = classify_domain("What is NAV in mutual funds?", demo_mode=True)
    assert decision.status == "authorized"


def test_off_topic_vacation_denied():
    decision = classify_domain("Plan a vacation in Japan", demo_mode=True)
    assert decision.status == "unauthorized"


def test_mixed_oscar_strips_non_finance():
    q = "What is an ETF? Also who won the Oscar?"
    extracted = extract_finance_query(q)
    assert "etf" in extracted.lower()
    assert "oscar" not in extracted.lower()
    assert classify_domain(q, demo_mode=True).status == "authorized"


def test_finance_complex_compare_allowed():
    decision = classify_domain("Compare ETFs and mutual funds for a beginner", demo_mode=True)
    assert decision.status == "authorized"


def test_geography_capital_question_denied():
    from agents.domain_classifier import load_domain_config

    load_domain_config.cache_clear()
    decision = classify_domain("What is the capital of Japan?", demo_mode=True)
    assert decision.status == "unauthorized"
    assert decision.layer == "rules_deny"


def test_capital_gains_still_finance():
    from agents.domain_classifier import load_domain_config

    load_domain_config.cache_clear()
    decision = classify_domain("How is short-term capital gains tax calculated?", demo_mode=True)
    assert decision.status == "authorized"
