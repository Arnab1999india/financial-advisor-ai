import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.market_data import _symbol_from_query, is_market_data_query


def test_recognises_equity_price_query():
    assert is_market_data_query("What is the latest price of TCS?") is True
    assert _symbol_from_query("What is the latest price of TCS?") == ("TCS.NS", "TCS")


def test_recognises_bse_symbol_request():
    assert is_market_data_query("What is the BSE price of TCS?") is True
    assert _symbol_from_query("What is the BSE price of TCS?") == ("TCS.BO", "TCS")


def test_recognises_metals_and_crypto():
    assert _symbol_from_query("gold price today") == ("GC=F", "Gold futures")
    assert _symbol_from_query("What is BTC price?") == ("BTC-USD", "Bitcoin")


def test_does_not_treat_general_finance_as_a_quote():
    assert is_market_data_query("Explain the price to earnings ratio") is False
