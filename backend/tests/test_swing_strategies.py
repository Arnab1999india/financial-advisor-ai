import numpy as np
import pandas as pd

from analysis.common.signal_schema import SignalStatus
from analysis.swing.breakout_retest import analyze_breakout_retest
from analysis.swing.support_resistance import analyze_support_resistance
from analysis.swing.trend_structure import analyze_trend_structure, classify_market_structure


def _trend_df(rows: int = 120, drift: float = 0.4) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    close = 100 + np.arange(rows) * drift
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(rows, 2000),
        }
    )


def test_insufficient_history_returns_insufficient_data():
    tiny = _trend_df(10)
    signal = analyze_trend_structure(tiny, symbol="TEST")
    assert signal.status == SignalStatus.INSUFFICIENT_DATA


def test_trend_structure_on_uptrend_returns_signal_object():
    df = _trend_df(120)
    signal = analyze_trend_structure(df, symbol="TEST", min_rows=80)
    assert signal.strategy == "trend_structure"
    assert signal.status in {SignalStatus.VALID, SignalStatus.INSUFFICIENT_DATA}


def test_support_resistance_runs_without_error():
    df = _trend_df(120)
    signal = analyze_support_resistance(df, symbol="TEST", min_rows=60)
    assert signal.strategy == "support_resistance"
    assert "levels" in signal.metadata or signal.status == SignalStatus.INSUFFICIENT_DATA


def test_breakout_retest_hold_when_no_setup():
    df = _trend_df(120)
    signal = analyze_breakout_retest(df, symbol="TEST", min_rows=80)
    assert signal.strategy == "breakout_retest"


def test_classify_market_structure_bullish():
    highs = [(10, 110.0), (20, 120.0)]
    lows = [(12, 100.0), (22, 105.0)]
    assert classify_market_structure(highs, lows) == "bullish"
