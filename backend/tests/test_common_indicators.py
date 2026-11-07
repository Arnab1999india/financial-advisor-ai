import numpy as np
import pandas as pd
import pytest

from analysis.common.indicators import (
    atr,
    compute_standard_indicators,
    ema,
    macd,
    obv,
    rsi,
    sma,
)


def _ohlcv(rows: int = 250, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, size=rows))
    high = close + rng.uniform(0.5, 2.0, size=rows)
    low = close - rng.uniform(0.5, 2.0, size=rows)
    open_ = close + rng.normal(0, 0.5, size=rows)
    volume = rng.integers(1_000, 5_000, size=rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def test_sma_insufficient_history_is_nan():
    series = pd.Series([1.0, 2.0, 3.0])
    out = sma(series, 5)
    assert out.isna().all()


def test_rsi_wilder_produces_values_with_enough_history():
    df = _ohlcv(100)
    values = rsi(df["Close"], 14)
    assert values.iloc[-1] == values.iloc[-1]
    assert values.iloc[:13].isna().all()
    assert values.iloc[13:].notna().any()


def test_macd_histogram_alignment():
    df = _ohlcv(120)
    line, signal, hist = macd(df["Close"])
    assert len(line) == len(hist)
    assert (hist - (line - signal)).dropna().abs().max() < 1e-9


def test_compute_standard_indicators_does_not_mutate_source():
    df = _ohlcv(250)
    original_cols = list(df.columns)
    enriched = compute_standard_indicators(df)
    assert list(df.columns) == original_cols
    assert "sma_20" in enriched.columns
    assert "obv" in enriched.columns


def test_obv_and_atr_require_positive_inputs():
    df = _ohlcv(50)
    atr_series = atr(df["High"], df["Low"], df["Close"], 14)
    assert atr_series.iloc[-1] > 0
    obv_series = obv(df["Close"], df["Volume"])
    assert obv_series.notna().all()


def test_invalid_period_raises():
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), 0)
