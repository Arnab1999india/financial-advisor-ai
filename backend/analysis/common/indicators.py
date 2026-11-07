"""Reusable OHLCV technical indicators (pandas / NumPy)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("Date", "Open", "High", "Low", "Close", "Volume")

SMA_PERIODS: tuple[int, ...] = (20, 44, 50, 200)
EMA_PERIODS: tuple[int, ...] = (9, 21, 50)
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_SMA_PERIOD = 20


def _validate_period(period: int, *, name: str = "period") -> None:
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_series(series: pd.Series, *, name: str) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"{name} must be numeric.")


def ensure_ohlcv_columns(df: pd.DataFrame) -> None:
    """Verify canonical OHLCV columns exist."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    missing = [col for col in REQUIRED_OHLCV if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    _validate_series(series, name="series")
    _validate_period(period)
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    _validate_series(series, name="series")
    _validate_period(period)
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    _validate_series(close, name="close")
    _validate_period(period, name="period")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range per bar."""
    _validate_series(high, name="high")
    _validate_series(low, name="low")
    _validate_series(close, name="close")
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    _validate_period(period, name="period")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return middle, upper, and lower Bollinger Bands."""
    _validate_series(close, name="close")
    _validate_period(period, name="period")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal_period: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return MACD line, signal line, and histogram."""
    _validate_series(close, name="close")
    for label, value in (("fast", fast), ("slow", slow), ("signal_period", signal_period)):
        _validate_period(value, name=label)
    if fast >= slow:
        raise ValueError("MACD fast period must be less than slow period.")
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def volume_sma(volume: pd.Series, period: int = VOLUME_SMA_PERIOD) -> pd.Series:
    """Simple moving average of volume."""
    _validate_series(volume, name="volume")
    return sma(volume, period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    _validate_series(close, name="close")
    _validate_series(volume, name="volume")
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def compute_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach standard indicator columns to a copy of ``df``."""
    ensure_ohlcv_columns(df)
    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"]

    for period in SMA_PERIODS:
        result[f"sma_{period}"] = sma(close, period)
    for period in EMA_PERIODS:
        result[f"ema_{period}"] = ema(close, period)

    result["rsi_14"] = rsi(close, RSI_PERIOD)
    result["atr_14"] = atr(high, low, close, ATR_PERIOD)

    bb_mid, bb_up, bb_low = bollinger_bands(close, BB_PERIOD, BB_STD)
    result["bb_middle_20"] = bb_mid
    result["bb_upper_20"] = bb_up
    result["bb_lower_20"] = bb_low

    macd_line, signal_line, hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    result["macd_line"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = hist

    result["volume_sma_20"] = volume_sma(volume, VOLUME_SMA_PERIOD)
    result["obv"] = obv(close, volume)
    return result


def latest_finite(values: Iterable[float | None]) -> float | None:
    """Return the last finite value from an iterable of floats."""
    last: float | None = None
    for value in values:
        if value is None:
            continue
        if isinstance(value, (float, int)) and np.isfinite(value):
            last = float(value)
    return last
