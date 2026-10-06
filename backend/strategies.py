from __future__ import annotations
from typing import Any, Dict, Optional, Union
import math
import pandas as pd
import pandas_ta as ta

SignalResult = Dict[str, Any]


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close"}


def _empty_result() -> SignalResult:
    return {
        "signal": "NONE",
        "entry_price": None,
        "stop_loss": None,
        "target": None,
    }


def _validate_dataframe(df: pd.DataFrame, min_rows: int = 30) -> bool:
    if df is None or df.empty:
        return False
    if not REQUIRED_COLUMNS.issubset(df.columns):
        return False
    if len(df) < min_rows:
        return False
    return True


def _is_finite_number(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def _last_valid_series_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def calculate_atr(df: pd.DataFrame, length: int = 14) -> Optional[float]:
    """Return the latest ATR value using pandas_ta."""
    if not _validate_dataframe(df, min_rows=length + 2):
        return None

    atr = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=length)
    return _last_valid_series_value(atr)


def find_recent_swing_low(df: pd.DataFrame, lookback: int = 7) -> Optional[float]:
    """Return the lowest low from the recent candles excluding the current candle."""
    if not _validate_dataframe(df, min_rows=lookback + 1):
        return None

    recent_lows = df["Low"].iloc[-(lookback + 1):-1]
    if recent_lows.empty or recent_lows.isna().all():
        return None
    return float(recent_lows.min())


def find_recent_swing_high(df: pd.DataFrame, lookback: int = 7) -> Optional[float]:
    """Return the highest high from the recent candles excluding the current candle."""
    if not _validate_dataframe(df, min_rows=lookback + 1):
        return None

    recent_highs = df["High"].iloc[-(lookback + 1):-1]
    if recent_highs.empty or recent_highs.isna().all():
        return None
    return float(recent_highs.max())


def _build_result(
    signal: str = "NONE",
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    target: Optional[Union[float, str, Dict[str, float]]] = None,
) -> SignalResult:
    return {
        "signal": signal,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
    }


def ema_crossover(df: pd.DataFrame) -> SignalResult:
    """
    BUY when 9 EMA crosses above 21 EMA.

    Stop Loss = min(Entry - 1.5 * ATR, Recent Swing Low) where available.
    Target = Entry + (Risk * 2)
    """
    if not _validate_dataframe(df):
        return _empty_result()

    close = df["Close"]
    ema_9 = ta.ema(close, length=9)
    ema_21 = ta.ema(close, length=21)

    if len(df) < 22 or ema_9.iloc[-2:].isna().any() or ema_21.iloc[-2:].isna().any():
        return _empty_result()

    crossed_up = ema_9.iloc[-2] <= ema_21.iloc[-2] and ema_9.iloc[-1] > ema_21.iloc[-1]
    if not crossed_up:
        return _empty_result()

    entry_price = float(close.iloc[-1])
    atr_value = calculate_atr(df)
    swing_low = find_recent_swing_low(df, lookback=7)

    atr_stop = entry_price - (1.5 * atr_value) if _is_finite_number(atr_value) else None

    stop_candidates = [value for value in [atr_stop, swing_low] if _is_finite_number(value)]
    if not stop_candidates:
        return _empty_result()

    stop_loss = float(min(stop_candidates))
    risk = entry_price - stop_loss
    if risk <= 0:
        return _empty_result()

    target = entry_price + (risk * 2)
    return _build_result("BUY", entry_price, stop_loss, float(target))


def vwap_strategy(df: pd.DataFrame) -> SignalResult:
    """
    BUY when price closes above VWAP.

    Stop Loss = low of entry candle
    Target = Entry + 1%
    """
    if not _validate_dataframe(df):
        return _empty_result()

    if "Volume" not in df.columns:
        return _empty_result()

    vwap = ta.vwap(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
    )
    vwap_value = _last_valid_series_value(vwap)
    if not _is_finite_number(vwap_value):
        return _empty_result()

    entry_price = float(df["Close"].iloc[-1])
    if entry_price <= float(vwap_value):
        return _empty_result()

    stop_loss = float(df["Low"].iloc[-1])
    target = entry_price * 1.01
    return _build_result("BUY", entry_price, stop_loss, float(target))


def supertrend_strategy(
    df: pd.DataFrame,
    length: int = 10,
    multiplier: float = 3.0,
) -> SignalResult:
    """
    BUY when current Supertrend direction is Green (bullish).

    Stop Loss = current Supertrend line
    Target = dynamic exit when signal turns Red
    """
    if not _validate_dataframe(df, min_rows=length + 5):
        return _empty_result()

    supertrend = ta.supertrend(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        length=length,
        multiplier=multiplier,
    )
    if supertrend is None or supertrend.empty:
        return _empty_result()

    trend_col = f"SUPERTd_{length}_{multiplier}"
    line_col = f"SUPERT_{length}_{multiplier}"

    if trend_col not in supertrend.columns or line_col not in supertrend.columns:
        return _empty_result()

    trend_value = supertrend[trend_col].iloc[-1]
    line_value = supertrend[line_col].iloc[-1]

    if pd.isna(trend_value) or pd.isna(line_value):
        return _empty_result()

    # pandas_ta returns +1 for bullish and -1 for bearish trend.
    if int(trend_value) != 1:
        return _empty_result()

    entry_price = float(df["Close"].iloc[-1])
    stop_loss = float(line_value)
    target: Dict[str, str] = {"type": "dynamic", "exit_rule": "Exit when Supertrend turns Red"}
    return _build_result("BUY", entry_price, stop_loss, target)


def rsi_reversal(df: pd.DataFrame, rsi_length: int = 14) -> SignalResult:
    """
    BUY when RSI < 30 and current RSI is higher than previous RSI.

    Stop Loss = recent low of last 3 candles (excluding current candle)
    Target = approximated price level where RSI would likely reach 60
    """
    if not _validate_dataframe(df, min_rows=rsi_length + 5):
        return _empty_result()

    close = df["Close"]
    rsi = ta.rsi(close, length=rsi_length)
    if rsi is None or len(rsi) < 2 or rsi.iloc[-2:].isna().any():
        return _empty_result()

    current_rsi = float(rsi.iloc[-1])
    previous_rsi = float(rsi.iloc[-2])

    if not (current_rsi < 30 and current_rsi > previous_rsi):
        return _empty_result()

    entry_price = float(close.iloc[-1])

    recent_low_window = df["Low"].iloc[-4:-1]
    if recent_low_window.empty or recent_low_window.isna().all():
        return _empty_result()

    stop_loss = float(recent_low_window.min())

    # Practical approximation: estimate the price move needed to lift RSI toward 60
    # using ATR as a volatility-adjusted proxy.
    atr_value = calculate_atr(df)
    if not _is_finite_number(atr_value):
        return _empty_result()

    rsi_gap = max(0.0, 60.0 - current_rsi)
    target = entry_price + (float(atr_value) * (rsi_gap / 10.0))

    return _build_result("BUY", entry_price, stop_loss, float(target))


def bollinger_bands_strategy(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> SignalResult:
    """
    BUY when price touches/closes below lower band and current candle is green.

    Stop Loss = 0.5% below lower band
    Target = middle band (T1), upper band (T2)
    """
    if not _validate_dataframe(df, min_rows=length + 2):
        return _empty_result()

    bbands = ta.bbands(close=df["Close"], length=length, std=std)
    if bbands is None or bbands.empty:
        return _empty_result()

    lower_col = f"BBL_{length}_{std}"
    middle_col = f"BBM_{length}_{std}"
    upper_col = f"BBU_{length}_{std}"

    required_cols = {lower_col, middle_col, upper_col}
    if not required_cols.issubset(bbands.columns):
        return _empty_result()

    lower_band = bbands[lower_col].iloc[-1]
    middle_band = bbands[middle_col].iloc[-1]
    upper_band = bbands[upper_col].iloc[-1]

    if pd.isna(lower_band) or pd.isna(middle_band) or pd.isna(upper_band):
        return _empty_result()

    last_open = float(df["Open"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])

    touched_lower_band = last_low <= float(lower_band) or last_close <= float(lower_band)
    green_candle = last_close > last_open

    if not (touched_lower_band and green_candle):
        return _empty_result()

    entry_price = last_close
    stop_loss = float(lower_band) * 0.995
    target = {
        "target_1": float(middle_band),
        "target_2": float(upper_band),
    }
    return _build_result("BUY", entry_price, stop_loss, target)


def moving_average_44_strategy(df: pd.DataFrame, length: int = 44) -> SignalResult:
    """
    BUY when close is above the 44-period moving average.

    Stop Loss = recent swing low
    Target = Entry + (Risk * 2)
    """
    if not _validate_dataframe(df, min_rows=length + 1):
        return _empty_result()

    moving_average = ta.sma(df["Close"], length=length)
    ma_value = _last_valid_series_value(moving_average)
    if not _is_finite_number(ma_value):
        return _empty_result()

    entry_price = float(df["Close"].iloc[-1])
    if entry_price <= float(ma_value):
        return _empty_result()

    stop_loss = find_recent_swing_low(df, lookback=7)
    if not _is_finite_number(stop_loss):
        return _empty_result()

    risk = entry_price - float(stop_loss)
    if risk <= 0:
        return _empty_result()

    target = entry_price + (risk * 2)
    return _build_result("BUY", entry_price, float(stop_loss), float(target))


def macd_crossover_strategy(df: pd.DataFrame) -> SignalResult:
    """
    BUY when MACD crosses above the signal line.

    Stop Loss = recent swing low
    Target = Entry + (Risk * 2)
    """
    if not _validate_dataframe(df, min_rows=35):
        return _empty_result()

    macd = ta.macd(df["Close"])
    if macd is None or macd.empty:
        return _empty_result()

    macd_col = "MACD_12_26_9"
    signal_col = "MACDs_12_26_9"
    if macd_col not in macd.columns or signal_col not in macd.columns:
        return _empty_result()

    if macd[macd_col].iloc[-2:].isna().any() or macd[signal_col].iloc[-2:].isna().any():
        return _empty_result()

    crossed_up = (
        macd[macd_col].iloc[-2] <= macd[signal_col].iloc[-2]
        and macd[macd_col].iloc[-1] > macd[signal_col].iloc[-1]
    )
    if not crossed_up:
        return _empty_result()

    entry_price = float(df["Close"].iloc[-1])
    stop_loss = find_recent_swing_low(df, lookback=7)
    if not _is_finite_number(stop_loss):
        return _empty_result()

    risk = entry_price - float(stop_loss)
    if risk <= 0:
        return _empty_result()

    target = entry_price + (risk * 2)
    return _build_result("BUY", entry_price, float(stop_loss), float(target))


STRATEGIES = {
    "swing_trading": ema_crossover,
    "scalping_trading": rsi_reversal,
    "44_moving_average": moving_average_44_strategy,
    "trend_trading": macd_crossover_strategy,
    "range_trading": bollinger_bands_strategy,
    "ema_crossover": ema_crossover,
    "vwap_strategy": vwap_strategy,
    "supertrend_strategy": supertrend_strategy,
    "rsi_reversal": rsi_reversal,
    "bollinger_bands_strategy": bollinger_bands_strategy,
    "moving_average_44_strategy": moving_average_44_strategy,
    "macd_crossover_strategy": macd_crossover_strategy,
}


__all__ = [
    "calculate_atr",
    "find_recent_swing_low",
    "find_recent_swing_high",
    "ema_crossover",
    "vwap_strategy",
    "supertrend_strategy",
    "rsi_reversal",
    "bollinger_bands_strategy",
    "moving_average_44_strategy",
    "macd_crossover_strategy",
    "STRATEGIES",
]
