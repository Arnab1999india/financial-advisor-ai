"""Trend structure (HH/HL/LH/LL) swing strategy on daily OHLCV."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.common.indicators import atr, ema, ensure_ohlcv_columns
from analysis.common.risk_management import atr_stops_and_target, risk_reward_ratio
from analysis.common.signal_schema import (
    Signal,
    SignalDirection,
    SignalStatus,
    insufficient_data_signal,
)
from analysis.swing.support_resistance import detect_confirmed_pivots


def classify_market_structure(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
) -> str:
    """Classify bullish, bearish, or sideways from the last two pivots."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "sideways"

    _, h1 = swing_highs[-2]
    _, h2 = swing_highs[-1]
    _, l1 = swing_lows[-2]
    _, l2 = swing_lows[-1]

    hh = h2 > h1
    hl = l2 > l1
    lh = h2 < h1
    ll = l2 < l1

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "sideways"


def analyze_trend_structure(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "1D",
    pivot_lookback: int = 5,
    min_rows: int = 80,
) -> Signal:
    """Evaluate market structure using confirmed swings and EMA trend filter."""
    strategy = "trend_structure"
    try:
        ensure_ohlcv_columns(df)
    except (TypeError, ValueError) as exc:
        return insufficient_data_signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            reason=str(exc),
        )

    if len(df) < min_rows:
        return insufficient_data_signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            reason=f"Need at least {min_rows} daily candles; got {len(df)}.",
        )

    work = df.sort_values("Date").reset_index(drop=True)
    swing_highs, swing_lows = detect_confirmed_pivots(work, lookback=pivot_lookback)
    structure = classify_market_structure(swing_highs, swing_lows)

    ema20 = ema(work["Close"], 20)
    ema50 = ema(work["Close"], 50)
    idx = len(work) - 1
    close = float(work["Close"].iloc[idx])
    e20 = _finite_at(ema20, idx)
    e50 = _finite_at(ema50, idx)

    atr14 = atr(work["High"], work["Low"], work["Close"], 14)
    atr_val = _finite_at(atr14, idx)

    evidence: dict[str, Any] = {
        "structure": structure,
        "swing_highs": swing_highs[-4:],
        "swing_lows": swing_lows[-4:],
        "ema_20": e20,
        "ema_50": e50,
    }

    if structure == "sideways" or e20 is None or e50 is None:
        return Signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.HOLD,
            status=SignalStatus.VALID,
            reason="Sideways or unconfirmed structure / EMAs not ready.",
            indicator_values={"close": close, "ema_20": e20, "ema_50": e50},
            timestamp=_last_timestamp(work),
            metadata=evidence,
        )

    if structure == "bullish" and close > e20 > e50:
        if atr_val is None:
            return Signal(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                direction=SignalDirection.HOLD,
                status=SignalStatus.VALID,
                reason="Bullish structure but ATR unavailable.",
                indicator_values={"close": close},
                metadata=evidence,
            )
        entry = close
        stop, target = atr_stops_and_target(entry, atr_val, "BUY")
        rr = risk_reward_ratio(entry, stop, target, "BUY")
        signal = Signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.BUY,
            status=SignalStatus.VALID,
            entry=entry,
            stop_loss=stop,
            target=target,
            risk_reward_ratio=rr,
            reason="Successive HH/HL with price above rising 20/50 EMA.",
            indicator_values={"close": close, "ema_20": e20, "ema_50": e50, "atr_14": atr_val},
            timestamp=_last_timestamp(work),
            metadata=evidence,
        )
        signal.validate()
        return signal

    if structure == "bearish" and close < e20 < e50:
        if atr_val is None:
            return Signal(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                direction=SignalDirection.HOLD,
                status=SignalStatus.VALID,
                reason="Bearish structure but ATR unavailable.",
                indicator_values={"close": close},
                metadata=evidence,
            )
        entry = close
        stop, target = atr_stops_and_target(entry, atr_val, "SELL")
        rr = risk_reward_ratio(entry, stop, target, "SELL")
        signal = Signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.SELL,
            status=SignalStatus.VALID,
            entry=entry,
            stop_loss=stop,
            target=target,
            risk_reward_ratio=rr,
            reason="Successive LH/LL with price below falling 20/50 EMA.",
            indicator_values={"close": close, "ema_20": e20, "ema_50": e50, "atr_14": atr_val},
            timestamp=_last_timestamp(work),
            metadata=evidence,
        )
        signal.validate()
        return signal

    return Signal(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        direction=SignalDirection.HOLD,
        status=SignalStatus.VALID,
        reason="Structure and EMA trend filter not aligned.",
        indicator_values={"close": close, "ema_20": e20, "ema_50": e50},
        timestamp=_last_timestamp(work),
        metadata=evidence,
    )


def _finite_at(series: pd.Series, index: int) -> float | None:
    value = series.iloc[index]
    if pd.isna(value):
        return None
    return float(value)


def _last_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    if "Date" not in df.columns or df.empty:
        return None
    ts = df["Date"].iloc[-1]
    return pd.Timestamp(ts) if pd.notna(ts) else None
