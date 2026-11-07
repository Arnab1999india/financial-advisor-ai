"""Breakout-and-retest swing strategy on daily OHLCV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.common.indicators import ensure_ohlcv_columns, volume_sma
from analysis.common.risk_management import atr_stops_and_target, risk_reward_ratio
from analysis.common.signal_schema import (
    Signal,
    SignalDirection,
    SignalStatus,
    insufficient_data_signal,
)
from analysis.swing.support_resistance import build_sr_levels, nearest_support_resistance


def analyze_breakout_retest(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "1D",
    pivot_lookback: int = 5,
    tolerance_pct: float = 0.015,
    min_touches: int = 2,
    breakout_buffer_pct: float = 0.003,
    retest_window: int = 10,
    volume_factor: float = 1.2,
    min_rows: int = 80,
) -> Signal:
    """Detect breakout, retest, and confirmation on confirmed S/R levels."""
    strategy = "breakout_retest"
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
    levels = build_sr_levels(
        work.iloc[:-1],
        pivot_lookback=pivot_lookback,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
    )

    idx = len(work) - 1
    row = work.iloc[idx]
    close = float(row["Close"])
    open_ = float(row["Open"])
    volume = float(row["Volume"]) if pd.notna(row["Volume"]) else np.nan
    vol_sma = volume_sma(work["Volume"], 20)
    vol_avg = _finite_at(vol_sma, idx)

    from analysis.common.indicators import atr

    atr14 = atr(work["High"], work["Low"], work["Close"], 14)
    atr_val = _finite_at(atr14, idx)

    support, resistance = nearest_support_resistance(levels, close)
    metadata: dict[str, Any] = {
        "resistance": resistance.price if resistance else None,
        "support": support.price if support else None,
    }

    if resistance is None and support is None:
        return Signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.HOLD,
            status=SignalStatus.VALID,
            reason="No levels available for breakout evaluation.",
            timestamp=_last_timestamp(work),
            metadata=metadata,
        )

    window = work.iloc[max(0, idx - retest_window) : idx + 1]

    buy_signal = _bullish_breakout_retest(
        window,
        resistance.price if resistance else None,
        breakout_buffer_pct=breakout_buffer_pct,
        volume=volume,
        vol_avg=vol_avg,
        volume_factor=volume_factor,
    )
    if buy_signal and atr_val is not None:
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
            reason="Bullish breakout above resistance, retest hold, bullish confirmation.",
            indicator_values={
                "close": close,
                "volume": volume,
                "volume_sma_20": vol_avg,
                "atr_14": atr_val,
            },
            timestamp=_last_timestamp(work),
            metadata={**metadata, **buy_signal},
        )
        signal.validate()
        return signal

    sell_signal = _bearish_breakout_retest(
        window,
        support.price if support else None,
        breakout_buffer_pct=breakout_buffer_pct,
        volume=volume,
        vol_avg=vol_avg,
        volume_factor=volume_factor,
    )
    if sell_signal and atr_val is not None:
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
            reason="Bearish breakdown below support, retest rejection, bearish confirmation.",
            indicator_values={
                "close": close,
                "volume": volume,
                "volume_sma_20": vol_avg,
                "atr_14": atr_val,
            },
            timestamp=_last_timestamp(work),
            metadata={**metadata, **sell_signal},
        )
        signal.validate()
        return signal

    return Signal(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        direction=SignalDirection.HOLD,
        status=SignalStatus.VALID,
        reason="No confirmed breakout-retest setup on the latest bar.",
        indicator_values={"close": close, "open": open_, "volume_sma_20": vol_avg},
        timestamp=_last_timestamp(work),
        metadata=metadata,
    )


def _bullish_breakout_retest(
    window: pd.DataFrame,
    resistance: float | None,
    *,
    breakout_buffer_pct: float,
    volume: float,
    vol_avg: float | None,
    volume_factor: float,
) -> dict[str, Any] | None:
    if resistance is None or len(window) < 3:
        return None

    closes = window["Close"].to_numpy(dtype=float)
    lows = window["Low"].to_numpy(dtype=float)
    level = resistance * (1 + breakout_buffer_pct)
    band = resistance * breakout_buffer_pct

    breakout_idx = None
    for i in range(len(window) - 2):
        if closes[i] > level:
            breakout_idx = i
            break
    if breakout_idx is None:
        return None

    if vol_avg is not None and np.isfinite(volume) and volume < vol_avg * volume_factor:
        return None

    retest = False
    for j in range(breakout_idx + 1, len(window) - 1):
        if abs(lows[j] - resistance) <= band * 2 and closes[j] >= resistance:
            retest = True
            break
    if not retest:
        return None

    last = window.iloc[-1]
    if float(last["Close"]) <= float(last["Open"]):
        return None

    return {"breakout_level": resistance, "side": "bullish"}


def _bearish_breakout_retest(
    window: pd.DataFrame,
    support: float | None,
    *,
    breakout_buffer_pct: float,
    volume: float,
    vol_avg: float | None,
    volume_factor: float,
) -> dict[str, Any] | None:
    if support is None or len(window) < 3:
        return None

    closes = window["Close"].to_numpy(dtype=float)
    highs = window["High"].to_numpy(dtype=float)
    level = support * (1 - breakout_buffer_pct)
    band = support * breakout_buffer_pct

    breakdown_idx = None
    for i in range(len(window) - 2):
        if closes[i] < level:
            breakdown_idx = i
            break
    if breakdown_idx is None:
        return None

    if vol_avg is not None and np.isfinite(volume) and volume < vol_avg * volume_factor:
        return None

    retest = False
    for j in range(breakdown_idx + 1, len(window) - 1):
        if abs(highs[j] - support) <= band * 2 and closes[j] <= support:
            retest = True
            break
    if not retest:
        return None

    last = window.iloc[-1]
    if float(last["Close"]) >= float(last["Open"]):
        return None

    return {"breakdown_level": support, "side": "bearish"}


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
