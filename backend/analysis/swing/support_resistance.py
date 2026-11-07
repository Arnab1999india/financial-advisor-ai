"""Support/resistance detection and swing signals from daily OHLCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analysis.common.indicators import atr, ema, ensure_ohlcv_columns, rsi
from analysis.common.risk_management import atr_stops_and_target, risk_reward_ratio
from analysis.common.signal_schema import (
    Signal,
    SignalDirection,
    SignalStatus,
    insufficient_data_signal,
)


@dataclass(frozen=True)
class PriceLevel:
    price: float
    kind: str  # "support" | "resistance"
    touches: int
    last_touch_index: int


def detect_confirmed_pivots(
    df: pd.DataFrame,
    *,
    lookback: int = 5,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return confirmed swing highs/lows as (index, price) without look-ahead."""
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    n = len(df)
    if n < 2 * lookback + 1:
        return highs, lows

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    for i in range(lookback, n - lookback):
        window_high = high[i - lookback : i + lookback + 1]
        window_low = low[i - lookback : i + lookback + 1]
        if np.isfinite(high[i]) and high[i] == np.nanmax(window_high):
            highs.append((i, float(high[i])))
        if np.isfinite(low[i]) and low[i] == np.nanmin(window_low):
            lows.append((i, float(low[i])))
    return highs, lows


def cluster_levels(
    prices: list[float],
    *,
    tolerance_pct: float = 0.015,
) -> list[float]:
    """Cluster nearby prices; return representative cluster means."""
    if tolerance_pct <= 0:
        raise ValueError("tolerance_pct must be positive.")
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = [[sorted_prices[0]]]
    for price in sorted_prices[1:]:
        anchor = clusters[-1][0]
        if abs(price - anchor) / anchor <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [float(np.mean(group)) for group in clusters if group]


def build_sr_levels(
    df: pd.DataFrame,
    *,
    pivot_lookback: int = 5,
    tolerance_pct: float = 0.015,
    min_touches: int = 2,
) -> list[PriceLevel]:
    """Identify meaningful support/resistance from pivots and reactions."""
    highs, lows = detect_confirmed_pivots(df, lookback=pivot_lookback)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)

    support_clusters = cluster_levels([p for _, p in lows], tolerance_pct=tolerance_pct)
    resistance_clusters = cluster_levels([p for _, p in highs], tolerance_pct=tolerance_pct)

    levels: list[PriceLevel] = []
    for price in support_clusters:
        touches = _count_touches(close, volume, price, tolerance_pct, kind="support")
        if touches >= min_touches:
            idx = _last_touch_index(close, price, tolerance_pct)
            levels.append(PriceLevel(price, "support", touches, idx))

    for price in resistance_clusters:
        touches = _count_touches(close, volume, price, tolerance_pct, kind="resistance")
        if touches >= min_touches:
            idx = _last_touch_index(close, price, tolerance_pct)
            levels.append(PriceLevel(price, "resistance", touches, idx))

    return sorted(levels, key=lambda item: item.price)


def nearest_support_resistance(
    levels: list[PriceLevel],
    price: float,
) -> tuple[PriceLevel | None, PriceLevel | None]:
    """Nearest support below and resistance above current price."""
    supports = [lvl for lvl in levels if lvl.kind == "support" and lvl.price <= price]
    resistances = [lvl for lvl in levels if lvl.kind == "resistance" and lvl.price >= price]
    support = max(supports, key=lambda lvl: lvl.price) if supports else None
    resistance = min(resistances, key=lambda lvl: lvl.price) if resistances else None
    return support, resistance


def analyze_support_resistance(
    df: pd.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "1D",
    pivot_lookback: int = 5,
    tolerance_pct: float = 0.015,
    min_touches: int = 2,
    proximity_pct: float = 0.01,
    min_rows: int = 60,
) -> Signal:
    """Generate BUY/SELL/HOLD signal near confirmed S/R with confirmation."""
    strategy = "support_resistance"
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
        work,
        pivot_lookback=pivot_lookback,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
    )
    if not levels:
        return Signal(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            direction=SignalDirection.HOLD,
            status=SignalStatus.VALID,
            reason="No confirmed support/resistance levels detected.",
            timestamp=_last_timestamp(work),
            metadata={"levels": []},
        )

    idx = len(work) - 1
    row = work.iloc[idx]
    price = float(row["Close"])
    support, resistance = nearest_support_resistance(levels, price)

    ema20 = ema(work["Close"], 20)
    rsi14 = rsi(work["Close"], 14)
    atr14 = atr(work["High"], work["Low"], work["Close"], 14)
    bullish = float(row["Close"]) > float(row["Open"])
    bearish = float(row["Close"]) < float(row["Open"])

    indicator_values = {
        "close": price,
        "ema_20": _finite_at(ema20, idx),
        "rsi_14": _finite_at(rsi14, idx),
        "atr_14": _finite_at(atr14, idx),
    }

    metadata: dict[str, Any] = {
        "levels": [
            {
                "price": lvl.price,
                "kind": lvl.kind,
                "touches": lvl.touches,
                "last_touch_index": lvl.last_touch_index,
            }
            for lvl in levels
        ],
        "nearest_support": support.price if support else None,
        "nearest_resistance": resistance.price if resistance else None,
    }

    if support and abs(price - support.price) / support.price <= proximity_pct:
        if bullish and _finite_at(rsi14, idx) is not None and _finite_at(rsi14, idx) > 45:
            entry = price
            atr_val = _finite_at(atr14, idx)
            if atr_val is None or atr_val <= 0:
                return Signal(
                    strategy=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=SignalDirection.HOLD,
                    status=SignalStatus.VALID,
                    reason="Support proximity without usable ATR for risk.",
                    indicator_values=indicator_values,
                    timestamp=_last_timestamp(work),
                    metadata=metadata,
                )
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
                reason="Price near confirmed support with bullish candle and RSI support.",
                indicator_values=indicator_values,
                timestamp=_last_timestamp(work),
                metadata=metadata,
            )
            signal.validate()
            return signal

    if resistance and abs(price - resistance.price) / resistance.price <= proximity_pct:
        if bearish and _finite_at(rsi14, idx) is not None and _finite_at(rsi14, idx) < 55:
            entry = price
            atr_val = _finite_at(atr14, idx)
            if atr_val is None or atr_val <= 0:
                return Signal(
                    strategy=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=SignalDirection.HOLD,
                    status=SignalStatus.VALID,
                    reason="Resistance proximity without usable ATR for risk.",
                    indicator_values=indicator_values,
                    timestamp=_last_timestamp(work),
                    metadata=metadata,
                )
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
                reason="Price near confirmed resistance with bearish candle and RSI weakness.",
                indicator_values=indicator_values,
                timestamp=_last_timestamp(work),
                metadata=metadata,
            )
            signal.validate()
            return signal

    return Signal(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        direction=SignalDirection.HOLD,
        status=SignalStatus.VALID,
        reason="No confirmed setup at nearest S/R levels.",
        indicator_values=indicator_values,
        timestamp=_last_timestamp(work),
        metadata=metadata,
    )


def _count_touches(
    close: np.ndarray,
    volume: np.ndarray,
    level: float,
    tolerance_pct: float,
    *,
    kind: str,
) -> int:
    band = level * tolerance_pct
    touches = 0
    for i, price in enumerate(close):
        if not np.isfinite(price):
            continue
        if abs(price - level) <= band:
            if kind == "support" and price >= level - band:
                touches += 1
            elif kind == "resistance" and price <= level + band:
                touches += 1
        elif kind == "support" and price < level and abs(price - level) <= band * 2:
            touches += 1
        elif kind == "resistance" and price > level and abs(price - level) <= band * 2:
            touches += 1
        _ = volume[i]
    return touches


def _last_touch_index(close: np.ndarray, level: float, tolerance_pct: float) -> int:
    band = level * tolerance_pct
    for i in range(len(close) - 1, -1, -1):
        price = close[i]
        if np.isfinite(price) and abs(price - level) <= band:
            return i
    return -1


def _finite_at(series: pd.Series, index: int) -> float | None:
    value = series.iloc[index]
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _last_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    if "Date" not in df.columns or df.empty:
        return None
    ts = df["Date"].iloc[-1]
    return pd.Timestamp(ts) if pd.notna(ts) else None
