"""Live market-data lookup backed by yfinance.

The module is deliberately independent from the LLM: a user asking for a
quote receives the provider result, timestamp, and currency rather than an
LLM-generated (and potentially stale) number.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from cache.market_cache import get_quote, put_quote
from config import settings
from resilience.external import ProviderUnavailableError, call_external
from security.tool_policy import assert_read_only_tool

try:  # Keep the API start-up error explicit if an environment missed yfinance.
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised in deployments without the extra
    yf = None  # type: ignore[assignment]


_SPECIAL_ASSETS = {
    "gold": ("GC=F", "Gold futures"),
    "silver": ("SI=F", "Silver futures"),
    "bitcoin": ("BTC-USD", "Bitcoin"),
    "btc": ("BTC-USD", "Bitcoin"),
    "ethereum": ("ETH-USD", "Ethereum"),
    "eth": ("ETH-USD", "Ethereum"),
}
# Equity venues resolved by _symbol_from_query (yfinance ticker suffixes).
SUPPORTED_EQUITY_EXCHANGES: tuple[dict[str, str], ...] = (
    {
        "id": "NSE",
        "label": "NSE — National Stock Exchange of India",
        "yfinance_suffix": ".NS",
    },
    {
        "id": "BSE",
        "label": "BSE — Bombay Stock Exchange",
        "yfinance_suffix": ".BO",
    },
)

_NOISE_WORDS = {
    "what", "is", "the", "price", "latest", "current", "live", "quote",
    "of", "for", "on", "in", "at", "today", "share", "stock", "stocks",
    "market", "rate", "value", "please", "tell", "me", "much", "how",
    "trading", "traded", "nse", "bse", "india", "indian", "crypto", "coin",
}


def is_market_data_query(query: str) -> bool:
    """Return true only for an explicit quote/price request."""
    lowered = (query or "").lower()
    price_words = ("price", "quote", "live rate", "current rate", "latest rate", "trading at")
    assets = ("gold", "silver", "bitcoin", "btc", "ethereum", "eth", "crypto")
    return any(word in lowered for word in price_words) and (
        bool(re.search(r"\b(?:nse|bse|nifty|sensex|gold|silver|bitcoin|btc|ethereum|eth|crypto)\b", lowered))
        or bool(re.search(r"\b(?:price|quote)\s+(?:of|for)\s+[a-z0-9.^=-]+", lowered))
    ) or ("how much is" in lowered and any(asset in lowered for asset in assets))


def _symbol_from_query(query: str) -> tuple[str | None, str]:
    """Resolve common asset names and NSE/BSE equity symbols from a user query."""
    lowered = (query or "").lower()
    for label, resolved in _SPECIAL_ASSETS.items():
        if re.search(rf"\b{re.escape(label)}\b", lowered):
            return resolved

    exchange = ".BO" if re.search(r"\bbse\b", lowered) else ".NS"
    explicit = re.search(r"\b([a-z0-9&^.-]+\.(?:ns|bo))\b", lowered, re.IGNORECASE)
    if explicit:
        symbol = explicit.group(1).upper()
        return symbol, symbol

    tokens = re.findall(r"[a-z0-9&^.-]+", lowered)
    candidates = [t for t in tokens if t not in _NOISE_WORDS and len(t) >= 2]
    if not candidates:
        return None, ""
    raw = candidates[-1].upper()
    return f"{raw}{exchange}", raw


def get_market_quote(query: str) -> dict[str, Any]:
    """Fetch the most recent available quote for NSE/BSE, metals, or crypto.

    yfinance may return delayed data depending on the exchange and instrument;
    the response therefore always carries the provider timestamp and is never
    described as guaranteed real-time data.
    """
    assert_read_only_tool("get_market_quote")
    if yf is None:
        return {"ok": False, "error": "Live market data is unavailable because yfinance is not installed."}

    symbol, display_name = _symbol_from_query(query)
    if not symbol:
        return {"ok": False, "error": "Please include a ticker or asset name, for example TCS, gold, BTC, or TCS.BO."}

    cache_key = symbol.strip().upper()
    cached = get_quote(cache_key)
    if cached is not None:
        return cached

    try:
        def _fetch():
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="1d", interval="1m", auto_adjust=False)
            if history.empty:
                history = ticker.history(period="5d", interval="1d", auto_adjust=False)
            if history.empty:
                raise ValueError(f"No quote was returned for {symbol}.")
            return ticker, history

        ticker, history = call_external(
            "market_data",
            _fetch,
            timeout_seconds=settings.market_data_timeout_seconds,
        )
        latest = history.iloc[-1]
        timestamp = history.index[-1]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        as_of = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        price = float(latest["Close"])
        previous_close = float(history.iloc[-2]["Close"]) if len(history.index) > 1 else None
        change_pct = ((price - previous_close) / previous_close * 100) if previous_close else None
        info = getattr(ticker, "fast_info", {}) or {}
        currency = getattr(info, "currency", None) or info.get("currency") or "USD"
        name = getattr(info, "long_name", None) or info.get("long_name") or display_name or symbol
        as_of_dt = timestamp if isinstance(timestamp, datetime) else None
        stale = _is_stale(as_of_dt, settings.market_data_freshness_minutes)
        payload = {
            "ok": True,
            "symbol": symbol,
            "name": name,
            "price": price,
            "currency": currency,
            "as_of": as_of,
            "change_percent": change_pct,
            "provider": "Yahoo Finance via yfinance",
            "is_stale": stale,
        }
        put_quote(cache_key, payload)
        return payload
    except ProviderUnavailableError:
        return {"ok": False, "symbol": symbol, "error": "Market data service is busy. Document-based answers may still be available."}
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": f"Unable to fetch live data: {exc}"}


def _is_stale(as_of: datetime | None, freshness_minutes: int) -> bool:
    if as_of is None:
        return True
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    age_minutes = (datetime.now(timezone.utc) - as_of).total_seconds() / 60.0
    return age_minutes > max(freshness_minutes, 1)
