"""Standardized trading signal schema for analysis strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

import math


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"


class SignalStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class Signal:
    """Canonical signal envelope returned by swing/intraday strategies."""

    strategy: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    status: SignalStatus
    entry: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    risk_reward_ratio: float | None = None
    reason: str = ""
    indicator_values: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate enum fields and directional price levels."""
        if not self.strategy.strip():
            raise ValueError("strategy is required.")
        if not self.symbol.strip():
            raise ValueError("symbol is required.")
        if not self.timeframe.strip():
            raise ValueError("timeframe is required.")

        if self.status == SignalStatus.INSUFFICIENT_DATA:
            if self.direction not in {SignalDirection.HOLD, SignalDirection.NO_SIGNAL}:
                raise ValueError(
                    "INSUFFICIENT_DATA signals must use HOLD or NO_SIGNAL direction."
                )
            return

        if self.status == SignalStatus.INVALID:
            return

        if self.status != SignalStatus.VALID:
            raise ValueError(f"Unsupported status: {self.status}")

        if self.direction in {SignalDirection.HOLD, SignalDirection.NO_SIGNAL}:
            return

        _validate_directional_levels(
            self.direction,
            self.entry,
            self.stop_loss,
            self.target,
        )

        if self.risk_reward_ratio is not None and self.risk_reward_ratio < 0:
            raise ValueError("risk_reward_ratio cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        payload = asdict(self)
        payload["direction"] = self.direction.value
        payload["status"] = self.status.value
        payload["timestamp"] = (
            self.timestamp.isoformat() if self.timestamp is not None else None
        )
        payload["entry"] = _serialize_optional_float(self.entry)
        payload["stop_loss"] = _serialize_optional_float(self.stop_loss)
        payload["target"] = _serialize_optional_float(self.target)
        payload["risk_reward_ratio"] = _serialize_optional_float(self.risk_reward_ratio)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Signal:
        """Build a Signal from a mapping (e.g. JSON payload)."""
        ts = data.get("timestamp")
        parsed_ts: datetime | None
        if ts is None or ts == "":
            parsed_ts = None
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = datetime.fromisoformat(str(ts))

        signal = cls(
            strategy=str(data.get("strategy", "")),
            symbol=str(data.get("symbol", "")),
            timeframe=str(data.get("timeframe", "")),
            direction=SignalDirection(str(data.get("direction", SignalDirection.NO_SIGNAL))),
            status=SignalStatus(str(data.get("status", SignalStatus.INVALID))),
            entry=_parse_optional_float(data.get("entry")),
            stop_loss=_parse_optional_float(data.get("stop_loss")),
            target=_parse_optional_float(data.get("target")),
            risk_reward_ratio=_parse_optional_float(data.get("risk_reward_ratio")),
            reason=str(data.get("reason", "")),
            indicator_values=dict(data.get("indicator_values") or {}),
            timestamp=parsed_ts,
            metadata=dict(data.get("metadata") or {}),
        )
        signal.validate()
        return signal


def insufficient_data_signal(
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    reason: str,
    indicator_values: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Signal:
    """Factory for graceful insufficient-history responses."""
    return Signal(
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        direction=SignalDirection.NO_SIGNAL,
        status=SignalStatus.INSUFFICIENT_DATA,
        reason=reason,
        indicator_values=indicator_values or {},
        metadata=metadata or {},
    )


def _validate_directional_levels(
    direction: SignalDirection,
    entry: float | None,
    stop_loss: float | None,
    target: float | None,
) -> None:
    if entry is None or stop_loss is None or target is None:
        raise ValueError("VALID directional signals require entry, stop_loss, and target.")

    for label, value in (("entry", entry), ("stop_loss", stop_loss), ("target", target)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a positive finite price.")

    if direction == SignalDirection.BUY:
        if not (stop_loss < entry < target):
            raise ValueError("BUY requires stop_loss < entry < target.")
    elif direction == SignalDirection.SELL:
        if not (target < entry < stop_loss):
            raise ValueError("SELL requires target < entry < stop_loss.")


def _serialize_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed
