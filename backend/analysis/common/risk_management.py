"""Deterministic risk and position-sizing helpers (no signal generation)."""

from __future__ import annotations

import math
from typing import Literal

Direction = Literal["BUY", "SELL"]


def risk_reward_ratio(
    entry: float,
    stop_loss: float,
    target: float,
    direction: Direction,
) -> float:
    """Compute reward-to-risk ratio for a directional trade."""
    _validate_positive_finite(entry, "entry")
    _validate_positive_finite(stop_loss, "stop_loss")
    _validate_positive_finite(target, "target")
    validate_directional_levels(entry, stop_loss, target, direction)

    if direction == "BUY":
        risk = entry - stop_loss
        reward = target - entry
    else:
        risk = stop_loss - entry
        reward = entry - target

    if risk <= 0:
        raise ValueError("Risk distance must be positive.")
    return reward / risk


def atr_stops_and_target(
    entry: float,
    atr_value: float,
    direction: Direction,
    *,
    stop_atr_multiple: float = 1.5,
    target_atr_multiple: float = 3.0,
) -> tuple[float, float]:
    """Derive stop-loss and target from ATR multiples."""
    _validate_positive_finite(entry, "entry")
    _validate_positive_finite(atr_value, "atr_value")
    if stop_atr_multiple <= 0 or target_atr_multiple <= 0:
        raise ValueError("ATR multiples must be positive.")

    if direction == "BUY":
        stop_loss = entry - stop_atr_multiple * atr_value
        target = entry + target_atr_multiple * atr_value
    else:
        stop_loss = entry + stop_atr_multiple * atr_value
        target = entry - target_atr_multiple * atr_value

    validate_directional_levels(entry, stop_loss, target, direction)
    return stop_loss, target


def position_size(
    account_equity: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    *,
    max_position_value: float | None = None,
) -> float:
    """Position size in units based on fixed fractional risk."""
    _validate_positive_finite(account_equity, "account_equity")
    _validate_positive_finite(entry, "entry")
    _validate_positive_finite(stop_loss, "stop_loss")
    if not (0 < risk_percent <= 100):
        raise ValueError("risk_percent must be in (0, 100].")

    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit <= 0:
        raise ValueError("entry and stop_loss must differ.")

    risk_capital = account_equity * (risk_percent / 100.0)
    units = risk_capital / risk_per_unit

    position_value = units * entry
    if max_position_value is not None:
        if max_position_value <= 0:
            raise ValueError("max_position_value must be positive when provided.")
        if position_value > max_position_value:
            units = max_position_value / entry

    return max(units, 0.0)


def validate_directional_levels(
    entry: float,
    stop_loss: float,
    target: float,
    direction: Direction,
) -> None:
    """Validate BUY/SELL geometry for entry, stop, and target."""
    _validate_positive_finite(entry, "entry")
    _validate_positive_finite(stop_loss, "stop_loss")
    _validate_positive_finite(target, "target")

    if direction == "BUY":
        if not (stop_loss < entry < target):
            raise ValueError("BUY requires stop_loss < entry < target.")
    elif direction == "SELL":
        if not (target < entry < stop_loss):
            raise ValueError("SELL requires target < entry < stop_loss.")
    else:
        raise ValueError("direction must be BUY or SELL.")


def _validate_positive_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
