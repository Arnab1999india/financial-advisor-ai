import pytest

from analysis.common.signal_schema import (
    Signal,
    SignalDirection,
    SignalStatus,
    insufficient_data_signal,
)


def test_valid_buy_signal_roundtrip():
    signal = Signal(
        strategy="demo",
        symbol="TCS",
        timeframe="1D",
        direction=SignalDirection.BUY,
        status=SignalStatus.VALID,
        entry=100.0,
        stop_loss=95.0,
        target=110.0,
        risk_reward_ratio=2.0,
        reason="test",
    )
    signal.validate()
    restored = Signal.from_dict(signal.to_dict())
    assert restored.direction == SignalDirection.BUY
    assert restored.entry == 100.0


def test_insufficient_data_signal():
    signal = insufficient_data_signal(
        strategy="demo",
        symbol="TCS",
        timeframe="1D",
        reason="too short",
    )
    signal.validate()
    assert signal.status == SignalStatus.INSUFFICIENT_DATA


def test_invalid_buy_geometry_rejected():
    signal = Signal(
        strategy="demo",
        symbol="TCS",
        timeframe="1D",
        direction=SignalDirection.BUY,
        status=SignalStatus.VALID,
        entry=100.0,
        stop_loss=101.0,
        target=110.0,
    )
    with pytest.raises(ValueError):
        signal.validate()
