import pytest

from analysis.common.risk_management import (
    atr_stops_and_target,
    position_size,
    risk_reward_ratio,
    validate_directional_levels,
)


def test_risk_reward_ratio_buy():
    rr = risk_reward_ratio(100.0, 95.0, 110.0, "BUY")
    assert rr == pytest.approx(2.0)


def test_risk_reward_ratio_sell():
    rr = risk_reward_ratio(100.0, 105.0, 90.0, "SELL")
    assert rr == pytest.approx(2.0)


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        validate_directional_levels(100.0, 101.0, 110.0, "BUY")


def test_atr_stops_and_target_buy():
    stop, target = atr_stops_and_target(100.0, 2.0, "BUY", stop_atr_multiple=1.0, target_atr_multiple=2.0)
    assert stop == 98.0
    assert target == 104.0


def test_position_size_respects_max_notional():
    units = position_size(
        account_equity=100_000,
        risk_percent=1.0,
        entry=100.0,
        stop_loss=95.0,
        max_position_value=5_000,
    )
    assert units * 100.0 <= 5_000 + 1e-9
