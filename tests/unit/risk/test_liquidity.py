"""Liquidity: entry caps, days-to-liquidate, concentration flag."""

from __future__ import annotations

import math

from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.liquidity import (
    days_to_liquidate,
    max_entry_weight,
    portfolio_liquidity,
)

CFG = RiskConfig()


NO_FLOOR = RiskConfig(min_adv_dollar=0.0)


def test_max_entry_weight_formula() -> None:
    # 3 days x 10% x $1M ADV / $100k equity = 3.0 (uncapped here; sizing clamps)
    assert math.isclose(max_entry_weight(1_000_000.0, 100_000.0, CFG), 3.0)
    assert math.isclose(max_entry_weight(10_000.0, 100_000.0, NO_FLOOR), 0.03)


def test_absolute_adv_floor_binds_where_the_relative_cap_cannot() -> None:
    """The relative cap scales with equity, so a small account could always
    satisfy it: at $1,000 of equity a name trading $267/day cleared a full
    8% position. Microstructure does not scale with the account."""
    thin = 267.0
    assert max_entry_weight(thin, 1_000.0, NO_FLOOR) >= CFG.per_position_cap
    assert max_entry_weight(thin, 1_000.0, CFG) == 0.0
    # The floor is absolute: a large account is refused the same name.
    assert max_entry_weight(thin, 10_000_000.0, CFG) == 0.0
    # It binds at the boundary, not above it.
    assert max_entry_weight(CFG.min_adv_dollar - 1.0, 100_000.0, CFG) == 0.0
    assert max_entry_weight(CFG.min_adv_dollar, 100_000.0, CFG) > 0.0


def test_max_entry_weight_fails_closed() -> None:
    assert max_entry_weight(float("nan"), 100_000.0, CFG) == 0.0
    assert max_entry_weight(0.0, 100_000.0, CFG) == 0.0
    assert max_entry_weight(1e6, 0.0, CFG) == 0.0
    assert max_entry_weight(1e6, float("inf"), CFG) == 0.0


def test_days_to_liquidate() -> None:
    assert math.isclose(days_to_liquidate(20_000.0, 10_000.0), 10.0)  # 20% ADV
    assert days_to_liquidate(0.0, 10_000.0) == 0.0
    assert days_to_liquidate(1.0, 0.0) == float("inf")
    assert days_to_liquidate(float("nan"), 10_000.0) == float("inf")


def test_concentration_flag_when_illiquid_positions_exceed_10pct() -> None:
    report = portfolio_liquidity(
        weights={"LIQ": 0.20, "ILLIQ": 0.15},
        equity=1_000_000.0,
        adv_dollar_20d={"LIQ": 1e9, "ILLIQ": 100_000.0},
    )
    # ILLIQ: $150k at 20% of $100k ADV = 7.5 sessions > 5 -> 15% of equity slow.
    assert report.days_to_liquidate["ILLIQ"] > 5.0
    assert report.concentration_flag
    assert math.isclose(report.illiquid_equity_fraction, 0.15)
    assert report.portfolio_days_to_liquidate >= 7.5


def test_no_flag_for_liquid_portfolio() -> None:
    report = portfolio_liquidity(
        weights={"A": 0.5, "B": 0.4},
        equity=100_000.0,
        adv_dollar_20d={"A": 1e8, "B": 1e8},
    )
    assert not report.concentration_flag
    assert report.illiquid_equity_fraction == 0.0
