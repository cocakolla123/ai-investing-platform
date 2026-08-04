"""Historical VaR/CVaR ordering properties and the Kupiec traffic light."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from risk_helpers import make_dates

from dynamo.risk.types import ReturnsPanel
from dynamo.risk.var import (
    VarTrafficLight,
    historical_var,
    kupiec_tighten_factor,
    kupiec_traffic_light,
)


@settings(max_examples=80, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    sessions=st.integers(min_value=80, max_value=600),
    w1=st.floats(min_value=0.0, max_value=0.5),
    w2=st.floats(min_value=0.0, max_value=0.5),
)
def test_var_ordering_properties(
    seed: int, sessions: int, w1: float, w2: float
) -> None:
    rng = np.random.default_rng(seed)
    values = rng.standard_t(4, size=(sessions, 2)) * 0.01
    panel = ReturnsPanel(make_dates(sessions), ("A", "B"), values)
    result = historical_var(panel, {"A": w1, "B": w2})
    assert result is not None
    assert result.var95 >= 0.0
    assert result.var99 >= result.var95
    assert result.cvar95 >= result.var95
    assert result.cvar99 >= result.var99
    assert result.n_obs <= 504


def test_var_fails_closed_on_missing_symbol_or_short_history() -> None:
    panel = ReturnsPanel(
        make_dates(100), ("A",), np.zeros((100, 1), dtype=np.float64)
    )
    assert historical_var(panel, {"MISSING": 0.1}) is None
    short = ReturnsPanel(make_dates(10), ("A",), np.zeros((10, 1), dtype=np.float64))
    assert historical_var(short, {"A": 0.1}) is None
    assert historical_var(panel, {"A": float("nan")}) is None


def test_all_cash_portfolio_has_zero_var() -> None:
    rng = np.random.default_rng(0)
    panel = ReturnsPanel(
        make_dates(200), ("A",), rng.normal(0, 0.01, size=(200, 1))
    )
    result = historical_var(panel, {})
    assert result is not None
    assert result.var95 == 0.0 and result.cvar99 == 0.0


def test_kupiec_zones_and_tighten_factor() -> None:
    # 95% VaR over 250 sessions: ~12.5 exceptions expected.
    assert kupiec_traffic_light(0) is VarTrafficLight.GREEN
    assert kupiec_traffic_light(12) is VarTrafficLight.GREEN
    assert kupiec_traffic_light(22) is VarTrafficLight.YELLOW
    assert kupiec_traffic_light(40) is VarTrafficLight.RED
    assert kupiec_traffic_light(-1) is VarTrafficLight.RED  # invalid fails closed
    assert kupiec_tighten_factor(VarTrafficLight.GREEN) == 1.0
    assert kupiec_tighten_factor(VarTrafficLight.YELLOW) == 1.0
    assert kupiec_tighten_factor(VarTrafficLight.RED) == 0.75


@settings(max_examples=60, deadline=None)
@given(exceptions=st.integers(min_value=0, max_value=250))
def test_kupiec_monotone_in_exceptions(exceptions: int) -> None:
    order = {
        VarTrafficLight.GREEN: 0,
        VarTrafficLight.YELLOW: 1,
        VarTrafficLight.RED: 2,
    }
    current = kupiec_traffic_light(exceptions)
    following = kupiec_traffic_light(exceptions + 1)
    assert order[following] >= order[current]
