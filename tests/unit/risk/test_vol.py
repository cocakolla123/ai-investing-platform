"""EWMA vol, portfolio realized vol, and ATR."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from risk_helpers import make_dates, make_panel

from dynamo.risk.types import ANNUALIZATION, SymbolBars
from dynamo.risk.vol import (
    atr,
    ewma_sigma_annual,
    ewma_sigma_daily,
    ewma_vol_by_symbol,
    portfolio_realized_vol_annual,
)


def test_constant_returns_have_zero_sigma() -> None:
    r = np.full(100, 0.001, dtype=np.float64)
    assert ewma_sigma_daily(r, 60) == 0.0


def test_sigma_scales_linearly() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, 300)
    assert math.isclose(ewma_sigma_daily(3.0 * r, 60), 3.0 * ewma_sigma_daily(r, 60), rel_tol=1e-9)


def test_annualization_factor() -> None:
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.01, 300)
    assert math.isclose(ewma_sigma_annual(r, 60), ewma_sigma_daily(r, 60) * ANNUALIZATION)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=9999),
    nan_count=st.integers(min_value=0, max_value=50),
)
def test_sigma_robust_to_nan_and_inf(seed: int, nan_count: int) -> None:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, 200)
    idx = rng.choice(200, size=nan_count, replace=False)
    r[idx[: nan_count // 2]] = np.nan
    r[idx[nan_count // 2 :]] = np.inf
    sigma = ewma_sigma_daily(r, 60)
    assert math.isfinite(sigma) and sigma >= 0.0


def test_unusable_series_gives_nan() -> None:
    assert math.isnan(ewma_sigma_daily(np.array([np.nan, np.inf]), 60))
    assert math.isnan(ewma_sigma_daily(np.array([0.01]), 60))
    assert math.isnan(ewma_sigma_daily(np.array([0.01, 0.02]), 1))


def test_vol_by_symbol_covers_all_symbols() -> None:
    panel = make_panel(1, ("A", "B", "C"), 300)
    vols = ewma_vol_by_symbol(panel, 60)
    assert set(vols) == {"A", "B", "C"}
    assert all(v > 0.0 for v in vols.values())


def test_portfolio_realized_vol_of_cash_is_nan_free_zero_weighting() -> None:
    panel = make_panel(2, ("A",), 100)
    vol = portfolio_realized_vol_annual(panel, {})
    assert vol == 0.0  # zero-weight portfolio has zero vol


def test_portfolio_realized_vol_missing_symbol_is_nan() -> None:
    panel = make_panel(3, ("A",), 100)
    assert math.isnan(portfolio_realized_vol_annual(panel, {"OTHER": 0.5}))


def test_atr_of_constant_range_bars() -> None:
    n = 40
    bars = SymbolBars(
        make_dates(n),
        np.full(n, 102.0),
        np.full(n, 98.0),
        np.full(n, 100.0),
    )
    assert math.isclose(atr(bars, 14), 4.0)


def test_atr_handles_short_or_bad_series() -> None:
    bars = SymbolBars(make_dates(1), np.array([1.0]), np.array([0.5]), np.array([0.8]))
    assert math.isnan(atr(bars))
    n = 30
    bad = SymbolBars(
        make_dates(n),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )
    assert math.isnan(atr(bad))
