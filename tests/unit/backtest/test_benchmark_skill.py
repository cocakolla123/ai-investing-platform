"""Beta-adjusted skill in the benchmark comparison.

awesome-systematic-trading's replication record: the median published
strategy carries a beta of +0.17, and removing it takes the median
information ratio from 0.37 to 0.21 — a large slice of "edge" is index
exposure. Our own committee campaign was the extreme case: +12.80% return,
Sharpe 2.08, and on inspection beta 0.908, alpha -1.0%/yr, t = -0.41,
because 86% of the book was the SPY cash ballast. These tests pin the
numbers that make that visible.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.backtest.metrics import benchmark_comparison

DAYS = pd.bdate_range("2026-01-02", periods=252)


def _curve(returns: np.ndarray, index: pd.DatetimeIndex = DAYS) -> pd.Series:
    return pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)


@pytest.fixture(scope="module")
def market() -> np.ndarray:
    return np.random.default_rng(7).normal(0.0005, 0.01, len(DAYS))


def test_a_pure_beta_portfolio_shows_no_skill(market: np.ndarray) -> None:
    """0.9 x the index plus tiny noise: a high Sharpe, and zero alpha."""
    noise = np.random.default_rng(1).normal(0.0, 0.0005, len(DAYS))
    cmp = benchmark_comparison(_curve(0.9 * market + noise), _curve(market))
    assert cmp.beta == pytest.approx(0.9, abs=0.02)
    assert abs(cmp.alpha_t_stat) < 1.96
    assert cmp.n_aligned == len(DAYS) - 1


def test_real_skill_is_measured_as_an_information_ratio(market: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    daily_alpha = 0.15 / 252  # 15%/yr of skill
    resid = rng.normal(0.0, 0.004, len(DAYS))
    cmp = benchmark_comparison(_curve(1.0 * market + daily_alpha + resid), _curve(market))
    te = cmp.tracking_error_annual
    assert te == pytest.approx(0.004 * math.sqrt(252), rel=0.15)
    assert cmp.information_ratio == pytest.approx(cmp.alpha_annual / te)
    assert cmp.alpha_t_stat == pytest.approx(cmp.information_ratio * math.sqrt(251 / 252))
    assert cmp.years_to_significance == pytest.approx((1.96 / cmp.information_ratio) ** 2)


def test_a_negative_edge_never_becomes_significant(market: np.ndarray) -> None:
    rng = np.random.default_rng(4)
    cmp = benchmark_comparison(
        _curve(market - 0.10 / 252 + rng.normal(0, 0.003, len(DAYS))), _curve(market)
    )
    assert cmp.information_ratio < 0
    assert cmp.years_to_significance is None


def test_returns_are_paired_by_date_not_position(market: np.ndarray) -> None:
    """The benchmark is missing one session. Positional pairing shifted every
    later return by a day; date pairing drops only the missing session."""
    strategy = _curve(1.2 * market)
    full = _curve(market)
    bench = cast(pd.Series, full[full.index != DAYS[100]])
    cmp = benchmark_comparison(strategy, bench)
    assert cmp.beta == pytest.approx(1.2, abs=0.05)
    # Sessions 99->101 span two strategy days but pair with one bench return,
    # so exactly the returns touching the gap are excluded, nothing else.
    assert cmp.n_aligned == len(DAYS) - 2
