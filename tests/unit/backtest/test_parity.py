"""Parity test (doc 08 section 9): event-driven vs vectorized MA-cross with
matched flat costs and no risk constraints -> terminal equity within 0.5%."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynamo.backtest.parity import (
    ma_cross_positions,
    run_parity_check,
    vectorized_ma_cross_equity,
)
from tests.unit.backtest.builders import sessions


def _seeded_series(n: int = 150, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]  # open at previous close (no overnight gap)
    idx = pd.Index(sessions(n))
    return pd.Series(opens, index=idx), pd.Series(closes, index=idx)


def test_ma_cross_positions_basic():
    closes = pd.Series([1.0] * 20 + [2.0] * 10)
    pos = ma_cross_positions(closes, fast=5, slow=20)
    assert pos.iloc[:20].sum() == 0.0  # warmup + flat trend
    assert pos.iloc[-1] == 1.0  # fast pulled above slow


def test_event_driven_matches_vectorized_within_half_percent():
    opens, closes = _seeded_series()
    parity = run_parity_check(opens, closes, symbol="SPY", fast=5, slow=20, cost_bps=5.0)
    assert parity.event_result.fills  # the strategy actually traded
    assert parity.relative_difference <= parity.tolerance, (
        f"event {parity.terminal_event_driven:.2f} vs "
        f"vectorized {parity.terminal_vectorized:.2f} "
        f"({parity.relative_difference:.4%})"
    )
    assert parity.passed


def test_vectorized_costs_reduce_equity():
    opens, closes = _seeded_series(seed=7)
    free = vectorized_ma_cross_equity(opens, closes, cost_bps=0.0)
    costly = vectorized_ma_cross_equity(opens, closes, cost_bps=25.0)
    assert float(costly.iloc[-1]) < float(free.iloc[-1])
