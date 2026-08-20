"""Cost model (doc 08 section 3) and metrics sanity, incl. deflated Sharpe."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from dynamo.backtest.costs import (
    CostParams,
    cost_per_side_fraction,
    half_spread_bps,
    impact_fraction,
)
from dynamo.backtest.metrics import (
    build_report,
    deflated_sharpe_ratio,
    max_drawdown,
    metrics_block,
    trade_stats,
)


class TestHalfSpread:
    @pytest.mark.parametrize(
        ("adv", "bps"),
        [
            (2e8, 2.0),  # deciles 1-2, > $100M
            (5e7, 5.0),  # deciles 3-5, $20-100M
            (1e7, 12.0),  # deciles 6-8, $5-20M
            (1e6, 25.0),  # deciles 9-10, < $5M
            (float("nan"), 25.0),  # unknown fails expensive
        ],
    )
    def test_liquidity_bands(self, adv, bps):
        assert half_spread_bps(adv) == bps


class TestImpactMonotonicity:
    def test_bigger_orders_never_cheaper(self):
        sizes = [1e3, 1e4, 1e5, 1e6, 5e6]
        fracs = [impact_fraction(v, 0.02, 5e7) for v in sizes]
        assert all(b >= a for a, b in pairwise(fracs))
        # total per-side cost is monotone too
        totals = [cost_per_side_fraction(v, 0.02, 5e7) for v in sizes]
        assert all(b >= a for a, b in pairwise(totals))

    def test_sqrt_model_value(self):
        # k * sigma * sqrt(V/ADV) = 1.0 * 0.02 * sqrt(1e6/1e8) = 0.002
        assert impact_fraction(1e6, 0.02, 1e8, k=1.0) == pytest.approx(0.002)

    def test_flat_override_for_parity(self):
        params = CostParams(flat_cost_bps=5.0)
        assert cost_per_side_fraction(1e9, 0.5, 1e3, params) == pytest.approx(5e-4)


class TestMetrics:
    def test_max_drawdown_hand_case(self):
        equity = pd.Series([100.0, 110.0, 99.0, 104.5, 121.0])
        dd, sessions = max_drawdown(equity)
        assert dd == pytest.approx(99.0 / 110.0 - 1.0)
        assert sessions == 2  # two sessions underwater (99, 104.5)

    def test_metrics_block_flat_curve(self):
        block = metrics_block(pd.Series([100.0] * 10))
        assert block.total_return == 0.0
        assert block.sharpe == 0.0
        assert block.max_drawdown == 0.0

    def test_trade_stats(self):
        rows = [
            ("2026-01-05", "AAA", 100.0),
            ("2026-01-06", "BBB", -50.0),
            ("2026-01-07", "CCC", 25.0),
        ]
        stats = trade_stats(rows)
        assert stats.n_closed == 3
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.profit_factor == pytest.approx(125.0 / 50.0)
        assert stats.worst_trades[0]["symbol"] == "BBB"


class TestReport:
    def test_build_report_json_serializable_with_regimes_and_benchmark(self):
        rng = np.random.default_rng(1)
        n = 100
        equity = pd.Series(100_000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n)))
        spy = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.009, n)))
        regimes = ["risk_on"] * 60 + ["risk_off"] * 40
        report = build_report(
            run_id="run-1",
            strategy_ref="toy",
            equity=equity,
            session_dates=[str(i) for i in range(n)],
            regime_by_session=regimes,
            gross_by_session=[0.9] * n,
            cash_by_session=[0.1] * n,
            one_way_turnover_total=0.8,
            realized_pnls=[("2026-01-05", "AAA", 120.0), ("2026-01-08", "BBB", -60.0)],
            trials=7,
            benchmark_closes=spy,
            constraint_violations=0,
            look_ahead_audit_passed=True,
        )
        payload = report.model_dump_json()
        assert '"engine":"event_driven"' in payload
        assert set(report.per_regime) == {"risk_on", "risk_off"}
        assert report.trial_number == 7
        assert report.benchmark is not None and report.benchmark.benchmark == "SPY"
        assert report.behavior.one_way_turnover_total == 0.8


class TestDeflatedSharpe:
    def test_more_trials_lower_dsr(self):
        n_obs = 504
        sr = 1.2
        values = [deflated_sharpe_ratio(sr, n_obs, trials) for trials in (1, 5, 25, 100, 400)]
        assert all(b < a for a, b in pairwise(values))

    def test_bounds_and_sanity(self):
        dsr = deflated_sharpe_ratio(2.0, 504, 10)
        assert 0.0 <= dsr <= 1.0
        # A strongly positive Sharpe with few trials should look probable...
        assert deflated_sharpe_ratio(2.5, 1260, 2) > 0.9
        # ...and a flat strategy should not.
        assert deflated_sharpe_ratio(0.0, 1260, 50) < 0.5

    def test_seeded_random_returns_dsr_near_zero_edge(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 0.01, 756)
        equity = pd.Series(100_000.0 * np.cumprod(1.0 + r))
        block = metrics_block(equity)
        dsr_few = deflated_sharpe_ratio(block.sharpe, 756, 2)
        dsr_many = deflated_sharpe_ratio(block.sharpe, 756, 200)
        assert dsr_many < dsr_few
        assert math.isfinite(dsr_many)
