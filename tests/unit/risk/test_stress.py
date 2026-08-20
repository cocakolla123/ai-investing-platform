"""Deterministic stress suite: replays, skip notes, synthetic shocks."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
from risk_helpers import make_panel

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.stress import (
    HISTORICAL_WINDOWS,
    run_stress_suite,
    synthetic_scenarios,
)
from dynamo.risk.types import ReturnsPanel


def _covid_panel() -> ReturnsPanel:
    """History that covers the 2020 window with a heavy drawdown."""
    start = date(2020, 1, 2)
    n = 120
    dates = tuple(start + timedelta(days=i) for i in range(n))
    rng = np.random.default_rng(20)
    values = rng.normal(0.0, 0.005, size=(n, 2))
    in_window = [date(2020, 2, 19) <= d <= date(2020, 3, 23) for d in dates]
    values[np.asarray(in_window), 0] = -0.02  # symbol A: crash inside the window
    return ReturnsPanel(dates, ("A", "SPY"), values)


def test_covered_window_replays_with_ladder_implication() -> None:
    results = run_stress_suite(_covid_panel(), {"A": 0.9}, 100_000.0, 100_000.0)
    by_name = {r.scenario: r for r in results}
    covid = by_name["2020_covid_crash"]
    assert "skipped" not in covid.details
    assert covid.equity_impact_pct < -0.15
    assert covid.ladder_implication is not None
    assert covid.ladder_implication != LadderRung.NORMAL
    assert covid.details.get("breaches_defensive_floor") == 1.0


def test_absent_history_notes_skipped_never_silently_drops() -> None:
    panel = make_panel(30, ("A", "SPY"), 300)  # dates end 2024: no 2008 data
    results = run_stress_suite(panel, {"A": 0.5}, 100_000.0, 100_000.0)
    names = [r.scenario for r in results]
    for window in HISTORICAL_WINDOWS:
        assert window.name in names
    by_name = {r.scenario: r for r in results}
    assert by_name["2008_q4"].details.get("skipped") == 1.0
    assert by_name["2008_q4"].equity_impact_pct == 0.0


def test_symbol_missing_from_window_uses_market_proxy() -> None:
    panel = _covid_panel()
    results = run_stress_suite(panel, {"NOT_IN_PANEL": 0.5}, 100_000.0, 100_000.0)
    covid = next(r for r in results if r.scenario == "2020_covid_crash")
    assert covid.details.get("proxied_symbols") == 1.0


def test_synthetic_gap_scales_with_gross_exposure() -> None:
    panel = make_panel(31, ("A", "B"), 300)
    results = synthetic_scenarios(panel, {"A": 0.3, "B": 0.3}, 100_000.0, 100_000.0)
    gap = next(r for r in results if r.scenario == "synthetic_gap_down_10pct_corr1")
    assert math.isclose(gap.equity_impact_pct, -0.10 * 0.6)


def test_synthetic_single_name_uses_largest_position() -> None:
    panel = make_panel(32, ("A", "B"), 300)
    results = synthetic_scenarios(panel, {"A": 0.02, "B": 0.08}, 100_000.0, 100_000.0)
    single = next(r for r in results if r.scenario == "synthetic_single_name_down_40pct")
    assert math.isclose(single.equity_impact_pct, -0.40 * 0.08)
    assert single.details["largest_weight"] == 0.08


def test_rate_shock_uses_tlt_beta_or_skips() -> None:
    with_tlt = make_panel(33, ("A", "TLT"), 400)
    results = synthetic_scenarios(with_tlt, {"A": 0.5}, 100_000.0, 100_000.0)
    rates = next(r for r in results if r.scenario == "synthetic_rates_up_200bps")
    assert "beta_A" in rates.details
    without_tlt = make_panel(33, ("A",), 400)
    results = synthetic_scenarios(without_tlt, {"A": 0.5}, 100_000.0, 100_000.0)
    rates = next(r for r in results if r.scenario == "synthetic_rates_up_200bps")
    assert rates.details.get("skipped") == 1.0


def test_full_suite_is_deterministic() -> None:
    panel = _covid_panel()
    first = run_stress_suite(panel, {"A": 0.4}, 100_000.0, 105_000.0)
    second = run_stress_suite(panel, {"A": 0.4}, 100_000.0, 105_000.0)
    assert first == second
