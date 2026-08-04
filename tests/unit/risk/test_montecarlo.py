"""Monte Carlo: seeded determinism, probability sanity, df clamping."""

from __future__ import annotations

import itertools

import numpy as np
from risk_helpers import make_panel

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.montecarlo import (
    DF_MAX,
    DF_MIN,
    MonteCarloResult,
    fit_student_df,
    run_monte_carlo,
)

WEIGHTS = {"A": 0.10, "B": 0.15, "C": 0.05}


def _run(seed: int, n_paths: int = 2000) -> MonteCarloResult:
    panel = make_panel(21, ("A", "B", "C"), 520)
    result = run_monte_carlo(
        panel,
        WEIGHTS,
        equity=100_000.0,
        high_water_mark=104_000.0,
        seed=seed,
        n_paths=n_paths,
    )
    assert result is not None
    return result


def test_same_seed_same_result_different_seed_differs() -> None:
    a, b, c = _run(1), _run(1), _run(2)
    assert a.loss_probabilities == b.loss_probabilities
    assert a.expected_shortfall_95 == b.expected_shortfall_95
    assert a.rung_hit_probabilities == b.rung_hit_probabilities
    assert (
        c.expected_shortfall_95 != a.expected_shortfall_95
        or c.loss_probabilities != a.loss_probabilities
    )


def test_probabilities_sane_and_monotone() -> None:
    result = _run(3)
    thresholds = sorted(result.loss_probabilities)
    probs = [result.loss_probabilities[t] for t in thresholds]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(probs))
    # Deeper ladder rungs cannot be more likely than shallower ones.
    order = [
        LadderRung.DD5,
        LadderRung.DD8,
        LadderRung.DD10,
        LadderRung.DD15,
        LadderRung.DD20,
    ]
    hit = [result.rung_hit_probabilities[r] for r in order]
    assert all(0.0 <= p <= 1.0 for p in hit)
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(hit))
    assert result.expected_shortfall_95 >= 0.0  # worst-5% mean is a loss here


def test_df_fit_clamped_to_3_30() -> None:
    rng = np.random.default_rng(5)
    fat = rng.standard_t(2.5, 2000) * 0.01  # fatter than the clamp
    thin = rng.normal(0.0, 0.01, 2000)  # ~ df -> inf
    assert DF_MIN <= fit_student_df(fat) <= DF_MAX
    assert fit_student_df(thin) >= 15.0
    assert fit_student_df(np.array([0.01] * 10)) == 5.0  # tiny sample fallback
    assert DF_MIN <= fit_student_df(np.full(100, np.nan)) <= DF_MAX


def test_df_reported_per_symbol() -> None:
    result = _run(4)
    assert set(result.df_by_symbol) == set(WEIGHTS)
    assert all(DF_MIN <= df <= DF_MAX for df in result.df_by_symbol.values())


def test_fails_closed_on_bad_inputs() -> None:
    panel = make_panel(22, ("A",), 520)
    kwargs = {"equity": 100_000.0, "high_water_mark": 100_000.0, "seed": 0}
    assert run_monte_carlo(panel, {"MISSING": 0.1}, **kwargs) is None
    assert run_monte_carlo(panel, {"A": float("nan")}, **kwargs) is None
    assert run_monte_carlo(panel, {}, **kwargs) is None
    assert (
        run_monte_carlo(
            panel, {"A": 0.1}, equity=float("nan"), high_water_mark=1.0, seed=0
        )
        is None
    )
