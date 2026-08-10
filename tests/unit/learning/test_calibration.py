"""Calibration math property tests (doc 10 §5, §9): Brier bounds, perfect and
anti-perfect forecasters, permutation invariance, empty-bucket safety."""

from __future__ import annotations

import random
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dynamo.learning.calibration import (
    CalibrationObservation,
    brier_score,
    compute_calibration,
    conformal_adjust,
    reliability_buckets,
    trade_outcome_achieved,
)

AS_OF = date(2026, 8, 3)


def obs(conviction: float, outcome: bool, scope_id: str = "arbiter") -> CalibrationObservation:
    return CalibrationObservation(
        scope_type="agent", scope_id=scope_id, conviction=conviction, outcome=outcome
    )


observations = st.lists(
    st.builds(
        obs,
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.booleans(),
    ),
    min_size=1,
    max_size=50,
)


@given(observations)
def test_brier_bounded_zero_one(items: list[CalibrationObservation]) -> None:
    score = brier_score(items)
    assert 0.0 <= score <= 1.0


@given(st.lists(st.booleans(), min_size=1, max_size=30))
def test_perfect_forecaster_scores_zero(outcomes: list[bool]) -> None:
    items = [obs(1.0 if o else 0.0, o) for o in outcomes]
    assert brier_score(items) == pytest.approx(0.0)


@given(st.lists(st.booleans(), min_size=1, max_size=30))
def test_anti_perfect_forecaster_scores_one(outcomes: list[bool]) -> None:
    items = [obs(0.0 if o else 1.0, o) for o in outcomes]
    assert brier_score(items) == pytest.approx(1.0)


@given(observations, st.randoms(use_true_random=False))
def test_permutation_invariance(
    items: list[CalibrationObservation], rng: random.Random
) -> None:
    shuffled = list(items)
    rng.shuffle(shuffled)
    assert brier_score(shuffled) == pytest.approx(brier_score(items))
    a, b = reliability_buckets(items), reliability_buckets(shuffled)
    assert set(a) == set(b)
    for key, value in a.items():
        assert b[key] == pytest.approx(value)


def test_empty_buckets_are_omitted_never_divided() -> None:
    items = [obs(0.55, True), obs(0.58, False)]  # everything in one decile
    stats = reliability_buckets(items)
    assert stats["0.5-0.6_n"] == 2
    assert stats["0.5-0.6_observed"] == pytest.approx(0.5)
    assert all(k.startswith("0.5-0.6") for k in stats)


def test_conviction_one_lands_in_top_bucket() -> None:
    stats = reliability_buckets([obs(1.0, True)])
    assert stats["0.9-1.0_n"] == 1


def test_out_of_range_conviction_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        brier_score([obs(1.5, True)])
    with pytest.raises(ValueError, match="outside"):
        reliability_buckets([obs(-0.1, False)])


def test_empty_observations_safe() -> None:
    assert brier_score([]) == 0.0
    assert reliability_buckets([]) == {}
    assert compute_calibration([], AS_OF) == []


def test_compute_calibration_groups_per_scope() -> None:
    items = [
        obs(0.9, True, "bull"),
        obs(0.9, False, "bear"),
        obs(0.1, False, "bull"),
    ]
    points = compute_calibration(items, AS_OF)
    assert [(p.scope_id, p.n) for p in points] == [("bear", 1), ("bull", 2)]
    bull = points[1]
    assert bull.as_of == AS_OF
    assert bull.brier == pytest.approx(((0.9 - 1) ** 2 + (0.1 - 0) ** 2) / 2)
    assert 0.0 <= bull.brier <= 1.0


# --- outcome definition: >= half expected upside before expected downside ---


def test_outcome_achieved_when_half_upside_hit_first() -> None:
    path = [1.0, 3.0, 5.1, -12.0]  # hits 5% (half of 10%) before -8%
    assert trade_outcome_achieved(path, 10.0, 8.0, horizon_days=10)


def test_outcome_failed_when_downside_hit_first() -> None:
    path = [-2.0, -8.5, 6.0, 9.0]  # breaches -8% before reaching +5%
    assert not trade_outcome_achieved(path, 10.0, 8.0, horizon_days=10)


def test_outcome_respects_horizon() -> None:
    path = [0.0] * 10 + [6.0]  # target reached only after the horizon
    assert not trade_outcome_achieved(path, 10.0, 8.0, horizon_days=10)
    assert trade_outcome_achieved(path, 10.0, 8.0, horizon_days=11)


def test_outcome_flat_path_is_failure() -> None:
    assert not trade_outcome_achieved([0.0, 0.1, -0.1], 10.0, 8.0, horizon_days=3)


# --- conformal_adjust: recalibrate estimates against realized outcomes ------


def _history(n: int, *, overconfident: bool = True) -> list[tuple[float, bool]]:
    rng = random.Random(7)
    out: list[tuple[float, bool]] = []
    for _ in range(n):
        p = rng.uniform(0.05, 0.95)
        realized_rate = p * 0.6 if overconfident else p
        out.append((p, rng.random() < realized_rate))
    return out


def test_conformal_adjust_identity_below_min_n() -> None:
    probs = [0.2, 0.5, 0.9]
    assert conformal_adjust(probs, _history(29), min_n=30) == probs


def test_conformal_adjust_identity_when_mapie_unimportable(monkeypatch) -> None:
    import dynamo.learning.calibration as calibration_mod

    class _BrokenImportlib:
        @staticmethod
        def import_module(name: str, package: str | None = None) -> object:
            raise ImportError("mapie unavailable")

    monkeypatch.setattr(calibration_mod, "importlib", _BrokenImportlib)
    probs = [0.2, 0.5, 0.9]
    assert conformal_adjust(probs, _history(60), min_n=30) == probs


def test_conformal_adjust_identity_on_degenerate_history() -> None:
    probs = [0.4, 0.6]
    all_wins = [(0.7, True)] * 40  # single-class history: nothing to learn
    assert conformal_adjust(probs, all_wins, min_n=30) == probs


def test_conformal_adjust_recalibrates_with_enough_history() -> None:
    pytest.importorskip("mapie")
    probs = [0.1, 0.5, 0.9]
    adjusted = conformal_adjust(probs, _history(200), min_n=30)
    assert len(adjusted) == len(probs)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    # calibrated confidence order is preserved, and the overconfident
    # forecaster's high-end estimate gets pulled toward reality
    assert adjusted == sorted(adjusted)
    assert adjusted[-1] < probs[-1]


def test_conformal_adjust_rejects_out_of_range_probs() -> None:
    with pytest.raises(ValueError):
        conformal_adjust([1.2], _history(60))


def test_conformal_adjust_empty_probs() -> None:
    assert conformal_adjust([], _history(60)) == []
