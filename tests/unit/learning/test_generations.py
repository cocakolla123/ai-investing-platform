"""Generational trainer tests (plan Phase 5): fold hygiene (no leakage,
final holdout excluded everywhere), the MIN-gate skip path, the frozen
control's identity calibration, the value-of-learning sign convention, and
the per-decision EV policy. Pure — no database, no network, no LLM."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.learning.generations import (
    ev_per_decision,
    final_holdout_period,
    frozen_control_probabilities,
    generation_folds,
    run_generation,
    usable_experiences,
)
from dynamo.learning.meta_model import MIN_TRAIN_EXPERIENCES, brier


def _exp(i: int, as_of: date, *, conviction: float, label: str, pnl: float) -> DecisionExperience:
    return DecisionExperience.model_validate(
        {
            "id": f"exp:test:{as_of.isoformat()}:{i}",
            "created_at": datetime(as_of.year, as_of.month, as_of.day, 12, tzinfo=UTC),
            "symbol": f"SYM{i % 7}",
            "as_of": as_of,
            "sleeve": "long_term",
            "regime": "neutral",
            "sector": "tech",
            "action": "buy",
            "conviction": conviction,
            "disagreement_score": 0.1,
            "quant_alignment": "agree",
            "expected_upside_pct": 8.0,
            "expected_downside_pct": 4.0,
            "horizon_days": 20,
            "resolved": True,
            "pnl_pct": pnl,
            "label": label,
        }
    )


def _month(year: int, month: int, n: int, *, informative: bool = True) -> list[DecisionExperience]:
    """n resolved experiences spread over one month. ``informative``:
    conviction predicts the outcome (high => win), so a trained model can
    beat noise; the split is deterministic."""
    out: list[DecisionExperience] = []
    for i in range(n):
        win = i % 2 == 0
        conviction = (0.85 if win else 0.15) if informative else 0.5
        out.append(
            _exp(
                i,
                date(year, month, 1 + (i % 27)),
                conviction=conviction,
                label="win" if win else "loss",
                pnl=5.0 if win else -5.0,
            )
        )
    return out


# --------------------------------------------------------------------------
# folds: structure, holdout exclusion, leakage guard
# --------------------------------------------------------------------------


def test_folds_expand_monthly_and_exclude_final_holdout() -> None:
    experiences = (
        _month(2026, 1, 10) + _month(2026, 2, 10) + _month(2026, 3, 10) + _month(2026, 4, 10)
    )
    folds = generation_folds(experiences, final_holdout_months=1)
    # Months Jan..Apr, holdout=Apr -> usable Jan..Mar -> gen1 (Jan->Feb),
    # gen2 (Jan+Feb->Mar).
    assert [f.generation for f in folds] == [1, 2]
    assert folds[0].train_start == date(2026, 1, 1)
    assert folds[0].test_start == date(2026, 2, 1)
    assert folds[1].train_end == date(2026, 2, 28)
    assert folds[1].test_start == date(2026, 3, 1)
    assert len(folds[1].train) == 20 and len(folds[1].test) == 10
    # THE holdout guarantee: April rows appear in NO fold, either side.
    for fold in folds:
        for e in (*fold.train, *fold.test):
            assert e.as_of < date(2026, 4, 1)


def test_no_leak_train_strictly_before_test_in_every_fold() -> None:
    experiences = _month(2026, 1, 8) + _month(2026, 2, 8) + _month(2026, 3, 8) + _month(2026, 4, 8)
    for fold in generation_folds(experiences, final_holdout_months=1):
        max_train = max((e.as_of, e.created_at) for e in fold.train)
        min_test = min((e.as_of, e.created_at) for e in fold.test)
        assert max_train < min_test


def test_zero_holdout_and_thin_history_edges() -> None:
    two_months = _month(2026, 1, 5) + _month(2026, 2, 5)
    # holdout=1 leaves one usable month -> no fold has a test month.
    assert generation_folds(two_months, final_holdout_months=1) == []
    # holdout=0 uses both months -> one generation.
    folds = generation_folds(two_months, final_holdout_months=0)
    assert [f.generation for f in folds] == [1]
    with pytest.raises(ValueError):
        generation_folds(two_months, final_holdout_months=-1)


def test_unusable_rows_are_filtered_from_folds() -> None:
    good = _month(2026, 1, 6) + _month(2026, 2, 6) + _month(2026, 3, 6)
    unresolved = _exp(99, date(2026, 1, 15), conviction=0.9, label="win", pnl=1.0)
    unresolved = unresolved.model_copy(update={"resolved": False})
    unlabeled = _exp(98, date(2026, 2, 15), conviction=0.9, label="win", pnl=1.0)
    unlabeled = unlabeled.model_copy(update={"label": None})
    no_conviction = _exp(97, date(2026, 2, 16), conviction=0.9, label="win", pnl=1.0)
    no_conviction = no_conviction.model_copy(update={"conviction": None})
    rows = usable_experiences([*good, unresolved, unlabeled, no_conviction])
    assert len(rows) == len(good)
    folds = generation_folds([*good, unresolved, unlabeled, no_conviction], final_holdout_months=1)
    ids = {e.id for f in folds for e in (*f.train, *f.test)}
    assert unresolved.id not in ids and unlabeled.id not in ids and no_conviction.id not in ids


def test_final_holdout_period_is_the_claimable_calendar_window() -> None:
    experiences = _month(2026, 1, 5) + _month(2026, 2, 5) + _month(2026, 4, 5)
    assert final_holdout_period(experiences, final_holdout_months=1) == (
        date(2026, 4, 1),
        date(2026, 4, 30),
    )
    assert final_holdout_period(experiences, final_holdout_months=0) is None
    # Everything would be holdout -> nothing left to train: no claimable period.
    assert final_holdout_period(_month(2026, 1, 5), final_holdout_months=1) is None


# --------------------------------------------------------------------------
# per-generation evaluation
# --------------------------------------------------------------------------


def test_below_min_generation_is_skipped_not_trained() -> None:
    experiences = _month(2026, 1, 40) + _month(2026, 2, 20) + _month(2026, 3, 20)
    folds = generation_folds(experiences, final_holdout_months=1)
    outcome = run_generation(folds[0], algo="logistic")
    assert outcome.skipped is True
    assert outcome.model is None
    assert str(MIN_TRAIN_EXPERIENCES) in outcome.skip_reason
    assert outcome.brier_meta is None and outcome.value_of_learning is None
    assert outcome.n_train == 40 and outcome.n_test == 20


def test_frozen_control_is_identity_on_raw_convictions() -> None:
    rows = [
        _exp(0, date(2026, 1, 5), conviction=0.9, label="win", pnl=3.0),
        _exp(1, date(2026, 1, 6), conviction=0.25, label="loss", pnl=-2.0),
        _exp(2, date(2026, 1, 7), conviction=1.0, label="win", pnl=1.0),
    ]
    # Identity calibration: control probabilities == raw convictions, exactly.
    assert frozen_control_probabilities(rows) == [0.9, 0.25, 1.0]


def test_trained_generation_reports_meta_vs_control_and_sign_convention() -> None:
    n = MIN_TRAIN_EXPERIENCES + 30
    experiences = _month(2026, 1, n) + _month(2026, 2, 40) + _month(2026, 3, 10)
    folds = generation_folds(experiences, final_holdout_months=1)
    assert folds[0].generation == 1 and len(folds[0].train) == n
    outcome = run_generation(folds[0], algo="logistic")
    assert outcome.skipped is False and outcome.model is not None
    assert outcome.model_version == "meta-gen001-logistic"
    # Control values are exactly the identity-calibrated raw convictions.
    test_outcomes = [e.label == "win" for e in folds[0].test]
    control = frozen_control_probabilities(folds[0].test)
    assert outcome.brier_control == pytest.approx(brier(control, test_outcomes))
    # SIGN CONVENTION: value_of_learning = brier_control - brier_meta.
    assert outcome.brier_meta is not None and outcome.brier_control is not None
    assert outcome.value_of_learning == pytest.approx(outcome.brier_control - outcome.brier_meta)
    # Deterministic: the same fold re-runs to the identical numbers.
    again = run_generation(folds[0], algo="logistic")
    assert again.brier_meta == outcome.brier_meta
    assert again.value_of_learning == outcome.value_of_learning
    # EV per decision exists for both scorers on the identical decisions.
    assert outcome.ev_meta is not None and outcome.ev_control is not None


def test_value_of_learning_positive_iff_meta_beats_control() -> None:
    outcomes = [True, False, True, False]
    perfect = [1.0, 0.0, 1.0, 0.0]
    bad = [0.4, 0.6, 0.4, 0.6]
    # brier is a LOSS: perfect scorer 0.0, bad scorer higher. Positive
    # (control - meta) therefore means the meta side is better.
    assert brier(bad, outcomes) - brier(perfect, outcomes) > 0.0


def test_ev_per_decision_take_predicted_winners_policy() -> None:
    probs = [0.9, 0.4, 0.6, 0.1]
    pnls = [5.0, -10.0, -2.0, 100.0]
    # Takes decisions 0 and 2 (p >= 0.5): (5.0 + -2.0) / 4 decisions.
    assert ev_per_decision(probs, pnls) == pytest.approx(0.75)
    assert ev_per_decision([], []) is None
    assert ev_per_decision([0.9], [None]) == 0.0  # missing pnl contributes 0
