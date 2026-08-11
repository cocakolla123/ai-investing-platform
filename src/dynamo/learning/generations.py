"""Generational meta-model training vs a frozen control (plan Phase 5).

PURE — data in, results out: experiences are PASSED IN (import-linter keeps
this module off persistence); the CLI (``dynamo.cli.generations``) does the
loading, the durable ``evaluation_claims`` bookkeeping, and the persistence
of results/artifacts.

Fold scheme (monthly, expanding): generation k trains on months 1..k and
tests on month k+1. The FINAL ``final_holdout_months`` months are excluded
from EVERY fold — that period is claimed once in ``evaluation_claims`` by the
CLI and refused to all training forever after.

Frozen control: the raw committee conviction passed through the IDENTITY
calibration — ``p_control = clip(conviction, 0, 1)`` on the *identical* test
decisions the meta-model scores. It is the "what if we never learned
anything" baseline, frozen by construction (no parameters, nothing to fit).

SIGN CONVENTION (documented, load-bearing):

    value_of_learning = brier_control - brier_meta

Brier is a LOSS (lower is better), so POSITIVE value_of_learning means the
learned meta-model beats the frozen control — learning helps; negative means
the control is still better and the meta-model has not earned trust.

Generations with fewer than ``MIN_TRAIN_EXPERIENCES`` training rows are
recorded as SKIPPED (never trained on thin data) — the gate is the same one
``MetaModel.train`` enforces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from dynamo.domain.models.learning import DecisionExperience
from dynamo.learning.hierarchy import CalibratedProbability
from dynamo.learning.meta_model import (
    MIN_TRAIN_EXPERIENCES,
    MetaModel,
    brier,
    build_feature_row,
    precision_by_conviction_bucket,
)

Month = tuple[int, int]  # (year, month)

__all__ = [
    "GenerationFold",
    "GenerationOutcome",
    "ev_per_decision",
    "final_holdout_period",
    "frozen_control_probabilities",
    "generation_folds",
    "run_generation",
    "usable_experiences",
]


def _month_of(d: date) -> Month:
    return (d.year, d.month)


def _month_start(m: Month) -> date:
    return date(m[0], m[1], 1)


def _month_end(m: Month) -> date:
    nxt = date(m[0] + 1, 1, 1) if m[1] == 12 else date(m[0], m[1] + 1, 1)
    return nxt - timedelta(days=1)


def usable_experiences(
    experiences: Sequence[DecisionExperience],
) -> list[DecisionExperience]:
    """Resolved rows with a win/loss label AND a conviction — the only rows
    both the trainer and the frozen control can score. Deterministic order:
    (as_of, created_at, id)."""
    rows = [
        e
        for e in experiences
        if e.resolved and e.label in ("win", "loss") and e.conviction is not None
    ]
    return sorted(rows, key=lambda e: (e.as_of, e.created_at, e.id))


@dataclass(frozen=True)
class GenerationFold:
    """Generation k: train on months 1..k, test on month k+1."""

    generation: int  # 1-based
    train: tuple[DecisionExperience, ...]
    test: tuple[DecisionExperience, ...]
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def generation_folds(
    experiences: Sequence[DecisionExperience],
    month_boundaries: Sequence[Month] | None = None,
    final_holdout_months: int = 1,
) -> list[GenerationFold]:
    """Expanding monthly folds with the final holdout EXCLUDED everywhere.

    ``month_boundaries`` defaults to the sorted distinct (year, month) values
    observed in the usable experiences. The last ``final_holdout_months``
    months are dropped from BOTH sides of every fold; the leakage guard
    asserts max(train ts) < min(test ts) inside the builder.
    """
    if final_holdout_months < 0:
        raise ValueError("final_holdout_months must be >= 0")
    rows = usable_experiences(experiences)
    months = (
        sorted(set(month_boundaries))
        if month_boundaries is not None
        else sorted({_month_of(e.as_of) for e in rows})
    )
    usable_months = months[:-final_holdout_months] if final_holdout_months else list(months)
    by_month: dict[Month, list[DecisionExperience]] = {m: [] for m in usable_months}
    for e in rows:
        m = _month_of(e.as_of)
        if m in by_month:
            by_month[m].append(e)

    folds: list[GenerationFold] = []
    for k in range(1, len(usable_months)):
        train_months = usable_months[:k]
        test_month = usable_months[k]
        train = tuple(e for m in train_months for e in by_month[m])
        test = tuple(by_month[test_month])
        if not train or not test:
            continue
        # THE leakage guard: every training timestamp strictly precedes every
        # test timestamp. Months are disjoint by construction; this assert
        # keeps that true against any future refactor.
        max_train_ts = max((e.as_of, e.created_at) for e in train)
        min_test_ts = min((e.as_of, e.created_at) for e in test)
        assert max_train_ts < min_test_ts, (
            f"generation {k}: train ts {max_train_ts} >= test ts {min_test_ts}"
        )
        folds.append(
            GenerationFold(
                generation=k,
                train=train,
                test=test,
                train_start=_month_start(train_months[0]),
                train_end=_month_end(train_months[-1]),
                test_start=_month_start(test_month),
                test_end=_month_end(test_month),
            )
        )
    return folds


def final_holdout_period(
    experiences: Sequence[DecisionExperience],
    month_boundaries: Sequence[Month] | None = None,
    final_holdout_months: int = 1,
) -> tuple[date, date] | None:
    """Calendar (start, end) of the excluded final holdout — what the CLI
    claims in ``evaluation_claims`` — or ``None`` when there is no holdout."""
    if final_holdout_months <= 0:
        return None
    rows = usable_experiences(experiences)
    months = (
        sorted(set(month_boundaries))
        if month_boundaries is not None
        else sorted({_month_of(e.as_of) for e in rows})
    )
    holdout = months[-final_holdout_months:]
    if not holdout or len(months) <= final_holdout_months:
        return None
    return _month_start(holdout[0]), _month_end(holdout[-1])


# --------------------------------------------------------------------------
# per-generation evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationOutcome:
    """One generation's result row — meta vs frozen control, or a skip."""

    generation: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    model_version: str
    n_train: int
    n_test: int
    skipped: bool
    skip_reason: str
    brier_meta: float | None
    brier_control: float | None
    precision_meta: dict[str, float]
    precision_control: dict[str, float]
    ev_meta: float | None
    ev_control: float | None
    value_of_learning: float | None
    model: MetaModel | None


def frozen_control_probabilities(
    experiences: Sequence[DecisionExperience],
) -> list[float]:
    """The frozen control: raw conviction under the IDENTITY calibration
    (clipped to [0, 1]) — nothing fit, nothing learned, frozen forever."""
    return [
        min(max(float(e.conviction if e.conviction is not None else 0.0), 0.0), 1.0)
        for e in experiences
    ]


def ev_per_decision(
    probs: Sequence[float], pnls: Sequence[float | None]
) -> float | None:
    """Realized EV per decision under a take-predicted-winners policy.

    For each test decision: if the scorer predicts a win (p >= 0.5) the
    policy takes the trade and realizes its actual ``pnl_pct``; otherwise it
    abstains (0). The mean over all test decisions is the per-decision EV —
    directly comparable between meta and control because both score the
    identical decisions.
    """
    if not probs:
        return None
    total = 0.0
    for p, pnl in zip(probs, pnls, strict=True):
        if p >= 0.5 and pnl is not None:
            total += float(pnl)
    return total / len(probs)


def _feature_rows(experiences: Sequence[DecisionExperience]) -> pd.DataFrame:
    """Deterministic feature frame via the SHARED builder (meta_model.py).

    Neighbors are deliberately empty here: neighbor retrieval needs the
    repository (persistence) and this module is pure — both meta and control
    see the identical degradation, so the comparison stays fair.
    """
    rows = []
    for exp in experiences:
        calib = (
            None
            if exp.calibrated_probability is None
            else CalibratedProbability(
                value=exp.calibrated_probability,
                scope_used=exp.calibration_scope_used or "global",
                n_effective=0,
            )
        )
        rows.append(build_feature_row(exp, calib, ()))
    return pd.DataFrame(rows)


def run_generation(fold: GenerationFold, *, algo: str = "gradient_boosting") -> GenerationOutcome:
    """Train generation k on its fold; score meta vs frozen control on the
    identical test decisions. Below ``MIN_TRAIN_EXPERIENCES`` the generation
    is recorded as skipped — never trained."""
    model_version = f"meta-gen{fold.generation:03d}-{algo}"

    def outcome(
        *,
        skipped: bool,
        skip_reason: str = "",
        brier_meta: float | None = None,
        brier_control: float | None = None,
        precision_meta: dict[str, float] | None = None,
        precision_control: dict[str, float] | None = None,
        ev_meta: float | None = None,
        ev_control: float | None = None,
        value_of_learning: float | None = None,
        model: MetaModel | None = None,
    ) -> GenerationOutcome:
        return GenerationOutcome(
            generation=fold.generation,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            model_version=model_version,
            n_train=len(fold.train),
            n_test=len(fold.test),
            skipped=skipped,
            skip_reason=skip_reason,
            brier_meta=brier_meta,
            brier_control=brier_control,
            precision_meta=precision_meta or {},
            precision_control=precision_control or {},
            ev_meta=ev_meta,
            ev_control=ev_control,
            value_of_learning=value_of_learning,
            model=model,
        )

    if len(fold.train) < MIN_TRAIN_EXPERIENCES:
        return outcome(
            skipped=True,
            skip_reason=(
                f"{len(fold.train)} train experiences < hard gate {MIN_TRAIN_EXPERIENCES}"
            ),
        )

    train_frame = _feature_rows(fold.train)
    train_labels = [e.label == "win" for e in fold.train]
    # Scalar (string) time keys: uniformly comparable for the trainer's
    # stable argsort, deterministic tie-break by id.
    train_order = [
        f"{e.as_of.isoformat()}|{e.created_at.isoformat()}|{e.id}" for e in fold.train
    ]
    model = MetaModel(algo=algo)
    model.train(train_frame, train_labels, train_order)

    test_frame = _feature_rows(fold.test)
    outcomes = [e.label == "win" for e in fold.test]
    pnls = [e.pnl_pct for e in fold.test]
    p_meta = model.predict_proba(test_frame)
    p_control = frozen_control_probabilities(fold.test)

    brier_meta = brier(p_meta, outcomes)
    brier_control = brier(p_control, outcomes)
    return outcome(
        skipped=False,
        brier_meta=brier_meta,
        brier_control=brier_control,
        precision_meta=precision_by_conviction_bucket(test_frame, p_meta, outcomes),
        precision_control=precision_by_conviction_bucket(test_frame, p_control, outcomes),
        ev_meta=ev_per_decision(p_meta, pnls),
        ev_control=ev_per_decision(p_control, pnls),
        # POSITIVE = the learned model beats the frozen control (module docstring).
        value_of_learning=brier_control - brier_meta,
        model=model,
    )
