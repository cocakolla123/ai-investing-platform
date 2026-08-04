"""Calibration tracking: Brier scores + reliability buckets (doc 10 §5).

Outcome definition per horizon: did the position achieve at least half the
expected upside before hitting the expected downside, within ``horizon_days``.
All functions are pure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from dynamo.domain.models.learning import CalibrationPoint

N_BUCKETS = 10


@dataclass(frozen=True)
class CalibrationObservation:
    """One scored forecast: conviction in [0, 1] against a realized outcome."""

    scope_type: str  # agent | committee | strategy | prediction_market
    scope_id: str
    conviction: float
    outcome: bool


def trade_outcome_achieved(
    daily_cum_returns_pct: Sequence[float],
    expected_upside_pct: float,
    expected_downside_pct: float,
    horizon_days: int,
) -> bool:
    """Walk the cumulative-return path: True iff >= half the expected upside is
    reached before the expected downside, within the horizon (doc 10 §5)."""
    if expected_upside_pct <= 0:
        return False
    target = expected_upside_pct / 2.0
    floor = -abs(expected_downside_pct)
    for r in daily_cum_returns_pct[:horizon_days]:
        if r <= floor:
            return False
        if r >= target:
            return True
    return False


def brier_score(observations: Sequence[CalibrationObservation]) -> float:
    """Mean squared error of conviction vs outcome. Bounded [0, 1]."""
    if not observations:
        return 0.0
    _validate(observations)
    total = sum((o.conviction - (1.0 if o.outcome else 0.0)) ** 2 for o in observations)
    return total / len(observations)


def reliability_buckets(
    observations: Sequence[CalibrationObservation], n_buckets: int = N_BUCKETS
) -> dict[str, float]:
    """Conviction deciles vs realized frequency. Empty buckets are simply
    omitted — never a division by zero."""
    _validate(observations)
    counts = [0] * n_buckets
    hits = [0] * n_buckets
    forecast_sum = [0.0] * n_buckets
    for o in observations:
        idx = min(int(o.conviction * n_buckets), n_buckets - 1)
        counts[idx] += 1
        hits[idx] += 1 if o.outcome else 0
        forecast_sum[idx] += o.conviction
    stats: dict[str, float] = {}
    for i in range(n_buckets):
        if counts[i] == 0:
            continue
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        label = f"{lo:.1f}-{hi:.1f}"
        stats[f"{label}_n"] = float(counts[i])
        stats[f"{label}_forecast"] = forecast_sum[i] / counts[i]
        stats[f"{label}_observed"] = hits[i] / counts[i]
    return stats


def compute_calibration(
    observations: Sequence[CalibrationObservation],
    as_of: date,
    n_buckets: int = N_BUCKETS,
) -> list[CalibrationPoint]:
    """One ``CalibrationPoint`` per (scope_type, scope_id): Brier + buckets."""
    _validate(observations)
    by_scope: dict[tuple[str, str], list[CalibrationObservation]] = {}
    for o in observations:
        by_scope.setdefault((o.scope_type, o.scope_id), []).append(o)
    points: list[CalibrationPoint] = []
    for (scope_type, scope_id) in sorted(by_scope):
        group = by_scope[(scope_type, scope_id)]
        points.append(
            CalibrationPoint(
                scope_type=scope_type,
                scope_id=scope_id,
                as_of=as_of,
                brier=brier_score(group),
                n=len(group),
                bucket_stats=reliability_buckets(group, n_buckets),
            )
        )
    return points


def _validate(observations: Sequence[CalibrationObservation]) -> None:
    for o in observations:
        if not 0.0 <= o.conviction <= 1.0:
            raise ValueError(f"conviction {o.conviction} outside [0, 1] for {o.scope_id}")
