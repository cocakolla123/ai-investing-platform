"""Learning-metrics suite over resolved DecisionExperience rows (plan Phase 6).

PURE — everything is computed off sequences PASSED IN (experiences, verdict
action counts, generation-result rows); import-linter keeps this module off
persistence. Every function is total and deterministic: empty input yields
``None`` (undefined) or an empty/zero-count container — never an exception.

Outcome definition: a row is *scorable* when it carries a conviction AND a
resolvable outcome — path-based ``target_achieved`` first, the win/loss
``label`` as fallback (:func:`dynamo.learning.hierarchy.experience_outcome`).
*Success* for pnl cuts means ``label == 'win'`` (realized pnl >= 0).

Sharpe / Sortino / max-drawdown assumptions (documented, load-bearing):

- the "daily pnl series" credits each resolved experience's FULL ``pnl_pct``
  on its exit date (``exit_at.date()``, falling back to ``as_of``), summed
  per day and equal-weighted across decisions;
- weekdays between the first and last exit date with no exits are zero-filled
  (weekends omitted — a business-day approximation of sessions);
- Sharpe = mean/std * sqrt(252), Sortino = mean/downside-dev * sqrt(252)
  (both vs a 0 hurdle), max drawdown = largest peak-to-trough drop of the
  CUMULATIVE-SUM curve, in percentage points.

This is a decision-level diagnostic — it ignores position sizing and capital
overlap, so it is NOT the account Sharpe (``equity_curve`` is the account).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from dynamo.learning.hierarchy import experience_outcome

if TYPE_CHECKING:
    from dynamo.domain.models.learning import DecisionExperience

N_BUCKETS = 10
TRADING_DAYS_PER_YEAR = 252
ABSTAIN_ACTIONS = frozenset({"no_action"})

# One metrics_daily-shaped row: (metric, scope_type, scope_id, value).
MetricRow = tuple[str, str, str, float]


# --- result containers -------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSummary:
    """Population calibration over scorable rows: Brier + expected
    calibration error (conviction deciles). ``None`` fields when n == 0."""

    n: int
    brier: float | None
    ece: float | None


@dataclass(frozen=True)
class BucketStat:
    """One conviction bucket: forecasts made vs how often they succeeded."""

    n: int
    mean_conviction: float
    success_rate: float


@dataclass(frozen=True)
class DistributionSummary:
    n: int
    mean: float | None
    median: float | None


@dataclass(frozen=True)
class RiskSummary:
    """Sharpe / Sortino / max drawdown over the experience-derived daily pnl
    series (module-docstring assumptions). ``None`` when undefined (fewer
    than two days, or zero variance / no downside)."""

    n_days: int
    sharpe: float | None
    sortino: float | None
    max_drawdown_pct: float | None


@dataclass(frozen=True)
class CutStat:
    """Per-group performance for :func:`cut_by`."""

    n: int
    wins: int
    win_rate: float
    mean_pnl_pct: float | None


# --- scorable pairs + calibration --------------------------------------------


def scored_pairs(experiences: Sequence[DecisionExperience]) -> list[tuple[float, bool]]:
    """(conviction, outcome) for every scorable resolved row, in input order.
    Convictions are clipped to [0, 1]; unscorable rows are skipped."""
    out: list[tuple[float, bool]] = []
    for exp in experiences:
        if not exp.resolved or exp.conviction is None:
            continue
        outcome = experience_outcome(exp)
        if outcome is None:
            continue
        out.append((min(max(float(exp.conviction), 0.0), 1.0), outcome))
    return out


def brier_score(experiences: Sequence[DecisionExperience]) -> float | None:
    """Mean squared error of conviction vs outcome; ``None`` on no scorable rows."""
    pairs = scored_pairs(experiences)
    if not pairs:
        return None
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in pairs) / len(pairs)


def expected_calibration_error(
    experiences: Sequence[DecisionExperience], n_buckets: int = N_BUCKETS
) -> float | None:
    """ECE over conviction buckets: sum_b (n_b/N) * |mean conviction_b -
    success rate_b|. ``None`` on no scorable rows."""
    buckets = success_rate_by_conviction_bucket(experiences, n_buckets)
    total = sum(b.n for b in buckets.values())
    if total == 0:
        return None
    return sum(
        (b.n / total) * abs(b.mean_conviction - b.success_rate) for b in buckets.values()
    )


def calibration_summary(
    experiences: Sequence[DecisionExperience], n_buckets: int = N_BUCKETS
) -> CalibrationSummary:
    pairs = scored_pairs(experiences)
    return CalibrationSummary(
        n=len(pairs),
        brier=brier_score(experiences),
        ece=expected_calibration_error(experiences, n_buckets),
    )


def success_rate_by_conviction_bucket(
    experiences: Sequence[DecisionExperience], n_buckets: int = N_BUCKETS
) -> dict[str, BucketStat]:
    """Per-conviction-decile success rate ("precision" of the stated
    conviction). Empty buckets are omitted — never a division by zero."""
    pairs = scored_pairs(experiences)
    counts = [0] * n_buckets
    hits = [0] * n_buckets
    conviction_sum = [0.0] * n_buckets
    for conviction, outcome in pairs:
        idx = min(int(conviction * n_buckets), n_buckets - 1)
        counts[idx] += 1
        hits[idx] += 1 if outcome else 0
        conviction_sum[idx] += conviction
    out: dict[str, BucketStat] = {}
    for i in range(n_buckets):
        if counts[i] == 0:
            continue
        label = f"{i / n_buckets:.1f}-{(i + 1) / n_buckets:.1f}"
        out[label] = BucketStat(
            n=counts[i],
            mean_conviction=conviction_sum[i] / counts[i],
            success_rate=hits[i] / counts[i],
        )
    return out


# --- pnl-based summaries ------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def benchmark_excess(experiences: Sequence[DecisionExperience]) -> DistributionSummary:
    """Mean/median realized excess over the benchmark (SPY) per decision, in
    percentage points, over resolved rows that carry the excess."""
    values = [
        float(e.benchmark_excess_pct)
        for e in experiences
        if e.resolved and e.benchmark_excess_pct is not None
    ]
    if not values:
        return DistributionSummary(n=0, mean=None, median=None)
    return DistributionSummary(
        n=len(values), mean=sum(values) / len(values), median=_median(values)
    )


def ev_per_decision(experiences: Sequence[DecisionExperience]) -> float | None:
    """Realized EV per decision: mean ``pnl_pct`` over resolved rows carrying
    one (equal-weighted; the sizing-blind per-decision expectation)."""
    values = [float(e.pnl_pct) for e in experiences if e.resolved and e.pnl_pct is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _exit_day(exp: DecisionExperience) -> date:
    return exp.exit_at.date() if exp.exit_at is not None else exp.as_of


def daily_pnl_series(experiences: Sequence[DecisionExperience]) -> list[tuple[date, float]]:
    """Experience pnl bucketed onto exit weekdays (module-docstring
    assumptions): sum of ``pnl_pct`` per exit date, zero-filled over weekdays
    between the first and last exit date. Weekend exits keep their own day."""
    by_day: dict[date, float] = {}
    for exp in experiences:
        if not exp.resolved or exp.pnl_pct is None:
            continue
        day = _exit_day(exp)
        by_day[day] = by_day.get(day, 0.0) + float(exp.pnl_pct)
    if not by_day:
        return []
    first, last = min(by_day), max(by_day)
    out: dict[date, float] = {}
    current = first
    while current <= last:
        if current.weekday() < 5:
            out[current] = 0.0
        current += timedelta(days=1)
    for day, value in by_day.items():
        out[day] = out.get(day, 0.0) + value
    return sorted(out.items())


def sharpe_ratio(daily_returns: Sequence[float]) -> float | None:
    """Annualized mean/std of the daily series; ``None`` below 2 points or on
    zero variance."""
    n = len(daily_returns)
    if n < 2:
        return None
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    if variance <= 0.0:
        return None
    return mean / math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: Sequence[float]) -> float | None:
    """Annualized mean/downside-deviation (vs 0); ``None`` below 2 points or
    with no downside observations."""
    n = len(daily_returns)
    if n < 2:
        return None
    mean = sum(daily_returns) / n
    downside = sum(min(r, 0.0) ** 2 for r in daily_returns) / n
    if downside <= 0.0:
        return None
    return mean / math.sqrt(downside) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown_pct(daily_returns: Sequence[float]) -> float | None:
    """Largest peak-to-trough drop of the cumulative-sum curve, in percentage
    points (>= 0); ``None`` on an empty series."""
    if not daily_returns:
        return None
    peak = 0.0
    cumulative = 0.0
    worst = 0.0
    for r in daily_returns:
        cumulative += r
        peak = max(peak, cumulative)
        worst = max(worst, peak - cumulative)
    return worst


def risk_summary(experiences: Sequence[DecisionExperience]) -> RiskSummary:
    series = [value for _, value in daily_pnl_series(experiences)]
    return RiskSummary(
        n_days=len(series),
        sharpe=sharpe_ratio(series),
        sortino=sortino_ratio(series),
        max_drawdown_pct=max_drawdown_pct(series),
    )


# --- abstention ---------------------------------------------------------------


def abstention_rate(action_counts: Mapping[str, int]) -> float | None:
    """Share of committee verdicts that abstained (``no_action``), over the
    decisions-considered denominator.

    Experiences alone cannot supply the denominator — only DECIDED-AND-CLOSED
    trades become experiences — so the caller passes committee_verdicts action
    counts (``{action: count}``). ``None`` when there are no verdicts."""
    total = sum(max(int(v), 0) for v in action_counts.values())
    if total == 0:
        return None
    abstained = sum(
        max(int(v), 0) for k, v in action_counts.items() if k in ABSTAIN_ACTIONS
    )
    return abstained / total


# --- generic cuts ---------------------------------------------------------------


def cut_by(
    experiences: Sequence[DecisionExperience],
    key_fn: Callable[[DecisionExperience], str | None],
) -> dict[str, CutStat]:
    """Group resolved win/loss-labeled rows by ``key_fn`` and summarize each
    group. Rows keyed ``None`` are skipped. Deterministic: dict keys are
    sorted, stats depend only on group membership. The regime / sector /
    sleeve / quant-alignment cuts below are thin wrappers; a prompt-version
    cut works the same way once the caller closes over a
    ``trade_id -> prompt_version`` mapping from the linked reviews."""
    groups: dict[str, list[DecisionExperience]] = {}
    for exp in experiences:
        if not exp.resolved or exp.label not in ("win", "loss"):
            continue
        key = key_fn(exp)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(exp)
    out: dict[str, CutStat] = {}
    for key in sorted(groups):
        rows = groups[key]
        wins = sum(1 for e in rows if e.label == "win")
        pnls = [float(e.pnl_pct) for e in rows if e.pnl_pct is not None]
        out[key] = CutStat(
            n=len(rows),
            wins=wins,
            win_rate=wins / len(rows),
            mean_pnl_pct=sum(pnls) / len(pnls) if pnls else None,
        )
    return out


def performance_by_quant_alignment(
    experiences: Sequence[DecisionExperience],
) -> dict[str, CutStat]:
    """Performance when the committee agreed / conflicted / was neutral vs
    the quant view."""
    return cut_by(experiences, lambda e: e.quant_alignment)


def performance_by_regime(experiences: Sequence[DecisionExperience]) -> dict[str, CutStat]:
    return cut_by(experiences, lambda e: e.regime)


def performance_by_sector(experiences: Sequence[DecisionExperience]) -> dict[str, CutStat]:
    return cut_by(experiences, lambda e: e.sector)


def performance_by_sleeve(experiences: Sequence[DecisionExperience]) -> dict[str, CutStat]:
    return cut_by(experiences, lambda e: e.sleeve)


def failure_class_frequency(experiences: Sequence[DecisionExperience]) -> dict[str, int]:
    """How often each failure class was named primary across resolved rows
    (the linked review's ``failure_primary``, denormalized onto the row)."""
    counts: dict[str, int] = {}
    for exp in experiences:
        if not exp.resolved or exp.failure_primary is None:
            continue
        key = exp.failure_primary.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --- value-of-learning passthrough ---------------------------------------------


def latest_value_of_learning(
    generation_rows: Sequence[Mapping[str, object]],
) -> tuple[float, int] | None:
    """(value_of_learning, n_test) of the highest generation carrying a
    value, from ``generation_results`` rows passed in (sign convention lives
    in ``learning/generations.py``: positive = learning beats the frozen
    control). ``None`` when no generation trained."""
    def _num(raw: object, default: float = 0.0) -> float:
        return float(raw) if isinstance(raw, int | float) else default

    best: tuple[int, float, int] | None = None
    for row in generation_rows:
        value = row.get("value_of_learning")
        if not isinstance(value, int | float):
            continue
        generation = int(_num(row.get("generation")))
        if best is None or generation > best[0]:
            best = (generation, float(value), int(_num(row.get("n_test"))))
    if best is None:
        return None
    return best[1], best[2]


# --- metrics_daily row builder (consumed by the learning_metrics asset) --------


def learning_metric_rows(
    experiences: Sequence[DecisionExperience],
    action_counts: Mapping[str, int],
    generation_rows: Sequence[Mapping[str, object]],
) -> list[MetricRow]:
    """The full suite as ``metrics_daily``-shaped rows, all namespaced
    ``learning.*``. Headline numbers use scope (platform, all); cuts use the
    dimension as ``scope_type`` and the group as ``scope_id``. Undefined
    metrics are simply omitted (never fabricated)."""
    rows: list[MetricRow] = []

    def headline(metric: str, value: float | None) -> None:
        if value is not None:
            rows.append((metric, "platform", "all", float(value)))

    calibration = calibration_summary(experiences)
    headline("learning.n_resolved", float(sum(1 for e in experiences if e.resolved)))
    headline("learning.n_scored", float(calibration.n))
    headline("learning.brier", calibration.brier)
    headline("learning.ece", calibration.ece)
    if calibration.n:
        pairs = scored_pairs(experiences)
        headline("learning.success_rate", sum(1 for _, o in pairs if o) / len(pairs))
    headline("learning.ev_per_decision", ev_per_decision(experiences))
    excess = benchmark_excess(experiences)
    headline("learning.benchmark_excess_mean", excess.mean)
    headline("learning.benchmark_excess_median", excess.median)
    risk = risk_summary(experiences)
    headline("learning.sharpe", risk.sharpe)
    headline("learning.sortino", risk.sortino)
    headline("learning.max_drawdown_pct", risk.max_drawdown_pct)
    headline("learning.abstention_rate", abstention_rate(action_counts))
    learned = latest_value_of_learning(generation_rows)
    if learned is not None:
        headline("learning.value_of_learning", learned[0])
        headline("learning.value_of_learning_n_test", float(learned[1]))

    for label, bucket in success_rate_by_conviction_bucket(experiences).items():
        rows.append(("learning.bucket_success_rate", "conviction_bucket", label,
                     bucket.success_rate))
        rows.append(("learning.bucket_n", "conviction_bucket", label, float(bucket.n)))

    for scope_type, cuts in (
        ("quant_alignment", performance_by_quant_alignment(experiences)),
        ("regime", performance_by_regime(experiences)),
        ("sector", performance_by_sector(experiences)),
        ("sleeve", performance_by_sleeve(experiences)),
    ):
        for scope_id, stat in cuts.items():
            rows.append(("learning.win_rate", scope_type, scope_id, stat.win_rate))
            rows.append(("learning.n", scope_type, scope_id, float(stat.n)))
            if stat.mean_pnl_pct is not None:
                rows.append(("learning.mean_pnl_pct", scope_type, scope_id, stat.mean_pnl_pct))

    for failure_class, count in failure_class_frequency(experiences).items():
        rows.append(("learning.failure_count", "failure_class", failure_class, float(count)))
    return rows
