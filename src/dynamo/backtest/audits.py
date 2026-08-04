"""Overfitting and leakage audits (doc 08 section 6). Required stage gates.

Look-ahead audit: every promotion-grade run is re-executed with all features
shifted +1 session (the advisory provider sees a one-session-staler window).
If performance does not degrade materially (default: Sharpe drop >= 50%), the
run FAILS — an edge that survives the shift was not flowing through the
point-in-time feature path, i.e. it was leakage.

Constraint-violation counter: the event loop checks every constructed target
against its session verdict; the count must be zero for promotion evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from dynamo.backtest.engine import (
    AdvisoryProvider,
    BacktestData,
    BacktestEngine,
    BacktestResult,
    SessionAdvice,
    compliance_violations,
)
from dynamo.backtest.metrics import metrics_block

DEFAULT_MIN_SHARPE_DROP = 0.5

__all__ = [
    "DEFAULT_MIN_SHARPE_DROP",
    "BaselineComparison",
    "LaggedProvider",
    "LookAheadAuditResult",
    "baseline_comparison",
    "compliance_violations",
    "constraint_violation_audit",
    "run_look_ahead_audit",
]


class LaggedProvider:
    """Wraps a provider so it sees features shifted +1 session (staler window)."""

    def __init__(self, inner: AdvisoryProvider) -> None:
        self._inner = inner

    def advise(
        self,
        as_of: date,
        visible_closes: pd.DataFrame,
        holdings: Mapping[str, float],
    ) -> SessionAdvice:
        lagged = visible_closes.iloc[:-1] if len(visible_closes) > 1 else visible_closes
        return self._inner.advise(as_of, lagged, holdings)


@dataclass(frozen=True)
class LookAheadAuditResult:
    baseline_sharpe: float
    shifted_sharpe: float
    sharpe_drop: float  # fraction of baseline Sharpe lost under the shift
    min_drop: float
    passed: bool  # False -> the edge survived the shift: leakage, run fails


def look_ahead_audit(
    baseline_sharpe: float,
    shifted_sharpe: float,
    min_drop: float = DEFAULT_MIN_SHARPE_DROP,
) -> LookAheadAuditResult:
    """Fail the run when the Sharpe drop under a +1-session shift is < min_drop."""
    if not math.isfinite(baseline_sharpe) or baseline_sharpe <= 0.0:
        # No positive edge to defend; the audit is vacuously passed and the
        # run will fail promotion on performance grounds instead.
        drop = 1.0
    else:
        drop = (baseline_sharpe - shifted_sharpe) / baseline_sharpe
    return LookAheadAuditResult(
        baseline_sharpe=baseline_sharpe,
        shifted_sharpe=shifted_sharpe,
        sharpe_drop=drop,
        min_drop=min_drop,
        passed=drop >= min_drop,
    )


def run_look_ahead_audit(
    engine: BacktestEngine,
    data: BacktestData,
    provider: AdvisoryProvider,
    min_drop: float = DEFAULT_MIN_SHARPE_DROP,
    baseline: BacktestResult | None = None,
) -> LookAheadAuditResult:
    """Run baseline and +1-session-shifted backtests and compare Sharpe."""
    base = baseline if baseline is not None else engine.run(data, provider)
    shifted = engine.run(data, LaggedProvider(provider))
    base_sharpe = metrics_block(base.equity_curve).sharpe
    shifted_sharpe = metrics_block(shifted.equity_curve).sharpe
    return look_ahead_audit(base_sharpe, shifted_sharpe, min_drop)


def constraint_violation_audit(result: BacktestResult) -> tuple[bool, int]:
    """(passed, count). Count must be 0 for any promotion-grade evidence."""
    return result.constraint_violation_count == 0, result.constraint_violation_count


@dataclass(frozen=True)
class BaselineComparison:
    """Strategy vs mandatory naive baselines on risk-adjusted terms (doc 08 s6)."""

    strategy_sharpe: float
    baseline_sharpes: tuple[tuple[str, float], ...]  # (name, sharpe), sorted
    beats_all: bool


def baseline_comparison(
    strategy_sharpe: float, baselines: Mapping[str, float]
) -> BaselineComparison:
    """Hook for the mandatory-baseline gate: SPY buy-and-hold, naive, bootstrap.

    Callers supply each baseline's Sharpe computed on the SAME window/folds;
    a strategy that cannot beat naive is retired regardless of absolute Sharpe.
    """
    rows = tuple(sorted(baselines.items()))
    beats = all(strategy_sharpe > s for _, s in rows) if rows else False
    return BaselineComparison(
        strategy_sharpe=strategy_sharpe,
        baseline_sharpes=rows,
        beats_all=beats,
    )
