"""Backtesting & validation — the machinery that keeps backtests honest.

Design: docs/design/08-backtesting-and-validation.md. The event-driven
validator imports the production risk/ and portfolio/ modules; only its
reports are admissible promotion evidence.
"""

from dynamo.backtest.audits import (
    LaggedProvider,
    LookAheadAuditResult,
    baseline_comparison,
    compliance_violations,
    constraint_violation_audit,
    look_ahead_audit,
    run_look_ahead_audit,
)
from dynamo.backtest.costs import (
    CostParams,
    cost_per_side_fraction,
    half_spread_bps,
    impact_fraction,
)
from dynamo.backtest.engine import (
    AdvisoryProvider,
    BacktestConfig,
    BacktestData,
    BacktestEngine,
    BacktestResult,
    SessionAdvice,
    in_universe,
)
from dynamo.backtest.metrics import (
    BacktestReport,
    MetricsBlock,
    build_report,
    deflated_sharpe_ratio,
    metrics_block,
)
from dynamo.backtest.promotion import (
    PROMOTION_STAGES,
    RegistryRepository,
    StageGate,
    StageResult,
    run_stage,
)
from dynamo.backtest.signal_audit import audit_factor
from dynamo.backtest.tearsheet import generate_tearsheet, write_backtest_artifacts

__all__ = [
    "PROMOTION_STAGES",
    "AdvisoryProvider",
    "BacktestConfig",
    "BacktestData",
    "BacktestEngine",
    "BacktestReport",
    "BacktestResult",
    "CostParams",
    "LaggedProvider",
    "LookAheadAuditResult",
    "MetricsBlock",
    "RegistryRepository",
    "SessionAdvice",
    "StageGate",
    "StageResult",
    "audit_factor",
    "baseline_comparison",
    "build_report",
    "compliance_violations",
    "constraint_violation_audit",
    "cost_per_side_fraction",
    "deflated_sharpe_ratio",
    "generate_tearsheet",
    "half_spread_bps",
    "impact_fraction",
    "in_universe",
    "look_ahead_audit",
    "metrics_block",
    "run_look_ahead_audit",
    "run_stage",
    "write_backtest_artifacts",
]
