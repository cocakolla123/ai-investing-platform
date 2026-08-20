"""QuantStats HTML tearsheets for finished event-driven backtest runs.

``quantstats`` ships in the optional ``analysis`` extra
(``uv sync --extra analysis``); the import is guarded, so without it
:func:`generate_tearsheet` returns ``None`` with a log line and the run's
JSON report is still written — the tearsheet is a bonus artifact, never a
dependency of the validation pipeline.

Artifact layout (written by :func:`write_backtest_artifacts` after a run)::

    {artifact_dir}/backtests/{run_id}/report.json    # BacktestReport JSON
    {artifact_dir}/backtests/{run_id}/tearsheet.html # quantstats full report

The dashboard Performance page lists every ``tearsheet.html`` under
``{artifact_dir}/backtests/``.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from dynamo.backtest.metrics import BacktestReport

log = structlog.get_logger(__name__)

__all__ = ["generate_tearsheet", "write_backtest_artifacts"]


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    series = pd.Series(
        np.asarray(equity, dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(equity.index)),
    )
    return series.pct_change().dropna()


def generate_tearsheet(
    returns_series: pd.Series,
    benchmark_series: pd.Series | None,
    output_html_path: str | Path,
) -> Path | None:
    """Write a quantstats HTML tearsheet; ``None`` when unavailable/failed.

    ``returns_series`` (and the optional ``benchmark_series``) are daily
    returns indexed by date. Guarded: quantstats missing or raising is a
    logged skip, never an error.
    """
    try:
        quantstats = importlib.import_module("quantstats")
    except Exception:
        log.info(
            "tearsheet_skipped",
            reason="quantstats not installed (uv sync --extra analysis)",
        )
        return None
    path = Path(output_html_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantstats.reports.html(
                returns_series,
                benchmark=benchmark_series,
                output=str(path),
                download_filename=path.name,
                title="Dynamo backtest tearsheet",
            )
    except Exception as exc:
        log.warning("tearsheet_failed", error=str(exc), path=str(path))
        return None
    if not path.exists():
        log.warning("tearsheet_failed", error="quantstats wrote no file", path=str(path))
        return None
    return path


def write_backtest_artifacts(
    report: BacktestReport,
    equity_curve: pd.Series,
    *,
    artifact_dir: str | Path,
    benchmark_closes: pd.Series | None = None,
) -> dict[str, Path]:
    """Persist a finished run's artifacts: report JSON + tearsheet HTML beside it."""
    run_id = report.run_id or (report.config_hash[:12] if report.config_hash else "run")
    out_dir = Path(artifact_dir) / "backtests" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    artifacts: dict[str, Path] = {"report": report_path}

    returns = _returns_from_equity(equity_curve)
    benchmark = _returns_from_equity(benchmark_closes) if benchmark_closes is not None else None
    if len(returns) < 2:
        log.info("tearsheet_skipped", reason="not enough sessions for returns")
        return artifacts
    tearsheet_path = generate_tearsheet(returns, benchmark, out_dir / "tearsheet.html")
    if tearsheet_path is not None:
        artifacts["tearsheet"] = tearsheet_path
    return artifacts
