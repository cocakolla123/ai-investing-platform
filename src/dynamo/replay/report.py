"""Replay run reporting (backtest-campaign Phase 5).

Maps a finished quant/portfolio replay's persisted rows (``replay_equity`` +
``replay_fills`` + ``replay_runs``) into the EXISTING backtest report
pipeline: ``backtest.metrics.build_report`` ->
``artifacts/backtests/<run_id>/report.json`` (+ quantstats tearsheet via
``write_backtest_artifacts`` — guarded, fine when the analysis extra is
absent). Composition-layer module: replay may import backtest, never the
reverse (import-linter contract 3).

Report inputs and where each comes from:

- equity / cash-fraction / gross-exposure / regime series: ``replay_equity``;
- realized trade P&L, win rate, total costs: ``replay_fills`` (the runner
  persists backtest-``FillRecord``-parity ``realized_pnl`` on sells);
- constraint violations: the per-session RiskEngine veto counts persisted in
  ``replay_equity.vetoes``;
- one-way turnover: re-derived from persisted fills + the equity curve with
  the runner's own formula (per fill session: traded value over the PREVIOUS
  session's equity — the decision equity — halved, summed);
- SPY benchmark closes: INJECTED by the caller. The CLI loads them through a
  ``HistoricalDataGateway`` fixed at the run's END date, so the benchmark
  read is bounded exactly like every other replay read (nothing after the
  window is visible); tests inject a synthetic series.

Committee runs persist no equity curve (committee mode is sizing-free by
design) — reporting one raises :class:`ReplayReportError`; report the
portfolio run that replayed its verdicts instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from dynamo.backtest.metrics import BacktestReport, build_report
from dynamo.backtest.tearsheet import write_backtest_artifacts


class ReplayReportError(RuntimeError):
    """The run cannot be reported (unknown, or no persisted equity curve)."""


class UnknownRunError(ReplayReportError):
    """No ``replay_runs`` row exists for the requested run id."""


class ReportSource(Protocol):
    """The ``ReplayRepository`` slice reporting reads (never writes)."""

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def equity_for(self, run_id: str) -> list[dict[str, Any]]: ...
    def fills_for(self, run_id: str) -> list[dict[str, Any]]: ...


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def equity_series(equity_rows: Sequence[Mapping[str, Any]]) -> pd.Series:
    """The persisted curve as a float Series indexed by session date."""
    ordered = sorted(equity_rows, key=lambda r: _as_date(r["as_of"]))
    return pd.Series(
        [float(r["equity"]) for r in ordered],
        index=[_as_date(r["as_of"]) for r in ordered],
        name="equity",
        dtype=float,
    )


def derive_turnover_total(
    equity_rows: Sequence[Mapping[str, Any]],
    fills: Sequence[Mapping[str, Any]],
) -> float:
    """One-way turnover re-derived from persisted rows.

    Mirrors the runner's accumulator: fills on session ``s`` executed the
    target queued on the PREVIOUS session, so their traded value is scaled by
    that previous session's equity (the decision equity), halved (one-way).
    """
    ordered = sorted(equity_rows, key=lambda r: _as_date(r["as_of"]))
    dates = [_as_date(r["as_of"]) for r in ordered]
    prev_equity = {dates[i]: float(ordered[i - 1]["equity"]) for i in range(1, len(dates))}
    traded_by_session: dict[date, float] = {}
    for f in fills:
        session = _as_date(f["session"])
        traded_by_session[session] = traded_by_session.get(session, 0.0) + float(f["trade_value"])
    total = 0.0
    for session in sorted(traded_by_session):
        decision_equity = prev_equity.get(session, 0.0)
        if decision_equity > 0.0:
            total += traded_by_session[session] / decision_equity / 2.0
    return total


def realized_pnls(
    fills: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, float]]:
    """(date_iso, symbol, realized_pnl) rows for closing (sell) fills."""
    return [
        (_as_date(f["session"]).isoformat(), str(f["symbol"]), float(f["realized_pnl"]))
        for f in fills
        if f.get("realized_pnl") is not None
    ]


def total_costs(fills: Sequence[Mapping[str, Any]]) -> float:
    """Total execution costs + commissions across every persisted fill."""
    return float(sum(float(f["cost"]) for f in fills))


def _config_hash(config_json: Mapping[str, Any]) -> str:
    """Reconstruct the deterministic config hash from the persisted config
    (extras like the preflight report are ignored by validation); empty on
    any mismatch — the run id already pins the identity."""
    try:
        from dynamo.replay.runner import ReplayConfig

        return ReplayConfig.model_validate(dict(config_json)).config_hash
    except Exception:
        return ""


def replay_report(
    run_id: str,
    repo: ReportSource,
    artifact_dir: str | Path,
    *,
    benchmark_closes: pd.Series | None = None,
    trials: int = 1,
) -> BacktestReport:
    """Build + persist the report for a finished quant/portfolio replay.

    Writes ``{artifact_dir}/backtests/{run_id}/report.json`` (and the
    quantstats tearsheet beside it when available) and returns the report.
    ``benchmark_closes`` is a SPY close series indexed by date covering the
    run window (see the module docstring for the bounded-read contract).
    """
    run = repo.get_run(run_id)
    if run is None:
        raise UnknownRunError(f"replay run {run_id!r} not found")
    equity_rows = repo.equity_for(run_id)
    if not equity_rows:
        raise ReplayReportError(
            f"replay run {run_id!r} has no persisted equity curve "
            f"(mode={run.get('mode', '?')!r}) — committee runs are sizing-free; "
            "report a quant/portfolio run (re-run it if it predates the "
            "replay_equity table)"
        )
    fills = repo.fills_for(run_id)
    curve = equity_series(equity_rows)
    ordered = sorted(equity_rows, key=lambda r: _as_date(r["as_of"]))
    session_dates = [_as_date(r["as_of"]).isoformat() for r in ordered]
    regimes = [str(r["regime"]) for r in ordered]
    gross = [float(r["gross"]) for r in ordered]
    cash_fractions = [
        float(r["cash"]) / float(r["equity"]) if float(r["equity"]) > 0 else 1.0 for r in ordered
    ]
    benchmark = None
    if benchmark_closes is not None:
        aligned = benchmark_closes.reindex(list(curve.index)).dropna()
        benchmark = aligned if len(aligned) >= 2 else None

    config_json = dict(run.get("config_json") or {})
    label = str(run.get("label", ""))
    strategy_ref = f"replay:{run.get('mode', '')}" + (f":{label}" if label else "")
    report = build_report(
        run_id=run_id,
        strategy_ref=strategy_ref,
        equity=curve,
        session_dates=session_dates,
        regime_by_session=regimes,
        gross_by_session=gross,
        cash_by_session=cash_fractions,
        one_way_turnover_total=derive_turnover_total(equity_rows, fills),
        realized_pnls=realized_pnls(fills),
        trials=trials,
        config_hash=_config_hash(config_json),
        benchmark_closes=benchmark,
        constraint_violations=int(sum(int(r["vetoes"]) for r in ordered)),
    )
    write_backtest_artifacts(report, curve, artifact_dir=artifact_dir, benchmark_closes=benchmark)
    return report
