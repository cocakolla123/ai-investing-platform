"""replay_report end-to-end on synthetic persisted rows (campaign Phase 5).

Headline numbers are hand-verified on a tiny curve; artifacts land in
``{artifact_dir}/backtests/{run_id}/report.json`` (tearsheet is optional —
quantstats absent is a logged skip, never an error).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dynamo.replay.report import (
    ReplayReportError,
    UnknownRunError,
    derive_turnover_total,
    equity_series,
    realized_pnls,
    replay_report,
    total_costs,
)

RUN_ID = "replay-feedface0001"
DATES = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)]
EQUITY = [100_000.0, 110_000.0, 99_000.0, 108_900.0]


def _equity_rows() -> list[dict[str, Any]]:
    return [
        {
            "run_id": RUN_ID,
            "as_of": d,
            "equity": e,
            "cash": e * 0.4,
            "gross": 0.6,
            "regime": "risk_on" if i < 2 else "risk_off",
            "ladder_rung": "normal",
            "vetoes": i,  # 0+1+2+3 = 6 total
        }
        for i, (d, e) in enumerate(zip(DATES, EQUITY, strict=True))
    ]


def _fills() -> list[dict[str, Any]]:
    return [
        {
            "run_id": RUN_ID, "session": DATES[1], "symbol": "AAA", "side": "buy",
            "qty": 100.0, "price": 50.0, "trade_value": 5_000.0, "cost": 5.0,
            "realized_pnl": None,
        },
        {
            "run_id": RUN_ID, "session": DATES[2], "symbol": "AAA", "side": "sell",
            "qty": 100.0, "price": 55.0, "trade_value": 5_500.0, "cost": 5.5,
            "realized_pnl": 489.5,
        },
        {
            "run_id": RUN_ID, "session": DATES[3], "symbol": "BBB", "side": "sell",
            "qty": 10.0, "price": 20.0, "trade_value": 200.0, "cost": 0.5,
            "realized_pnl": -12.0,
        },
    ]


class FakeReportRepo:
    def __init__(
        self,
        run: dict[str, Any] | None,
        equity_rows: list[dict[str, Any]],
        fills: list[dict[str, Any]],
    ) -> None:
        self._run = run
        self._equity = equity_rows
        self._fills = fills

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._run

    def equity_for(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._equity)

    def fills_for(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._fills)


def _run_row(mode: str = "quant") -> dict[str, Any]:
    return {
        "id": RUN_ID,
        "label": "synthetic",
        "mode": mode,
        "start_as_of": DATES[0],
        "end_as_of": DATES[-1],
        "config_json": {"initial_equity": 100_000.0},
        "status": "completed",
    }


def test_report_headline_numbers_hand_verified(tmp_path: Path) -> None:
    repo = FakeReportRepo(_run_row(), _equity_rows(), _fills())
    spy = pd.Series([500.0, 505.0, 495.0, 510.0], index=DATES)
    report = replay_report(RUN_ID, repo, tmp_path, benchmark_closes=spy)

    m = report.metrics
    assert m.n_sessions == 4
    assert m.total_return == pytest.approx(0.089)  # 100k -> 108.9k
    # Peak 110k -> trough 99k: exactly -10%.
    assert m.max_drawdown == pytest.approx(-0.10)
    assert report.constraint_violations == 6  # 0+1+2+3 persisted veto counts
    assert report.strategy_ref == "replay:quant:synthetic"
    assert report.window_start == "2026-06-01"
    assert report.window_end == "2026-06-04"

    # Trades: one win (+489.5), one loss (-12.0).
    assert report.trades.n_closed == 2
    assert report.trades.win_rate == pytest.approx(0.5)
    assert report.trades.avg_win == pytest.approx(489.5)
    assert report.trades.avg_loss == pytest.approx(-12.0)

    # Turnover: traded value over the PREVIOUS session's equity, halved.
    expected_turnover = (
        5_000.0 / 100_000.0 / 2.0  # buy on d2, decision equity d1
        + 5_500.0 / 110_000.0 / 2.0  # sell on d3, decision equity d2
        + 200.0 / 99_000.0 / 2.0  # sell on d4, decision equity d3
    )
    assert report.behavior.one_way_turnover_total == pytest.approx(expected_turnover)
    assert report.behavior.avg_gross_exposure == pytest.approx(0.6)
    assert report.behavior.avg_cash_fraction == pytest.approx(0.4)

    # Benchmark: SPY 500 -> 510 = +2%; excess = 8.9% - 2%.
    assert report.benchmark is not None
    assert report.benchmark.benchmark_total_return == pytest.approx(0.02)
    assert report.benchmark.excess_total_return == pytest.approx(0.089 - 0.02)

    # Regimes split as persisted.
    assert set(report.per_regime) == {"risk_on", "risk_off"}

    # report.json landed and round-trips.
    report_path = tmp_path / "backtests" / RUN_ID / "report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["run_id"] == RUN_ID
    assert payload["metrics"]["total_return"] == pytest.approx(0.089)


def test_unknown_run_raises(tmp_path: Path) -> None:
    repo = FakeReportRepo(None, [], [])
    with pytest.raises(UnknownRunError):
        replay_report(RUN_ID, repo, tmp_path)


def test_committee_run_without_curve_raises(tmp_path: Path) -> None:
    repo = FakeReportRepo(_run_row(mode="committee"), [], [])
    with pytest.raises(ReplayReportError, match="no persisted equity curve"):
        replay_report(RUN_ID, repo, tmp_path)
    assert not (tmp_path / "backtests").exists()


def test_helpers_on_string_dates() -> None:
    """DB drivers may hand back ISO strings — helpers normalize them."""
    rows = [dict(r, as_of=r["as_of"].isoformat()) for r in _equity_rows()]
    fills = [dict(f, session=f["session"].isoformat()) for f in _fills()]
    curve = equity_series(rows)
    assert list(curve.index) == DATES
    assert float(curve.iloc[-1]) == 108_900.0
    assert derive_turnover_total(rows, fills) > 0.0
    assert realized_pnls(fills) == [
        ("2026-06-03", "AAA", 489.5),
        ("2026-06-04", "BBB", -12.0),
    ]
    assert total_costs(fills) == pytest.approx(11.0)


def test_short_benchmark_is_dropped(tmp_path: Path) -> None:
    """A benchmark overlapping fewer than 2 sessions is omitted, not crashed on."""
    repo = FakeReportRepo(_run_row(), _equity_rows(), _fills())
    spy = pd.Series([500.0], index=[DATES[0]])
    report = replay_report(RUN_ID, repo, tmp_path, benchmark_closes=spy)
    assert report.benchmark is None
