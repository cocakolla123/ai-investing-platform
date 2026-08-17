"""CLI report/compare subcommands (campaign Phase 5) — repo monkeypatched,
NO network, NO database."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from dynamo.cli import replay as replay_cli

DATES = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)]

RUNS: dict[str, dict[str, Any]] = {
    "replay-aaa": {
        "id": "replay-aaa",
        "label": "1k",
        "mode": "portfolio",
        "start_as_of": DATES[0],
        "end_as_of": DATES[-1],
        "config_json": {"initial_equity": 1_000.0},
        "status": "completed",
    },
    "replay-bbb": {
        "id": "replay-bbb",
        "label": "100k",
        "mode": "portfolio",
        "start_as_of": DATES[0],
        "end_as_of": DATES[-1],
        "config_json": {"initial_equity": 100_000.0},
        "status": "completed",
    },
    "replay-cmt": {
        "id": "replay-cmt",
        "label": "committee",
        "mode": "committee",
        "start_as_of": DATES[0],
        "end_as_of": DATES[-1],
        "config_json": {},
        "status": "completed",
    },
}

EQUITY: dict[str, list[float]] = {
    "replay-aaa": [1_000.0, 1_010.0, 990.0, 1_050.0],  # +5.0%
    "replay-bbb": [100_000.0, 101_000.0, 103_000.0, 102_000.0],  # +2.0%
}

FILLS: dict[str, list[dict[str, Any]]] = {
    "replay-aaa": [
        {
            "run_id": "replay-aaa", "session": DATES[1], "symbol": "AAA",
            "side": "buy", "qty": 5.0, "price": 100.0, "trade_value": 500.0,
            "cost": 1.0, "realized_pnl": None,
        },
        {
            "run_id": "replay-aaa", "session": DATES[2], "symbol": "AAA",
            "side": "sell", "qty": 5.0, "price": 108.0, "trade_value": 540.0,
            "cost": 1.0, "realized_pnl": 38.0,
        },
    ],
    "replay-bbb": [
        {
            "run_id": "replay-bbb", "session": DATES[2], "symbol": "BBB",
            "side": "sell", "qty": 10.0, "price": 50.0, "trade_value": 500.0,
            "cost": 2.0, "realized_pnl": -20.0,
        },
    ],
}


class FakeRepo:
    def __init__(self, engine: Any) -> None:
        pass

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return RUNS.get(run_id)

    def equity_for(self, run_id: str) -> list[dict[str, Any]]:
        run = RUNS.get(run_id)
        if run is None or run_id not in EQUITY:
            return []
        return [
            {
                "run_id": run_id,
                "as_of": d,
                "equity": e,
                "cash": e,
                "gross": 0.0,
                "regime": "risk_on",
                "ladder_rung": "normal",
                "vetoes": 0,
            }
            for d, e in zip(DATES, EQUITY[run_id], strict=True)
        ]

    def fills_for(self, run_id: str) -> list[dict[str, Any]]:
        return list(FILLS.get(run_id, []))


@pytest.fixture(autouse=True)
def _patched(monkeypatch: pytest.MonkeyPatch) -> None:
    import dynamo.persistence.repositories.replay as repo_mod

    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())
    monkeypatch.setattr(repo_mod, "ReplayRepository", FakeRepo)
    monkeypatch.setattr(replay_cli, "_benchmark_closes", lambda engine, run: None)


def test_report_unknown_run_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(["report", "--run-id", "replay-missing"])
    assert rc == 2
    assert "not found" in capsys.readouterr().out


def test_report_committee_run_without_curve_exits_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = replay_cli.main(["report", "--run-id", "replay-cmt"])
    assert rc == 1
    assert "no persisted equity curve" in capsys.readouterr().out


def test_report_writes_artifacts_and_prints_headline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = replay_cli.main(
        ["report", "--run-id", "replay-aaa", "--artifact-dir", str(tmp_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "total return:     +5.00%" in out
    assert "cost drag:        $2.00 (0.20% of initial equity)" in out
    assert "benchmark excess: n/a" in out
    assert (tmp_path / "backtests" / "replay-aaa" / "report.json").exists()


def test_compare_unknown_run_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(
        ["compare", "--run-ids", "replay-aaa", "replay-missing"]
    )
    assert rc == 2
    assert "not found" in capsys.readouterr().out


def test_compare_run_without_curve_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(["compare", "--run-ids", "replay-aaa", "replay-cmt"])
    assert rc == 1
    assert "nothing to compare" in capsys.readouterr().out


def test_compare_table_values_and_order(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(["compare", "--run-ids", "replay-aaa", "replay-bbb"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("replay-")]
    assert len(lines) == 2
    # Rows come back in the requested order.
    assert lines[0].startswith("replay-aaa") and lines[1].startswith("replay-bbb")
    # Hand-checked values: initial equity from config_json, return %, win rate.
    assert "1,000.00" in lines[0] and "+5.00%" in lines[0]
    assert "100.0%" in lines[0]  # one winning sell out of one
    assert "100,000.00" in lines[1] and "+2.00%" in lines[1]
    assert "0.0%" in lines[1]  # one losing sell out of one


def test_compare_png_written_when_matplotlib_present(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    png = tmp_path / "compare.png"
    rc = replay_cli.main(
        ["compare", "--run-ids", "replay-aaa", "replay-bbb", "--png", str(png)]
    )
    assert rc == 0
    assert png.exists() and png.stat().st_size > 0


def test_compare_png_missing_matplotlib_is_a_note_not_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib

    real_import = importlib.import_module

    def block_matplotlib(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", block_matplotlib)
    png = tmp_path / "compare.png"
    rc = replay_cli.main(
        ["compare", "--run-ids", "replay-aaa", "--png", str(png)]
    )
    assert rc == 0
    assert "png skipped" in capsys.readouterr().out
    assert not png.exists()
