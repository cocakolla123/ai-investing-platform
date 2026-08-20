"""`replay discovery-report` wiring — exit codes and the universe read.

The wiring assertion that matters is the last one: the command must load
closes for the universe the panel SCANNED, not for the names it flagged.
Loading only flagged names leaves no unflagged baseline, every daily spread
collapses to zero, and the report reads "no detector has an edge" — the one
input mistake that fails silently and looks like a finding.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

import dynamo.cli.replay as replay_cli

SESSIONS = [d.date() for d in pd.bdate_range("2026-03-02", periods=60)]
UNIVERSE = [f"SYM{i:02d}" for i in range(8)]
FLAGGED = UNIVERSE[0]

RUNS: dict[str, dict[str, Any]] = {
    "replay-disc": {
        "run_id": "replay-disc",
        "start_as_of": SESSIONS[0],
        "end_as_of": SESSIONS[-1],
        "config_json": {"universe_tiers": [1, 2, 3], "discovery_mode": "code"},
    },
    "replay-noflags": {
        "run_id": "replay-noflags",
        "start_as_of": SESSIONS[0],
        "end_as_of": SESSIONS[-1],
        "config_json": {"universe_tiers": [1, 3], "discovery_mode": "off"},
    },
}

FLAGS = [
    {
        "run_id": "replay-disc",
        "as_of": day,
        "symbol": FLAGGED,
        "detector": "range_bound",
        "severity": 0.7,
        "promoted": True,
    }
    for day in SESSIONS[:-25]
]


class FakeRepo:
    def __init__(self, engine: Any) -> None:
        pass

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return RUNS.get(run_id)

    def flags_for(self, run_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [f for f in FLAGS if f["run_id"] == run_id]

    def decisions_for(self, run_id: str) -> list[dict[str, Any]]:
        return [{"symbol": FLAGGED}, {"symbol": "SYM07"}]


class FakeGateway:
    """Records what the command asked for — that is the thing under test."""

    requested: list[str] = []
    tiers: tuple[int, ...] | None = None

    def __init__(self, engine: Any, as_of: date) -> None:
        self.as_of = as_of

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]:
        type(self).tiers = tiers
        return list(UNIVERSE)

    def bars(self, symbols: Any, lookback_days: int) -> pd.DataFrame:
        type(self).requested = list(symbols)
        rows = []
        for i, sym in enumerate(symbols):
            for j, day in enumerate(SESSIONS):
                # The flagged name drifts up; everything else wobbles flat.
                drift = 1.015 if sym == FLAGGED else 1.0
                px = 100.0 * (drift**j) * (1.0 + 0.004 * ((i + j) % 7 - 3))
                rows.append({"symbol": sym, "ts": pd.Timestamp(day), "close": px})
        return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _patched(monkeypatch: pytest.MonkeyPatch) -> None:
    import dynamo.persistence.repositories.replay as repo_mod
    import dynamo.replay.gateway as gw_mod

    FakeGateway.requested = []
    FakeGateway.tiers = None
    monkeypatch.setattr(replay_cli, "_engine", lambda url: object())
    monkeypatch.setattr(repo_mod, "ReplayRepository", FakeRepo)
    monkeypatch.setattr(gw_mod, "HistoricalDataGateway", FakeGateway)


def test_unknown_run_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(["discovery-report", "--run-id", "replay-missing"])
    assert rc == 2
    assert "not found" in capsys.readouterr().out


def test_a_run_with_no_flags_exits_1_and_says_why(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = replay_cli.main(["discovery-report", "--run-id", "replay-noflags"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "no flags" in out
    assert "--discovery off" in out, "the likely cause must be named, not just the symptom"


def test_reports_the_gate_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    rc = replay_cli.main(["discovery-report", "--run-id", "replay-disc", "--horizon-sessions", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "range_bound" in out
    assert "distinct promoted: 1" in out
    assert "promoted share:    50.0%" in out
    assert "pass: 30-120/day" in out


def test_it_loads_the_scanned_universe_not_just_the_flagged_names() -> None:
    """The silent-failure guard. One symbol was flagged; the command must
    still request all eight, or there is nothing to difference against."""
    assert replay_cli.main(["discovery-report", "--run-id", "replay-disc"]) == 0
    assert set(UNIVERSE) <= set(FakeGateway.requested)
    assert "SPY" in FakeGateway.requested
    assert FakeGateway.tiers == (1, 2, 3), "tiers come from the run's own config"


def test_the_gateway_is_pinned_past_the_window_so_forward_returns_exist() -> None:
    """Post-hoc evaluation, so the read deliberately runs past the window end.
    Pinning the gateway AT the end date would leave every flag unscored."""
    captured: list[date] = []
    original = FakeGateway.__init__

    def record(self: FakeGateway, engine: Any, as_of: date) -> None:
        captured.append(as_of)
        original(self, engine, as_of)

    FakeGateway.__init__ = record  # type: ignore[method-assign]
    try:
        replay_cli.main(["discovery-report", "--run-id", "replay-disc", "--horizon-sessions", "20"])
    finally:
        FakeGateway.__init__ = original  # type: ignore[method-assign]

    assert captured, "the gateway must be constructed"
    assert captured[0] > SESSIONS[-1] + timedelta(days=20)
