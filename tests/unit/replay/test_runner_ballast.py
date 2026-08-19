"""Replay with the idle-cash index ballast: config identity + behavior.

Reuses the fake-gateway harness from ``test_runner``. Covers: ballast
participates in ``config_hash``/``run_id`` (a ballast baseline can never
collide with the cash baseline), a fixture run ends near-fully-invested with
the residual riding SPY, active picks are untouched, and determinism.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_runner import (
    END,
    HISTORY_SESSIONS,
    RUN_SESSIONS,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    make_bars,
    make_scores,
    nyse_sessions,
)

BALLAST = "SPY"


@pytest.fixture(scope="module")
def fixture_frames() -> Any:
    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    return sessions, make_bars(sessions), make_scores(sessions)


def _run_with_ballast(fixture_frames: Any) -> tuple[Any, FakeRunRepo]:
    sessions, bars, scores = fixture_frames
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(
        start=window[0],
        end=window[-1],
        label="ballast",
        horizon_sessions=3,
        ballast_symbol=BALLAST,
    )
    run_repo = FakeRunRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=run_repo,
        config=config,
    )
    return runner.run(), run_repo


@pytest.fixture(scope="module")
def ballast_result(fixture_frames: Any) -> tuple[Any, FakeRunRepo]:
    return _run_with_ballast(fixture_frames)


def test_ballast_participates_in_config_hash_and_run_id() -> None:
    base = ReplayConfig(start=date(2026, 6, 1), end=date(2026, 6, 30))
    ballast = ReplayConfig(
        start=date(2026, 6, 1), end=date(2026, 6, 30), ballast_symbol=BALLAST
    )
    assert base.config_hash != ballast.config_hash
    assert base.run_id != ballast.run_id


def test_run_ends_near_fully_invested_in_ballast(ballast_result: Any) -> None:
    res, _ = ballast_result
    assert res.status == "completed"
    # SPY bought through the ordinary fill path (marketable at next open).
    assert any(f.symbol == BALLAST and f.side == "buy" for f in res.fills)
    # After the first fill day, the residual rides SPY: gross ~ 1 - floor.
    final = res.session_records[-1]
    assert final.gross > 0.80, f"gross {final.gross} should be near 1 - cash floor"
    assert final.cash / res.equity_curve[-1][1] < 0.20


def test_ballast_never_crowds_out_active_picks(ballast_result: Any) -> None:
    res, _ = ballast_result
    # The liquid top-ranked name is still bought (same as the no-ballast run).
    assert any(f.symbol == "LIQ" and f.side == "buy" for f in res.fills)
    # The vetoed illiquid name still never trades.
    assert not any(f.symbol == "ILLQ" for f in res.fills)


def test_ballast_run_is_deterministic(fixture_frames: Any) -> None:
    res_a, _ = _run_with_ballast(fixture_frames)
    res_b, _ = _run_with_ballast(fixture_frames)
    assert res_a.run_id == res_b.run_id
    assert res_a.equity_curve == res_b.equity_curve
    assert res_a.fills == res_b.fills
    assert res_a.decisions == res_b.decisions


def test_ballast_recorded_in_run_config(ballast_result: Any) -> None:
    _, run_repo = ballast_result
    assert run_repo.runs[0]["config"]["ballast_symbol"] == BALLAST
