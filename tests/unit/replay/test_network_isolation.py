"""Network isolation (backtest-campaign Phase 4).

Strongest practical assertion, documented choice: every fixture here is
IN-MEMORY (fake gateway / repos — no Postgres), so the socket layer can be
blocked COMPLETELY — ``socket.socket`` and ``socket.create_connection`` both
raise on any use. A quant replay, a portfolio replay and a committee replay
(injected recording stub) all complete under the total block, proving no code
path opens any connection: no LLM egress, no HF hub fetch, no telemetry.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from dynamo.domain.models.research import Action
from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_committee import RUN_SESSIONS
from tests.unit.replay.test_runner import (
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    fixture_frames,  # noqa: F401  (module-scoped fixture reused here)
)


@pytest.fixture()
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted during a replay test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def test_quant_replay_full_run_under_total_socket_block(
    fixture_frames: Any,  # noqa: F811
    no_network: None,
) -> None:
    sessions, bars, scores = fixture_frames
    window = sessions[-RUN_SESSIONS:]
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=ReplayConfig(
            start=window[0], end=window[-1], label="socket-block", horizon_sessions=3
        ),
    )
    result = runner.run()
    assert result.status == "completed"
    assert result.decisions, "the run actually did work under the block"


def test_portfolio_replay_full_run_under_total_socket_block(
    fixture_frames: Any,  # noqa: F811
    no_network: None,
) -> None:
    from tests.unit.replay.test_runner_portfolio import (
        SOURCE_RUN_ID,
        _verdict,
        _verdict_rows,
    )

    sessions, bars, scores = fixture_frames
    window = sessions[-RUN_SESSIONS:]
    verdicts = [_verdict("LIQ", window[0], Action.BUY, 0.8)]
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=ReplayConfig(
            start=window[0],
            end=window[-1],
            mode="portfolio",
            label="socket-block-portfolio",
            horizon_sessions=3,
            source_run_id=SOURCE_RUN_ID,
        ),
        verdict_source=lambda run_id: _verdict_rows(verdicts),
    )
    result = runner.run()
    assert result.status == "completed"
    assert result.fills, "the recorded verdict produced a fill under the block"


def test_committee_replay_only_egress_is_the_injected_client(
    fixture_frames: Any,  # noqa: F811
    no_network: None,
) -> None:
    """Committee mode under the total socket block: the run completes, and
    EVERY outbound request went through the injected recording stub — the
    Anthropic client object is the single egress seam by construction."""
    import tests.unit.replay.test_committee as tc

    replay, model, run_repo, _exp, _config = tc._replay(fixture_frames)
    result = replay.run()
    assert result.status == "completed"
    assert result.deep_dives
    # All stage traffic is accounted for on the stub: every persisted context
    # row corresponds to a request the stub actually received, and the stub's
    # sync path (complete_json) is booby-trapped and was never hit.
    submitted_ids = {str(r["custom_id"]) for batch in model.submitted for r in batch}
    assert submitted_ids == {r["id"] for r in run_repo.contexts}
    assert len(model.submitted) % 3 == 0  # dossier / bull+bear / arbiter per day
