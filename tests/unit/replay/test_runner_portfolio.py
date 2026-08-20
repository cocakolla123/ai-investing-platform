"""Verdict-reuse portfolio mode (backtest-campaign Phase 3).

The runner replays a committee run's PERSISTED verdicts keyed (as_of, symbol)
with ZERO LLM anything: the token firewall is structural (no LLM import, no
client seam) and asserted here by making every LLM/synthesis path explode.
Days without verdicts open no new risk but existing holdings are still
managed (horizon exits); verdicts for unknown symbols are ignored gracefully;
three bankroll configs over one source run get three deterministic run ids.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from dynamo.domain.models.research import Action, CommitteeVerdict
from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_runner import (
    RUN_SESSIONS,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    fixture_frames,  # noqa: F401  (module-scoped fixture reused here)
)

SOURCE_RUN_ID = "replay-committee-src"


def _verdict(symbol: str, as_of: date, action: Action, conviction: float) -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"rv-{SOURCE_RUN_ID}-{as_of.isoformat()}-{symbol}",
        symbol=symbol,
        as_of=as_of,
        action=action,
        conviction=conviction,
        thesis="committee deep dive (recorded)",
        bull_case="bull",
        bear_case="bear",
        expected_upside_pct=8.0,
        expected_downside_pct=5.0,
        horizon_days=3,
        prompt_version="committee-v1",
        model_id="claude-test",
    )


def _verdict_rows(verdicts: list[CommitteeVerdict]) -> list[dict[str, Any]]:
    """Rows shaped exactly like ReplayRepository.verdicts_for output."""
    return [
        {
            "id": v.id,
            "run_id": SOURCE_RUN_ID,
            "symbol": v.symbol,
            "as_of": v.as_of,
            "action": v.action.value,
            "conviction": v.conviction,
            "disagreement_score": 0.0,
            "payload": v.model_dump(mode="json"),
        }
        for v in verdicts
    ]


def _config(window: list[date], **overrides: Any) -> ReplayConfig:
    payload: dict[str, Any] = {
        "start": window[0],
        "end": window[-1],
        "mode": "portfolio",
        "label": "portfolio-test",
        "horizon_sessions": 3,
        "source_run_id": SOURCE_RUN_ID,
    }
    payload.update(overrides)
    return ReplayConfig(**payload)


def _run(
    fixture: Any,
    verdicts: list[CommitteeVerdict],
    monkeypatch: pytest.MonkeyPatch | None = None,
    **config_overrides: Any,
) -> tuple[Any, FakeExperienceRepo, FakeRunRepo]:
    sessions, bars, scores = fixture
    window = sessions[-RUN_SESSIONS:]
    config = _config(window, **config_overrides)
    exp_repo = FakeExperienceRepo()
    run_repo = FakeRunRepo()

    calls: list[str] = []

    def verdict_source(run_id: str) -> list[dict[str, Any]]:
        calls.append(run_id)
        return _verdict_rows(verdicts)

    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=exp_repo,
        run_repo=run_repo,
        config=config,
        verdict_source=verdict_source,
    )
    result = runner.run()
    assert calls == [SOURCE_RUN_ID]  # preloaded ONCE at run start
    return result, exp_repo, run_repo


def _window(fixture: Any) -> list[date]:
    sessions, _, _ = fixture
    return sessions[-RUN_SESSIONS:]


# -- mode guard --------------------------------------------------------------


def test_portfolio_mode_requires_source_run_id() -> None:
    with pytest.raises(ValueError, match="source_run_id"):
        ReplayRunner(
            gateway_factory=lambda d: None,  # type: ignore[arg-type,return-value]
            experience_repo=FakeExperienceRepo(),
            run_repo=FakeRunRepo(),
            config=ReplayConfig(start=date(2026, 6, 1), end=date(2026, 6, 5), mode="portfolio"),
            verdict_source=lambda run_id: [],
        )


def test_portfolio_mode_requires_verdict_source() -> None:
    with pytest.raises(ValueError, match="verdict_source"):
        ReplayRunner(
            gateway_factory=lambda d: None,  # type: ignore[arg-type,return-value]
            experience_repo=FakeExperienceRepo(),
            run_repo=FakeRunRepo(),
            config=ReplayConfig(
                start=date(2026, 6, 1),
                end=date(2026, 6, 5),
                mode="portfolio",
                source_run_id=SOURCE_RUN_ID,
            ),
        )


# -- token firewall ----------------------------------------------------------


def test_portfolio_run_never_touches_llm_or_synthesis(
    fixture_frames: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The firewall test: a full portfolio run on fixtures with (a) the
    AnthropicClient constructor rigged to explode, (b) the quant-mode verdict
    synthesizer rigged to explode, and (c) an exploding bundle factory —
    completes without touching any of them."""
    import dynamo.llm.anthropic_client as llm_mod
    import dynamo.replay.runner as runner_mod

    def explode(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("portfolio mode must never construct an LLM client")

    monkeypatch.setattr(llm_mod.AnthropicClient, "__init__", explode)
    monkeypatch.setattr(
        runner_mod,
        "synthesize_verdicts",
        lambda *a, **k: pytest.fail("portfolio mode must not synthesize verdicts"),
    )

    window = _window(fixture_frames)
    verdicts = [_verdict("LIQ", window[0], Action.BUY, 0.9)]
    _sessions, bars, scores = fixture_frames
    exp_repo = FakeExperienceRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=exp_repo,
        run_repo=FakeRunRepo(),
        config=_config(window),
        bundle_factory=lambda version: pytest.fail("no bundle load without a version"),
        verdict_source=lambda run_id: _verdict_rows(verdicts),
    )
    result = runner.run()
    assert result.status == "completed"
    assert any(f.symbol == "LIQ" and f.side == "buy" for f in result.fills)


# -- verdict keying ----------------------------------------------------------


def test_verdicts_apply_on_their_day_and_fill_next_open(fixture_frames: Any) -> None:  # noqa: F811
    window = _window(fixture_frames)
    verdicts = [_verdict("LIQ", window[0], Action.BUY, 0.9)]
    result, _, _ = _run(fixture_frames, verdicts)
    assert result.status == "completed"
    buys = [f for f in result.fills if f.symbol == "LIQ" and f.side == "buy"]
    assert buys and buys[0].session == window[1]  # decision day 0, NEXT open fill
    # The decision row on day 0 carries the reused committee verdict payload.
    day0 = [d for d in result.decisions if d["as_of"] == window[0] and d["symbol"] == "LIQ"]
    assert day0 and day0[0]["payload"]["verdict"]["prompt_version"] == "committee-v1"
    assert day0[0]["payload"]["verdict"]["model_id"] == "claude-test"


def test_missing_days_manage_holdings_only(fixture_frames: Any) -> None:  # noqa: F811
    """One BUY on day 0, NOTHING after: no new entries on verdict-less days,
    and the horizon exit still fires portfolio-side, resolving an experience."""
    window = _window(fixture_frames)
    verdicts = [_verdict("LIQ", window[0], Action.BUY, 0.9)]
    result, exp_repo, _ = _run(fixture_frames, verdicts)
    buys = [f for f in result.fills if f.side == "buy"]
    assert {f.symbol for f in buys} == {"LIQ"}
    sells = [f for f in result.fills if f.symbol == "LIQ" and f.side == "sell"]
    # horizon_sessions=3: entry fills at window[1]; small weight-pinning trims
    # may occur earlier, but the FULL exit (the last sell) fires only once the
    # lot is 3 sessions old, filling the following open.
    assert sells and sells[-1].session >= window[4]
    assert exp_repo.inserted and exp_repo.inserted[0].symbol == "LIQ"
    assert exp_repo.inserted[0].source == "replay"
    # Bankroll context flows into the replay experience (campaign Phase 3).
    assert exp_repo.inserted[0].equity_at_decision == pytest.approx(100_000.0)
    assert exp_repo.inserted[0].position_notional == pytest.approx(
        buys[0].qty * buys[0].price, abs=0.51
    )


def test_unknown_symbol_verdict_ignored_gracefully(fixture_frames: Any) -> None:  # noqa: F811
    window = _window(fixture_frames)
    verdicts = [
        _verdict("LIQ", window[0], Action.BUY, 0.9),
        _verdict("ZZZ", window[0], Action.BUY, 0.95),  # not in universe, no bars
    ]
    result, _, _ = _run(fixture_frames, verdicts)
    assert result.status == "completed"
    assert not any(f.symbol == "ZZZ" for f in result.fills)
    assert not any(d["symbol"] == "ZZZ" for d in result.decisions)
    assert any(f.symbol == "LIQ" and f.side == "buy" for f in result.fills)


def test_same_config_twice_is_bit_identical(fixture_frames: Any) -> None:  # noqa: F811
    window = _window(fixture_frames)
    verdicts = [
        _verdict("LIQ", window[0], Action.BUY, 0.9),
        _verdict("AAA", window[2], Action.BUY, 0.7),
    ]
    res_a, exp_a, _ = _run(fixture_frames, verdicts)
    res_b, exp_b, _ = _run(fixture_frames, verdicts)
    assert res_a.run_id == res_b.run_id
    assert res_a.decisions == res_b.decisions
    assert res_a.equity_curve == res_b.equity_curve
    assert res_a.fills == res_b.fills
    assert [e.model_dump() for e in exp_a.inserted] == [e.model_dump() for e in exp_b.inserted]


# -- run-id determinism across bankrolls -------------------------------------


def test_three_bankrolls_same_source_three_distinct_run_ids() -> None:
    base: dict[str, Any] = {
        "start": date(2026, 2, 2),
        "end": date(2026, 7, 31),
        "mode": "portfolio",
        "label": "campaign",
        "source_run_id": SOURCE_RUN_ID,
    }
    configs = [
        ReplayConfig(**base, initial_equity=equity) for equity in (1_000.0, 10_000.0, 100_000.0)
    ]
    assert len({c.run_id for c in configs}) == 3
    # Same config twice -> same run id (idempotent re-runs).
    again = ReplayConfig(**base, initial_equity=1_000.0)
    assert again.run_id == configs[0].run_id
    # Realism knobs are part of the hash too: the whole-shares sensitivity
    # run can never collide with the fractional run.
    sensitivity = ReplayConfig(**base, initial_equity=1_000.0, whole_shares=True)
    assert sensitivity.run_id != configs[0].run_id
    with_commission = ReplayConfig(**base, initial_equity=1_000.0, commission_per_order=1.0)
    with_min = ReplayConfig(**base, initial_equity=1_000.0, min_order_notional=25.0)
    assert len({configs[0].run_id, with_commission.run_id, with_min.run_id}) == 3
