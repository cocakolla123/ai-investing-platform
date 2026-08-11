"""Replay firewall test (plan Phase 4): a full replay run against a real
Postgres leaves EVERY live table untouched.

Seeds live rows (orders, trades, decision_context, committee_verdicts,
journal_entries) plus the market data a run needs, executes a real
``ReplayRunner`` over the real ``HistoricalDataGateway`` and repositories,
then asserts the live-table row counts are unchanged while
replay_runs / replay_decisions / decision_experience(source='replay') grew.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest

from tests.unit.replay.test_runner import (
    END,
    HISTORY_SESSIONS,
    RUN_SESSIONS,
    UNIVERSE,
    make_bars,
    make_scores,
    nyse_sessions,
)

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]

LIVE_TABLES = (
    "orders",
    "trades",
    "decision_context",
    "committee_verdicts",
    "journal_entries",
    "quant_scores",
    "watchlist",
    "target_portfolios",
    "risk_verdicts",
    "agent_opinions",
)


@pytest.fixture()
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine, text

    from dynamo.persistence.schema import metadata

    eng = create_engine(INTEGRATION_URL, future=True)
    with eng.begin() as cx:
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    metadata.create_all(eng)
    yield eng
    metadata.drop_all(eng)
    eng.dispose()


def _count(engine, table_name: str) -> int:  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from dynamo.persistence import schema

    table = schema.metadata.tables[table_name]
    with engine.connect() as cx:
        return int(cx.execute(select(func.count()).select_from(table)).scalar() or 0)


def _seed(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.learning import JournalEntry, JournalKind
    from dynamo.domain.models.market import Bar, Instrument, Tier, UniverseMembership
    from dynamo.domain.models.portfolio import DecisionContext
    from dynamo.domain.models.research import Sleeve
    from dynamo.persistence import schema
    from dynamo.persistence.repositories.learning import (
        DecisionContextRepository,
        JournalRepository,
    )
    from dynamo.persistence.repositories.market import MarketDataRepository

    now = datetime(2026, 6, 1, 12, tzinfo=UTC)
    repo = MarketDataRepository(engine)
    sessions = nyse_sessions(END, HISTORY_SESSIONS)

    # Market data the run consumes.
    bars = make_bars(sessions)
    repo.upsert_bars(
        Bar(
            symbol=str(r["symbol"]),
            ts=r["ts"].to_pydatetime() if hasattr(r["ts"], "to_pydatetime") else r["ts"],
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=int(r["volume"]),
            adj_close=float(r["adj_close"]),
            source="tiingo",
        )
        for r in bars.to_dict(orient="records")
    )
    repo.upsert_universe_membership(
        UniverseMembership(
            symbol=s, tier=Tier.CORE, valid_from=date(2026, 1, 2), reason="seed"
        )
        for s in UNIVERSE
    )
    repo.upsert_instruments(
        Instrument(symbol=s, name=s, sector="tech") for s in UNIVERSE
    )
    repo.upsert_rows(
        schema.quant_scores,
        [
            {k: r[k] for k in (
                "symbol", "as_of", "model_version", "expected_excess_return",
                "downside_prob", "vol_forecast", "rank", "confidence",
                "feature_coverage",
            )}
            for r in make_scores(sessions[-RUN_SESSIONS:]).to_dict(orient="records")
        ],
    )

    # Live rows that must survive untouched.
    repo.upsert_rows(
        schema.committee_verdicts,
        [
            {
                "id": "v-live-1", "symbol": "LIQ", "as_of": date(2026, 6, 1),
                "action": "buy", "conviction": 0.7, "disagreement_score": 0.0,
                "payload": {"id": "v-live-1"},
            }
        ],
    )
    DecisionContextRepository(engine).append(
        DecisionContext(
            id="dc-live-1",
            created_at=now,
            env="PAPER",
            symbol="LIQ",
            sleeve=Sleeve.LONG_TERM,
            verdict_id="v-live-1",
            target_portfolio_id="tp-live-1",
            payload={},
            expected_upside_pct=8.0,
            expected_downside_pct=4.0,
            horizon_days=20,
        )
    )
    JournalRepository(engine).insert(
        JournalEntry(
            id="j-live-1",
            kind=JournalKind.THESIS,
            symbols=("LIQ",),
            text="live thesis",
            created_at=now,
        )
    )
    with engine.begin() as cx:
        cx.execute(
            schema.orders.insert(),
            [
                {
                    "client_order_id": "ord-live-1",
                    "decision_context_id": "dc-live-1",
                    "symbol": "LIQ",
                    "side": "buy",
                    "qty": 1.0,
                    "order_type": "marketable_limit",
                    "state": "filled",
                    "created_at": now,
                    "updated_at": now,
                }
            ],
        )
        cx.execute(
            schema.trades.insert(),
            [
                {
                    "client_order_id": "ord-live-1",
                    "symbol": "LIQ",
                    "side": "buy",
                    "qty": 1.0,
                    "fill_price": 100.0,
                    "fees": 0.0,
                    "filled_at": now,
                }
            ],
        )


def test_replay_never_touches_live_tables(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import ReplayConfig, ReplayRunner

    _seed(engine)
    before = {t: _count(engine, t) for t in LIVE_TABLES}
    assert before["orders"] == 1 and before["committee_verdicts"] == 1

    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(
        start=window[0], end=window[-1], label="firewall", horizon_sessions=3
    )
    runner = ReplayRunner(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        experience_repo=ExperienceRepository(engine),
        run_repo=ReplayRepository(engine),
        config=config,
    )
    result = runner.run()
    assert result.status == "completed"
    assert result.decisions, "the run should have produced decisions"
    assert result.experiences, "the run should have resolved experiences"

    # THE firewall: every live table count is unchanged.
    after = {t: _count(engine, t) for t in LIVE_TABLES}
    assert after == before

    # Replay-side rows landed where they belong.
    replay_repo = ReplayRepository(engine)
    run = replay_repo.get_run(config.run_id)
    assert run is not None and run["status"] == "completed"
    assert len(replay_repo.decisions_for(config.run_id)) == len(result.decisions)
    exp_repo = ExperienceRepository(engine)
    assert exp_repo.count(source="replay") == len(result.experiences)
    assert exp_repo.count(source="live") == 0
    by_run = exp_repo.by_run(config.run_id)
    assert {e.id for e in by_run} == {e.id for e in result.experiences}

    # Re-running the same config is idempotent (same deterministic ids).
    runner2 = ReplayRunner(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        experience_repo=ExperienceRepository(engine),
        run_repo=ReplayRepository(engine),
        config=config,
    )
    result2 = runner2.run()
    assert {t: _count(engine, t) for t in LIVE_TABLES} == before
    assert exp_repo.count(source="replay") == len(result.experiences)
    assert result2.decisions == result.decisions


def test_committee_replay_never_touches_live_tables(engine) -> None:  # type: ignore[no-untyped-def]
    """Firewall extension (plan Phase 5): a committee replay (fake model, real
    gateway + repositories) leaves committee_verdicts and every other live
    table untouched; verdicts land in replay_verdicts only."""
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.budget import ReplayBudget
    from dynamo.replay.committee import CommitteeReplay
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import ReplayConfig
    from tests.unit.replay.test_committee import (
        PROMPTS,
        SCHEMAS,
        TIER_MODELS,
        FakeResearchModel,
    )

    _seed(engine)
    before = {t: _count(engine, t) for t in LIVE_TABLES}
    assert before["committee_verdicts"] == 1

    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    window = sessions[-5:]
    config = ReplayConfig(
        start=window[0],
        end=window[-1],
        mode="committee",
        label="committee-firewall",
        horizon_sessions=2,
        committee_deep_dives_per_day=2,
    )
    run_repo = ReplayRepository(engine)
    budget = ReplayBudget(cap_usd=100.0, run_repo=run_repo, run_id=config.run_id)
    replay = CommitteeReplay(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        research_model=FakeResearchModel(),
        budget=budget,
        run_repo=run_repo,
        experience_repo=ExperienceRepository(engine),
        config=config,
        tier_models=TIER_MODELS,
        prompts=PROMPTS,
        schemas=SCHEMAS,
        sleep_fn=lambda s: None,
    )
    result = replay.run()
    assert result.status == "completed"
    assert result.deep_dives, "the run should have produced deep dives"

    # THE firewall: every live table count is unchanged — especially
    # committee_verdicts.
    after = {t: _count(engine, t) for t in LIVE_TABLES}
    assert after == before

    # Replay-side rows landed where they belong.
    verdicts = run_repo.verdicts_for(config.run_id)
    assert len(verdicts) == len(result.deep_dives)
    assert _count(engine, "replay_verdicts") == len(verdicts)
    exp_repo = ExperienceRepository(engine)
    assert {e.id for e in exp_repo.by_run(config.run_id)} == {
        e.id for e in result.experiences
    }
    assert exp_repo.count(source="live") == 0
