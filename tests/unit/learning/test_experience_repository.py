"""ExperienceRepository against a real Postgres (plan Phase 1).

Integration-marked (the live-insert idempotency rides on the partial unique
index ``decision_context_id WHERE source='live'`` — Postgres-only); skipped
unless INTEGRATION_DB_URL is set, matching the Phase 0 pattern.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import (
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    OutcomeQuality,
)

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")
NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]


@pytest.fixture()
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine

    from dynamo.persistence.schema import decision_experience, metadata

    eng = create_engine(INTEGRATION_URL, future=True)
    metadata.create_all(eng, tables=[decision_experience])
    yield eng
    metadata.drop_all(eng, tables=[decision_experience])
    eng.dispose()


def _exp(i: int, *, source: str = "live", resolved: bool = True, **overrides: object):
    base: dict[str, object] = {
        "id": f"exp:{source}:dc-{i}",
        "decision_context_id": f"dc-{i}",
        "trade_id": f"AAPL-2026-08-{i:02d}",
        "source": source,
        "created_at": NOW,
        "symbol": "AAPL",
        "as_of": date(2026, 8, i),
        "sleeve": "long_term",
        "regime": "neutral",
        "resolved": resolved,
        "pnl_pct": 3.0,
        "forecast_quality": ForecastQuality.ACCURATE,
        "decision_quality": DecisionQuality.SOUND,
        "outcome_quality": OutcomeQuality.WIN,
        "label": "win",
        # bankroll context (campaign Phase 3; migration 0015)
        "equity_at_decision": 50_000.0,
        "position_notional": 2_500.0,
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


def test_live_insert_is_idempotent_per_decision_context(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)  # type: ignore[arg-type]
    assert repo.insert(_exp(1)) is True
    # Same decision context again (even under a different row id): no-op.
    assert repo.insert(_exp(1)) is False
    assert repo.insert(_exp(1, id="exp:live:dc-1-rerun")) is False
    assert repo.count(source="live") == 1
    assert repo.live_decision_context_ids() == frozenset({"dc-1"})


def test_replay_rows_do_not_collide_with_live(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)  # type: ignore[arg-type]
    assert repo.insert(_exp(1)) is True
    # Replay may revisit the SAME decision context many times across runs —
    # the partial unique index must not block it.
    for run in ("run-a", "run-b"):
        assert (
            repo.insert(
                _exp(1, source="replay", id=f"exp:{run}:AAPL", replay_run_id=run)
            )
            is True
        )
    # But a replay re-run with the same deterministic id is idempotent.
    assert (
        repo.insert(_exp(1, source="replay", id="exp:run-a:AAPL", replay_run_id="run-a"))
        is False
    )
    assert repo.count() == 3
    assert repo.count(source="live") == 1
    assert repo.count(source="replay") == 2


def test_live_readers_filter_source_and_resolution(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)  # type: ignore[arg-type]
    repo.insert(_exp(1))
    repo.insert(_exp(2, resolved=False, pnl_pct=None, label=None))
    repo.insert(_exp(3, source="replay", id="exp:run-a:3", replay_run_id="run-a"))

    resolved = repo.resolved_live()
    assert [e.decision_context_id for e in resolved] == ["dc-1"]
    # Round trip preserves enums and values.
    assert resolved[0].outcome_quality is OutcomeQuality.WIN
    assert resolved[0].pnl_pct == 3.0
    # Bankroll context round-trips (campaign Phase 3; migration 0015).
    assert resolved[0].equity_at_decision == 50_000.0
    assert resolved[0].position_notional == 2_500.0

    assert [e.decision_context_id for e in repo.unresolved_live()] == ["dc-2"]
    assert repo.resolved_live(since=date(2026, 8, 2)) == []  # dc-1 is as_of 08-01

    by_run = repo.by_run("run-a")
    assert [e.id for e in by_run] == ["exp:run-a:3"]
    assert by_run[0].source == "replay"


def test_update_calibration_round_trip(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)  # type: ignore[arg-type]
    repo.insert(_exp(1))
    resolved = repo.resolved_live()
    assert resolved[0].calibrated_probability is None
    assert resolved[0].calibration_scope_used is None

    repo.update_calibration("exp:live:dc-1", 0.62, "sleeve=long_term|regime=neutral")
    updated = repo.resolved_live()[0]
    assert updated.calibrated_probability == 0.62
    assert updated.calibration_scope_used == "sleeve=long_term|regime=neutral"


def test_similar_hides_future_and_ranks_by_boosts(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)  # type: ignore[arg-type]
    query_as_of = date(2026, 8, 10)
    # Same day as the query -> invisible (strictly as_of < query).
    repo.insert(_exp(10, symbol="NVDA"))
    # Unresolved and replay rows -> invisible regardless of date.
    repo.insert(_exp(2, resolved=False, pnl_pct=None, label=None))
    repo.insert(_exp(3, source="replay", id="exp:run-a:3", replay_run_id="run-a"))
    # Prior live cases: same sector+regime beats a fresher mismatch.
    repo.insert(_exp(4, symbol="AMD", sector="tech", regime="risk_off"))
    repo.insert(_exp(5, symbol="XOM", sector="energy", regime="risk_on"))

    cases = repo.similar(
        symbol="NVDA", as_of=query_as_of, sector="tech", regime="risk_off", limit=5
    )
    assert [c.decision_context_id for c in cases] == ["dc-4", "dc-5"]
    assert all(c.source == "live" and c.resolved for c in cases)
    assert all(c.as_of < query_as_of for c in cases)

    # Same-symbol boost outranks everything else.
    repo.insert(_exp(6, symbol="NVDA", sector="tech", regime="risk_off"))
    cases = repo.similar(
        symbol="NVDA", as_of=query_as_of, sector="tech", regime="risk_off", limit=2
    )
    assert [c.symbol for c in cases] == ["NVDA", "AMD"]
