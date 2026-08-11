"""Learning repositories against a real database (doc 10 §2, §6-8).

Integration-marked; skipped unless INTEGRATION_DB_URL points at a Postgres
with the pgvector extension available.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest

from dynamo.domain.models.learning import (
    ImprovementProposal,
    JournalEntry,
    JournalKind,
    ModelVersion,
    ProposalKind,
    ProposalStatus,
    RegistryStatus,
)
from dynamo.domain.models.portfolio import DecisionContext
from dynamo.domain.models.research import Sleeve
from dynamo.learning.promotion import PromotionError, promote

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")
NOW = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]


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


def _dc(i: int) -> DecisionContext:
    return DecisionContext(
        id=f"dc-{i}",
        created_at=NOW + timedelta(minutes=i),
        env="PAPER",
        symbol="AAPL",
        sleeve=Sleeve.LONG_TERM,
        verdict_id=f"v-{i}",
        target_portfolio_id="tp-1",
        payload={"seq": i},
        expected_upside_pct=10.0,
        expected_downside_pct=5.0,
        horizon_days=30,
    )


def test_decision_context_chain_append_and_verify(engine) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text

    from dynamo.persistence.repositories.learning import DecisionContextRepository

    repo = DecisionContextRepository(engine)  # type: ignore[arg-type]
    for i in range(5):
        # append() returns the id callers store on orders — the SAME key
        # get() looks up (returning row_hash here once broke the join spine).
        assert repo.append(_dc(i)) == f"dc-{i}"
    assert repo.verify().ok
    loaded = repo.get("dc-3")
    assert loaded is not None and loaded.payload == {"seq": 3}
    # Round trip through the stored reference: id in, same id out.
    assert repo.get(repo.append(_dc(5))) is not None

    # Tamper with one historical row → the verifier must detect it.
    with engine.begin() as cx:  # type: ignore[attr-defined]
        cx.execute(text("UPDATE decision_context SET symbol = 'TSLA' WHERE id = 'dc-2'"))
    result = repo.verify()
    assert not result.ok
    assert result.index == 2


def test_journal_insert_and_text_search(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.learning import JournalRepository

    repo = JournalRepository(engine)  # type: ignore[arg-type]
    entry = JournalEntry(
        id="j-1",
        kind=JournalKind.LESSON,
        symbols=("AAPL",),
        strategy="earnings-momo",
        text="Semis corrected hard after the cycle peak.",
        created_at=NOW,
    )
    repo.insert(entry, embedding=[0.1] * 384, embedding_version="bge-small-v1")
    repo.insert(
        JournalEntry(
            id="j-2", kind=JournalKind.INCIDENT, text="Provider outage.", created_at=NOW
        )
    )

    hits = repo.search(text_query="semis", kinds=[JournalKind.LESSON])
    assert [e.id for e, _ in hits] == ["j-1"]
    vector_hits = repo.search(embedding=[0.1] * 384, top_k=1)
    assert [e.id for e, _ in vector_hits] == ["j-1"]
    assert vector_hits[0][1] == pytest.approx(1.0, abs=1e-6)  # identical vector

    # Idempotent insert: re-deriving the same entry id is a no-op, not an error.
    repo.insert(entry, embedding=[0.1] * 384, embedding_version="bge-small-v1")
    assert len(repo.search(kinds=[JournalKind.LESSON])) == 1


def test_journal_search_as_of_hides_future_entries(engine) -> None:  # type: ignore[no-untyped-def]
    """Write -> search round trip with the hash-fallback embedder; an entry
    created after ``as_of`` is invisible (leakage guard for replay)."""
    from dynamo.llm.embeddings import LocalEmbedder
    from dynamo.persistence.repositories.learning import JournalRepository

    embedder = LocalEmbedder()
    repo = JournalRepository(engine)  # type: ignore[arg-type]
    past = JournalEntry(
        id="j-past", kind=JournalKind.LESSON, symbols=("AAPL",),
        text="AAPL: cut losers at the stop.", created_at=NOW,
    )
    future = JournalEntry(
        id="j-future", kind=JournalKind.LESSON, symbols=("AAPL",),
        text="AAPL: future hindsight entry.", created_at=NOW + timedelta(days=5),
    )
    for entry in (past, future):
        repo.insert(entry, embedder.embed([entry.text])[0], embedding_version=embedder.backend)

    query = embedder.embed(["AAPL investment lessons"])[0]
    unbounded = repo.search(embedding=query, top_k=5)
    assert {e.id for e, _ in unbounded} == {"j-past", "j-future"}
    bounded = repo.search(embedding=query, top_k=5, as_of=NOW + timedelta(days=1))
    assert {e.id for e, _ in bounded} == {"j-past"}  # the future entry is invisible


def test_proposal_calibration_metrics_roundtrip(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.learning.calibration import CalibrationObservation, compute_calibration
    from dynamo.persistence.repositories.learning import (
        CalibrationRepository,
        MetricsRepository,
        ProposalRepository,
    )

    proposals = ProposalRepository(engine)  # type: ignore[arg-type]
    proposals.insert(
        ImprovementProposal(
            id="p-1", created_at=NOW, kind=ProposalKind.RETRAIN, target="ranker-v1"
        )
    )
    assert [p["id"] for p in proposals.open_proposals()] == ["p-1"]
    proposals.set_status("p-1", ProposalStatus.ACCEPTED)
    assert proposals.open_proposals() == []

    cal = CalibrationRepository(engine)  # type: ignore[arg-type]
    points = compute_calibration(
        [
            CalibrationObservation(
                scope_type="agent", scope_id="bull", conviction=0.8, outcome=True
            )
        ],
        date(2026, 8, 3),
    )
    cal.upsert(points)
    cal.upsert(points)  # idempotent

    metrics = MetricsRepository(engine)  # type: ignore[arg-type]
    metrics.upsert_metric("sharpe_1y", "portfolio", "all", date(2026, 8, 3), 1.1)
    metrics.upsert_metric("sharpe_1y", "portfolio", "all", date(2026, 8, 3), 1.2)  # restate


def test_registry_status_changes_only_via_apply_promotion(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.learning import ModelRegistryRepository

    repo = ModelRegistryRepository(engine)  # type: ignore[arg-type]
    repo.insert_candidate(
        ModelVersion(id="m-1", model_type="ranker", algo="lgbm", created_at=NOW)
    )
    assert repo.status_of("m-1") == RegistryStatus.CANDIDATE

    record = promote("m-1", RegistryStatus.CANDIDATE, RegistryStatus.VALIDATING, "owner", NOW)
    repo.apply_promotion(record)
    assert repo.status_of("m-1") == RegistryStatus.VALIDATING

    # A stale record (wrong from_status) is refused.
    stale = promote("m-1", RegistryStatus.CANDIDATE, RegistryStatus.VALIDATING, "owner", NOW)
    with pytest.raises(ValueError, match="stale"):
        repo.apply_promotion(stale)

    # Illegal transitions cannot even build a record.
    with pytest.raises(PromotionError):
        promote("m-1", RegistryStatus.VALIDATING, RegistryStatus.PRODUCTION, "owner", NOW)

    assert repo.increment_trials("ranker") == 1
    assert repo.increment_trials("ranker") == 2
