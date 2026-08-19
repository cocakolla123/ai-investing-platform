"""journal_seed CLI: firewall opt-in, embedder gate, idempotent writes.

Every seam (engine, repositories, embedder) is monkeypatched — no database,
no model download.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from dynamo.cli import journal_seed as cli
from dynamo.domain.models.learning import (
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    OutcomeQuality,
)
from dynamo.llm.embeddings import BACKEND_HASH_FALLBACK, DIM

# Captured before any fixture patches it, so the gate test can restore it.
REAL_EMBEDDER_GATE = cli._embedder

LIVE = DecisionExperience(
    id="exp:live:AAPL-2026-05-01",
    source="live",
    created_at=datetime(2026, 6, 1, tzinfo=UTC),
    symbol="AAPL",
    as_of=date(2026, 5, 1),
    action="buy",
    conviction=0.6,
    holding_days=10,
    exit_at=datetime(2026, 5, 15, tzinfo=UTC),
    resolved=True,
    pnl_pct=2.0,
    outcome_quality=OutcomeQuality.WIN,
    decision_quality=DecisionQuality.SOUND,
    forecast_quality=ForecastQuality.ACCURATE,
)
REPLAY = LIVE.model_copy(
    update={
        "id": "exp:replay-r1:KO-2026-05-01",
        "source": "replay",
        "replay_run_id": "r1",
        "symbol": "KO",
    }
)


class FakeJournal:
    def __init__(self) -> None:
        self.inserted: list[tuple[str, str]] = []
        self.updated: list[tuple[str, str]] = []
        self.stale: list[Any] = []

    def insert(self, entry: Any, embedding: Any = None, embedding_version: str = "") -> None:
        self.inserted.append((entry.id, embedding_version))

    def entries_with_embedding_version(self, version: str) -> list[Any]:
        return list(self.stale)

    def update_embedding(self, entry_id: str, embedding: Any, version: str) -> None:
        self.updated.append((entry_id, version))


class FakeExperienceRepo:
    def __init__(self, engine: Any) -> None:
        pass

    def resolved_live(self, since: Any = None) -> list[DecisionExperience]:
        return [LIVE]

    def by_run(self, replay_run_id: str) -> list[DecisionExperience]:
        return [REPLAY] if replay_run_id == "r1" else []


class FakeEmbedder:
    def __init__(self, backend: str = "sentence-transformers") -> None:
        self.backend = backend
        self.calls = 0

    def embed(self, texts: Any) -> list[list[float]]:
        texts = list(texts)
        self.calls += 1
        return [[0.0] * DIM for _ in texts]


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> FakeJournal:
    journal = FakeJournal()
    import orchestration.resources as resources_mod

    import dynamo.persistence.repositories.experience as exp_mod
    import dynamo.persistence.repositories.learning as learn_mod

    monkeypatch.setattr(exp_mod, "ExperienceRepository", FakeExperienceRepo)
    monkeypatch.setattr(learn_mod, "JournalRepository", lambda engine: journal)
    monkeypatch.setattr(
        resources_mod.DynamoResource, "engine", lambda self: object(), raising=False
    )
    monkeypatch.setattr(cli, "_embedder", lambda allow, log: FakeEmbedder())
    return journal


def test_live_only_by_default_replay_rows_need_the_opt_in(wired: FakeJournal) -> None:
    assert cli.main([]) == 0
    seeded = {entry_id for entry_id, _ in wired.inserted}
    assert any("AAPL" in entry_id for entry_id in seeded)
    assert not any("KO" in entry_id for entry_id in seeded)

    wired.inserted.clear()
    assert cli.main(["--replay-run-id", "r1"]) == 0
    seeded = {entry_id for entry_id, _ in wired.inserted}
    assert any("KO" in entry_id for entry_id in seeded)


def test_records_the_embedding_backend_that_ran(wired: FakeJournal) -> None:
    assert cli.main([]) == 0
    assert {version for _, version in wired.inserted} == {"sentence-transformers"}


def test_refuses_the_hash_fallback_unless_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch, wired: FakeJournal, caplog: pytest.LogCaptureFixture
) -> None:
    import dynamo.llm.embeddings as embeddings_mod

    monkeypatch.setattr(cli, "_embedder", REAL_EMBEDDER_GATE)  # restore the real gate
    monkeypatch.setattr(
        embeddings_mod, "LocalEmbedder", lambda *a, **k: FakeEmbedder(BACKEND_HASH_FALLBACK)
    )
    assert cli.main([]) == 2
    assert wired.inserted == []
    assert cli.main(["--allow-hash-embedder"]) == 0
    assert {version for _, version in wired.inserted} == {BACKEND_HASH_FALLBACK}


def test_dry_run_writes_nothing(wired: FakeJournal, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--dry-run"]) == 0
    assert wired.inserted == []
    assert "dry run" in capsys.readouterr().out


def test_rerun_reinserts_the_same_ids(wired: FakeJournal) -> None:
    """Idempotency lives in the deterministic ids: the repository insert is
    conflict-do-nothing, so a second run offers exactly the same rows."""
    assert cli.main([]) == 0
    first = list(wired.inserted)
    wired.inserted.clear()
    assert cli.main([]) == 0
    assert wired.inserted == first


def test_reembed_upgrades_entries_written_under_the_fallback(
    wired: FakeJournal, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Entry:
        id = "review:exp:live:AAPL-2026-05-01"
        text = "Post-trade review AAPL"

    wired.stale = [_Entry()]
    assert cli.main(["--reembed"]) == 0
    assert wired.updated == [("review:exp:live:AAPL-2026-05-01", "sentence-transformers")]
    assert wired.inserted == []
