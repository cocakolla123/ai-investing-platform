"""The LIVE retrieval path honours the hash-fallback abstention.

``_journal_hits_for`` is what actually fills ``## Journal hits`` in a
committee dossier. The pure gate is tested in
``tests/unit/learning/test_journal_guard.py``; this file checks it is wired
in — that a fallback embedder means no vector search runs at all, that the
structured prior-case evidence keeps flowing anyway, and that the explicit
setting turns the old behaviour back on. No database, no network.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

import orchestration.assets.research as research_mod
import pytest
from orchestration.assets.research import _journal_hits_for

from dynamo.domain.models.learning import JournalEntry, JournalKind
from dynamo.learning.journal import HASH_FALLBACK_BACKEND

AS_OF = date(2026, 6, 16)
REAL_BACKEND = "sentence-transformers"


class _Embedder:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.calls = 0

    def embed(self, texts: Any) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2, 0.3] for _ in list(texts)]


class _JournalRepo:
    """The two JournalRepository methods the live path touches."""

    def __init__(self, versions: dict[str, str]) -> None:
        self.versions = versions
        self.searches = 0
        self.allow_flags: list[Any] = []

    def search(self, **kwargs: Any) -> list[tuple[JournalEntry, float]]:
        self.searches += 1
        self.allow_flags.append(kwargs.get("allow_hash_fallback"))
        # Deliberately does NOT apply the repository's own SQL exclusion:
        # this fake exists to prove the caller-side filter still bites if
        # that exclusion ever regresses.
        return [
            (
                JournalEntry(
                    id=entry_id,
                    kind=JournalKind.LESSON,
                    symbols=("ACME",),
                    text=f"lesson {entry_id}",
                    created_at=datetime(2026, 6, 1, tzinfo=UTC),
                ),
                0.9,
            )
            for entry_id in self.versions
        ]

    def embedding_versions(self, entry_ids: Any) -> dict[str, str]:
        return {k: v for k, v in self.versions.items() if k in set(entry_ids)}


class _Settings:
    journal_retrieval_enabled = True
    experience_retrieval_top_k = 3
    journal_allow_hash_fallback = False


class _Dynamo:
    def __init__(self, settings: Any) -> None:
        self._settings = settings

    def settings(self) -> Any:
        return self._settings

    def engine(self) -> Any:
        return object()

    def experience_repo(self) -> Any:
        return object()


PRIOR_CASE = ("exp:live:1", "Prior similar case (evidence, not rule): ACME buy ...")


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = _JournalRepo({"j1": HASH_FALLBACK_BACKEND, "j2": REAL_BACKEND})
    embedder = _Embedder(HASH_FALLBACK_BACKEND)
    settings = _Settings()

    monkeypatch.setattr(
        "dynamo.persistence.repositories.learning.JournalRepository", lambda _e: repo
    )
    monkeypatch.setattr(research_mod, "_journal_embedder", lambda: embedder)
    # Structured retrieval: no embedding involved, must survive the gate.
    monkeypatch.setattr(research_mod, "_prior_case_hits", lambda *a, **k: {"ACME": [PRIOR_CASE]})
    monkeypatch.setattr(research_mod, "_sectors_for", lambda *a, **k: {"ACME": "Tech"})
    monkeypatch.setattr(research_mod, "_regime_at", lambda *a, **k: None)
    return {"repo": repo, "embedder": embedder, "settings": settings}


def _run(wired: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    return _journal_hits_for(_Dynamo(wired["settings"]), ["ACME"], AS_OF)  # type: ignore[arg-type]


def test_a_fallback_embedder_runs_no_vector_search_at_all(
    wired: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Not "search and then filter" — the query vector is noise, so the
    search is not worth issuing and its hits must not reach the prompt."""
    with caplog.at_level(logging.WARNING):
        hits = _run(wired)
    assert wired["repo"].searches == 0
    assert wired["embedder"].calls == 0
    assert [text for _, text in hits["ACME"]] == [PRIOR_CASE[1]]


def test_the_abstention_is_logged_loudly_with_the_remediation(
    wired: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _run(wired)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ABSTAINING" in m for m in messages)
    assert any("uv sync --extra embeddings" in m for m in messages)


def test_structured_prior_cases_still_flow_while_retrieval_abstains(
    wired: dict[str, Any],
) -> None:
    """The abstention removes the noise, not the evidence: prior-similar-case
    lines come from decision_experience via SQL and use no embedding."""
    assert _run(wired)["ACME"] == [PRIOR_CASE]


def test_a_real_backend_still_searches_and_drops_stored_fallback_rows(
    wired: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The other half of the guard: a good query vector against stored random
    ones is just as meaningless, so ``j1`` must not survive."""
    wired["embedder"].backend = REAL_BACKEND
    with caplog.at_level(logging.WARNING):
        hits = _run(wired)
    assert wired["repo"].searches == 1
    ids = [entry_id for entry_id, _ in hits["ACME"]]
    assert "j2" in ids and "j1" not in ids
    assert any("hash-fallback vector" in r.getMessage() for r in caplog.records)


def test_the_setting_restores_the_old_behaviour(
    wired: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Explicit opt-in: search runs, nothing is dropped, and the log still
    says the committee is being shown random precedent."""
    wired["settings"].journal_allow_hash_fallback = True
    with caplog.at_level(logging.WARNING):
        hits = _run(wired)
    assert wired["repo"].searches == 1
    ids = [entry_id for entry_id, _ in hits["ACME"]]
    assert "j1" in ids and "j2" in ids
    assert any("RANDOM" in r.getMessage() for r in caplog.records)


def test_retrieval_disabled_short_circuits_before_the_gate(
    wired: dict[str, Any],
) -> None:
    wired["settings"].journal_retrieval_enabled = False
    assert _journal_hits_for(_Dynamo(wired["settings"]), ["ACME"], AS_OF) == {"ACME": []}  # type: ignore[arg-type]
    assert wired["repo"].searches == 0


def test_the_setting_is_passed_down_to_the_sql_exclusion(
    wired: dict[str, Any],
) -> None:
    """The repository excludes fallback-stamped rows in SQL by default; the
    escape hatch has to reach it, or opting in would still return nothing."""
    wired["embedder"].backend = REAL_BACKEND
    _run(wired)
    assert wired["repo"].allow_flags == [False]

    wired["settings"].journal_allow_hash_fallback = True
    _run(wired)
    assert wired["repo"].allow_flags[-1] is True
