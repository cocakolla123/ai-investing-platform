"""Journal retrieval abstains on hash-fallback embeddings.

``LocalEmbedder`` falls back to a random Gaussian seeded from sha256(text)
when the ``embeddings`` extra is absent, and all 113 rows in
``journal_entries`` were written that way. Retrieval is ON by default and
feeds its hits to live committee prompts as "similar past situations", so the
degradation is not a worse search box — it is fabricated precedent.

These tests pin the abstention and the explicit escape hatch. Pure functions
only: no database, no network, no embedding backend.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dynamo.domain.models.learning import JournalEntry, JournalKind
from dynamo.learning.journal import (
    HASH_FALLBACK_BACKEND,
    HASH_FALLBACK_REMEDIATION,
    drop_hash_fallback_entries,
    journal_retrieval_gate,
)

REAL_BACKEND = "sentence-transformers"


def _entry(entry_id: str) -> JournalEntry:
    return JournalEntry(
        id=entry_id,
        kind=JournalKind.LESSON,
        symbols=("ACME",),
        text=f"lesson {entry_id}",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


# --- the constant this whole guard keys on ---------------------------------


def test_the_mirrored_backend_name_matches_the_adapter() -> None:
    """``dynamo.learning`` may not import ``dynamo.llm`` (import-linter), so
    the backend name is duplicated. If it ever drifts the guard silently
    stops recognising the fallback — which is the failure it exists to
    prevent."""
    from dynamo.llm.embeddings import BACKEND_HASH_FALLBACK

    assert HASH_FALLBACK_BACKEND == BACKEND_HASH_FALLBACK


# --- the query-side gate ---------------------------------------------------


def test_a_real_backend_is_allowed_silently() -> None:
    gate = journal_retrieval_gate(query_backend=REAL_BACKEND)
    assert gate.allowed is True
    assert gate.reason == ""
    assert bool(gate) is True


def test_the_hash_fallback_is_refused_by_default() -> None:
    """The default, and the whole point: no setting had to be changed for
    this to start protecting the committee."""
    gate = journal_retrieval_gate(query_backend=HASH_FALLBACK_BACKEND)
    assert gate.allowed is False
    assert bool(gate) is False


def test_the_refusal_says_what_is_wrong_and_how_to_fix_it() -> None:
    """A silent abstention is its own kind of bug. The reason has to name the
    cause and carry the exact remediation."""
    reason = journal_retrieval_gate(query_backend=HASH_FALLBACK_BACKEND).reason
    assert "ABSTAINING" in reason
    assert "sha256" in reason
    assert HASH_FALLBACK_REMEDIATION in reason
    assert "uv sync --extra embeddings" in reason
    assert "journal_seed --reembed" in reason


def test_the_escape_hatch_allows_it_and_still_shouts() -> None:
    """An operator reproducing a historical run may opt in — but the run log
    must say the committee is seeing random precedent."""
    gate = journal_retrieval_gate(query_backend=HASH_FALLBACK_BACKEND, allow_hash_fallback=True)
    assert gate.allowed is True
    assert "RANDOM" in gate.reason
    assert HASH_FALLBACK_REMEDIATION in gate.reason


# --- the stored-vector side ------------------------------------------------


def test_stored_fallback_entries_are_dropped() -> None:
    """A real query vector ranked against stored random ones is exactly as
    meaningless as the reverse."""
    hits = [(_entry("a"), 0.9), (_entry("b"), 0.8)]
    versions = {"a": HASH_FALLBACK_BACKEND, "b": REAL_BACKEND}
    kept, dropped = drop_hash_fallback_entries(hits, versions)
    assert [e.id for e, _ in kept] == ["b"]
    assert dropped == 1


def test_a_partially_reembedded_journal_keeps_only_the_upgraded_rows() -> None:
    """`--reembed` can be interrupted; both kinds then live in the table at
    once and only the stamp tells them apart."""
    hits = [(_entry(x), 0.5) for x in ("a", "b", "c", "d")]
    versions = {
        "a": REAL_BACKEND,
        "b": HASH_FALLBACK_BACKEND,
        "c": REAL_BACKEND,
        "d": HASH_FALLBACK_BACKEND,
    }
    kept, dropped = drop_hash_fallback_entries(hits, versions)
    assert [e.id for e, _ in kept] == ["a", "c"]
    assert dropped == 2


def test_an_unstamped_entry_is_assumed_good() -> None:
    """An absent ``embedding_version`` predates the column; it does not deny
    the vector. Dropping those would silently blank retrieval for anything
    written before the stamp existed."""
    hits = [(_entry("old"), 0.7)]
    kept, dropped = drop_hash_fallback_entries(hits, {})
    assert [e.id for e, _ in kept] == ["old"]
    assert dropped == 0


def test_the_escape_hatch_keeps_stored_fallback_entries_too() -> None:
    hits = [(_entry("a"), 0.9), (_entry("b"), 0.8)]
    versions = {"a": HASH_FALLBACK_BACKEND, "b": HASH_FALLBACK_BACKEND}
    kept, dropped = drop_hash_fallback_entries(hits, versions, allow_hash_fallback=True)
    assert len(kept) == 2 and dropped == 0


def test_dropping_preserves_order_and_scores() -> None:
    hits = [(_entry("a"), 0.9), (_entry("b"), 0.8), (_entry("c"), 0.7)]
    kept, _ = drop_hash_fallback_entries(hits, {"b": HASH_FALLBACK_BACKEND})
    assert kept == [(hits[0][0], 0.9), (hits[2][0], 0.7)]
