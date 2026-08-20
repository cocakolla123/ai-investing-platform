"""Investment journal: entry builders + similar-situation ranking (doc 10 §6).

Builders turn domain events (verdict theses, reviews, lessons, incidents) into
``JournalEntry`` rows; ``rank_similar`` merges vector similarity with the
structured filters used during dossier assembly.

THE HASH-FALLBACK ABSTENTION
----------------------------
``dynamo.llm.embeddings.LocalEmbedder`` falls back to a **random Gaussian
seeded from sha256(text)** when the optional ``embeddings`` extra is absent.
Those vectors are deterministic, correctly normalized, the right dimension —
and carry zero semantic signal. Cosine similarity over them is a random
ordering of the journal. All 113 rows in ``journal_entries`` were written
that way.

That is a tolerable degradation for an offline search box. It is not one
here: journal retrieval is ON by default and injects its hits into live
committee prompts under the heading "similar past situations". Randomly
chosen past trades, presented to the committee as precedent, are worse than
no precedent at all — nothing downstream can tell them from real ones.

So retrieval **abstains** instead of degrading. :func:`journal_retrieval_gate`
is that decision, kept pure and here rather than in the composition roots so
the live and replay paths cannot drift apart on it. Both halves are checked:
the QUERY backend, and each candidate's stored ``embedding_version`` — a real
query vector ranked against 113 stored random ones is exactly as meaningless
as the reverse.

The stored-vector half is enforced twice, on purpose.
``JournalRepository.search`` excludes fallback-stamped rows in SQL, because
it is the one seam EVERY caller goes through — live, replay, and whatever is
added next — so a new caller is protected by default rather than by
remembering. :func:`drop_hash_fallback_entries` applies the same rule at the
call site, which is the only place that can count what it removed and log it.

REMEDIATION (the whole fix, two commands)::

    uv sync --extra embeddings
    uv run python -m dynamo.cli.journal_seed --reembed

The first installs sentence-transformers/bge-small (384-d, local, zero API
cost); the second re-embeds in place every entry stamped ``hash-fallback``,
touching only the vector, never the text or the write time. Retrieval turns
itself back on once the entries stop being fallback-stamped — no setting has
to be flipped back. See ``docs/journal_embeddings.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from dynamo.domain.models.learning import JournalEntry, JournalKind, PostTradeReview
from dynamo.domain.models.research import CommitteeVerdict

DEFAULT_TOP_K = 5
_FILTER_BOOST = 0.05  # per matched structured filter, on top of cosine sim

# Mirrors ``dynamo.llm.embeddings.BACKEND_HASH_FALLBACK``. Duplicated rather
# than imported: ``dynamo.learning`` may not import ``dynamo.llm``
# (import-linter contract "core logic does not import adapters"). The equality
# is asserted in ``tests/unit/learning/test_journal_guard.py``.
HASH_FALLBACK_BACKEND = "hash-fallback"

HASH_FALLBACK_REMEDIATION = (
    "uv sync --extra embeddings && uv run python -m dynamo.cli.journal_seed --reembed"
)


@dataclass(frozen=True)
class RetrievalGate:
    """Whether journal vector hits may enter a committee context, and why not."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def journal_retrieval_gate(
    *, query_backend: str, allow_hash_fallback: bool = False
) -> RetrievalGate:
    """Refuse to run journal vector search on the hash fallback.

    ``query_backend`` is ``LocalEmbedder.backend``. When it is the fallback,
    the query vector is noise and every hit is a random journal entry dressed
    up as a similar past situation. Callers must log the returned ``reason``
    at WARNING and skip retrieval entirely — NOT quietly substitute text
    search, which would change what the committee reads without saying so.

    ``allow_hash_fallback`` (settings: ``journal_allow_hash_fallback``) exists
    for the operator who explicitly wants the old behaviour, e.g. to reproduce
    a historical run byte-for-byte. It defaults to off, and it still logs.
    """
    if query_backend != HASH_FALLBACK_BACKEND:
        return RetrievalGate(allowed=True)
    if allow_hash_fallback:
        return RetrievalGate(
            allowed=True,
            reason=(
                f"journal retrieval is running on the {HASH_FALLBACK_BACKEND} "
                "embedder because journal_allow_hash_fallback is ON — the committee "
                "is being shown RANDOM past situations as precedent. Fix: "
                f"{HASH_FALLBACK_REMEDIATION}"
            ),
        )
    return RetrievalGate(
        allowed=False,
        reason=(
            f"journal retrieval ABSTAINING: the embedder resolved to "
            f"{HASH_FALLBACK_BACKEND}, whose vectors are a random Gaussian seeded "
            "from sha256(text) and carry no semantic signal. Injecting those hits "
            "would put randomly chosen past trades in front of the committee as "
            f"'similar past situations'. Fix: {HASH_FALLBACK_REMEDIATION}"
        ),
    )


def drop_hash_fallback_entries(
    hits: Sequence[tuple[JournalEntry, float]],
    versions: Mapping[str, str],
    *,
    allow_hash_fallback: bool = False,
) -> tuple[list[tuple[JournalEntry, float]], int]:
    """(kept hits, number dropped) — drop entries whose STORED vector is fallback.

    The gate above catches a fallback query. This catches the other half, and
    the mixed state a partial re-embed leaves behind: the journal can hold
    real and random vectors at once, and only the stamp tells them apart.
    ``versions`` maps entry id -> ``embedding_version``; an entry missing from
    it is assumed good, because an absent stamp predates the column rather
    than denying it.
    """
    if allow_hash_fallback:
        return list(hits), 0
    kept = [
        (entry, score)
        for entry, score in hits
        if versions.get(entry.id, "") != HASH_FALLBACK_BACKEND
    ]
    return kept, len(hits) - len(kept)


# --- entry builders ---


def thesis_entry(
    verdict: CommitteeVerdict,
    created_at: datetime,
    *,
    strategy: str | None = None,
    regime_at_write: str | None = None,
    sectors: Sequence[str] = (),
) -> JournalEntry:
    text = (
        f"[{verdict.action.value}] {verdict.symbol} (conviction {verdict.conviction:.2f}): "
        f"{verdict.thesis}"
    )
    return JournalEntry(
        id=f"thesis:{verdict.id}",
        kind=JournalKind.THESIS,
        symbols=(verdict.symbol,),
        strategy=strategy,
        regime_at_write=regime_at_write,
        sectors=tuple(sectors),
        text=text,
        refs={"verdict_id": verdict.id},
        created_at=created_at,
    )


def review_entry(
    review: PostTradeReview,
    created_at: datetime,
    *,
    strategy: str | None = None,
    regime_at_write: str | None = None,
    sectors: Sequence[str] = (),
) -> JournalEntry:
    parts = [
        f"Post-trade review {review.symbol}: pnl {review.outcome_pnl_pct:+.2f}% "
        f"over {review.holding_days}d.",
    ]
    if review.what_worked:
        parts.append("Worked: " + "; ".join(review.what_worked))
    if review.what_failed:
        parts.append("Failed: " + "; ".join(review.what_failed))
    if review.failure_primary is not None:
        parts.append(f"Primary failure class: {review.failure_primary.value}.")
    return JournalEntry(
        id=f"review:{review.trade_id}",
        kind=JournalKind.REVIEW,
        symbols=(review.symbol,),
        strategy=strategy,
        regime_at_write=regime_at_write,
        sectors=tuple(sectors),
        text=" ".join(parts),
        refs={
            "trade_id": review.trade_id,
            "decision_context_id": review.decision_context_id,
        },
        created_at=created_at,
    )


def lesson_entries(
    review: PostTradeReview,
    created_at: datetime,
    *,
    strategy: str | None = None,
    regime_at_write: str | None = None,
) -> list[JournalEntry]:
    """One journal entry per extracted lesson (doc 10 §6)."""
    return [
        JournalEntry(
            id=f"lesson:{review.trade_id}:{i}",
            kind=JournalKind.LESSON,
            symbols=(review.symbol,),
            strategy=strategy,
            regime_at_write=regime_at_write,
            text=lesson,
            refs={"trade_id": review.trade_id},
            created_at=created_at,
        )
        for i, lesson in enumerate(review.lessons)
    ]


def incident_entry(
    incident_id: str,
    description: str,
    created_at: datetime,
    *,
    symbols: Sequence[str] = (),
    refs: dict[str, str] | None = None,
) -> JournalEntry:
    return JournalEntry(
        id=f"incident:{incident_id}",
        kind=JournalKind.INCIDENT,
        symbols=tuple(symbols),
        text=description,
        refs=refs or {"incident_id": incident_id},
        created_at=created_at,
    )


# --- similar-situation ranking (doc 10 §6 pre-recommendation consultation) ---


@dataclass(frozen=True)
class JournalFilters:
    kinds: frozenset[JournalKind] = frozenset()
    symbols: frozenset[str] = frozenset()
    strategy: str | None = None
    regime: str | None = None
    sectors: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RankedEntry:
    entry: JournalEntry
    similarity: float  # raw vector similarity
    score: float = field(default=0.0)  # similarity + structured-filter boosts


def _matches_hard_filters(entry: JournalEntry, filters: JournalFilters) -> bool:
    return not (filters.kinds and entry.kind not in filters.kinds)


def _boost(entry: JournalEntry, filters: JournalFilters) -> float:
    boost = 0.0
    if filters.symbols and set(entry.symbols) & filters.symbols:
        boost += _FILTER_BOOST
    if filters.strategy is not None and entry.strategy == filters.strategy:
        boost += _FILTER_BOOST
    if filters.regime is not None and entry.regime_at_write == filters.regime:
        boost += _FILTER_BOOST
    if filters.sectors and set(entry.sectors) & filters.sectors:
        boost += _FILTER_BOOST
    return boost


def rank_similar(
    candidates: Sequence[tuple[JournalEntry, float]],  # (entry, vector similarity)
    filters: JournalFilters | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RankedEntry]:
    """Merge vector similarity with structured filters: kind is a hard filter,
    the rest boost the score; deduped by entry id, best score wins."""
    filters = filters if filters is not None else JournalFilters()
    best: dict[str, RankedEntry] = {}
    for entry, similarity in candidates:
        if not _matches_hard_filters(entry, filters):
            continue
        score = similarity + _boost(entry, filters)
        current = best.get(entry.id)
        if current is None or score > current.score:
            best[entry.id] = RankedEntry(entry=entry, similarity=similarity, score=score)
    ranked = sorted(best.values(), key=lambda r: (-r.score, r.entry.id))
    return ranked[:top_k]
