"""Investment journal: entry builders + similar-situation ranking (doc 10 §6).

Builders turn domain events (verdict theses, reviews, lessons, incidents) into
``JournalEntry`` rows; ``rank_similar`` merges vector similarity with the
structured filters used during dossier assembly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from dynamo.domain.models.learning import JournalEntry, JournalKind, PostTradeReview
from dynamo.domain.models.research import CommitteeVerdict

DEFAULT_TOP_K = 5
_FILTER_BOOST = 0.05  # per matched structured filter, on top of cosine sim


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
