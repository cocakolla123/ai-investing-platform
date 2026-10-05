"""Backfill plan for journal metadata (docs/journal_metadata.md).

Pure: given the entries' current metadata and the sources, decide what each
NULL or empty field can be filled with, using only facts knowable at the
entry's ``created_at``. Existing values are never overwritten, so applying
the plan twice changes nothing. The DB wrapper is
``scripts/journal_backfill_metadata.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from dynamo.learning.journal_meta import entry_sectors, regime_on_or_before

FIELDS = ("strategy", "regime_at_write", "sectors")


@dataclass(frozen=True)
class EntryMeta:
    id: str
    kind: str
    symbols: tuple[str, ...]
    created_at: datetime
    refs: Mapping[str, Any]
    strategy: str | None
    regime_at_write: str | None
    sectors: tuple[str, ...]


@dataclass(frozen=True)
class SleeveSource:
    """A record carrying a sleeve, with when it came into existence."""

    sleeve: str | None
    created_at: datetime


@dataclass(frozen=True)
class Sources:
    decision_contexts: Mapping[str, SleeveSource] = field(default_factory=dict)
    decision_context_by_verdict: Mapping[str, SleeveSource] = field(default_factory=dict)
    experiences: Mapping[str, SleeveSource] = field(default_factory=dict)
    live_regimes: Sequence[tuple[date, str]] = ()
    replay_regimes: Sequence[tuple[date, str]] = ()
    sector_by_symbol: Mapping[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Fill:
    id: str
    strategy: str | None  # None = leave as is
    regime_at_write: str | None
    sectors: tuple[str, ...] | None


def _knowable(src: SleeveSource | None, at: datetime) -> str | None:
    if src is None or src.sleeve is None or src.created_at > at:
        return None
    return src.sleeve


def _strategy(entry: EntryMeta, s: Sources) -> tuple[str | None, str]:
    refs = entry.refs
    for key, table in (
        ("decision_context_id", s.decision_contexts),
        ("experience_id", s.experiences),
    ):
        ref = refs.get(key)
        if ref and (sleeve := _knowable(table.get(str(ref)), entry.created_at)):
            return sleeve, f"from {key}"
    verdict = refs.get("verdict_id")
    if verdict:
        dc = s.decision_context_by_verdict.get(str(verdict))
        if (sleeve := _knowable(dc, entry.created_at)) is not None:
            return sleeve, "from decision_context of verdict"
        if dc is not None and dc.created_at > entry.created_at:
            return None, "sleeve assigned after the entry was written"
        return None, "no decision context for the verdict"
    if refs.get("experience_id"):
        return None, "linked experience has no sleeve"
    return None, "no linked trade record"


def plan(entries: Sequence[EntryMeta], s: Sources) -> tuple[list[Fill], Counter[str]]:
    """-> (fills for entries with something to fill, reasons a field stays NULL)."""
    fills: list[Fill] = []
    unfilled: Counter[str] = Counter()
    for e in entries:
        strategy = regime = None
        sectors: tuple[str, ...] | None = None
        if e.strategy is None:
            strategy, why = _strategy(e, s)
            if strategy is None:
                unfilled[f"{e.kind}.strategy: {why}"] += 1
        if e.regime_at_write is None:
            replay = e.refs.get("source") == "replay" or bool(e.refs.get("replay_run_id"))
            history = s.replay_regimes if replay else s.live_regimes
            regime = regime_on_or_before(history, e.created_at.date())
            if regime is None:
                src = "replay" if replay else "regime_states"
                unfilled[f"{e.kind}.regime_at_write: no {src} session on or before write date"] += 1
        if not e.sectors:
            found = entry_sectors(e.symbols, s.sector_by_symbol)
            if found:
                sectors = found
            else:
                unfilled[f"{e.kind}.sectors: no instruments.sector for {list(e.symbols)}"] += 1
        if strategy is not None or regime is not None or sectors is not None:
            fills.append(Fill(e.id, strategy, regime, sectors))
    return fills, unfilled


def counts(entries: Sequence[EntryMeta]) -> dict[str, dict[str, int]]:
    """Per kind: total and how many have each field filled."""
    out: dict[str, dict[str, int]] = {}
    for e in entries:
        row = out.setdefault(e.kind, {"total": 0, **dict.fromkeys(FIELDS, 0)})
        row["total"] += 1
        row["strategy"] += e.strategy is not None
        row["regime_at_write"] += e.regime_at_write is not None
        row["sectors"] += bool(e.sectors)
    return dict(sorted(out.items()))


def applied(entries: Sequence[EntryMeta], fills: Sequence[Fill]) -> list[EntryMeta]:
    """The entries as they will look after ``fills`` (for the after-counts)."""
    by_id = {f.id: f for f in fills}
    out = []
    for e in entries:
        f = by_id.get(e.id)
        if f is None:
            out.append(e)
            continue
        out.append(
            EntryMeta(
                id=e.id,
                kind=e.kind,
                symbols=e.symbols,
                created_at=e.created_at,
                refs=e.refs,
                strategy=e.strategy if e.strategy is not None else f.strategy,
                regime_at_write=e.regime_at_write
                if e.regime_at_write is not None
                else f.regime_at_write,
                sectors=e.sectors or (f.sectors or ()),
            )
        )
    return out
