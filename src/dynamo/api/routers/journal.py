"""Journal search endpoint (JournalRepository under the hood)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from dynamo.api.deps import DepsDep
from dynamo.api.envelopes import Page
from dynamo.domain.models.learning import JournalEntry, JournalKind

router = APIRouter(prefix="/journal", tags=["journal"])


@router.get("/search")
def search_journal(
    deps: DepsDep,
    q: str | None = None,
    kind: str | None = None,
    symbol: str | None = None,
    limit: int = 50,
) -> Page[JournalEntry]:
    if kind is not None:
        try:
            JournalKind(kind)
        except ValueError as exc:
            valid = ", ".join(k.value for k in JournalKind)
            raise HTTPException(400, f"unknown journal kind {kind!r}; valid: {valid}") from exc
    entries = deps.journal.search(q, kind, symbol.upper() if symbol else None, min(limit, 500))
    return Page.of(entries)
