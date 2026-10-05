"""Journal entry metadata derivation (docs/journal_metadata.md).

Pure helpers shared by every write path and by the backfill, so a field
means the same thing however the entry was written:

- ``regime_on_or_before``: the regime on the write date, from a per-session
  history, never using a session dated after it;
- ``entry_sectors``: today's ``instruments.sector`` for the entry's symbols.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, Sequence
from datetime import date

#: Sector values that mean "not classified"; never written as a sector.
_NO_SECTOR = frozenset({"", "unknown"})


def regime_on_or_before(history: Sequence[tuple[date, str]], day: date) -> str | None:
    """The last regime in ``history`` dated on or before ``day``.

    ``history`` is (session date, regime) sorted ascending by date. A session
    after ``day`` is never used: if every session is later, the answer is
    None (unknown), not the nearest future value.
    """
    dates = [d for d, _ in history]
    i = bisect.bisect_right(dates, day)
    if i == 0:
        return None
    regime = history[i - 1][1]
    return regime or None


def entry_sectors(
    symbols: Iterable[str], sector_by_symbol: Mapping[str, str | None]
) -> tuple[str, ...]:
    """Distinct, sorted sectors of ``symbols``; unclassified symbols are skipped."""
    found = {sector_by_symbol.get(s) for s in symbols}
    return tuple(sorted(s for s in found if s is not None and s.strip() not in _NO_SECTOR))
