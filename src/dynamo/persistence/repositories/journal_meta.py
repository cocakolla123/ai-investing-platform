"""Read-side sources for journal metadata (docs/journal_metadata.md)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sqlalchemy import Engine, text


def live_regime_history(engine: Engine) -> list[tuple[date, str]]:
    """(session, regime) from the nightly ``regime_states`` table, ascending."""
    with engine.connect() as c:
        rows = c.execute(
            text("select as_of, regime from regime_states where regime <> '' order by as_of")
        ).all()
    return [(r[0], str(r[1])) for r in rows]


def replay_regime_history(engine: Engine) -> list[tuple[date, str]]:
    """(session, regime) computed by replay runs, one row per session, ascending.

    Replay runs compute the regime from data on or before each session and
    store it in ``replay_equity.regime``. Runs agree per session (checked
    2026-10-05: no session has two different regimes), so ``min`` only picks
    a deterministic representative. This is the same source the retrieval
    labels use (scripts/eval/retrieval_build_labels.py).
    """
    with engine.connect() as c:
        rows = c.execute(
            text(
                "select as_of, min(regime) from replay_equity where regime <> '' "
                "group by as_of order by as_of"
            )
        ).all()
    return [(r[0], str(r[1])) for r in rows]


def sector_map(engine: Engine, symbols: Sequence[str]) -> dict[str, str | None]:
    """Today's ``instruments.sector`` per symbol (not vintaged)."""
    if not symbols:
        return {}
    with engine.connect() as c:
        rows = c.execute(
            text("select symbol, sector from instruments where symbol = any(:s)"),
            {"s": list(symbols)},
        ).all()
    return {str(r[0]): (str(r[1]) if r[1] else None) for r in rows}
