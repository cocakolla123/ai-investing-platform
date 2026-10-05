"""Backfill journal metadata (docs/journal_metadata.md). DRY RUN by default.

    PYTHONPATH=. .venv/bin/python scripts/journal_backfill_metadata.py          # dry run
    PYTHONPATH=. .venv/bin/python scripts/journal_backfill_metadata.py --apply  # backup, then write

Fills only NULL ``strategy`` / ``regime_at_write`` and empty ``sectors``, with
facts knowable at each entry's ``created_at``. Never reads or writes ``text``,
``embedding`` or ``embedding_version``. Idempotent: existing values are never
overwritten. ``--apply`` first copies ``journal_entries`` to
``journal_entries_backup_<UTC timestamp>`` and refuses to write if that fails.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.learning.journal_backfill import (
    EntryMeta,
    SleeveSource,
    Sources,
    applied,
    counts,
    plan,
)
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.journal_meta import (
    live_regime_history,
    replay_regime_history,
    sector_map,
)


def _utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="back up, then write (default: dry run)")
    args = ap.parse_args()
    engine = make_engine(Settings().database_url)

    with engine.connect() as c:
        rows = c.execute(
            text(
                "select id, kind, symbols, created_at, refs, strategy, regime_at_write, sectors "
                "from journal_entries order by id"
            )
        ).all()
        dcs = c.execute(
            text("select id, verdict_id, sleeve, created_at from decision_context")
        ).all()
        exps = c.execute(text("select id, sleeve, created_at from decision_experience")).all()
    entries = [
        EntryMeta(
            id=r[0],
            kind=r[1],
            symbols=tuple(r[2] or ()),
            created_at=_utc(r[3]),
            refs=r[4] or {},
            strategy=r[5],
            regime_at_write=r[6],
            sectors=tuple(r[7] or ()),
        )
        for r in rows
    ]
    sources = Sources(
        decision_contexts={d[0]: SleeveSource(d[2], _utc(d[3])) for d in dcs},
        decision_context_by_verdict={
            d[1]: SleeveSource(d[2], _utc(d[3])) for d in dcs if d[1] is not None
        },
        experiences={x[0]: SleeveSource(x[1], _utc(x[2])) for x in exps},
        live_regimes=live_regime_history(engine),
        replay_regimes=replay_regime_history(engine),
        sector_by_symbol=sector_map(engine, sorted({s for e in entries for s in e.symbols})),
    )
    fills, unfilled = plan(entries, sources)
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "entries": len(entries),
        "entries_to_update": len(fills),
        "before": counts(entries),
        "after": counts(applied(entries, fills)),
        "left_null_reasons": dict(sorted(unfilled.items())),
    }

    if args.apply and fills:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup = f"journal_entries_backup_{stamp}"
        # The backup commits on its own, BEFORE any update, so it exists even if
        # the update transaction fails (which rolls back as a whole).
        with engine.begin() as c:
            c.execute(text(f"create table {backup} as select * from journal_entries"))
        with engine.connect() as c:
            copied = c.execute(text(f"select count(*) from {backup}")).scalar()
        if copied != len(entries):
            raise SystemExit(f"backup {backup} has {copied} rows, expected {len(entries)}")
        with engine.begin() as c:
            for f in fills:
                c.execute(
                    text(
                        "update journal_entries set "
                        "strategy = coalesce(strategy, :strategy), "
                        "regime_at_write = coalesce(regime_at_write, :regime), "
                        "sectors = case when cardinality(sectors) = 0 and :has_sectors "
                        "then cast(:sectors as text[]) else sectors end "
                        "where id = :id"
                    ),
                    {
                        "id": f.id,
                        "strategy": f.strategy,
                        "regime": f.regime_at_write,
                        "has_sectors": f.sectors is not None,
                        "sectors": list(f.sectors or ()),
                    },
                )
        report["backup_table"] = backup
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
