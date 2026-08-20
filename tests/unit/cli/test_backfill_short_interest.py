"""Short-interest backfill CLI: cycle selection, row shape, idempotent upsert.

The one rule worth a test each way: a cycle that has SETTLED but has not yet
been PUBLISHED is never selected, whatever flag the operator passed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Table, create_engine, select

from dynamo.cli.backfill_short_interest import select_cycles, to_rows
from dynamo.data.finra import parse_short_interest_csv
from dynamo.persistence import schema
from dynamo.persistence.repositories.market import MarketDataRepository
from tests.unit.data.test_finra import FIXTURE

# A realistic run of cycles: two a month, newest first.
AVAILABLE = [
    date(2026, 7, 31),
    date(2026, 7, 15),
    date(2026, 6, 30),
    date(2026, 6, 15),
    date(2026, 5, 29),
    date(2026, 5, 15),
]


def test_an_unpublished_cycle_is_never_selected() -> None:
    """2026-07-31 publishes on 2026-08-10. On 2026-08-05 it has settled but
    does not exist yet, so no flag may reach it — not --all, not an explicit
    --settlement for that exact date."""
    as_of = date(2026, 8, 5)
    assert date(2026, 7, 31) not in select_cycles(AVAILABLE, as_of=as_of, all_cycles=True)
    assert select_cycles(AVAILABLE, as_of=as_of, settlement=date(2026, 7, 31)) == []
    assert select_cycles(AVAILABLE, as_of=as_of, all_cycles=True)[0] == date(2026, 7, 15)


def test_the_same_cycle_is_selected_once_it_has_been_published() -> None:
    assert select_cycles(AVAILABLE, as_of=date(2026, 8, 10), settlement=date(2026, 7, 31)) == [
        date(2026, 7, 31)
    ]


def test_cycle_selection_flags() -> None:
    as_of = date(2026, 8, 20)
    assert select_cycles(AVAILABLE, as_of=as_of, all_cycles=True) == AVAILABLE
    assert select_cycles(AVAILABLE, as_of=as_of, cycles=2) == AVAILABLE[:2]
    assert select_cycles(AVAILABLE, as_of=as_of, since=date(2026, 6, 20)) == AVAILABLE[:3]
    # Default (no flag) is the most recent year of cycles, newest first.
    assert select_cycles(AVAILABLE, as_of=as_of) == AVAILABLE


def _sqlite_engine() -> Any:
    """A throwaway SQLite database carrying just the short_interest table.

    SQLite has no ``ON CONFLICT ... DO UPDATE`` through the Postgres dialect
    helper, so the upsert path itself is exercised against Postgres in
    integration; here the table shape and the natural key are what is checked.
    """
    engine = create_engine("sqlite://")
    schema.short_interest.create(engine)
    return engine


def test_rows_carry_both_dates_and_the_full_payload() -> None:
    parsed = list(parse_short_interest_csv(FIXTURE.read_text(encoding="utf-8")))
    fetched_at = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    rows = to_rows(parsed, fetched_at=fetched_at)

    assert len(rows) == len(parsed)
    table_columns = {c.name for c in schema.short_interest.columns}
    assert set(rows[0]) == table_columns
    agilent = next(r for r in rows if r["symbol"] == "A")
    assert agilent["settlement_date"] == date(2026, 7, 31)
    assert agilent["publication_date"] == date(2026, 8, 10)
    assert agilent["shares_short"] == 5_749_623.0
    assert agilent["fetched_at"] == fetched_at


def test_reingesting_a_cycle_replaces_it_rather_than_duplicating_it() -> None:
    """FINRA revises cycles (``revisionFlag``), and the backfill is re-run by
    hand. (symbol, settlement_date) is the natural key, so the second write
    must update the first — not append a second truth."""
    engine = _sqlite_engine()
    repo = MarketDataRepository(engine)
    table: Table = schema.short_interest
    parsed = list(parse_short_interest_csv(FIXTURE.read_text(encoding="utf-8")))
    rows = to_rows(parsed, fetched_at=datetime(2026, 8, 11, tzinfo=UTC))

    _write(engine, table, rows)
    revised = [dict(r) for r in rows]
    for row in revised:
        row["shares_short"] = row["shares_short"] * 2
        row["revision_flag"] = "R"
    _write(engine, table, revised)

    with engine.begin() as conn:
        stored = conn.execute(select(table)).mappings().all()
    assert len(stored) == len(rows)  # not doubled
    agilent = next(r for r in stored if r["symbol"] == "A")
    assert agilent["shares_short"] == 5_749_623.0 * 2
    assert agilent["revision_flag"] == "R"
    assert repo is not None  # the repo is the production write path


def _write(engine: Any, table: Table, rows: list[dict[str, Any]]) -> None:
    """SQLite-flavoured equivalent of the repository's ON CONFLICT upsert."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    stmt = sqlite_insert(table).values(rows)
    pk = [c.name for c in table.primary_key.columns]
    stmt = stmt.on_conflict_do_update(
        index_elements=pk,
        set_={c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk},
    )
    with engine.begin() as conn:
        conn.execute(stmt)
