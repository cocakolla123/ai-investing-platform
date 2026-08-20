"""FINRA short-interest backfill CLI — a separate entry point by design.

``dynamo.cli.backfill`` is already 1,200 lines covering nine sources; short
interest gets its own module rather than a tenth subcommand there.

Usage::

    # every cycle FINRA still serves that has actually been PUBLISHED
    python -m dynamo.cli.backfill_short_interest --all

    # the most recent N published cycles (two cycles a month)
    python -m dynamo.cli.backfill_short_interest --cycles 24

    # one explicit settlement date, or a settlement-date range
    python -m dynamo.cli.backfill_short_interest --settlement 2026-07-31
    python -m dynamo.cli.backfill_short_interest --since 2025-01-01

    # what would be fetched, without fetching it
    python -m dynamo.cli.backfill_short_interest --all --dry-run

Free and key-less: FINRA's Query API needs no registration, so unlike the
Tiingo/FRED paths there is no "missing key" branch here.

POINT-IN-TIME. A cycle whose PUBLICATION date is after ``--as-of`` (default:
today) is refused, not silently written — writing it would leave rows in the
table that a point-in-time read at ``as_of`` could legitimately see only by
accident of the query bound. ``--as-of`` also makes the whole backfill
reproducible: running it with a past as-of writes exactly the cycles that
existed then.

Writes go through ``MarketDataRepository.upsert_rows``, which is
``ON CONFLICT (symbol, settlement_date) DO UPDATE`` — re-running any cycle is
idempotent, and a revised cycle (FINRA's ``revisionFlag``) correctly replaces
the figures it revises.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Any

from dynamo.data.finra import FinraClient, publication_date_for
from dynamo.domain.models.market import ShortInterest

DEFAULT_CYCLES = 24  # ~1 year: FINRA settles twice a month


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-backfill-short-interest",
        description="Backfill FINRA consolidated short interest (free, no API key)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="every published cycle on offer")
    scope.add_argument(
        "--cycles", type=int, default=None, help=f"most recent N (default {DEFAULT_CYCLES})"
    )
    scope.add_argument(
        "--settlement", type=date.fromisoformat, default=None, help="one settlement cycle"
    )
    scope.add_argument(
        "--since", type=date.fromisoformat, default=None, help="cycles from this date on"
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="knowledge date (default today); cycles published later are refused",
    )
    parser.add_argument("--dry-run", action="store_true", help="list cycles, fetch nothing")
    parser.add_argument("--database-url", default=None)
    return parser


def select_cycles(
    available: Sequence[date],
    *,
    as_of: date,
    all_cycles: bool = False,
    cycles: int | None = None,
    settlement: date | None = None,
    since: date | None = None,
) -> list[date]:
    """The settlement cycles to fetch, newest first — pure, so it is testable.

    Every branch is filtered by publication date first: a cycle that has
    settled but not yet been disseminated is never selectable, whatever the
    caller asked for. An explicit ``--settlement`` that is unpublished at
    ``as_of`` therefore comes back empty rather than being fetched anyway.
    """
    published = sorted((d for d in available if publication_date_for(d) <= as_of), reverse=True)
    if settlement is not None:
        return [d for d in published if d == settlement]
    if since is not None:
        return [d for d in published if d >= since]
    if all_cycles:
        return published
    return published[: cycles or DEFAULT_CYCLES]


def to_rows(items: Iterable[ShortInterest], *, fetched_at: datetime) -> list[dict[str, Any]]:
    """Domain rows -> ``short_interest`` table rows."""
    return [
        {
            "symbol": item.symbol,
            "settlement_date": item.settlement_date,
            "publication_date": item.publication_date,
            "shares_short": item.shares_short,
            "previous_shares_short": item.previous_shares_short,
            "avg_daily_volume": item.avg_daily_volume,
            "days_to_cover": item.days_to_cover,
            "change_pct": item.change_pct,
            "market_class": item.market_class,
            "issue_name": item.issue_name,
            "split_flag": item.split_flag,
            "revision_flag": item.revision_flag,
            "fetched_at": fetched_at,
        }
        for item in items
    ]


def _repo(database_url: str | None) -> Any:
    from dynamo.config.settings import Settings
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.market import MarketDataRepository

    return MarketDataRepository(make_engine(database_url or Settings().database_url))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    as_of = args.as_of or datetime.now(UTC).date()
    client = FinraClient()

    available = client.settlement_dates()
    wanted = select_cycles(
        available,
        as_of=as_of,
        all_cycles=args.all,
        cycles=args.cycles,
        settlement=args.settlement,
        since=args.since,
    )
    if not wanted:
        print(
            f"no published short-interest cycles match (as-of {as_of.isoformat()}; "
            f"{len(available)} cycles offered, newest {available[0].isoformat()} "
            f"publishes {publication_date_for(available[0]).isoformat()})"
        )
        return 0

    print(f"short interest: {len(wanted)} cycle(s) published on or before {as_of.isoformat()}")
    for settlement in wanted:
        published = publication_date_for(settlement).isoformat()
        print(f"  {settlement.isoformat()} -> published {published}")
    if args.dry_run:
        print("dry run: nothing fetched, nothing written")
        return 0

    from dynamo.persistence import schema

    repo = _repo(args.database_url)
    fetched_at = datetime.now(UTC)
    total = 0
    for settlement in wanted:
        rows = to_rows(client.short_interest([settlement]), fetched_at=fetched_at)
        written = repo.upsert_rows(schema.short_interest, rows)
        total += written
        print(f"short interest: {settlement.isoformat()}: {written} rows")
    print(f"short-interest backfill complete: {total} rows across {len(wanted)} cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
