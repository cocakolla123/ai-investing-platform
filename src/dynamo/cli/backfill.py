"""One-shot historical backfill CLI (plan Phase 4) — manual only.

Bulk range loads that bypass the one-day-per-partition Dagster ingestion,
feeding the replay foundation:

    python -m dynamo.cli.backfill bars --start 2025-01-01 --end 2026-08-01 \
        --symbols AAPL MSFT            # or --universe
    python -m dynamo.cli.backfill fundamentals --universe
    python -m dynamo.cli.backfill macro --start 2020-01-01

- ``bars``: Tiingo adjusted EOD history over an arbitrary range (delisted
  names included; the adapter's 5 req/s throttle applies). Skips gracefully
  with a clear message when TIINGO_API_KEY is unset — no network attempted.
- ``fundamentals``: EDGAR companyfacts XBRL, ``filed_at``-vintaged (point-in-
  time honest by construction).
- ``macro``: FRED/ALFRED series WITH vintages (``series_vintages``), default
  series = the production ingestion set. Skips when FRED_API_KEY is unset.

All writes are the same idempotent natural-key upserts the nightly pipeline
uses — re-running a backfill is safe. NOTE: this command performs real HTTP
calls when run with keys configured; tests mock every adapter.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

# Production FRED series (mirrors orchestration.resources.DEFAULT_FRED_SERIES;
# duplicated here so the CLI works without the orchestration package on path).
DEFAULT_FRED_SERIES: tuple[str, ...] = (
    "DGS3MO",
    "DGS2",
    "DGS10",
    "T10Y3M",
    "T10Y2Y",
    "CPIAUCSL",
    "PCEPILFE",
    "UNRATE",
    "BAMLH0A0HYM2",
    "NFCI",
    "DFF",
    "VIXCLS",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-backfill",
        description="One-shot historical range loads (bars / fundamentals / macro)",
    )
    parser.add_argument("--database-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    bars = sub.add_parser("bars", help="Tiingo daily bars over [start, end]")
    bars.add_argument("--start", required=True, type=date.fromisoformat)
    bars.add_argument("--end", required=True, type=date.fromisoformat)
    group = bars.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")

    fund = sub.add_parser("fundamentals", help="EDGAR companyfacts (filed_at-vintaged)")
    group = fund.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")

    macro = sub.add_parser("macro", help="FRED series with vintages")
    macro.add_argument("--start", required=True, type=date.fromisoformat)
    macro.add_argument("--series", nargs="+", default=list(DEFAULT_FRED_SERIES))
    return parser


# -- injectable composition seams (tests monkeypatch these) -----------------


def _settings() -> Any:
    from dynamo.config.settings import Settings

    return Settings()


def _repo(database_url: str | None) -> Any:
    from dynamo.config.settings import Settings
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.market import MarketDataRepository

    return MarketDataRepository(make_engine(database_url or Settings().database_url))


def _tiingo(settings: Any) -> Any:
    from dynamo.data.tiingo import TiingoClient

    return TiingoClient(settings.tiingo_api_key)


def _edgar(settings: Any) -> Any:
    from dynamo.data.edgar import EdgarClient

    return EdgarClient(settings.edgar_user_agent)


def _fred(settings: Any) -> Any:
    from dynamo.data.fred import FredClient

    return FredClient(settings.fred_api_key)


def _universe_symbols(repo: Any, as_of: date) -> list[str]:
    return sorted({m.symbol for m in repo.universe_at(as_of)})


def _cik_map(repo: Any, symbols: list[str]) -> dict[str, str]:
    """symbol -> CIK from the instruments table (empty CIKs dropped)."""
    from sqlalchemy import select

    from dynamo.persistence import schema

    inst = schema.instruments
    stmt = select(inst.c.symbol, inst.c.cik).where(inst.c.symbol.in_(symbols))
    with repo._engine.connect() as cx:  # composition-layer seam
        rows = cx.execute(stmt).all()
    return {str(r.symbol): str(r.cik) for r in rows if r.cik}


# -- subcommands ------------------------------------------------------------


def _cmd_bars(args: argparse.Namespace) -> int:
    settings = _settings()
    if not settings.tiingo_api_key:
        print(
            "TIINGO_API_KEY is not set — skipping bars backfill "
            "(no network calls made). Set the key and re-run."
        )
        return 0
    repo = _repo(args.database_url)
    symbols = list(args.symbols) if args.symbols else _universe_symbols(repo, args.end)
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
    tiingo = _tiingo(settings)
    total = 0
    for sym in symbols:
        rows = repo.upsert_bars(tiingo.daily_bars([sym], args.start, args.end))
        total += rows
        print(f"bars: {sym} {args.start} -> {args.end}: {rows} rows")
    print(f"bars backfill complete: {total} rows across {len(symbols)} symbols")
    return 0


def _cmd_fundamentals(args: argparse.Namespace) -> int:
    settings = _settings()
    repo = _repo(args.database_url)
    symbols = (
        list(args.symbols) if args.symbols else _universe_symbols(repo, date.today())
    )
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
    ciks = _cik_map(repo, symbols)
    missing = sorted(set(symbols) - set(ciks))
    if missing:
        print(f"fundamentals: no CIK on file for {missing} — skipped")
    edgar = _edgar(settings)
    total = 0
    for sym in sorted(ciks):
        rows = repo.upsert_xbrl_facts(edgar.company_facts(ciks[sym]))
        total += rows
        print(f"fundamentals: {sym} (CIK {ciks[sym]}): {rows} facts")
    print(f"fundamentals backfill complete: {total} facts across {len(ciks)} symbols")
    return 0


def _cmd_macro(args: argparse.Namespace) -> int:
    settings = _settings()
    if not settings.fred_api_key:
        print(
            "FRED_API_KEY is not set — skipping macro backfill "
            "(no network calls made). Set the key and re-run."
        )
        return 0
    repo = _repo(args.database_url)
    fred = _fred(settings)
    total = 0
    for series_id in args.series:
        rows = repo.upsert_macro(
            fred.series_vintages(series_id, args.start, realtime_start=args.start)
        )
        total += rows
        print(f"macro: {series_id} since {args.start}: {rows} vintaged points")
    print(f"macro backfill complete: {total} points across {len(args.series)} series")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "bars":
        return _cmd_bars(args)
    if args.command == "fundamentals":
        return _cmd_fundamentals(args)
    return _cmd_macro(args)


if __name__ == "__main__":
    sys.exit(main())
