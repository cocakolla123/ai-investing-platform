"""Market-wide instrument reference data: CIKs and sectors.

Usage::

    python -m dynamo.cli.reference ciks              # instruments.cik
    python -m dynamo.cli.reference sectors           # sector + industry
    python -m dynamo.cli.reference all               # both, in order
    python -m dynamo.cli.reference coverage          # report, writes nothing

Why this exists separately from ``backfill fundamentals --seed-ciks``
---------------------------------------------------------------------
That path seeds CIKs only for the symbols it is about to fetch XBRL facts
for — i.e. the ACTIVE universe (Tier 1 + promotions), ~55 names. It did its
job: 43 of 11,505 ``instruments`` rows carried a CIK. But ``xbrl_facts`` is
keyed by CIK, so the 13+ quality/value/growth features in
``research/quant/catalog.py`` and the Phase-2 fundamentals dossier resolved
to NaN for 99.6% of the market — including every Tier-2 name the discovery
funnel is supposed to promote ON THE STRENGTH of its fundamentals.

The scope is the difference, not the mechanism: this command seeds the
whole ``instruments`` table (plus anything in ``universe_membership``), so
a name that gets promoted tomorrow already has its CIK and sector today.

Both writes are additive. ``seed_instrument_ciks`` /
``seed_instrument_sectors`` / ``seed_instrument_industries`` each touch
exactly one column, so re-running is safe and the three compose in any
order. Crypto pairs are excluded throughout: they have no EDGAR filings,
and their sector (the literal ``"crypto"``) is owned by the crypto_bars
asset.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from typing import Any


def _settings() -> Any:
    from dynamo.config.settings import Settings

    return Settings()


def _repo(database_url: str | None) -> tuple[Any, Any]:
    """(engine, MarketDataRepository) — the engine is needed for the plain
    reference reads below, which are one-way sweeps, not hot paths worth a
    bespoke repository method."""
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.market import MarketDataRepository

    engine = make_engine(database_url or _settings().database_url)
    return engine, MarketDataRepository(engine)


def _edgar(settings: Any) -> Any:
    from dynamo.data.edgar import EdgarClient

    return EdgarClient(settings.edgar_user_agent)


def _sec_reference(settings: Any) -> Any:
    from dynamo.data.sec_reference import SecReferenceClient

    return SecReferenceClient(settings.edgar_user_agent)


def known_symbols(engine: Any) -> list[str]:
    """Every equity symbol the platform knows: ``instruments`` plus universe.

    Reads through the repository's engine rather than adding a bespoke
    repository method — this is a one-way reference sweep, not a hot path.
    """
    from sqlalchemy import select

    from dynamo.domain.symbols import is_crypto_symbol
    from dynamo.persistence import schema

    with engine.connect() as conn:
        symbols = {str(s) for (s,) in conn.execute(select(schema.instruments.c.symbol))}
        symbols |= {
            str(s) for (s,) in conn.execute(select(schema.universe_membership.c.symbol))
        }
    return sorted(s for s in symbols if s and not is_crypto_symbol(s))


def instrument_ciks(engine: Any) -> dict[str, str]:
    """symbol → CIK already on file (blank CIKs dropped)."""
    from sqlalchemy import select

    from dynamo.persistence import schema

    t = schema.instruments
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.symbol, t.c.cik)).all()
    return {str(sym): str(cik) for sym, cik in rows if cik}


def _coverage(engine: Any) -> dict[str, int]:
    from sqlalchemy import func, select

    from dynamo.persistence import schema

    t = schema.instruments
    with engine.connect() as conn:
        row = conn.execute(
            select(
                func.count(),
                func.count(t.c.cik),
                func.count(t.c.sector),
                func.count(t.c.industry),
            ).select_from(t)
        ).one()
    return {"rows": row[0], "cik": row[1], "sector": row[2], "industry": row[3]}


def _print_coverage(label: str, counts: dict[str, int]) -> None:
    rows = max(counts["rows"], 1)
    print(
        f"{label}: {counts['rows']} instruments | "
        f"cik {counts['cik']} ({100 * counts['cik'] / rows:.1f}%) | "
        f"sector {counts['sector']} ({100 * counts['sector'] / rows:.1f}%) | "
        f"industry {counts['industry']} ({100 * counts['industry'] / rows:.1f}%)"
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _cmd_ciks(args: argparse.Namespace) -> int:
    engine, repo = _repo(args.database_url)
    _print_coverage("before", _coverage(engine))

    symbols = set(known_symbols(engine))
    mapping = _edgar(_settings()).company_ticker_map()
    print(f"SEC company_tickers.json: {len(mapping)} symbol->CIK entries")

    subset = {s: cik for s, cik in mapping.items() if s in symbols}
    unmapped = len(symbols) - len(subset)
    if args.dry_run:
        print(f"dry-run: would seed {len(subset)} CIKs ({unmapped} symbols unmapped by SEC)")
        return 0
    seeded = repo.seed_instrument_ciks(subset)
    print(f"seeded {seeded} CIKs ({unmapped} of {len(symbols)} symbols unmapped by SEC)")
    _print_coverage("after", _coverage(engine))
    return 0


def _cmd_sectors(args: argparse.Namespace) -> int:
    from dynamo.data.sec_reference import recent_quarters, sectors_for_symbols

    engine, repo = _repo(args.database_url)
    _print_coverage("before", _coverage(engine))

    symbol_to_cik = instrument_ciks(engine)
    if not symbol_to_cik:
        print("no instruments carry a CIK — run `reference ciks` first", file=sys.stderr)
        return 1
    print(f"{len(symbol_to_cik)} instruments carry a CIK")

    quarters = list(args.quarters) if args.quarters else recent_quarters(date.today(), 4)
    print(f"SEC DERA financial statement data sets: {', '.join(quarters)}")
    cik_to_sic = _sec_reference(_settings()).cik_to_sic(quarters)
    print(f"CIK->SIC entries: {len(cik_to_sic)}")

    sectors, industries = sectors_for_symbols(symbol_to_cik, cik_to_sic)
    print(f"resolved {len(sectors)} sectors, {len(industries)} industries")
    if args.dry_run:
        return 0
    n_sectors = repo.seed_instrument_sectors(sectors)
    n_industries = repo.seed_instrument_industries(industries)
    print(f"seeded {n_sectors} sectors and {n_industries} industries")
    _print_coverage("after", _coverage(engine))
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    _print_coverage("coverage", _coverage(_repo(args.database_url)[0]))
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    return _cmd_ciks(args) or _cmd_sectors(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-reference",
        description="Seed instruments.cik / sector / industry market-wide from free SEC data",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written, write nothing"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ciks", help="seed instruments.cik from SEC company_tickers.json")
    sectors = sub.add_parser(
        "sectors", help="seed instruments.sector/industry from SEC DERA SIC codes"
    )
    sectors.add_argument(
        "--quarters",
        nargs="+",
        default=None,
        metavar="YYYYqN",
        help="DERA quarters, newest first (default: the 4 most recent published)",
    )
    sub.add_parser("all", help="ciks then sectors")
    sub.add_parser("coverage", help="report current coverage; writes nothing")
    return parser


_COMMANDS = {
    "ciks": _cmd_ciks,
    "sectors": _cmd_sectors,
    "all": _cmd_all,
    "coverage": _cmd_coverage,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not hasattr(args, "quarters"):
        args.quarters = None
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
