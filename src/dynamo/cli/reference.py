"""Market-wide instrument reference data: CIKs and sectors.

Usage::

    python -m dynamo.cli.reference ciks              # instruments.cik
    python -m dynamo.cli.reference sectors           # sector + industry
    python -m dynamo.cli.reference etfs              # instruments.is_etf
    python -m dynamo.cli.reference all               # all three, in order
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

Every write is additive. ``seed_instrument_ciks`` /
``seed_instrument_sectors`` / ``seed_instrument_industries`` /
``seed_instrument_etf_flags`` each touch exactly one column, so re-running is
safe and they compose in any order. Crypto pairs are excluded throughout:
they have no EDGAR filings, and their sector (the literal ``"crypto"``) is
owned by the crypto_bars asset.

``etfs``: fund-vs-operating-company from the data already fetched
--------------------------------------------------------------------
``instruments.is_etf`` was ``false`` on all 11,505 rows — no classification
existed at all. ``dynamo.research.quant.catalog.asset_class_of`` therefore
read the whole universe as operating companies and charged funds for the
issuer-only feature groups (insider, earnings, short-interest-vs-float) they
can never produce, deflating the coverage denominator.

No paid classification feed is needed, because the two SEC sweeps above
already separate the populations as a side effect:

1. **Absent from SEC ``company_tickers.json``** — that file lists every EDGAR
   filer with a listed ticker, i.e. operating companies. A symbol the platform
   trades that is missing from it is not an issuer: 5,419 of 11,505 symbols
   sampled as VTWO, IEFA, EWA, GSY, AVLV, BSCZ, ROM.
2. **Has a CIK but no DERA/SIC row** — the DERA Financial Statement Data Sets
   carry one row per filer that filed an XBRL *financial statement*. Funds
   file (N-CSR, 24F-2NT) but never appear there, so they hold a CIK and no
   SIC. This is the signal that catches SPY and QQQ, which signal 1 misses.

Anything else — a CIK plus a DERA SIC row — is an operating company.

Known limits, stated rather than discovered later:

* ``is_etf`` here means "not an operating-company equity", which is the
  distinction ``asset_class_of`` actually needs. The bucket also holds
  closed-end funds, trusts, units, warrants and preferreds. Calling that
  column ``is_etf`` is the schema's word, not a claim that every flagged row
  is an exchange-traded fund.
* Signal 2 would misfire on an operating company that filed no XBRL financial
  statement inside the DERA window; ``--quarters`` widens that window.
* By default the two signals are read from the ``cik`` and ``sector`` columns
  ``ciks``/``sectors`` already wrote — they ARE the persisted results of the
  same two fetches, so classifying costs no network. One caveat comes with
  that shortcut: ``sectors`` also drops a filer whose SIC maps to no sector
  (``dynamo.data.sic``), so a DERA-present filer with an exotic SIC reads as
  a fund. ``--refetch`` avoids it by testing DERA membership directly, at the
  cost of four ~70MB quarter downloads.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

# Why a symbol was classified as a fund (printed, and useful in tests).
ETF_ABSENT_FROM_SEC = "absent-from-sec-company-tickers"
ETF_CIK_WITHOUT_SIC = "cik-without-dera-sic"
EQUITY_CIK_WITH_SIC = "cik-with-dera-sic"


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


def instrument_sectors(engine: Any) -> dict[str, str]:
    """symbol → sector already on file (blank sectors dropped).

    Presence, not the value, is signal 2: a sector on an instrument row means
    the DERA sweep found a SIC for its CIK, i.e. it filed an XBRL financial
    statement — which funds do not.
    """
    from sqlalchemy import select

    from dynamo.persistence import schema

    t = schema.instruments
    with engine.connect() as conn:
        rows = conn.execute(select(t.c.symbol, t.c.sector)).all()
    return {str(sym): str(sector) for sym, sector in rows if sector}


# --------------------------------------------------------------------------
# the fund / operating-company classifier (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EtfClassification:
    """``symbol -> is_etf`` plus the signal that decided each one."""

    flags: dict[str, bool]
    reasons: dict[str, str]

    @property
    def etfs(self) -> list[str]:
        return sorted(s for s, flag in self.flags.items() if flag)

    @property
    def equities(self) -> list[str]:
        return sorted(s for s, flag in self.flags.items() if not flag)

    def reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.reasons.values():
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def classify_etfs(
    symbols: Iterable[str],
    *,
    sec_ciks: Mapping[str, str],
    sic_by_cik: Mapping[str, Any],
) -> EtfClassification:
    """Fund vs operating company from the two free SEC signals.

    ``sec_ciks`` is symbol → CIK as SEC's ``company_tickers.json`` reports it;
    ``sic_by_cik`` is CIK → anything truthy for every CIK the DERA financial
    statement data sets carry a SIC for. Only membership is read from the
    second map, so the caller may pass CIK → SIC or CIK → sector.

    The rule, in order:

    * no CIK in ``sec_ciks``          -> fund (not an EDGAR ticker filer)
    * CIK present, absent from DERA   -> fund (files, but no XBRL statements)
    * CIK present and in DERA         -> operating company

    Crypto pairs are not classified at all: they are neither, and their asset
    class is decided by :func:`dynamo.domain.symbols.is_crypto_symbol`
    upstream of ``is_etf``. They are simply absent from the result.
    """
    from dynamo.domain.symbols import is_crypto_symbol

    flags: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for symbol in sorted({str(s) for s in symbols if s}):
        if is_crypto_symbol(symbol):
            continue
        cik = sec_ciks.get(symbol)
        if not cik:
            flags[symbol], reasons[symbol] = True, ETF_ABSENT_FROM_SEC
        elif str(cik) not in sic_by_cik:
            flags[symbol], reasons[symbol] = True, ETF_CIK_WITHOUT_SIC
        else:
            flags[symbol], reasons[symbol] = False, EQUITY_CIK_WITH_SIC
    return EtfClassification(flags=flags, reasons=reasons)


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
        etfs = conn.execute(
            select(func.count()).select_from(t).where(t.c.is_etf.is_(True))
        ).scalar_one()
    return {
        "rows": row[0],
        "cik": row[1],
        "sector": row[2],
        "industry": row[3],
        "is_etf": int(etfs),
    }


def _print_coverage(label: str, counts: dict[str, int]) -> None:
    rows = max(counts["rows"], 1)
    print(
        f"{label}: {counts['rows']} instruments | "
        f"cik {counts['cik']} ({100 * counts['cik'] / rows:.1f}%) | "
        f"sector {counts['sector']} ({100 * counts['sector'] / rows:.1f}%) | "
        f"industry {counts['industry']} ({100 * counts['industry'] / rows:.1f}%) | "
        f"is_etf {counts['is_etf']} ({100 * counts['is_etf'] / rows:.1f}%)"
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


SPOT_CHECK_ETFS = ("SPY", "QQQ", "IWM")
SPOT_CHECK_EQUITIES = ("AAPL", "JPM", "XOM")


def _etf_signals(args: argparse.Namespace, engine: Any) -> tuple[dict[str, str], dict[str, Any]]:
    """(symbol -> CIK, CIK -> DERA evidence) for :func:`classify_etfs`."""
    if not args.refetch:
        sec_ciks = instrument_ciks(engine)
        sectors = instrument_sectors(engine)
        sic_by_cik = {cik: sectors[symbol] for symbol, cik in sec_ciks.items() if symbol in sectors}
        print(
            f"signals from the database: {len(sec_ciks)} symbols carry a CIK, "
            f"{len(sic_by_cik)} of those resolved to a DERA sector"
        )
        return sec_ciks, sic_by_cik

    from dynamo.data.sec_reference import recent_quarters

    settings = _settings()
    symbols = set(known_symbols(engine))
    mapping = _edgar(settings).company_ticker_map()
    sec_ciks = {s: cik for s, cik in mapping.items() if s in symbols}
    quarters = list(args.quarters) if args.quarters else recent_quarters(date.today(), 4)
    print(f"SEC company_tickers.json: {len(mapping)} entries, {len(sec_ciks)} of ours")
    print(f"SEC DERA financial statement data sets: {', '.join(quarters)}")
    cik_to_sic = _sec_reference(settings).cik_to_sic(quarters)
    print(f"CIK->SIC entries: {len(cik_to_sic)}")
    return sec_ciks, dict(cik_to_sic)


def _cmd_etfs(args: argparse.Namespace) -> int:
    """Classify every known symbol as fund or operating company -> is_etf."""
    engine, repo = _repo(args.database_url)
    _print_coverage("before", _coverage(engine))

    symbols = known_symbols(engine)
    sec_ciks, sic_by_cik = _etf_signals(args, engine)
    result = classify_etfs(symbols, sec_ciks=sec_ciks, sic_by_cik=sic_by_cik)

    etfs, equities = len(result.etfs), len(result.equities)
    total = max(etfs + equities, 1)
    print(
        f"classified {etfs + equities} symbols: "
        f"is_etf=true {etfs} ({100 * etfs / total:.1f}%), "
        f"is_etf=false {equities} ({100 * equities / total:.1f}%)"
    )
    for reason, count in sorted(result.reason_counts().items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    # Spot checks BEFORE writing: a classifier that calls SPY an operating
    # company is not one to persist over 11,505 rows. Symbols the platform
    # does not track are skipped — absent is not wrong.
    wrong: list[str] = []
    for expected, group in ((True, SPOT_CHECK_ETFS), (False, SPOT_CHECK_EQUITIES)):
        for symbol in group:
            got = result.flags.get(symbol)
            if got is None:
                print(f"  spot-check {symbol}: not tracked — skipped")
                continue
            print(f"  spot-check {symbol}: is_etf={got} ({result.reasons[symbol]})")
            if got is not expected:
                wrong.append(symbol)
    if wrong:
        print(f"ERROR: spot checks failed for {wrong} — refusing to write", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry-run: would write {len(result.flags)} is_etf flags")
        return 0
    written = repo.seed_instrument_etf_flags(result.flags)
    print(f"seeded {written} is_etf flags")
    _print_coverage("after", _coverage(engine))
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    _print_coverage("coverage", _coverage(_repo(args.database_url)[0]))
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    return _cmd_ciks(args) or _cmd_sectors(args) or _cmd_etfs(args)


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
    etfs = sub.add_parser(
        "etfs",
        help="seed instruments.is_etf from the two free SEC signals (see module doc)",
    )
    etfs.add_argument(
        "--refetch",
        action="store_true",
        help="re-derive both signals from SEC instead of the seeded columns "
        "(four ~70MB DERA downloads; avoids the SIC-mapping caveat)",
    )
    etfs.add_argument(
        "--quarters",
        nargs="+",
        default=None,
        metavar="YYYYqN",
        help="DERA quarters for --refetch (default: the 4 most recent published)",
    )
    sub.add_parser("all", help="ciks, then sectors, then etfs")
    sub.add_parser("coverage", help="report current coverage; writes nothing")
    return parser


_COMMANDS = {
    "ciks": _cmd_ciks,
    "sectors": _cmd_sectors,
    "etfs": _cmd_etfs,
    "all": _cmd_all,
    "coverage": _cmd_coverage,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not hasattr(args, "quarters"):
        args.quarters = None
    if not hasattr(args, "refetch"):
        args.refetch = False
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
