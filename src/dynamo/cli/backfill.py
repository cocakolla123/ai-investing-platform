"""One-shot historical backfill CLI (plan Phase 4 + backtest-campaign Phase 1).

Bulk range loads that bypass the one-day-per-partition Dagster ingestion,
feeding the replay foundation:

    python -m dynamo.cli.backfill bars --start 2025-01-01 --end 2026-08-01 \
        --symbols AAPL MSFT            # or --universe, or --symbols-file FILE
    python -m dynamo.cli.backfill fundamentals --universe --seed-ciks
    python -m dynamo.cli.backfill macro --start 2020-01-01
    python -m dynamo.cli.backfill universe --as-of 2024-12-01
    python -m dynamo.cli.backfill news --start 2026-01-01 --end 2026-08-01 --universe
    python -m dynamo.cli.backfill transcripts --universe
    python -m dynamo.cli.backfill crypto-bars --start 2024-01-01 --end 2026-08-01
    python -m dynamo.cli.backfill universe-wide --as-of 2026-08-17
    python -m dynamo.cli.backfill bars --provider alpaca --tiers 2 \
        --start 2024-01-01 --end 2026-08-16 --universe
    python -m dynamo.cli.backfill surveillance-sweep --as-of 2026-08-14 --tiers 2

- ``bars``: daily OHLCV over an arbitrary range. ``--provider tiingo``
  (default) walks symbols ONE PER REQUEST for adjusted EOD history (delisted
  names included; the adapter's 5 req/s throttle applies). Missing
  TIINGO_API_KEY is a LOUD failure (exit 1, no network attempted);
  ``--allow-missing-tiingo`` restores the old graceful skip.
  ``--provider alpaca`` is the WIDE path: bars from the free
  IEX feed, ``--chunk-size`` symbols per request (default 200), resumable
  (``--skip-covered``, on by default), per-chunk failures degrade to
  per-symbol retries instead of killing the run. Alpaca's free feed is
  single-venue, so its ``volume`` UNDERSTATES the consolidated tape — measured
  at ~6.6% of it on this repo's data. Dollar-volume gates therefore convert to
  consolidated-equivalent terms first; see ``dynamo.domain.volume_source`` for
  the measurement, its caveats, and how to re-measure it.
- ``fundamentals``: EDGAR companyfacts XBRL, ``filed_at``-vintaged (point-in-
  time honest by construction). ``instruments.cik`` has NO automatic
  population path today (only crypto instrument rows are auto-upserted, and
  those carry no CIK) — ``--seed-ciks`` fills it from SEC's public
  ``company_tickers.json`` ticker->CIK mapping before fetching facts.
- ``macro``: FRED/ALFRED series WITH vintages (``series_vintages``), default
  series = the production ingestion set. Skips when FRED_API_KEY is unset.
- ``universe``: backdate the CORE seed for the replay campaign — in ONE
  transaction, delete the seed-written CORE rows (approved_by='seed', plus
  any prior 'replay-backdate' rows) for the starter symbols and re-insert
  them with ``valid_from = --as-of`` (``approved_by='replay-backdate'``).
  Idempotent: re-running with the same as-of changes nothing; a different
  as-of replaces the intervals again. Crypto starters are included when
  ``settings.crypto_enabled``.
- ``news``: Finnhub historical company news over [start, end] via
  ``news_range`` (monthly chunks per symbol, token-bucket paced). Crypto
  pairs are skipped: finnhub ``/company-news`` is equities-only (its crypto
  coverage lives under candles/general categories, not per-pair company
  news), so crypto news backfill is unsupported and reported as skipped.
- ``transcripts``: bulk historical earnings-call transcripts via
  defeatbeta-api (the guarded optional ``transcripts`` extra). Missing extra
  is a LOUD failure (exit 1) unless ``--allow-missing-extra``.
- ``crypto-bars``: Alpaca crypto daily bars (free feed, existing Alpaca keys
  optional) into ``bars_1d`` with ``source='alpaca_crypto'``.
- ``universe-wide``: build the Tier-2 SURVEILLANCE universe from Alpaca's
  tradable-asset list — the whole-market breadth the three-tier funnel needs
  (doc 02 §2). Filters to real surveillance candidates (see
  :func:`filter_surveillance_candidates`), upserts ``instruments``
  ADDITIVELY (name/exchange only — existing CIK/sector/industry survive) and
  writes Tier-2 ``universe_membership`` rows. Existing Tier-1 CORE members
  are excluded from the Tier-2 write and verified unchanged afterwards.
- ``surveillance-sweep``: run the tier-2 bar detectors
  (``dynamo.data.detectors``) over a universe tier for one as-of date and
  write real ``surveillance_flags`` rows. ZERO LLM calls — this is the
  code-only half of the tier-2 scan; the Haiku triage that promotes flagged
  names to Tier 3 lives in the ``tier2_scan`` Dagster asset.

All writes are the same idempotent natural-key upserts the nightly pipeline
uses — re-running a backfill is safe. NOTE: this command performs real HTTP
calls when run with keys configured; tests mock every adapter.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps the CLI import-light)
    from dynamo.domain.models.market import Instrument

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

CRYPTO_NEWS_SKIP_MESSAGE = (
    "crypto news backfill unsupported (finnhub company-news is equities-only) — skipped"
)

# Finnhub free-tier 429 handling for the news backfill: retries per symbol and
# the pause when the response carries no Retry-After header.
_NEWS_RATE_LIMIT_RETRIES = 3
_NEWS_RATE_LIMIT_PAUSE_S = 65.0

# -- Tier-2 universe construction (``universe-wide``) -----------------------

# US primary listing venues we surveil. Alpaca's asset list also carries OTC
# (`exchange='OTC'`) — that bucket is where the unsponsored-ADR / pink-sheet
# noise lives (UPMMY, ASMIY in the sample) and it has no reliable IEX bar
# coverage, so it is excluded wholesale.
DEFAULT_SURVEILLANCE_EXCHANGES: tuple[str, ...] = ("NYSE", "NASDAQ", "ARCA", "BATS", "AMEX")

# Non-common-share security classes, matched on the asset NAME because that is
# the only class signal the ``Instrument`` port carries (Alpaca's asset payload
# has no security-type field; symbol suffixes alone are unreliable — `UNIT`,
# `UNTY`, `RIGL`, `WRB` are ordinary common stocks). Word-boundary anchored so
# "United"/"Community"/"Wright" are not caught.
NON_COMMON_NAME_RE = re.compile(
    r"\b("
    r"warrants?|rights?|units?|"
    r"preferred|preference|"
    r"notes?|debentures?|subordinated|"
    r"when[- ]issued|when\s+distributed"
    r")\b",
    re.IGNORECASE,
)

# NYSE/NYSE-American class suffixes Alpaca renders on the symbol itself:
# ``.PR*`` preferred, ``.WS*`` warrants, ``.RT`` rights, ``.U`` units. These
# are anchored to a DOT so ordinary five-letter tickers ending in W/R/U are
# untouched.
NON_COMMON_SYMBOL_RE = re.compile(r"\.(PR[A-Z]?|WS[A-Z]?|RT|U|W|R)$", re.IGNORECASE)

# Wide-bar defaults: symbols per Alpaca multi-symbol request, and the calendar
# slack allowed when deciding a symbol is already covered for a range.
DEFAULT_BARS_CHUNK_SIZE = 200
DEFAULT_COVERAGE_TOLERANCE_DAYS = 7


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-backfill",
        description=(
            "One-shot historical range loads (bars / fundamentals / macro / universe / "
            "universe-wide / news / transcripts / crypto-bars / surveillance-sweep)"
        ),
    )
    parser.add_argument("--database-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    bars = sub.add_parser("bars", help="daily bars over [start, end] (tiingo or alpaca)")
    bars.add_argument("--start", required=True, type=date.fromisoformat)
    bars.add_argument("--end", required=True, type=date.fromisoformat)
    group = bars.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument(
        "--symbols-file",
        type=Path,
        help="file with one symbol per line ('#' comments and blank lines "
        "ignored) — the universe-breadth path: backfill a wide TRAINING "
        "panel (e.g. config/training_universe.txt) without touching the "
        "trading universe (see config/training_universe.txt.example)",
    )
    group.add_argument("--universe", action="store_true")
    bars.add_argument(
        "--allow-missing-tiingo",
        action="store_true",
        help="missing TIINGO_API_KEY skips gracefully (exit 0) instead of failing",
    )
    bars.add_argument(
        "--provider",
        choices=("tiingo", "alpaca"),
        default="tiingo",
        help=(
            "tiingo (default): adjusted EOD, one request per symbol. "
            "alpaca: raw IEX bars, --chunk-size symbols per request — the "
            "wide-universe path (single-venue volume understates the "
            "consolidated tape ~15x; see dynamo.domain.volume_source)"
        ),
    )
    bars.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=None,
        help=(
            "with --universe, restrict to these membership tiers "
            "(1=core, 2=surveillance, 3=promoted). Default: every tier. "
            "NOTE --universe resolves membership AS OF --end, so --end must "
            "be >= the valid_from the tier rows were written with."
        ),
    )
    bars.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_BARS_CHUNK_SIZE,
        help=f"--provider alpaca: symbols per request (default {DEFAULT_BARS_CHUNK_SIZE})",
    )
    bars.add_argument(
        "--force",
        action="store_true",
        help=(
            "--provider alpaca: re-fetch symbols already covered for [start, end] "
            "(default skips them, making the backfill resumable)"
        ),
    )
    bars.add_argument(
        "--coverage-tolerance-days",
        type=int,
        default=DEFAULT_COVERAGE_TOLERANCE_DAYS,
        help=(
            "--provider alpaca: calendar slack when judging existing coverage "
            f"(default {DEFAULT_COVERAGE_TOLERANCE_DAYS}); symbols that listed "
            "mid-range never look covered and are re-fetched every run"
        ),
    )
    bars.add_argument("--limit", type=int, default=None, help="cap the symbol count (smoke runs)")

    fund = sub.add_parser("fundamentals", help="EDGAR companyfacts (filed_at-vintaged)")
    group = fund.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")
    fund.add_argument(
        "--seed-ciks",
        action="store_true",
        help=(
            "populate instruments.cik from SEC company_tickers.json first "
            "(there is no automatic CIK ingestion path)"
        ),
    )

    macro = sub.add_parser("macro", help="FRED series with vintages")
    macro.add_argument("--start", required=True, type=date.fromisoformat)
    macro.add_argument("--series", nargs="+", default=list(DEFAULT_FRED_SERIES))

    uni = sub.add_parser(
        "universe",
        help="backdate the CORE starter seed to --as-of (replay campaign, one transaction)",
    )
    uni.add_argument("--as-of", required=True, type=date.fromisoformat)

    wide = sub.add_parser(
        "universe-wide",
        help="build the Tier-2 surveillance universe from Alpaca's tradable-asset list",
    )
    wide.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help=(
            "valid_from for the Tier-2 rows (default: today). Alpaca's asset "
            "list is a TODAY snapshot with no delisted names, so backdating it "
            "injects survivorship bias — do that only deliberately."
        ),
    )
    wide.add_argument(
        "--exchanges",
        nargs="+",
        default=list(DEFAULT_SURVEILLANCE_EXCHANGES),
        help=f"listing venues to keep (default: {' '.join(DEFAULT_SURVEILLANCE_EXCHANGES)})",
    )
    wide.add_argument("--limit", type=int, default=None, help="cap the symbol count (smoke runs)")
    wide.add_argument(
        "--dry-run",
        action="store_true",
        help="report the filter funnel and write nothing",
    )

    sweep = sub.add_parser(
        "surveillance-sweep",
        help="run the tier-2 bar detectors over a universe tier and write surveillance_flags",
    )
    sweep.add_argument("--as-of", required=True, type=date.fromisoformat)
    sweep.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        default=[2],
        help="membership tiers to scan (default: 2 = surveillance)",
    )
    sweep.add_argument(
        "--lookback-days",
        type=int,
        default=400,
        help=(
            "calendar days of bar history loaded per symbol (default 400 ~ 275 "
            "sessions, covering range_break's 252-session lookback)"
        ),
    )
    sweep.add_argument(
        "--min-dollar-volume",
        type=float,
        default=0.0,
        help=(
            "drop symbols whose median daily dollar volume over the window is "
            "below this (default 0 = no floor). Expressed in CONSOLIDATED-"
            "EQUIVALENT dollars: IEX-sourced volume is scaled up to whole-tape "
            "terms before the comparison (dynamo.domain.volume_source), so the "
            "number means the same thing on either feed."
        ),
    )
    sweep.add_argument("--limit", type=int, default=None, help="cap the symbol count (smoke runs)")
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="run the detectors and report, but write no surveillance_flags rows",
    )

    news = sub.add_parser("news", help="Finnhub historical company news over [start, end]")
    news.add_argument("--start", required=True, type=date.fromisoformat)
    news.add_argument("--end", required=True, type=date.fromisoformat)
    group = news.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")

    tr = sub.add_parser(
        "transcripts", help="bulk historical transcripts via defeatbeta (transcripts extra)"
    )
    group = tr.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")
    tr.add_argument(
        "--since",
        type=date.fromisoformat,
        default=date(2020, 1, 1),
        help="earliest report_date to fetch (default 2020-01-01)",
    )
    tr.add_argument(
        "--allow-missing-extra",
        action="store_true",
        help="missing `transcripts` extra skips gracefully (exit 0) instead of failing",
    )

    cb = sub.add_parser(
        "crypto-bars", help="Alpaca crypto daily bars over [start, end] (free feed)"
    )
    cb.add_argument("--start", required=True, type=date.fromisoformat)
    cb.add_argument("--end", required=True, type=date.fromisoformat)
    cb.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="crypto pairs (default: settings.crypto_universe)",
    )
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


def _finnhub(settings: Any) -> Any:
    from dynamo.data.finnhub import FinnhubClient

    return FinnhubClient(settings.finnhub_api_key)


def _defeatbeta() -> Any:
    from dynamo.data.transcripts_defeatbeta import DefeatBetaTranscripts

    return DefeatBetaTranscripts()


def _alpaca_crypto(settings: Any) -> Any:
    from dynamo.data.alpaca_crypto import AlpacaCryptoData

    return AlpacaCryptoData(settings.alpaca_api_key, settings.alpaca_api_secret)


def _alpaca_market(settings: Any) -> Any:
    from dynamo.data.alpaca_data import AlpacaMarketData

    return AlpacaMarketData(settings.alpaca_api_key, settings.alpaca_api_secret)


def _universe_symbols(repo: Any, as_of: date, tiers: Sequence[int] | None = None) -> list[str]:
    wanted = set(tiers) if tiers else None
    return sorted(
        {m.symbol for m in repo.universe_at(as_of) if wanted is None or int(m.tier) in wanted}
    )


def _cik_map(repo: Any, symbols: list[str]) -> dict[str, str]:
    """symbol -> CIK from the instruments table (empty CIKs dropped)."""
    from sqlalchemy import select

    from dynamo.persistence import schema

    inst = schema.instruments
    stmt = select(inst.c.symbol, inst.c.cik).where(inst.c.symbol.in_(symbols))
    with repo._engine.connect() as cx:  # composition-layer seam
        rows = cx.execute(stmt).all()
    return {str(r.symbol): str(r.cik) for r in rows if r.cik}


def _split_crypto(symbols: list[str]) -> tuple[list[str], list[str]]:
    """(equities, crypto pairs) preserving order."""
    from dynamo.domain.symbols import is_crypto_symbol

    equities = [s for s in symbols if not is_crypto_symbol(s)]
    crypto = [s for s in symbols if is_crypto_symbol(s)]
    return equities, crypto


def is_non_common_share(symbol: str, name: str) -> bool:
    """True for warrants / rights / units / preferreds / notes.

    The ``Instrument`` port exposes only ``symbol``/``name``/``exchange``, and
    Alpaca's asset payload carries no security-type field at all, so the NAME
    is the classifier: ``"Biofrontera Inc. Warrants"``, ``"... Units"``,
    ``"... 9.5% Series A Perpetual Preferred Stock"``. Word boundaries keep
    ordinary names ("United", "Community Bank", "Wright") out of the net; a
    bare ``W``/``R``/``U`` ticker suffix is NOT used as a rule because real
    common stocks end in those letters (UNIT, RIGL, WRB). NYSE renders its
    class suffixes with a dot (``MTB.PRJ``, ``ABC.WS``) — that IS reliable.

    Exchange-listed American Depositary Shares (BABA, TSM) are deliberately
    KEPT: they are ordinary surveillance candidates. The unsponsored-ADR junk
    the raw list carries (UPMMY, ASMIY) sits on ``exchange='OTC'`` and is
    removed by the venue filter instead.
    """
    return bool(NON_COMMON_SYMBOL_RE.search(symbol) or NON_COMMON_NAME_RE.search(name or ""))


def filter_surveillance_candidates(
    instruments: Iterable[Instrument],
    *,
    exchanges: Sequence[str] = DEFAULT_SURVEILLANCE_EXCHANGES,
) -> tuple[list[Instrument], dict[str, int]]:
    """Alpaca's tradable list -> real Tier-2 surveillance candidates.

    Returns ``(kept, stage_counts)`` where ``stage_counts`` reports the funnel:
    ``fetched`` (the adapter already applies active + us_equity + tradable),
    ``dropped_exchange``, ``dropped_class``, ``dropped_duplicate``, ``kept``.
    """
    allowed = {e.upper() for e in exchanges}
    counts = {
        "fetched": 0,
        "dropped_exchange": 0,
        "dropped_class": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }
    kept: dict[str, Instrument] = {}
    for inst in instruments:
        counts["fetched"] += 1
        if (inst.exchange or "").upper() not in allowed:
            counts["dropped_exchange"] += 1
            continue
        if is_non_common_share(inst.symbol, inst.name):
            counts["dropped_class"] += 1
            continue
        if inst.symbol in kept:
            counts["dropped_duplicate"] += 1
            continue
        kept[inst.symbol] = inst
    counts["kept"] = len(kept)
    return ([kept[s] for s in sorted(kept)], counts)


def chunked(items: Sequence[str], size: int) -> list[list[str]]:
    """Split ``items`` into consecutive chunks of at most ``size``."""
    step = max(1, int(size))
    return [list(items[i : i + step]) for i in range(0, len(items), step)]


def covered_symbols(
    coverage: dict[str, tuple[date, date]],
    start: date,
    end: date,
    *,
    tolerance_days: int = DEFAULT_COVERAGE_TOLERANCE_DAYS,
) -> set[str]:
    """Symbols whose existing ``(first_ts, last_ts)`` already spans [start, end].

    ``tolerance_days`` of calendar slack absorbs weekends/holidays at both
    edges. A symbol that listed mid-range legitimately never satisfies the
    left edge and is therefore re-fetched on every run — the honest cost of
    not tracking per-symbol listing dates.
    """
    slack = timedelta(days=max(0, tolerance_days))
    return {
        symbol
        for symbol, (first, last) in coverage.items()
        if first <= start + slack and last >= end - slack
    }


def _bar_coverage(repo: Any, source: str) -> dict[str, tuple[date, date]]:
    """symbol -> (first bar date, last bar date) for one ``bars_1d`` source."""
    from sqlalchemy import func, select

    from dynamo.persistence import schema

    t = schema.bars_1d
    stmt = (
        select(t.c.symbol, func.min(t.c.ts), func.max(t.c.ts))
        .where(t.c.source == source)
        .group_by(t.c.symbol)
    )
    with repo._engine.connect() as cx:  # composition-layer seam (see _cik_map)
        rows = cx.execute(stmt).all()
    return {str(r[0]): (r[1].date(), r[2].date()) for r in rows}


def read_symbols_file(path: Path) -> list[str]:
    """Symbols from a text file: one per line, ``#`` starts a comment
    (full-line or trailing), blank lines skipped, duplicates dropped
    keeping first occurrence, case preserved (crypto pairs like BTC/USD).

    The universe-breadth prep path: a TRAINING universe file may be much
    wider than the trading universe (standard quant practice — train on
    breadth, trade a subset). See config/training_universe.txt.example for
    the documented format and the survivorship-bias caveats.
    """
    seen: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = line.split("#", 1)[0].strip()
        if symbol:
            seen.setdefault(symbol, None)
    return list(seen)


# -- subcommands ------------------------------------------------------------


def _cmd_bars(args: argparse.Namespace) -> int:
    settings = _settings()
    provider = getattr(args, "provider", "tiingo")
    if provider == "tiingo" and not settings.tiingo_api_key:
        if args.allow_missing_tiingo:
            print(
                "TIINGO_API_KEY is not set — skipping bars backfill "
                "(no network calls made; --allow-missing-tiingo). Set the key and re-run."
            )
            return 0
        print(
            "ERROR: TIINGO_API_KEY is not set — bars backfill cannot run "
            "(no network calls made). Set the key and re-run, or pass "
            "--allow-missing-tiingo to skip gracefully."
        )
        return 1
    if provider == "alpaca" and not (settings.alpaca_api_key and settings.alpaca_api_secret):
        print(
            "ERROR: ALPACA_API_KEY / ALPACA_API_SECRET are not set — "
            "--provider alpaca cannot run (no network calls made)."
        )
        return 1
    if args.symbols_file is not None:
        # Fail fast on an unusable file BEFORE any engine/network work.
        raw = read_symbols_file(args.symbols_file)
        if not raw:
            print(f"ERROR: --symbols-file {args.symbols_file} contains no symbols")
            return 1
        print(f"symbols-file: {len(raw)} symbols from {args.symbols_file}")
        repo = _repo(args.database_url)
    elif args.symbols:
        raw = list(args.symbols)
        repo = _repo(args.database_url)
    else:
        repo = _repo(args.database_url)
        tiers = getattr(args, "tiers", None)
        raw = _universe_symbols(repo, args.end, tiers)
        scope = f"tiers {sorted(tiers)}" if tiers else "all tiers"
        print(f"universe: {len(raw)} symbols at {args.end} ({scope})")
    symbols, crypto = _split_crypto(raw)
    if crypto:
        # The equity endpoints 404 on pairs; crypto history comes from `crypto-bars`.
        print(f"skipping crypto pairs (use `backfill crypto-bars`): {', '.join(crypto)}")
    limit = getattr(args, "limit", None)
    if limit is not None and limit > 0:
        symbols = symbols[:limit]
        print(f"--limit {limit}: truncated to {len(symbols)} symbols")
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
    if provider == "alpaca":
        return _bars_alpaca(args, settings, repo, symbols)
    tiingo = _tiingo(settings)
    total = 0
    failed: list[str] = []
    for sym in symbols:
        try:
            rows = repo.upsert_bars(tiingo.daily_bars([sym], args.start, args.end))
        except Exception as exc:  # per-symbol resilience: report at the end, keep going
            failed.append(sym)
            print(f"bars: {sym} FAILED: {exc}")
            continue
        total += rows
        print(f"bars: {sym} {args.start} -> {args.end}: {rows} rows")
    print(f"bars backfill complete: {total} rows across {len(symbols) - len(failed)} symbols")
    if failed:
        print(f"ERROR: {len(failed)} symbols failed: {', '.join(failed)}")
        return 1
    return 0


def _bars_alpaca(args: argparse.Namespace, settings: Any, repo: Any, symbols: list[str]) -> int:
    """Wide, batched, resumable Alpaca daily-bar backfill.

    Many symbols per request (``--chunk-size``, the adapter passes the list
    straight to ``symbol_or_symbols`` and alpaca-py follows page tokens), and
    a failing chunk degrades to per-symbol retries so one bad ticker cannot
    take out 200 good ones.
    """
    requested = list(symbols)
    if not args.force:
        coverage = _bar_coverage(repo, "alpaca")
        already = covered_symbols(
            coverage, args.start, args.end, tolerance_days=args.coverage_tolerance_days
        )
        symbols = [s for s in symbols if s not in already]
        print(
            f"resume: {len(requested) - len(symbols)}/{len(requested)} symbols already "
            f"cover {args.start} -> {args.end} (source=alpaca) — skipping them"
        )
        if not symbols:
            print("bars backfill complete: nothing to do (everything already covered)")
            return 0

    adapter = _alpaca_market(settings)
    chunks = chunked(symbols, args.chunk_size)
    started = time.monotonic()
    total_rows = 0
    with_bars: set[str] = set()
    failed: list[str] = []
    for n, chunk in enumerate(chunks, start=1):
        rows, ok, bad = _fetch_bar_chunk(adapter, repo, chunk, args.start, args.end)
        total_rows += rows
        with_bars |= ok
        failed.extend(bad)
        elapsed = time.monotonic() - started
        done = min(n * args.chunk_size, len(symbols))
        print(
            f"bars[alpaca] chunk {n}/{len(chunks)} ({len(chunk)} symbols): "
            f"{rows} rows, {len(ok)} with data, {len(bad)} failed | "
            f"{total_rows} rows / {done} symbols in {elapsed:.0f}s "
            f"({done / max(elapsed, 1e-9):.1f} sym/s)"
        )
    elapsed = time.monotonic() - started
    empty = sorted(set(symbols) - with_bars - set(failed))
    print(
        f"bars backfill complete [alpaca IEX]: {total_rows} rows, "
        f"{len(with_bars)}/{len(symbols)} symbols returned data, "
        f"{len(empty)} returned none, {len(failed)} failed, "
        f"{elapsed:.0f}s ({total_rows / max(elapsed, 1e-9):.0f} rows/s)"
    )
    print(
        "NOTE: the free Alpaca feed is IEX-only — bar VOLUME understates the "
        "consolidated tape (measured ~6.6% of it, i.e. ~15x). Compare these "
        "rows against a threshold only after converting with "
        "dynamo.domain.volume_source.consolidated_equivalent_dollar_volume; a "
        "floor calibrated on Tiingo data will otherwise over-reject ~15x."
    )
    if empty:
        print(f"symbols with no IEX bars in range (first 25): {empty[:25]}")
    if failed:
        print(f"ERROR: {len(failed)} symbols failed: {failed[:25]}")
        return 1
    return 0


def _fetch_bar_chunk(
    adapter: Any, repo: Any, chunk: list[str], start: date, end: date
) -> tuple[int, set[str], list[str]]:
    """One multi-symbol request; on failure retry the chunk symbol-by-symbol."""
    try:
        bars = list(adapter.daily_bars(chunk, start, end))
    except Exception as exc:
        if len(chunk) == 1:
            print(f"bars[alpaca]: {chunk[0]} FAILED: {exc}")
            return (0, set(), list(chunk))
        print(f"bars[alpaca]: chunk of {len(chunk)} FAILED ({exc}) — retrying per symbol")
        rows_total, ok_total, bad_total = 0, set[str](), list[str]()
        for sym in chunk:
            rows, ok, bad = _fetch_bar_chunk(adapter, repo, [sym], start, end)
            rows_total += rows
            ok_total |= ok
            bad_total.extend(bad)
        return (rows_total, ok_total, bad_total)
    with_data = {b.symbol for b in bars}
    return (repo.upsert_bars(bars), with_data, [])


def _cmd_fundamentals(args: argparse.Namespace) -> int:
    settings = _settings()
    repo = _repo(args.database_url)
    symbols = list(args.symbols) if args.symbols else _universe_symbols(repo, date.today())
    symbols, crypto = _split_crypto(symbols)
    if crypto:
        print(f"fundamentals: crypto pairs have no EDGAR filings — skipped: {crypto}")
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
    edgar = _edgar(settings)
    if args.seed_ciks:
        mapping = edgar.company_ticker_map()
        subset = {s: mapping[s] for s in symbols if s in mapping}
        seeded = repo.seed_instrument_ciks(subset)
        unmapped = sorted(set(symbols) - set(subset))
        print(f"seed-ciks: {seeded} instruments updated from SEC company_tickers.json")
        if unmapped:
            print(f"seed-ciks: no SEC ticker mapping for {unmapped}")
    ciks = _cik_map(repo, symbols)
    missing = sorted(set(symbols) - set(ciks))
    if missing:
        print(
            f"fundamentals: no CIK on file for {missing} — skipped "
            "(hint: --seed-ciks populates instruments.cik from SEC data)"
        )
    from dynamo.data.errors import DataGap, ProviderDown

    total = 0
    no_facts: list[str] = []
    for sym in sorted(ciks):
        try:
            rows = repo.upsert_xbrl_facts(edgar.company_facts(ciks[sym]))
        except (DataGap, ProviderDown) as exc:
            # ETFs/trusts (e.g. DIA, SPY) appear in SEC's ticker map but have
            # no us-gaap companyfacts JSON (404) — skip loudly, keep going.
            no_facts.append(sym)
            print(f"fundamentals: {sym} (CIK {ciks[sym]}): no companyfacts — skipped ({exc})")
            continue
        total += rows
        print(f"fundamentals: {sym} (CIK {ciks[sym]}): {rows} facts")
    if no_facts:
        print(f"fundamentals: no us-gaap companyfacts for {no_facts} (ETF/trust filers) — skipped")
    print(
        f"fundamentals backfill complete: {total} facts across {len(ciks) - len(no_facts)} symbols"
    )
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


def _cmd_universe(args: argparse.Namespace) -> int:
    """Backdate the CORE starter seed to --as-of (replay campaign)."""
    from dynamo.data.universe import seed_symbol_origins
    from dynamo.domain.models.market import Tier, UniverseMembership

    settings = _settings()
    repo = _repo(args.database_url)
    as_of: date = args.as_of
    origins = seed_symbol_origins(include_crypto=bool(settings.crypto_enabled))
    memberships = [
        UniverseMembership(
            symbol=symbol,
            tier=Tier.CORE,
            valid_from=as_of,
            valid_to=None,
            reason=f"replay backfill campaign backdate ({origin})",
            approved_by="replay-backdate",
        )
        for symbol, origin in origins
    ]
    deleted, inserted = repo.backdate_core_universe(memberships)
    starters = {m.symbol for m in memberships}
    equities, crypto = _split_crypto(sorted(starters))
    print(
        f"universe backdate to {as_of}: deleted {deleted} seed rows, "
        f"wrote {inserted} CORE rows ({len(equities)} equities/ETFs, "
        f"{len(crypto)} crypto pairs)"
    )

    # GICS sectors for the equity starters — sector-relative labels, dossier
    # sector context, and risk sector caps all read instruments.sector.
    from dynamo.data.universe import STARTER_SECTORS

    n_sectors = repo.seed_instrument_sectors(STARTER_SECTORS)
    print(f"seeded GICS sectors for {n_sectors} equity instruments")

    # Verification: the starter set must be fully active both at the
    # backdated as-of AND today (open-ended intervals) — anything else means
    # the backdate left a gap or a stale interval behind.
    expected = len(starters)
    failures = 0
    for label, day in (("as-of", as_of), ("today", date.today())):
        members = repo.universe_at(day)
        by_tier: dict[int, int] = {}
        for m in members:
            by_tier[int(m.tier)] = by_tier.get(int(m.tier), 0) + 1
        tier_counts = ", ".join(f"tier {t}: {n}" for t, n in sorted(by_tier.items()))
        core_starters = {
            m.symbol for m in members if int(m.tier) == int(Tier.CORE) and m.symbol in starters
        }
        status = "OK" if len(core_starters) == expected else "MISMATCH"
        print(
            f"verify universe_at({label}={day}): {len(core_starters)}/{expected} "
            f"starter symbols active [{tier_counts or 'empty'}] {status}"
        )
        if len(core_starters) != expected:
            failures += 1
            missing = sorted(starters - core_starters)
            print(f"  missing at {day}: {missing}")
    return 1 if failures else 0


def _cmd_universe_wide(args: argparse.Namespace) -> int:
    """Build the Tier-2 SURVEILLANCE universe from Alpaca's tradable assets."""
    from dynamo.domain.models.market import Tier, UniverseMembership
    from dynamo.persistence import schema

    settings = _settings()
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        print(
            "ERROR: ALPACA_API_KEY / ALPACA_API_SECRET are not set — "
            "universe-wide cannot run (no network calls made)."
        )
        return 1
    as_of: date = args.as_of or date.today()
    adapter = _alpaca_market(settings)
    started = time.monotonic()
    candidates, counts = filter_surveillance_candidates(
        adapter.tradable_instruments(), exchanges=args.exchanges
    )
    print(
        f"alpaca tradable_instruments: {counts['fetched']} active/tradable US-equity assets\n"
        f"  - {counts['dropped_exchange']} off-venue (kept: {', '.join(args.exchanges)})\n"
        f"  - {counts['dropped_class']} warrants/rights/units/preferreds/notes\n"
        f"  - {counts['dropped_duplicate']} duplicate symbols\n"
        f"  = {counts['kept']} surveillance candidates ({time.monotonic() - started:.1f}s)"
    )
    if args.limit is not None and args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"--limit {args.limit}: truncated to {len(candidates)} candidates")
    if not candidates:
        print("ERROR: no candidates survived the filter — refusing to write")
        return 1

    repo = _repo(args.database_url)
    core_before = {m.symbol for m in repo.universe_at(as_of, Tier.CORE)}
    new_members = [i for i in candidates if i.symbol not in core_before]
    print(
        f"tier-1 CORE at {as_of}: {len(core_before)} symbols "
        f"({len(candidates) - len(new_members)} of them also in the Alpaca list) — "
        f"{len(new_members)} symbols to write at Tier 2"
    )
    if args.dry_run:
        print("--dry-run: no writes performed")
        return 0

    # Additive instrument upsert: only symbol/name/exchange are supplied, so
    # `upsert_rows` builds an ON CONFLICT DO UPDATE touching those columns
    # ONLY — existing cik/sector/industry/status survive untouched (the same
    # pattern seed_instrument_ciks / seed_instrument_sectors rely on).
    inst_rows = [{"symbol": i.symbol, "name": i.name, "exchange": i.exchange} for i in candidates]
    n_inst = repo.upsert_rows(schema.instruments, inst_rows)
    print(f"instruments: {n_inst} rows upserted (name/exchange only; cik/sector preserved)")

    memberships = [
        UniverseMembership(
            symbol=i.symbol,
            tier=Tier.SURVEILLANCE,
            valid_from=as_of,
            valid_to=None,
            reason=f"alpaca tradable universe ({i.exchange})",
            approved_by="universe-wide",
        )
        for i in new_members
    ]
    n_mem = repo.upsert_universe_membership(memberships)
    print(f"universe_membership: {n_mem} Tier-2 rows written with valid_from={as_of}")

    # Verification: tier-1 must be byte-identical, tier-2 must be populated.
    core_after = {m.symbol for m in repo.universe_at(as_of, Tier.CORE)}
    tier2_after = {m.symbol for m in repo.universe_at(as_of, Tier.SURVEILLANCE)}
    core_ok = core_after == core_before
    print(
        f"verify tier-1 CORE at {as_of}: {len(core_after)} symbols "
        f"({'UNCHANGED' if core_ok else 'CHANGED — BUG'})"
    )
    print(f"verify tier-2 SURVEILLANCE at {as_of}: {len(tier2_after)} symbols")
    if not core_ok:
        print(f"  added: {sorted(core_after - core_before)[:20]}")
        print(f"  lost: {sorted(core_before - core_after)[:20]}")
        return 1
    return 0


def _cmd_surveillance_sweep(args: argparse.Namespace) -> int:
    """Run the tier-2 bar detectors over the wide universe; write real flags.

    ZERO LLM calls: this is the code-only half of the funnel. The Haiku
    triage that turns flags into Tier-3 promotions lives in the ``tier2_scan``
    Dagster asset and is deliberately NOT invoked here.
    """
    from datetime import UTC, datetime

    from dynamo.data.detectors import (
        price_zscore,
        range_break,
        volume_spike,
    )
    from dynamo.domain.volume_source import (
        IEX_CONSOLIDATED_VOLUME_SHARE,
        dominant_source,
        volume_source_share,
    )

    repo = _repo(args.database_url)
    as_of: date = args.as_of
    symbols = _universe_symbols(repo, as_of, args.tiers)
    _, crypto = _split_crypto(symbols)
    if crypto:
        print(f"surveillance-sweep: {len(crypto)} crypto pairs included in the scan")
    if args.limit is not None and args.limit > 0:
        symbols = sorted(symbols)[: args.limit]
    if not symbols:
        print(f"surveillance-sweep: no members in tiers {args.tiers} at {as_of} — nothing to scan")
        return 0
    start = as_of - timedelta(days=args.lookback_days)
    print(
        f"surveillance-sweep: {len(symbols)} symbols in tiers {sorted(args.tiers)} "
        f"at {as_of}; loading bars {start} -> {as_of}"
    )
    t0 = time.monotonic()
    bars = repo.bars_frame(symbols, start, as_of)
    load_s = time.monotonic() - t0
    if bars.empty:
        print("surveillance-sweep: no bars in range — nothing to scan")
        return 0
    scanned = sorted({str(s) for s in bars["symbol"].unique()})
    print(
        f"bars loaded: {len(bars)} rows / {len(scanned)} symbols with data "
        f"({len(symbols) - len(scanned)} had none) in {load_s:.0f}s"
    )

    if args.min_dollar_volume > 0:
        # --min-dollar-volume is CONSOLIDATED-EQUIVALENT dollars (the whole
        # tape). Observed volume is scaled up per row's source before the
        # comparison, so the flag means the same thing whether the bars came
        # from Tiingo (consolidated) or Alpaca (IEX-only, ~1/15th of it).
        # See dynamo.domain.volume_source for the measurement and its caveats.
        observed = (bars["close"] * bars["volume"]).groupby(bars["symbol"]).median()
        share = (
            bars.groupby("symbol")["source"]
            .agg(dominant_source)
            .map(volume_source_share)
            .reindex(observed.index)
            .fillna(1.0)
        )
        consolidated = observed / share
        liquid = {str(s) for s in consolidated[consolidated >= args.min_dollar_volume].index}
        bars = bars[bars["symbol"].isin(liquid)]
        print(
            f"liquidity floor: {len(liquid)}/{len(scanned)} symbols clear a "
            f"${args.min_dollar_volume:,.0f} median daily dollar volume "
            f"(consolidated-equivalent; IEX rows scaled up by "
            f"{1.0 / IEX_CONSOLIDATED_VOLUME_SHARE:.1f}x)"
        )
        scanned = sorted(liquid)
        if bars.empty:
            print("surveillance-sweep: liquidity floor removed every symbol")
            return 0

    ts = datetime(as_of.year, as_of.month, as_of.day, 21, 0, tzinfo=UTC)
    t0 = time.monotonic()
    flags = [
        *price_zscore(bars, as_of=ts),
        *volume_spike(bars, as_of=ts),
        *range_break(bars, as_of=ts),
    ]
    scan_s = time.monotonic() - t0
    by_detector: dict[str, int] = {}
    for f in flags:
        by_detector[f.detector] = by_detector.get(f.detector, 0) + 1
    print(
        f"detectors ran in {scan_s:.0f}s: {len(flags)} flags over "
        f"{len({f.symbol for f in flags})} distinct symbols "
        f"({len(scanned)} scanned)"
    )
    for detector, n in sorted(by_detector.items()):
        print(f"  {detector}: {n}")
    print(
        "NOT run here (no wide-universe input data): earnings_surprise, "
        "filing_event, insider_cluster, news_burst — those detectors read "
        "event tables that are only ingested for Tier 1. rel_strength needs "
        "instruments.sector, which the Alpaca asset list does not provide."
    )
    for f in sorted(flags, key=lambda f: -f.severity)[:15]:
        print(f"  FLAG {f.symbol:<8} {f.detector:<14} sev={f.severity:.2f} {dict(f.details)}")
    if args.dry_run:
        print("--dry-run: no surveillance_flags rows written")
        return 0
    written = repo.insert_surveillance_flags(flags)
    print(f"surveillance_flags: {written} rows written (ts={ts.isoformat()})")
    return 0


def _cmd_news(args: argparse.Namespace) -> int:
    settings = _settings()
    if not settings.finnhub_api_key:
        print(
            "ERROR: FINNHUB_API_KEY is not set — news backfill cannot run "
            "(no network calls made). Set the key and re-run."
        )
        return 1
    repo = _repo(args.database_url)
    symbols = list(args.symbols) if args.symbols else _universe_symbols(repo, args.end)
    symbols, crypto = _split_crypto(symbols)
    if crypto:
        print(f"{CRYPTO_NEWS_SKIP_MESSAGE}: {crypto}")
    if not symbols:
        print("no equity symbols to backfill (empty universe?)")
        return 0
    from dynamo.data.errors import RateLimited

    finnhub = _finnhub(settings)
    total = 0
    failed: list[str] = []
    for sym in symbols:
        # Finnhub free tier 429s in practice even under the 1 req/s client
        # pacing (observed mid-campaign after ~60 requests). Honor Retry-After
        # and retry the whole symbol (idempotent upserts) before giving up.
        for attempt in range(_NEWS_RATE_LIMIT_RETRIES + 1):
            try:
                rows = repo.upsert_news(finnhub.news_range([sym], args.start, args.end))
            except RateLimited as exc:
                if attempt >= _NEWS_RATE_LIMIT_RETRIES:
                    failed.append(sym)
                    print(f"news: {sym}: rate-limited after retries — giving up ({exc})")
                    break
                pause = exc.retry_after_s or _NEWS_RATE_LIMIT_PAUSE_S
                print(f"news: {sym}: 429 rate-limited — pausing {pause:.0f}s then retrying")
                time.sleep(pause)
                continue
            total += rows
            print(f"news: {sym} {args.start} -> {args.end}: {rows} items")
            break
    if failed:
        print(f"news backfill INCOMPLETE: rate-limited symbols (re-run to fill): {failed}")
    print(f"news backfill complete: {total} items across {len(symbols) - len(failed)} symbols")
    return 1 if failed else 0


def _cmd_transcripts(args: argparse.Namespace) -> int:
    provider = _defeatbeta()
    if not provider.available:
        if args.allow_missing_extra:
            print(
                "transcripts extra not installed — skipping transcripts backfill "
                "(--allow-missing-extra). Install with `uv sync --extra transcripts`."
            )
            return 0
        print(
            "ERROR: the `transcripts` extra is not installed — transcripts "
            "backfill cannot run. Install with `uv sync --extra transcripts` "
            "and re-run, or pass --allow-missing-extra to skip gracefully."
        )
        return 1
    repo = _repo(args.database_url)
    symbols = list(args.symbols) if args.symbols else _universe_symbols(repo, date.today())
    symbols, crypto = _split_crypto(symbols)
    if crypto:
        print(f"transcripts: crypto pairs have no earnings calls — skipped: {crypto}")
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
    fetched = provider.transcripts(symbols, since=args.since)
    rows = repo.upsert_transcripts(fetched)
    linked = 0
    for transcript in fetched:
        linked += repo.link_transcript_ref(transcript.symbol, transcript.event_date, transcript.ref)
    print(
        f"transcripts backfill complete: {rows} transcripts across "
        f"{len(symbols)} symbols since {args.since} "
        f"({linked} earnings events linked)"
    )
    return 0


def _cmd_crypto_bars(args: argparse.Namespace) -> int:
    settings = _settings()
    repo = _repo(args.database_url)
    symbols = list(args.symbols) if args.symbols else list(settings.crypto_universe)
    if not symbols:
        print("no crypto symbols to backfill (crypto universe empty?)")
        return 0
    adapter = _alpaca_crypto(settings)
    bars = list(adapter.daily_bars(symbols, args.start, args.end))
    per_symbol: dict[str, int] = {}
    for b in bars:
        per_symbol[b.symbol] = per_symbol.get(b.symbol, 0) + 1
    rows = repo.upsert_bars(bars)
    for sym in symbols:
        print(f"crypto-bars: {sym} {args.start} -> {args.end}: {per_symbol.get(sym, 0)} bars")
    print(f"crypto-bars backfill complete: {rows} rows across {len(symbols)} symbols")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "bars": _cmd_bars,
        "fundamentals": _cmd_fundamentals,
        "macro": _cmd_macro,
        "universe": _cmd_universe,
        "universe-wide": _cmd_universe_wide,
        "surveillance-sweep": _cmd_surveillance_sweep,
        "news": _cmd_news,
        "transcripts": _cmd_transcripts,
        "crypto-bars": _cmd_crypto_bars,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
