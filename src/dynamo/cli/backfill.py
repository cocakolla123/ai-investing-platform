"""One-shot historical backfill CLI (plan Phase 4 + backtest-campaign Phase 1).

Bulk range loads that bypass the one-day-per-partition Dagster ingestion,
feeding the replay foundation:

    python -m dynamo.cli.backfill bars --start 2025-01-01 --end 2026-08-01 \
        --symbols AAPL MSFT            # or --universe
    python -m dynamo.cli.backfill fundamentals --universe --seed-ciks
    python -m dynamo.cli.backfill macro --start 2020-01-01
    python -m dynamo.cli.backfill universe --as-of 2024-12-01
    python -m dynamo.cli.backfill news --start 2026-01-01 --end 2026-08-01 --universe
    python -m dynamo.cli.backfill transcripts --universe
    python -m dynamo.cli.backfill crypto-bars --start 2024-01-01 --end 2026-08-01

- ``bars``: Tiingo adjusted EOD history over an arbitrary range (delisted
  names included; the adapter's 5 req/s throttle applies). Missing
  TIINGO_API_KEY is a LOUD failure (exit 1, no network attempted);
  ``--allow-missing-tiingo`` restores the old graceful skip.
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

All writes are the same idempotent natural-key upserts the nightly pipeline
uses — re-running a backfill is safe. NOTE: this command performs real HTTP
calls when run with keys configured; tests mock every adapter.
"""

from __future__ import annotations

import argparse
import sys
import time
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

CRYPTO_NEWS_SKIP_MESSAGE = (
    "crypto news backfill unsupported (finnhub company-news is equities-only) — skipped"
)

# Finnhub free-tier 429 handling for the news backfill: retries per symbol and
# the pause when the response carries no Retry-After header.
_NEWS_RATE_LIMIT_RETRIES = 3
_NEWS_RATE_LIMIT_PAUSE_S = 65.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-backfill",
        description=(
            "One-shot historical range loads "
            "(bars / fundamentals / macro / universe / news / transcripts / crypto-bars)"
        ),
    )
    parser.add_argument("--database-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    bars = sub.add_parser("bars", help="Tiingo daily bars over [start, end]")
    bars.add_argument("--start", required=True, type=date.fromisoformat)
    bars.add_argument("--end", required=True, type=date.fromisoformat)
    group = bars.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", nargs="+")
    group.add_argument("--universe", action="store_true")
    bars.add_argument(
        "--allow-missing-tiingo",
        action="store_true",
        help="missing TIINGO_API_KEY skips gracefully (exit 0) instead of failing",
    )

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


def _split_crypto(symbols: list[str]) -> tuple[list[str], list[str]]:
    """(equities, crypto pairs) preserving order."""
    from dynamo.domain.symbols import is_crypto_symbol

    equities = [s for s in symbols if not is_crypto_symbol(s)]
    crypto = [s for s in symbols if is_crypto_symbol(s)]
    return equities, crypto


# -- subcommands ------------------------------------------------------------


def _cmd_bars(args: argparse.Namespace) -> int:
    settings = _settings()
    if not settings.tiingo_api_key:
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
    repo = _repo(args.database_url)
    raw = list(args.symbols) if args.symbols else _universe_symbols(repo, args.end)
    symbols, crypto = _split_crypto(raw)
    if crypto:
        # Tiingo's equity endpoint 404s on pairs; crypto history comes from `crypto-bars`.
        print(f"skipping crypto pairs (use `backfill crypto-bars`): {', '.join(crypto)}")
    if not symbols:
        print("no symbols to backfill (empty universe?)")
        return 0
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


def _cmd_fundamentals(args: argparse.Namespace) -> int:
    settings = _settings()
    repo = _repo(args.database_url)
    symbols = (
        list(args.symbols) if args.symbols else _universe_symbols(repo, date.today())
    )
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
        print(
            f"fundamentals: no us-gaap companyfacts for {no_facts} "
            "(ETF/trust filers) — skipped"
        )
    print(
        f"fundamentals backfill complete: {total} facts across "
        f"{len(ciks) - len(no_facts)} symbols"
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
    symbols = (
        list(args.symbols) if args.symbols else _universe_symbols(repo, date.today())
    )
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
        linked += repo.link_transcript_ref(
            transcript.symbol, transcript.event_date, transcript.ref
        )
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
        "news": _cmd_news,
        "transcripts": _cmd_transcripts,
        "crypto-bars": _cmd_crypto_bars,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
