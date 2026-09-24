"""Ingestion assets — doc 02 §4 cadence table.

Each asset: fetch via provider adapter -> upsert via MarketDataRepository ->
return row-count metadata. Missing API key -> log skip, empty metadata; the
platform runs partially configured (doc 01 §6 per-asset degradation).
"""

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, time, timedelta

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    OpExecutionContext,
    job,
    op,
)
from dagster import asset as dagster_asset

from dynamo.data.errors import RateLimited
from orchestration.partitions import daily_partitions
from orchestration.resources import DEFAULT_FRED_SERIES, DynamoResource, as_of_from_partition

INGESTION_GROUP = "ingestion"

# EDGAR poller cadence: every 10 minutes; overlap generously for idempotent upserts.
FILINGS_POLL_LOOKBACK = timedelta(minutes=30)


def _skipped(reason: str) -> MaterializeResult:
    return MaterializeResult(metadata={"skipped": reason, "rows": 0})


def _tolerate_rate_limit[T](
    items: Iterable[T],
    context: AssetExecutionContext,
    *,
    provider: str,
    flags: dict[str, bool],
    asset: str = "news",
) -> Iterator[T]:
    """Yield from ``items`` until the provider rate-limits, then stop cleanly.

    A 429 mid-sweep is an expected operating condition on the free tiers
    (Finnhub caps at 60 req/min against a universe several times that), not
    an incident: ``get_json`` deliberately does not retry it (doc 01 §6, "no
    retry storm"). Letting it escape, though, cost us the whole asset — every
    symbol after the limit AND the second, unrelated provider downstream.

    Stopping iteration instead keeps what was already fetched (the upserts
    materialise the generator, so partial results persist) and records the
    truncation in metadata rather than swallowing it silently.

    The same shape applies to the EOD price ingest: Tiingo's free tier allows
    500 unique symbols a month, and once a research backfill has spent them
    the nightly Tiingo leg 429s — which must not turn the run red when the
    Alpaca leg (the one every symbol actually relies on) already landed.
    """
    try:
        yield from items
    except RateLimited as exc:
        flags[provider] = True
        context.log.warning(
            "%s: %s rate-limited mid-sweep (%s) — persisting partial results",
            asset,
            provider,
            exc,
        )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def eod_bars(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """EOD bars, full market, both sources (Alpaca IEX + Tiingo adjusted)."""
    from dynamo.domain.symbols import is_crypto_symbol

    as_of = as_of_from_partition(context.partition_key)
    repo = dynamo.market_repo()
    # Crypto pairs share the universe table but not the equity endpoints: one
    # "AVAX/USD" in a batch fails the whole batch ("invalid symbol", HTTP 400).
    # Their bars come from crypto_bars, exactly as the backfill CLI splits them.
    symbols = [s for s in dynamo.universe_symbols(as_of) if not is_crypto_symbol(s)]
    if not symbols:
        context.log.warning("eod_bars: universe empty at %s — seed it first", as_of)
        return _skipped("empty universe")
    # Tiingo is the ADJUSTED-price source for the tier-1 core (canonical_bars:
    # "only 55 names carry Tiingo"); Alpaca covers the whole market. Tiingo is
    # one request per symbol, and the free plan allows 50/hour and 500 unique
    # symbols/month — asking it for all ~11,000 names could never succeed.
    tier1 = set(dynamo.universe_symbols(as_of, tiers=(1,)))
    scope = {"alpaca": symbols, "tiingo": [s for s in symbols if s in tier1]}
    counts: dict[str, int] = {}
    limited: dict[str, bool] = {}
    for name, provider in (("alpaca", dynamo.alpaca()), ("tiingo", dynamo.tiingo())):
        if provider is None:
            context.log.warning("eod_bars: %s not configured — skipping source", name)
            counts[name] = 0
            continue
        bars = _tolerate_rate_limit(
            provider.daily_bars(scope[name], as_of, as_of),
            context,
            provider=name,
            flags=limited,
            asset="eod_bars",
        )
        counts[name] = repo.upsert_bars(bars, timeframe="1d")
    return MaterializeResult(
        metadata={
            "rows": sum(counts.values()),
            "symbols": len(symbols),
            **counts,
            "rate_limited": sorted(limited) or "none",
        }
    )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def crypto_bars(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Daily crypto bars for the sleeve universe (ADR-0013).

    Crypto trades 24/7, so this runs every calendar day (21:35 ET, right
    after the equities EOD ingest). A short trailing window is fetched each
    night — idempotent upserts backfill weekend/holiday gaps and refresh the
    in-progress UTC day next run. Crypto instrument rows (sector="crypto")
    are upserted alongside so dossiers/risk sector caps see the sleeve.
    """
    from dynamo.data.universe import crypto_instruments

    as_of = as_of_from_partition(context.partition_key)
    provider = dynamo.alpaca_crypto()
    if provider is None:
        context.log.info("crypto_bars: crypto sleeve disabled — skipping")
        return _skipped("crypto disabled")
    symbols = list(dynamo.settings().crypto_universe)
    if not symbols:
        return _skipped("empty crypto universe")
    repo = dynamo.market_repo()
    bars = provider.daily_bars(symbols, as_of - timedelta(days=7), as_of)
    rows = repo.upsert_bars(bars, timeframe="1d")
    instruments = repo.upsert_instruments(crypto_instruments(symbols))
    return MaterializeResult(
        metadata={"rows": rows, "symbols": len(symbols), "instruments": instruments}
    )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def corporate_actions(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Splits/dividends from both sources (doc 02 §5)."""
    as_of = as_of_from_partition(context.partition_key)
    start = as_of - timedelta(days=7)
    repo = dynamo.market_repo()
    total = 0
    limited: dict[str, bool] = {}
    for name, provider in (("alpaca", dynamo.alpaca()), ("tiingo", dynamo.tiingo())):
        if provider is None:
            context.log.warning("corporate_actions: %s not configured — skipping", name)
            continue
        actions = _tolerate_rate_limit(
            provider.corporate_actions(start, as_of),
            context,
            provider=name,
            flags=limited,
            asset="corporate_actions",
        )
        total += repo.upsert_corporate_actions(actions)
    return MaterializeResult(metadata={"rows": total, "rate_limited": sorted(limited) or "none"})


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def insider_tx(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Form 4 insider transactions from EDGAR (hourly schedule, daily partition).

    Filtered to the CIKs of the tracked universe, and filtered BEFORE the
    ownership XML is fetched. The unfiltered path took the first
    ``max_insider_filings`` (50) Form 4s of the day out of ~1,000 filed
    market-wide — i.e. an arbitrary 5% of an arbitrary slice of the market,
    almost none of it names the platform trades. ``insider_cluster`` needs
    two insiders buying the SAME symbol inside 14 days, which that sample
    could only ever produce by luck.

    Falls back to the whole market when no instrument carries a CIK, so a
    fresh database still ingests something rather than silently nothing.
    """
    as_of = as_of_from_partition(context.partition_key)
    edgar = dynamo.edgar()
    if edgar is None:  # pragma: no cover - EDGAR needs no key
        return _skipped("edgar not configured")
    since = as_of - timedelta(days=1)
    ciks = dynamo.instrument_ciks()
    if not ciks:
        context.log.warning(
            "insider_tx: no instrument carries a CIK — sweeping the whole market; "
            "run `python -m dynamo.cli.reference ciks` to scope this"
        )
    txs = edgar.insider_transactions_in_range(since, as_of, ciks=list(ciks.values()) or None)
    rows = dynamo.market_repo().upsert_insider_tx(txs)
    return MaterializeResult(metadata={"rows": rows, "scoped_to_ciks": len(ciks)})


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def news(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Finnhub headlines for Tier 1 + promoted (30-min cadence, tiered scope)."""
    as_of = as_of_from_partition(context.partition_key)
    finnhub = dynamo.finnhub()
    if finnhub is None:
        context.log.warning("news: FINNHUB_API_KEY unset — skipping")
        return _skipped("finnhub not configured")
    symbols = dynamo.universe_symbols(as_of, tiers=(1, 3))
    if not symbols:
        return _skipped("no tier-1/promoted symbols")
    since = datetime.combine(as_of - timedelta(days=1), time.min, tzinfo=UTC)
    repo = dynamo.market_repo()
    # Each source is fetched under its own rate-limit guard: a 429 from one
    # provider must not cost us the other's headlines (they are independent
    # APIs, and TickerTick is keyless).
    limited: dict[str, bool] = {}
    rows = repo.upsert_news(
        _tolerate_rate_limit(
            finnhub.news(symbols, since), context, provider="finnhub", flags=limited
        )
    )
    # Second source: TickerTick (free, keyless). Ids are namespaced per source
    # ('finnhub:<id>' vs 'tickertick:<id>') so the shared upsert needs no dedupe.
    tickertick = dynamo.tickertick()
    tickertick_rows = 0
    if tickertick is None:
        context.log.info("news: tickertick disabled — finnhub only")
    else:
        tickertick_rows = repo.upsert_news(
            _tolerate_rate_limit(
                tickertick.news(symbols, since), context, provider="tickertick", flags=limited
            )
        )
    return MaterializeResult(
        metadata={
            "rows": rows + tickertick_rows,
            "finnhub": rows,
            "tickertick": tickertick_rows,
            "symbols": len(symbols),
            "rate_limited": sorted(limited) or "none",
        }
    )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def macro_series(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """FRED macro series with vintages (daily 08:00 ET)."""
    as_of = as_of_from_partition(context.partition_key)
    fred = dynamo.fred()
    if fred is None:
        context.log.warning("macro_series: FRED_API_KEY unset — skipping")
        return _skipped("fred not configured")
    repo = dynamo.market_repo()
    start = as_of - timedelta(days=45)
    rows = 0
    for series_id in DEFAULT_FRED_SERIES:
        rows += repo.upsert_macro(fred.series(series_id, start))
    return MaterializeResult(metadata={"rows": rows, "series": len(DEFAULT_FRED_SERIES)})


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def earnings_calendar(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Upcoming earnings events (Finnhub calendar, daily).

    Alongside the (destructive) ``earnings_events`` upsert, each fetch
    APPENDS to ``earnings_estimate_vintages`` when an estimate changed vs
    the latest stored vintage (migration 0019) — the estimate history that
    makes the estimate-revision features live for future dates.
    """
    as_of = as_of_from_partition(context.partition_key)
    finnhub = dynamo.finnhub()
    if finnhub is None:
        context.log.warning("earnings_calendar: FINNHUB_API_KEY unset — skipping")
        return _skipped("finnhub not configured")
    events = list(finnhub.earnings_calendar(as_of - timedelta(days=7), as_of + timedelta(days=21)))
    repo = dynamo.market_repo()
    rows = repo.upsert_earnings(events)
    vintages = repo.append_estimate_vintages(events)
    return MaterializeResult(metadata={"rows": rows, "estimate_vintages": vintages})


@dagster_asset(
    group_name=INGESTION_GROUP,
    partitions_def=daily_partitions,
    deps=["earnings_calendar"],
)
def transcripts(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Earnings-call transcripts for yesterday's + today's reporters.

    Alpha Vantage first (fresh, hard 25/day free quota -> settings cap),
    defeatbeta-api as fallback/backfill for whatever AV could not serve.
    Each source degrades independently; no source configured -> logged skip.
    """
    as_of = as_of_from_partition(context.partition_key)
    av = dynamo.alphavantage_transcripts()
    defeatbeta = dynamo.defeatbeta_transcripts()
    if av is None and defeatbeta is None:
        context.log.warning(
            "transcripts: ALPHAVANTAGE_API_KEY unset and transcripts extra not installed — skipping"
        )
        return _skipped("no transcript source configured")
    universe = set(dynamo.universe_symbols(as_of))
    if not universe:
        return _skipped("empty universe")
    reporters = [
        r
        for r in dynamo.earnings_reporters(as_of - timedelta(days=1), as_of)
        if r["symbol"] in universe
    ][: dynamo.settings().transcript_daily_cap]
    if not reporters:
        return MaterializeResult(metadata={"rows": 0, "reporters": 0})

    fetched = []
    missing = []
    for r in reporters:
        transcript = av.fetch(r["symbol"], r["event_date"], period=r["period"]) if av else None
        if transcript is not None:
            fetched.append(transcript)
        else:
            missing.append(r)
    av_count = len(fetched)
    if missing and defeatbeta is not None:
        earliest = min(r["event_date"] for r in missing)
        fetched.extend(
            defeatbeta.transcripts(
                sorted({r["symbol"] for r in missing}),
                since=earliest - timedelta(days=7),
            )
        )
    repo = dynamo.market_repo()
    rows = repo.upsert_transcripts(fetched)
    linked = 0
    for transcript in fetched:
        linked += repo.link_transcript_ref(transcript.symbol, transcript.event_date, transcript.ref)
    return MaterializeResult(
        metadata={
            "rows": rows,
            "reporters": len(reporters),
            "alphavantage": av_count,
            "defeatbeta": len(fetched) - av_count,
            "linked_earnings_events": linked,
        }
    )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def prediction_market_snapshot(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Daily read-only snapshot of Kalshi + Polymarket (doc 02 §7)."""
    repo = dynamo.market_repo()
    markets_total = 0
    prices_total = 0
    for name, provider in (("kalshi", dynamo.kalshi()), ("polymarket", dynamo.polymarket())):
        if provider is None:
            context.log.warning("prediction_market_snapshot: %s not configured", name)
            continue
        markets = list(provider.markets())
        markets_total += repo.upsert_prediction_markets(markets)
        prices_total += repo.upsert_prediction_market_prices(
            provider.prices([m.market_id for m in markets])
        )
    return MaterializeResult(metadata={"markets": markets_total, "prices": prices_total})


# One Gamma /markets page per matcher run — be gentle with the keyless API.
PM_MATCH_PAGE_LIMIT = 500


def _pm_matcher_fallback(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Title-match Kalshi (stored) vs Polymarket (one live Gamma page) pairs.

    Runs only when OddsPipe produced ZERO pairs (see ``oddspipe_spreads``).
    Every accepted match is written to ``pm_cross_venue_spreads`` with both
    venues' YES prices — Kalshi's from the latest stored
    ``prediction_market_prices`` row, Polymarket's from the Gamma listing —
    and logged as a CANDIDATE PAIR NEEDING VERIFICATION: a title match is not
    resolution equivalence, and the arb sleeve keeps hard-gating trades on
    ``pm_arb_verified_pairs``.
    """
    from dynamo.domain.models.pm import CrossVenueSpread
    from dynamo.research.pm_matching import match_markets

    kalshi_markets = [
        m
        for m in dynamo.stored_prediction_markets("kalshi")
        if m.resolved_outcome is None  # resolved markets cannot be paired
    ]
    if not kalshi_markets:
        context.log.warning(
            "oddspipe_spreads: matcher fallback found no stored kalshi markets — "
            "run prediction_market_snapshot first"
        )
        return _skipped("no stored kalshi markets")
    polymarket = dynamo.polymarket()
    if polymarket is None:  # defensive: the real resource is always available
        return _skipped("polymarket not configured")
    listed = polymarket.open_markets_with_yes(limit=PM_MATCH_PAGE_LIMIT)
    pm_yes = {market.market_id: yes for market, yes in listed}
    result = match_markets(
        kalshi_markets,
        [market for market, _yes in listed],
        min_score=dynamo.settings().pm_match_min_score,
    )
    kalshi_yes = dynamo.latest_prediction_market_yes("kalshi")

    now = datetime.now(UTC)
    spreads: list[CrossVenueSpread] = []
    unpriced = 0
    for pair in result.pairs:
        k_yes = kalshi_yes.get(pair.kalshi_market_id)
        p_yes = pm_yes.get(pair.polymarket_id)
        if k_yes is None or p_yes is None or not 0.0 <= p_yes <= 1.0:
            unpriced += 1
            continue
        spreads.append(
            CrossVenueSpread(
                question=pair.kalshi_question,
                kalshi_market_id=pair.kalshi_market_id,
                kalshi_yes=k_yes,
                polymarket_id=pair.polymarket_id,
                polymarket_yes=p_yes,
                yes_diff=k_yes - p_yes,
                as_of=now,
            )
        )
    rows = dynamo.market_repo().upsert_cross_venue_spreads(spreads)

    # Compact candidate table for the human verification pass.
    for spread in spreads:
        context.log.info(
            "pm_matcher CANDIDATE %s|%s  kalshi_yes=%.2f polymarket_yes=%.2f gap=%+.2f  %r",
            spread.kalshi_market_id,
            spread.polymarket_id,
            spread.kalshi_yes,
            spread.polymarket_yes,
            spread.yes_diff,
            spread.question,
        )
    for ambiguous in result.ambiguous:
        context.log.info(
            "pm_matcher AMBIGUOUS %s matches %s — not paired, resolve by hand",
            ambiguous.kalshi_market_id,
            ",".join(ambiguous.polymarket_ids),
        )
    if spreads:
        context.log.info(
            "pm_matcher: %d candidate pair(s) written. To trade one in sim, open BOTH "
            "venues' full market rules, confirm resolution criteria/source/expiry are "
            "identical, then add the pair id to PM_ARB_VERIFIED_PAIRS='[\"KALSHI|pmid\", ...]'. "
            "Until then candidates stay research rows (find_cross_venue_arbs -> unverified).",
            len(spreads),
        )
    return MaterializeResult(
        metadata={
            "rows": rows,
            "pairs": len(spreads),
            "source": "matcher",
            "ambiguous": len(result.ambiguous),
            "unpriced": unpriced,
            "unmatched_kalshi": result.unmatched_counts.kalshi,
            "unmatched_polymarket": result.unmatched_counts.polymarket,
        }
    )


@dagster_asset(
    group_name=INGESTION_GROUP,
    partitions_def=daily_partitions,
    deps=["prediction_market_snapshot"],
)
def oddspipe_spreads(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Cross-venue matched pairs: OddsPipe primary, title matcher fallback.

    Runs in the 20:00 ET snapshot chain, after the primary Kalshi/Polymarket
    snapshot. OddsPipe stays the primary source whenever it returns pairs;
    when it returns ZERO (key unset, API down, or — as observed — an empty
    catalog with working auth) the in-house title matcher
    (:mod:`dynamo.research.pm_matching`) pairs the stored Kalshi snapshot
    against one live Gamma page instead.

    Provenance note: ``pm_cross_venue_spreads`` has no source column and a
    committee replay forbids migrations right now, so matcher rows are NOT
    tagged in the table — they are written ONLY when the OddsPipe fetch
    returned zero pairs, and each run's metadata carries ``source``
    ('oddspipe' | 'matcher') so run history disambiguates any mixed rows.
    """
    provider = dynamo.oddspipe()
    spreads = provider.spreads() if provider is not None else []
    if spreads:
        rows = dynamo.market_repo().upsert_cross_venue_spreads(spreads)
        return MaterializeResult(
            metadata={"rows": rows, "pairs": len(spreads), "source": "oddspipe"}
        )
    if provider is None:
        context.log.warning(
            "oddspipe_spreads: ODDSPIPE_API_KEY unset — falling back to the title matcher"
        )
    else:
        context.log.info(
            "oddspipe_spreads: no matched pairs returned — falling back to the title matcher"
        )
    return _pm_matcher_fallback(context, dynamo)


# --- EDGAR 10-minute poller: schedule-driven op job, not an asset (doc 02 §4) ---


@op
def poll_filings(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    edgar = dynamo.edgar()
    if edgar is None:  # pragma: no cover - EDGAR needs no key
        context.log.warning("filings_poll: edgar not configured — skipping")
        return
    since = datetime.now(UTC) - FILINGS_POLL_LOOKBACK
    rows = dynamo.market_repo().upsert_filings(edgar.recent_filings(since))
    context.log.info("filings_poll upserted %d filings since %s", rows, since.isoformat())


@job
def filings_poll_job() -> None:
    """10-minute EDGAR poller on market days."""
    poll_filings()


INGESTION_ASSETS = [
    eod_bars,
    crypto_bars,
    corporate_actions,
    insider_tx,
    news,
    macro_series,
    earnings_calendar,
    transcripts,
    prediction_market_snapshot,
    oddspipe_spreads,
]
