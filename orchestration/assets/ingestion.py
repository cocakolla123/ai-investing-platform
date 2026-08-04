"""Ingestion assets — doc 02 §4 cadence table.

Each asset: fetch via provider adapter -> upsert via MarketDataRepository ->
return row-count metadata. Missing API key -> log skip, empty metadata; the
platform runs partially configured (doc 01 §6 per-asset degradation).
"""

from datetime import UTC, datetime, time, timedelta

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    OpExecutionContext,
    job,
    op,
)
from dagster import asset as dagster_asset

from orchestration.partitions import daily_partitions
from orchestration.resources import DEFAULT_FRED_SERIES, DynamoResource, as_of_from_partition

INGESTION_GROUP = "ingestion"

# EDGAR poller cadence: every 10 minutes; overlap generously for idempotent upserts.
FILINGS_POLL_LOOKBACK = timedelta(minutes=30)


def _skipped(reason: str) -> MaterializeResult:
    return MaterializeResult(metadata={"skipped": reason, "rows": 0})


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def eod_bars(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """EOD bars, full market, both sources (Alpaca IEX + Tiingo adjusted)."""
    as_of = as_of_from_partition(context.partition_key)
    repo = dynamo.market_repo()
    symbols = dynamo.universe_symbols(as_of)
    if not symbols:
        context.log.warning("eod_bars: universe empty at %s — seed it first", as_of)
        return _skipped("empty universe")
    counts: dict[str, int] = {}
    for name, provider in (("alpaca", dynamo.alpaca()), ("tiingo", dynamo.tiingo())):
        if provider is None:
            context.log.warning("eod_bars: %s not configured — skipping source", name)
            counts[name] = 0
            continue
        bars = provider.daily_bars(symbols, as_of, as_of)
        counts[name] = repo.upsert_bars(bars, timeframe="1d")
    return MaterializeResult(
        metadata={"rows": sum(counts.values()), "symbols": len(symbols), **counts}
    )


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def corporate_actions(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Splits/dividends from both sources (doc 02 §5)."""
    as_of = as_of_from_partition(context.partition_key)
    start = as_of - timedelta(days=7)
    repo = dynamo.market_repo()
    total = 0
    for name, provider in (("alpaca", dynamo.alpaca()), ("tiingo", dynamo.tiingo())):
        if provider is None:
            context.log.warning("corporate_actions: %s not configured — skipping", name)
            continue
        total += repo.upsert_corporate_actions(provider.corporate_actions(start, as_of))
    return MaterializeResult(metadata={"rows": total})


@dagster_asset(group_name=INGESTION_GROUP, partitions_def=daily_partitions)
def insider_tx(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Form 4 insider transactions from EDGAR (hourly schedule, daily partition)."""
    as_of = as_of_from_partition(context.partition_key)
    edgar = dynamo.edgar()
    if edgar is None:  # pragma: no cover - EDGAR needs no key
        return _skipped("edgar not configured")
    since = datetime.combine(as_of - timedelta(days=1), time.min, tzinfo=UTC)
    rows = dynamo.market_repo().upsert_insider_tx(edgar.insider_transactions(since))
    return MaterializeResult(metadata={"rows": rows})


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
    rows = dynamo.market_repo().upsert_news(finnhub.news(symbols, since))
    return MaterializeResult(metadata={"rows": rows, "symbols": len(symbols)})


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
    """Upcoming earnings events (Finnhub calendar, daily)."""
    as_of = as_of_from_partition(context.partition_key)
    finnhub = dynamo.finnhub()
    if finnhub is None:
        context.log.warning("earnings_calendar: FINNHUB_API_KEY unset — skipping")
        return _skipped("finnhub not configured")
    rows = dynamo.market_repo().upsert_earnings(
        finnhub.earnings_calendar(as_of - timedelta(days=7), as_of + timedelta(days=21))
    )
    return MaterializeResult(metadata={"rows": rows})


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
    corporate_actions,
    insider_tx,
    news,
    macro_series,
    earnings_calendar,
    prediction_market_snapshot,
]
