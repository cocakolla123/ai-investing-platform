"""Social awareness scan — Reddit + StockTwits mentions -> burst flags.

Owner-approved design: social activity is a DETECTOR INPUT for the Tier-2
surveillance funnel (awareness tripwire -> Haiku materiality triage), never a
direct buy signal. Mentions persist to the ``news`` table (source='reddit' /
'stocktwits'); flags share the same triage path as every other detector.
"""

from dagster import AssetExecutionContext, MaterializeResult
from dagster import asset as dagster_asset
from sqlalchemy import select

from dynamo.persistence import schema
from orchestration.assets.research import _read_df, _rows_to_models, _since, _triage_flags
from orchestration.resources import DynamoResource, utcnow

SOCIAL_GROUP = "research"


@dagster_asset(group_name=SOCIAL_GROUP, deps=["data_quality"])
def social_scan(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Fetch social mentions, persist, detect bursts, triage material ones."""
    from dynamo.data.reddit import RedditSocialProvider
    from dynamo.data.social import run_social_scan
    from dynamo.data.stocktwits import StockTwitsProvider
    from dynamo.domain.models.market import NewsItem

    settings = dynamo.settings()
    repo = dynamo.market_repo()
    now = utcnow()
    as_of = now.date()

    universe = set(dynamo.universe_symbols(as_of, tiers=(1, 2, 3)))
    if not universe:
        context.log.warning("social_scan: empty universe — skipping")
        return MaterializeResult(metadata={"mentions": 0, "flags": 0, "skipped": "no universe"})

    reddit_items: list[NewsItem] = []
    reddit = RedditSocialProvider(
        settings.reddit_client_id,
        settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        subreddits=tuple(s.strip() for s in settings.reddit_subreddits.split(",") if s.strip()),
    )
    if reddit.configured:
        reddit_items = reddit.mentions(universe)
    else:
        context.log.info("social_scan: reddit creds unset — reddit skipped")

    st_items: list[NewsItem] = []
    if settings.stocktwits_enabled:
        tier1 = dynamo.universe_symbols(as_of, tiers=(1, 3))
        st_items = StockTwitsProvider(enabled=True).mentions(tier1)

    fetched = [*reddit_items, *st_items]
    if fetched:
        repo.upsert_news(fetched, fetched_at=now)

    # Burst detection runs over the persisted 14-day social history so the
    # baseline survives restarts, not just this run's fetch.
    n = schema.news
    lookback = _since(as_of, settings.social_lookback_days + 1)
    history = _rows_to_models(
        NewsItem,
        _read_df(
            dynamo,
            select(n).where(
                n.c.published_at >= lookback, n.c.source.in_(("reddit", "stocktwits"))
            ),
        ),
    )
    flags = run_social_scan(
        [i for i in history if i.source == "reddit"],
        [i for i in history if i.source == "stocktwits"],
        as_of=now,
        zscore_threshold=settings.social_burst_zscore,
        min_mentions=settings.social_min_mentions,
        lookback_days=settings.social_lookback_days,
    )
    if flags:
        repo.insert_surveillance_flags(flags)
    # Persist first, then triage the TABLE: `_triage_flags` reads the
    # un-triaged queue rather than taking a list, so tonight's social flags
    # and anything earlier runs left behind are judged together.
    triaged = _triage_flags(context, dynamo, as_of)
    return MaterializeResult(
        metadata={
            "mentions": len(fetched),
            "reddit": len(reddit_items),
            "stocktwits": len(st_items),
            "flags": len(flags),
            "triaged": triaged,
        }
    )


SOCIAL_ASSETS = [social_scan]
