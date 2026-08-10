"""social_burst detector tests: bursts, noise floor, robust fallback, determinism."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dynamo.data.social import mention_counts, run_social_scan, social_burst
from dynamo.domain.models.market import NewsItem

AS_OF = datetime(2026, 8, 4, 22, 30, tzinfo=UTC)


def make_items(
    symbol: str, *, days_ago: int, count: int, source: str = "reddit"
) -> list[NewsItem]:
    # +30min keeps day-14 items inside the strict lookback window while
    # staying on the intended calendar day; day-0 items must precede as_of.
    offset = timedelta(minutes=-30) if days_ago == 0 else timedelta(minutes=30)
    published = AS_OF - timedelta(days=days_ago) + offset
    return [
        NewsItem(
            id=f"{source}:{symbol}-{days_ago}-{i}",
            symbols=(symbol,),
            headline=f"{symbol} mention {i}",
            source=source,
            published_at=published,
        )
        for i in range(count)
    ]


def steady(symbol: str, per_day: int, days: int = 14) -> list[NewsItem]:
    items: list[NewsItem] = []
    for d in range(1, days + 1):
        items += make_items(symbol, days_ago=d, count=per_day)
    return items


def test_mention_counts_shape_dedup_and_as_of_discipline() -> None:
    items = make_items("ACME", days_ago=0, count=3)
    items += items[:1]  # duplicate id counted once
    items.append(  # published after as_of: excluded
        NewsItem(
            id="reddit:future",
            symbols=("ACME",),
            headline="future",
            source="reddit",
            published_at=AS_OF + timedelta(hours=2),
        )
    )
    frame = mention_counts(items, as_of=AS_OF, lookback_days=14)
    assert list(frame.columns) == ["symbol", "day", "source", "count"]
    assert len(frame) == 1
    assert int(frame.iloc[0]["count"]) == 3


def test_planted_burst_flagged_with_sources_breakdown() -> None:
    items = steady("BURST", 1)
    items += make_items("BURST", days_ago=0, count=15, source="reddit")
    items += make_items("BURST", days_ago=0, count=5, source="stocktwits")
    flags = social_burst(mention_counts(items, as_of=AS_OF, lookback_days=14), as_of=AS_OF)
    assert len(flags) == 1
    flag = flags[0]
    assert flag.symbol == "BURST"
    assert flag.detector == "social_burst"
    assert flag.ts == AS_OF
    assert 0.0 < flag.severity <= 1.0
    assert flag.details["today_count"] == 20.0
    assert flag.details["baseline_mean"] == 1.0
    assert flag.details["sources"] == "reddit=15,stocktwits=5"
    zscore = flag.details["zscore"]
    assert isinstance(zscore, float) and zscore > 4.0


def test_quiet_symbol_not_flagged() -> None:
    # Steadily chatty (12/day incl. today, above min_mentions) but no burst.
    items = steady("QUIET", 12) + make_items("QUIET", days_ago=0, count=12)
    counts = mention_counts(items, as_of=AS_OF, lookback_days=14)
    assert social_burst(counts, as_of=AS_OF) == []


def test_min_mentions_floor_beats_huge_zscore() -> None:
    # Zero baseline then 5 mentions today: infinite relative jump, still noise.
    items = make_items("TINY", days_ago=0, count=5)
    counts = mention_counts(items, as_of=AS_OF, lookback_days=14)
    assert social_burst(counts, as_of=AS_OF, min_mentions=10) == []
    # Same data clears a lower floor.
    assert len(social_burst(counts, as_of=AS_OF, min_mentions=5)) == 1


def test_flat_baseline_uses_robust_fallback() -> None:
    # Constant baseline → std == 0 → median/MAD fallback (excess over median).
    items = steady("FLAT", 2) + make_items("FLAT", days_ago=0, count=30)
    flags = social_burst(mention_counts(items, as_of=AS_OF, lookback_days=14), as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["zscore"] == 28.0  # 30 - median(2)


def variable_baseline(symbol: str) -> list[NewsItem]:
    items: list[NewsItem] = []
    for d in range(1, 15):
        items += make_items(symbol, days_ago=d, count=4 if d % 2 else 6)
    return items


def test_severity_monotone_in_zscore() -> None:
    items = variable_baseline("LO") + make_items("LO", days_ago=0, count=10)
    items += variable_baseline("HI") + make_items("HI", days_ago=0, count=12)
    flags = social_burst(mention_counts(items, as_of=AS_OF, lookback_days=14), as_of=AS_OF)
    by_symbol = {f.symbol: f for f in flags}
    assert set(by_symbol) == {"LO", "HI"}
    assert by_symbol["HI"].severity > by_symbol["LO"].severity
    assert by_symbol["HI"].severity < 1.0  # not saturated, ordering is real


def test_run_social_scan_deterministic() -> None:
    reddit = steady("BURST", 1) + make_items("BURST", days_ago=0, count=18)
    stocktwits = make_items("BURST", days_ago=0, count=4, source="stocktwits")
    first = run_social_scan(reddit, stocktwits, as_of=AS_OF)
    second = run_social_scan(reddit, stocktwits, as_of=AS_OF)
    assert first == second
    assert len(first) == 1
