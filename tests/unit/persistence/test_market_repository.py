def test_upsert_news_dedupes_within_batch() -> None:
    """A provider repeating a story id inside ONE fetch must not blow up the
    multi-VALUES upsert (CardinalityViolation) — failed live every 30 min on
    night one. Last occurrence wins."""
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from dynamo.domain.models.market import NewsItem
    from dynamo.persistence.repositories.market import MarketDataRepository

    repo = MarketDataRepository.__new__(MarketDataRepository)
    captured: dict = {}
    repo.upsert_rows = MagicMock(
        side_effect=lambda t, rows: captured.setdefault("rows", rows) and len(rows) or len(rows)
    )  # type: ignore[method-assign]
    dup = NewsItem(
        id="story-1",
        symbols=("AAPL",),
        headline="first",
        summary="",
        source="s",
        published_at=datetime(2026, 9, 14, tzinfo=UTC),
        url="u",
    )
    dup2 = NewsItem(
        id="story-1",
        symbols=("AAPL",),
        headline="second",
        summary="",
        source="s",
        published_at=datetime(2026, 9, 14, tzinfo=UTC),
        url="u",
    )
    other = NewsItem(
        id="story-2",
        symbols=("MSFT",),
        headline="x",
        summary="",
        source="s",
        published_at=datetime(2026, 9, 14, tzinfo=UTC),
        url="u",
    )
    repo.upsert_news([dup, dup2, other])
    rows = captured["rows"]
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id["story-1"]["headline"] == "second", "last occurrence wins"
