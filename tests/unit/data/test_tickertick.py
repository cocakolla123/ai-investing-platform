"""TickerTick adapter: query syntax, parsing, since filter, caps, dedupe."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from dynamo.data.ratelimit import TokenBucket
from dynamo.data.tickertick import TickerTickClient

NOW_MS = 1786000000000  # 2026-08-06T17:06:40Z
OLD_MS = 1600000000000  # 2020 — before `since`

FEED = {
    "stories": [
        {
            "id": "426800427666858718",
            "title": "Acme beats on earnings",
            "url": "https://news.test/acme",
            "site": "news.test",
            "time": NOW_MS,
            "tickers": ["acme", "beta"],
            "description": "Q2 beat across the board.",
        },
        {"id": "111", "title": "Old story", "time": OLD_MS, "tickers": ["acme"]},
        {"title": "No id", "time": NOW_MS},  # dropped
        {"id": "222", "title": "Untagged story", "time": NOW_MS},  # falls back to query symbol
    ],
    "last_id": "-1",
}


def make_client(handler, **kwargs) -> TickerTickClient:
    return TickerTickClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
        **kwargs,
    )


def test_news_parses_and_filters_since() -> None:
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/feed"
        seen_queries.append(request.url.params["q"])
        assert request.url.params["n"] == "50"
        return httpx.Response(200, json=FEED)

    since = datetime(2026, 8, 1, tzinfo=UTC)
    items = list(make_client(handler).news(["ACME"], since))
    assert seen_queries == ["z:acme"]
    assert [i.id for i in items] == ["tickertick:426800427666858718", "tickertick:222"]
    story = items[0]
    assert story.source == "tickertick"
    assert story.headline == "Acme beats on earnings"
    assert story.summary == "Q2 beat across the board."
    assert story.symbols == ("ACME",)  # tagged tickers intersected with the universe
    assert story.published_at == datetime.fromtimestamp(NOW_MS / 1000, tz=UTC)
    assert story.url == "https://news.test/acme"
    assert items[1].symbols == ("ACME",)  # untagged -> the queried symbol


def test_news_dedupes_across_symbols_and_maps_shared_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=FEED)

    since = datetime(2026, 8, 1, tzinfo=UTC)
    items = list(make_client(handler).news(["ACME", "BETA"], since))
    # the shared story appears once, tagged with both universe symbols
    shared = [i for i in items if i.id == "tickertick:426800427666858718"]
    assert len(shared) == 1
    assert shared[0].symbols == ("ACME", "BETA")


def test_symbol_cap_limits_requests() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"stories": []})

    since = datetime(2026, 8, 1, tzinfo=UTC)
    list(make_client(handler, symbol_cap=2).news(["A", "B", "C", "D"], since))
    assert calls["n"] == 2
