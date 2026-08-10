"""StockTwits adapter tests: parsing, sentiment prefix, graceful degradation."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from dynamo.data.ratelimit import TokenBucket
from dynamo.data.stocktwits import StockTwitsProvider

STREAM = {
    "messages": [
        {
            "id": 555,
            "body": "ACME breaking out, loading calls",
            "created_at": "2026-08-04T12:34:56Z",
            "entities": {"sentiment": {"basic": "Bullish"}},
            "symbols": [{"symbol": "ACME"}, {"symbol": "TSLA"}],
        },
        {
            "id": 556,
            "body": "no sentiment tag on this one",
            "created_at": "2026-08-04T13:00:00Z",
            "entities": {"sentiment": None},
            "symbols": [{"symbol": "ACME"}],
        },
        {"id": 557, "body": "", "created_at": "2026-08-04T13:01:00Z"},  # dropped
    ]
}


def make_provider(handler, *, bucket=None, **kwargs) -> StockTwitsProvider:
    return StockTwitsProvider(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=bucket or TokenBucket(rate=10_000.0, capacity=10_000.0),
        base_url="https://st.test/api/2",
        **kwargs,
    )


def test_parses_messages_with_sentiment_prefix() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/2/streams/symbol/ACME.json"
        return httpx.Response(200, json=STREAM)

    items = make_provider(handler).mentions(["ACME"])
    assert [i.id for i in items] == ["stocktwits:555", "stocktwits:556"]
    first = items[0]
    assert first.summary == "[bullish] ACME breaking out, loading calls"
    assert first.headline == "ACME breaking out, loading calls"
    assert first.symbols == ("ACME", "TSLA")  # API-tagged symbols kept
    assert first.source == "stocktwits"
    assert first.published_at == datetime(2026, 8, 4, 12, 34, 56, tzinfo=UTC)
    assert items[1].summary == "no sentiment tag on this one"  # no prefix


def test_disabled_returns_empty_without_requests() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("disabled provider must not hit the network")

    assert make_provider(handler, enabled=False).mentions(["ACME"]) == []


def test_403_treated_as_provider_down_not_crash() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(403, json={"errors": [{"message": "auth required"}]})

    items = make_provider(handler).mentions(["ACME", "TSLA", "GME"])
    assert items == []
    assert len(calls) == 1  # stops after first failure, no per-symbol retry storm


def test_symbol_cap_prioritizes_list_order() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"messages": []})

    make_provider(handler, max_symbols_per_scan=2).mentions(["AAA", "BBB", "CCC", "DDD"])
    assert calls == [
        "/api/2/streams/symbol/AAA.json",
        "/api/2/streams/symbol/BBB.json",
    ]


def test_rate_limit_bucket_used() -> None:
    acquired: list[float] = []

    class CountingBucket(TokenBucket):
        def acquire(self, tokens: float = 1.0) -> None:
            acquired.append(tokens)
            super().acquire(tokens)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"messages": []})

    make_provider(handler, bucket=CountingBucket(rate=10_000.0, capacity=10_000.0)).mentions(
        ["ACME", "TSLA"]
    )
    assert len(acquired) == 2
