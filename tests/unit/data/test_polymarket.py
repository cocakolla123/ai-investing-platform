"""Polymarket Gamma adapter tests with httpx.MockTransport."""

from __future__ import annotations

import httpx

from dynamo.data.polymarket import PolymarketClient
from dynamo.data.ratelimit import TokenBucket

MARKET = {
    "id": "500001",
    "question": "Will the Fed cut rates in September 2026?",
    "category": "Economics",
    "endDate": "2026-09-17T00:00:00Z",
    "closed": False,
    "outcomePrices": '["0.65", "0.35"]',
    "volumeNum": 100_000.0,
    "liquidityNum": 40_000.0,
    "spread": 0.01,
}


def make_client(handler: httpx.MockTransport) -> PolymarketClient:
    return PolymarketClient(
        min_volume_usd=25_000.0,
        max_spread=0.03,
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_markets_parse_and_paginate_stop() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.params["closed"] == "false"
        return httpx.Response(200, json=[MARKET])  # < page size → single page

    markets = list(make_client(httpx.MockTransport(handler)).markets())
    assert calls["n"] == 1
    assert len(markets) == 1
    m = markets[0]
    assert m.venue == "polymarket"
    assert m.market_id == "500001"
    assert m.category == "Economics"
    assert m.resolved_outcome is None


def test_prices_apply_thin_market_trust_flag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/markets/500001")
        return httpx.Response(200, json=MARKET)

    prices = list(make_client(httpx.MockTransport(handler)).prices(["500001"]))
    assert len(prices) == 1
    p = prices[0]
    assert p.implied_prob == 0.65
    assert p.trusted  # 100k volume, 1c spread

    thin_market = {**MARKET, "volumeNum": 500.0}
    thin = list(
        make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=thin_market))).prices(
            ["500001"]
        )
    )
    assert not thin[0].trusted

    wide_market = {**MARKET, "spread": 0.10}
    wide = list(
        make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=wide_market))).prices(
            ["500001"]
        )
    )
    assert not wide[0].trusted


def test_closed_market_resolution_from_prices() -> None:
    closed = {**MARKET, "closed": True, "outcomePrices": '["1", "0"]'}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[closed] if "offset" in request.url.params else [])

    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=[closed])))
    market = next(iter(client.markets()))
    assert market.resolved_outcome is True


def test_quotes_open_market() -> None:
    """quotes() (ADR-0011 sim-sleeve input): fresh YES price, not resolved."""
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=MARKET)))
    quotes = list(client.quotes(["500001"]))
    assert len(quotes) == 1
    q = quotes[0]
    assert q.market_id == "500001"
    assert q.yes_price == 0.65
    assert q.closed is False
    assert q.resolved_yes is None


def test_quotes_closed_market_resolution() -> None:
    won = {**MARKET, "closed": True, "outcomePrices": '["0.999", "0.001"]'}
    lost = {**MARKET, "closed": True, "outcomePrices": '["0.005", "0.995"]'}
    unclear = {**MARKET, "closed": True, "outcomePrices": '["0.6", "0.4"]'}
    for payload, expected in ((won, True), (lost, False), (unclear, None)):
        client = make_client(
            httpx.MockTransport(lambda req, p=payload: httpx.Response(200, json=p))
        )
        q = next(iter(client.quotes(["500001"])))
        assert q.closed is True
        assert q.resolved_yes is expected


def test_quotes_skip_unknown_markets() -> None:
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    assert list(client.quotes(["999999"])) == []


def test_clob_token_ids_parse_json_string() -> None:
    market = dict(MARKET, clobTokenIds='["11111", "22222"]')

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/markets/500001")
        return httpx.Response(200, json=market)

    tokens = make_client(httpx.MockTransport(handler)).clob_token_ids("500001")
    assert tokens == ("11111", "22222")  # (yes, no) in outcome order


def test_clob_token_ids_missing_or_malformed_return_none() -> None:
    payloads = [
        MARKET,  # no clobTokenIds field
        dict(MARKET, clobTokenIds="not-json"),
        dict(MARKET, clobTokenIds='["only-one"]'),
        dict(MARKET, clobTokenIds='["", "22222"]'),  # empty token id
        {},  # unknown market
    ]
    for payload in payloads:
        handler = httpx.MockTransport(lambda _r, p=payload: httpx.Response(200, json=p))
        assert make_client(handler).clob_token_ids("500001") is None
