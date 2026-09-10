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


RESOLVED = {
    # A resolved-YES market: final price 1, ~24h-earlier price derivable from
    # oneDayPriceChange (Gamma stores no true pre-settlement snapshot).
    "id": "600001",
    "question": "Did the favorite win?",
    "closed": True,
    "umaResolutionStatus": "resolved",
    "outcomePrices": '["1", "0"]',
    "oneDayPriceChange": 0.06,
    "volumeNum": 10_000.0,
}


def test_resolution_mapping_derives_the_prior_price() -> None:
    to_resolution = PolymarketClient._to_resolution

    won = to_resolution(RESOLVED)
    assert won is not None
    assert won.market_id == "600001"
    assert won.resolved_yes is True
    assert won.final_price == 1.0
    assert won.prior_price is not None and abs(won.prior_price - 0.94) < 1e-12
    assert won.volume == 10_000.0

    lost = to_resolution({**RESOLVED, "outcomePrices": '["0", "1"]', "oneDayPriceChange": -0.06})
    assert lost is not None
    assert lost.resolved_yes is False
    assert lost.prior_price is not None and abs(lost.prior_price - 0.06) < 1e-12


def test_resolution_mapping_drops_unclear_rows() -> None:
    to_resolution = PolymarketClient._to_resolution
    assert to_resolution({**RESOLVED, "closed": False}) is None
    assert to_resolution({**RESOLVED, "umaResolutionStatus": "disputed"}) is None
    # Closed but the final price is away from 0/1: not clearly resolved.
    assert to_resolution({**RESOLVED, "outcomePrices": '["0.55", "0.45"]'}) is None
    # No change field -> no derivable prior price, but the outcome still maps.
    no_change = dict(RESOLVED)
    del no_change["oneDayPriceChange"]
    row = to_resolution(no_change)
    assert row is not None and row.prior_price is None


def test_resolved_markets_paginates_closed_listings() -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        assert request.url.params["closed"] == "true"
        return httpx.Response(200, json=[RESOLVED])  # < page size -> single page

    rows = list(make_client(httpx.MockTransport(handler)).resolved_markets(max_pages=3))
    assert len(calls) == 1
    assert len(rows) == 1 and rows[0].resolved_yes is True


EVENT = {
    "id": "16183",
    "title": "Who wins the nomination?",
    "negRisk": True,
    "markets": [
        MARKET,
        {**MARKET, "id": "500002", "question": "Or this one?", "closed": True},
        "not-a-dict",
    ],
}


def test_open_events_parse_neg_risk_and_price_closed_legs_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url.path).endswith("/events")
        assert request.url.params["closed"] == "false"
        return httpx.Response(200, json=[EVENT, {"title": "no id"}, "not-a-dict"])

    events = make_client(httpx.MockTransport(handler)).open_events(limit=50)
    assert len(events) == 1  # malformed events dropped, never guessed at
    event = events[0]
    assert event.event_id == "16183" and event.neg_risk is True
    assert len(event.markets) == 2  # the non-dict nested row is dropped
    open_market, open_price = event.markets[0]
    assert open_market.market_id == "500001"
    assert open_price is not None and open_price.implied_prob == 0.65
    assert open_price.trusted  # thin-market policy applied from the nested payload
    closed_market, closed_price = event.markets[1]
    assert closed_market.market_id == "500002"
    assert closed_price is None  # closed legs carry no price -> fail closed downstream
