"""OddsPipe adapter tests: graceful skip, defensive parsing, request shape."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from dynamo.data.oddspipe import CANDLE_INTERVALS, OddsPipeProvider
from dynamo.data.ratelimit import TokenBucket

GOOD_ITEM = {
    "title": "Fed funds above 4.00% in December?",
    "match_score": 91.2,
    "sources": {
        "polymarket": {"id": "514729", "yes_price": 0.41, "no_price": 0.59, "volume_usd": 90e3},
        "kalshi": {"ticker": "FED-26DEC-T4.00", "yes_price": 0.47, "no_price": 0.53},
    },
    "spread": {"yes_diff": 0.06, "direction": "kalshi_higher"},
    "polymarket_url": "https://polymarket.com/event/fed-funds-above-4",
    "kalshi_url": "https://kalshi.com/markets/fed-26dec",
}

# Ids only derivable from the venue URLs (defensive fallback path).
URL_ONLY_ITEM = {
    "title": "Will X happen?",
    "sources": {
        "polymarket": {"yes_price": 0.10},
        "kalshi": {"yes_price": 0.90},
    },
    "spread": {"yes_diff": 0.80},
    "polymarket_url": "https://polymarket.com/event/x-happens-2026",
    "kalshi_url": "https://kalshi.com/markets/kxhappen",
}

MALFORMED_ITEMS: list[Any] = [
    "not-a-mapping",
    {"sources": {"kalshi": {"yes_price": 0.5}, "polymarket": {"yes_price": 0.4}}},  # no title
    {"title": "no sources at all"},
    {"title": "half a pair", "sources": {"kalshi": {"ticker": "T", "yes_price": 0.5}}},
    {  # missing prices
        "title": "no prices",
        "sources": {"kalshi": {"ticker": "T"}, "polymarket": {"id": "1"}},
    },
    {  # out-of-range price
        "title": "bad price",
        "sources": {
            "kalshi": {"ticker": "T", "yes_price": 1.7},
            "polymarket": {"id": "1", "yes_price": 0.4},
        },
    },
    {  # no ids derivable anywhere
        "title": "no ids",
        "sources": {"kalshi": {"yes_price": 0.5}, "polymarket": {"yes_price": 0.4}},
    },
]


def make_provider(
    handler: Any = None, api_key: str = "op-key", min_spread: float = 0.03
) -> OddsPipeProvider:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no network call expected")

    return OddsPipeProvider(
        api_key,
        min_spread=min_spread,
        client=httpx.Client(transport=httpx.MockTransport(handler or refuse)),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def spreads_handler(payload: Any, seen: list[httpx.Request] | None = None) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return handler


def test_missing_key_skips_without_network() -> None:
    provider = make_provider(api_key="")
    assert not provider.configured
    assert provider.spreads() == []  # no network call, no exception
    assert provider.candlesticks("123") == []


def test_spreads_parsing_and_request_shape() -> None:
    seen: list[httpx.Request] = []
    payload = {"items": [GOOD_ITEM, URL_ONLY_ITEM, *MALFORMED_ITEMS]}
    provider = make_provider(spreads_handler(payload, seen), min_spread=0.05)
    spreads = provider.spreads()

    # Request: authenticated, min_spread threaded through.
    assert len(seen) == 1
    assert seen[0].headers["X-API-Key"] == "op-key"
    assert seen[0].url.path == "/v1/spreads"
    assert seen[0].url.params["min_spread"] == "0.05"

    # Malformed rows are skipped; the two good rows survive.
    assert len(spreads) == 2
    good = spreads[0]
    assert good.question.startswith("Fed funds")
    assert good.kalshi_market_id == "FED-26DEC-T4.00"
    assert good.kalshi_yes == 0.47
    assert good.polymarket_id == "514729"
    assert good.polymarket_yes == 0.41
    assert good.yes_diff == 0.06
    assert good.as_of.tzinfo is not None

    # URL fallback: last path segment; kalshi id uppercased (ticker convention).
    url_only = spreads[1]
    assert url_only.kalshi_market_id == "KXHAPPEN"
    assert url_only.polymarket_id == "x-happens-2026"


def test_spreads_min_spread_override() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(spreads_handler({"items": []}, seen))
    assert provider.spreads(min_spread=0.2) == []
    assert seen[0].url.params["min_spread"] == "0.2"


def test_spreads_api_down_returns_empty() -> None:
    provider = make_provider(lambda request: httpx.Response(500))
    assert provider.spreads() == []  # ProviderDown swallowed: enrichment only


def test_spreads_auth_rejected_returns_empty() -> None:
    provider = make_provider(
        lambda request: httpx.Response(401, json={"detail": "Missing X-API-Key header"})
    )
    assert provider.spreads() == []


def test_spreads_unexpected_payload_shapes() -> None:
    for payload in ({"weird": 1}, [], "nope", None):
        provider = make_provider(spreads_handler(payload))
        assert provider.spreads() == []


def test_yes_diff_computed_when_missing() -> None:
    item = {**GOOD_ITEM, "spread": {}}
    provider = make_provider(spreads_handler({"items": [item]}))
    (spread,) = provider.spreads()
    assert spread.yes_diff == pytest.approx(0.47 - 0.41)


CANDLES = {
    "candles": [
        {
            "ts": "2026-08-01T00:00:00Z",
            "open": 0.40,
            "high": 0.45,
            "low": 0.39,
            "close": 0.44,
            "volume": 1200,
        },
        {"ts": "2026-08-01T01:00:00Z", "open": 0.44, "high": 0.5, "low": 0.44, "close": 0.5},
        {"open": 0.1, "high": 0.2, "low": 0.1, "close": 0.2},  # no timestamp -> skipped
        {"ts": "2026-08-01T02:00:00Z", "open": None, "high": 0.5, "low": 0.4, "close": 0.5},
        "garbage",
    ]
}


def test_candlesticks_parsing() -> None:
    seen: list[httpx.Request] = []
    provider = make_provider(spreads_handler(CANDLES, seen))
    start = datetime(2026, 8, 1, tzinfo=UTC)
    end = datetime(2026, 8, 2, tzinfo=UTC)
    candles = provider.candlesticks("42", interval="1h", start=start, end=end)

    assert seen[0].url.path == "/v1/markets/42/candlesticks"
    assert seen[0].url.params["interval"] == "1h"
    assert seen[0].url.params["start"] == start.isoformat()
    assert seen[0].url.params["end"] == end.isoformat()

    assert len(candles) == 2  # malformed rows skipped
    first = candles[0]
    assert first.market_id == "42"
    assert first.ts == datetime(2026, 8, 1, tzinfo=UTC)
    assert (first.open, first.high, first.low, first.close) == (0.40, 0.45, 0.39, 0.44)
    assert first.volume == 1200.0
    assert candles[1].volume == 0.0  # missing volume defaults to 0


def test_candlesticks_bare_list_payload() -> None:
    provider = make_provider(spreads_handler(json.loads(json.dumps(CANDLES))["candles"]))
    assert len(provider.candlesticks("42")) == 2


def test_candlesticks_invalid_interval_raises() -> None:
    provider = make_provider()
    with pytest.raises(ValueError, match="interval"):
        provider.candlesticks("42", interval="3h")
    assert "3h" not in CANDLE_INTERVALS


def test_candlesticks_api_down_returns_empty() -> None:
    provider = make_provider(lambda request: httpx.Response(503))
    assert provider.candlesticks("42") == []
