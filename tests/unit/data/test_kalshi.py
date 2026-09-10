"""Kalshi adapter tests: graceful skip when unconfigured + mapping logic."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from dynamo.data.kalshi import KalshiClient
from dynamo.data.ratelimit import TokenBucket

MARKET = {
    "ticker": "FED-26DEC-T4.00",
    "title": "Fed funds above 4.00% in December?",
    "category": "Economics",
    "close_time": "2026-12-18T21:00:00Z",
    "status": "active",
    "result": "",
    "yes_bid": 42,
    "yes_ask": 44,
    "last_price": 43,
    "volume": 50_000,
    "liquidity": 120_000,
}


def make_client(configured: bool = False) -> KalshiClient:
    return KalshiClient(
        api_key_id="kid" if configured else "",
        private_key_path="/nonexistent.pem" if configured else "",
        min_volume_usd=25_000.0,
        max_spread=0.03,
        client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500))),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_unconfigured_skips_gracefully() -> None:
    client = make_client(configured=False)
    assert client.venue == "kalshi"
    assert not client.configured
    assert list(client.markets()) == []  # no network call, no exception
    assert list(client.prices(["FED-26DEC-T4.00"])) == []


def test_missing_key_file_also_skips_gracefully() -> None:
    # Key id set but the PEM can't be loaded → degrade per-asset, don't raise.
    client = make_client(configured=True)
    assert not client.configured
    assert list(client.markets()) == []


def test_market_mapping() -> None:
    market = make_client()._to_market(MARKET)
    assert market is not None
    assert market.venue == "kalshi"
    assert market.market_id == "FED-26DEC-T4.00"
    assert market.question.startswith("Fed funds")
    assert market.close_ts == datetime(2026, 12, 18, 21, 0, tzinfo=UTC)
    assert market.resolved_outcome is None
    resolved = make_client()._to_market({**MARKET, "result": "yes"})
    assert resolved is not None and resolved.resolved_outcome is True


def test_price_mapping_and_trust_flag() -> None:
    ts = datetime(2026, 8, 3, tzinfo=UTC)
    price = make_client()._to_price(MARKET, ts)
    assert price is not None
    assert price.implied_prob == 0.43  # last_price in cents
    assert price.volume == 50_000.0
    assert price.trusted  # volume >= 25k and spread 0.02 <= 0.03

    thin = make_client()._to_price({**MARKET, "volume": 100}, ts)
    assert thin is not None and not thin.trusted  # volume floor

    wide = make_client()._to_price({**MARKET, "yes_bid": 30, "yes_ask": 56}, ts)
    assert wide is not None and not wide.trusted  # spread cap


SETTLED_DOLLARS = {
    # The newer settled-listing schema serves decimal-string *_dollars fields
    # and no integer-cent twins (observed live 2026-09).
    "ticker": "KXEV-26SEP-A",
    "result": "yes",
    "last_price_dollars": "0.9905",
    "previous_price_dollars": "0.9400",
    "volume_fp": "504.43",
    "settlement_ts": "2026-09-10T06:53:55.572734Z",
}

SETTLED_CENTS = {
    # The classic schema: integer cents.
    "ticker": "KXEV-26SEP-B",
    "result": "no",
    "last_price": 3,
    "previous_price": 6,
    "volume": 1200,
    "settlement_ts": "2026-09-09T00:00:00Z",
}


def test_settlement_mapping_parses_both_price_schemas() -> None:
    to_settlement = KalshiClient._to_settlement

    dollars = to_settlement(SETTLED_DOLLARS)
    assert dollars is not None
    assert dollars.market_id == "KXEV-26SEP-A"
    assert dollars.result_yes is True
    assert dollars.last_price == 0.9905
    assert dollars.previous_price == 0.94
    assert dollars.volume == 504.43
    assert dollars.settled_ts is not None and dollars.settled_ts.tzinfo is not None

    cents = to_settlement(SETTLED_CENTS)
    assert cents is not None
    assert cents.result_yes is False
    assert cents.last_price == 0.03
    assert cents.previous_price == 0.06
    assert cents.volume == 1200.0


def test_settlement_mapping_drops_unusable_rows() -> None:
    to_settlement = KalshiClient._to_settlement
    assert to_settlement({**SETTLED_DOLLARS, "ticker": ""}) is None
    # Void/scalar results are not YES/NO outcomes -> dropped, not guessed.
    assert to_settlement({**SETTLED_DOLLARS, "result": ""}) is None
    assert to_settlement({**SETTLED_DOLLARS, "result": "void"}) is None
    # Malformed prices become None fields, not exceptions.
    mangled = to_settlement({**SETTLED_DOLLARS, "previous_price_dollars": "n/a"})
    assert mangled is not None and mangled.previous_price is None


def test_settled_markets_skips_when_unconfigured() -> None:
    assert list(make_client(configured=False).settled_markets()) == []
