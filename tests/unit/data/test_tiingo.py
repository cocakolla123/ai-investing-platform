"""Tiingo adapter parsing tests (adjClose, derived corp actions)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from dynamo.data.ratelimit import TokenBucket
from dynamo.data.tiingo import TiingoClient

PRICES = [
    {
        "date": "2026-07-30T00:00:00.000Z",
        "open": 10.0,
        "high": 11.0,
        "low": 9.5,
        "close": 10.5,
        "volume": 1000,
        "adjClose": 10.4,
        "splitFactor": 1.0,
        "divCash": 0.0,
    },
    {
        "date": "2026-07-31T00:00:00.000Z",
        "open": 10.5,
        "high": 12.0,
        "low": 10.0,
        "close": 11.0,
        "volume": 2000,
        "adjClose": 11.0,
        "splitFactor": 2.0,
        "divCash": 0.25,
    },
]


def make_client(handler: httpx.MockTransport) -> TiingoClient:
    return TiingoClient(
        "tok",
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_daily_bars_parse_adj_close_and_provenance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Token tok"
        assert "/tiingo/daily/acme/prices" in str(request.url)
        return httpx.Response(200, json=PRICES)

    client = make_client(httpx.MockTransport(handler))
    assert client.source == "tiingo"
    bars = list(client.daily_bars(["ACME"], date(2026, 7, 30), date(2026, 7, 31)))
    assert len(bars) == 2
    assert bars[0].symbol == "ACME"
    assert bars[0].ts == datetime(2026, 7, 30, tzinfo=UTC)
    assert bars[0].adj_close == 10.4
    assert bars[0].source == "tiingo"


def test_corporate_actions_for_derives_splits_and_dividends() -> None:
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=PRICES)))
    actions = list(client.corporate_actions_for(["ACME"], date(2026, 7, 30), date(2026, 7, 31)))
    kinds = {(a.action_type, a.ex_date) for a in actions}
    assert ("split", date(2026, 7, 31)) in kinds
    assert ("dividend", date(2026, 7, 31)) in kinds
    split = next(a for a in actions if a.action_type == "split")
    assert split.ratio == 2.0
    div = next(a for a in actions if a.action_type == "dividend")
    assert div.amount == 0.25


def test_latest_quotes_maps_tngo_last() -> None:
    payload = [{"ticker": "acme", "tngoLast": 11.25}, {"ticker": "XYZ", "last": 5.0}]
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
    quotes = client.latest_quotes(["ACME", "XYZ"])
    assert quotes == {"ACME": 11.25, "XYZ": 5.0}


def test_market_wide_corp_actions_and_assets_are_noops() -> None:
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=[])))
    assert list(client.corporate_actions(date(2026, 1, 1), date(2026, 2, 1))) == []
    assert list(client.tradable_instruments()) == []


def test_daily_bars_maps_dot_share_classes_to_hyphen_urls() -> None:
    """BRK.B-style tickers hit Tiingo as brk-b but keep the canonical symbol on bars."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/tiingo/daily/brk-b/prices" in str(request.url)
        return httpx.Response(200, json=PRICES)

    client = make_client(httpx.MockTransport(handler))
    bars = list(client.daily_bars(["BRK.B"], date(2026, 7, 30), date(2026, 7, 31)))
    assert bars and all(b.symbol == "BRK.B" for b in bars)


def test_monthly_symbol_cap_body_is_a_rate_limit_not_a_data_gap() -> None:
    """Tiingo answers an exhausted plan quota with HTTP 200 and a detail body
    (verbatim below, captured 2026-09-23). Read as DataGap it failed the whole
    nightly ingest; it is a rate limit, which the ingest tolerates."""
    import pytest

    from dynamo.data.errors import DataGap, RateLimited

    body = {
        "detail": "You have run over your 500 symbol look up for this month. Please "
        "upgrade at https://api.tiingo.com/pricing to have your limits increased."
    }
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    with pytest.raises(RateLimited):
        list(client.daily_bars(["AAA"], date(2026, 9, 23), date(2026, 9, 23)))

    odd = make_client(httpx.MockTransport(lambda r: httpx.Response(200, json={"x": 1})))
    with pytest.raises(DataGap):  # genuinely malformed payloads still are DataGap
        list(odd.daily_bars(["AAA"], date(2026, 9, 23), date(2026, 9, 23)))
