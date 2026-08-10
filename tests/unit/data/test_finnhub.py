"""Finnhub adapter parsing tests (company news + earnings calendar)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from dynamo.data.finnhub import FinnhubClient
from dynamo.data.ratelimit import TokenBucket

NEWS = [
    {
        "id": 101,
        "datetime": 1785400000,  # 2026-07-30T18:26:40Z
        "headline": "Acme announces merger",
        "summary": "Acme to combine with XYZ.",
        "source": "Reuters",
        "url": "https://n.test/1",
    },
    {"id": 102, "datetime": 1600000000, "headline": "Old news"},  # before `since`
    {"datetime": 1785400000, "headline": "No id"},  # dropped
]

CALENDAR = {
    "earningsCalendar": [
        {
            "symbol": "ACME",
            "date": "2026-08-05",
            "epsEstimate": 1.2,
            "epsActual": None,
            "revenueEstimate": 5.0e8,
            "revenueActual": None,
            "quarter": 2,
            "year": 2026,
        },
        {"date": "2026-08-06"},  # no symbol → dropped
    ]
}


def make_client(handler: httpx.MockTransport) -> FinnhubClient:
    return FinnhubClient(
        "tok",
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_news_parses_and_filters_since() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Finnhub-Token"] == "tok"
        assert request.url.params["symbol"] == "ACME"
        return httpx.Response(200, json=NEWS)

    since = datetime(2026, 7, 1, tzinfo=UTC)
    items = list(make_client(httpx.MockTransport(handler)).news(["ACME"], since))
    assert len(items) == 1
    item = items[0]
    assert item.id == "finnhub:101"
    assert item.symbols == ("ACME",)
    assert item.headline == "Acme announces merger"
    assert item.source == "Reuters"
    assert item.published_at == datetime.fromtimestamp(1785400000, tz=UTC)


def test_earnings_calendar_parses_period() -> None:
    client = make_client(httpx.MockTransport(lambda req: httpx.Response(200, json=CALENDAR)))
    events = list(client.earnings_calendar(date(2026, 8, 1), date(2026, 8, 31)))
    assert len(events) == 1
    e = events[0]
    assert e.symbol == "ACME"
    assert e.event_date == date(2026, 8, 5)
    assert e.period == "2026Q2"
    assert e.eps_est == 1.2
    assert e.eps_act is None
