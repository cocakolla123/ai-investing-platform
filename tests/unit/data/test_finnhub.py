"""Finnhub adapter parsing tests (company news + news_range + calendar)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from dynamo.data.finnhub import FinnhubClient, month_chunks
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


def test_month_chunks_partial_and_interior_months() -> None:
    chunks = list(month_chunks(date(2026, 1, 15), date(2026, 3, 10)))
    assert chunks == [
        (date(2026, 1, 15), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 10)),
    ]


def test_month_chunks_year_boundary_and_single_day() -> None:
    chunks = list(month_chunks(date(2025, 12, 20), date(2026, 1, 5)))
    assert chunks == [
        (date(2025, 12, 20), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 5)),
    ]
    assert list(month_chunks(date(2026, 6, 3), date(2026, 6, 3))) == [
        (date(2026, 6, 3), date(2026, 6, 3))
    ]
    assert list(month_chunks(date(2026, 6, 3), date(2026, 6, 1))) == []


def test_news_range_requests_monthly_chunks_per_symbol() -> None:
    calls: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        calls.append((p["symbol"], p["from"], p["to"]))
        return httpx.Response(200, json=[])

    client = make_client(httpx.MockTransport(handler))
    list(client.news_range(["ACME", "OTHR"], date(2026, 1, 15), date(2026, 2, 20)))
    assert calls == [
        ("ACME", "2026-01-15", "2026-01-31"),
        ("ACME", "2026-02-01", "2026-02-20"),
        ("OTHR", "2026-01-15", "2026-01-31"),
        ("OTHR", "2026-02-01", "2026-02-20"),
    ]


def test_news_range_maps_items_and_enforces_bounds() -> None:
    inside = 1770000000  # 2026-02-02T02:40:00Z
    before = 1767000000  # 2025-12-30T09:20:00Z (provider slop, outside range)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": 7,
                    "datetime": inside,
                    "headline": "In range",
                    "source": "Reuters",
                    "url": "https://n.test/7",
                },
                {"id": 8, "datetime": before, "headline": "Out of range"},
                {"datetime": inside, "headline": "No id"},  # dropped
            ],
        )

    client = make_client(httpx.MockTransport(handler))
    items = list(client.news_range(["ACME"], date(2026, 2, 1), date(2026, 2, 28)))
    assert [i.id for i in items] == ["finnhub:7"]
    item = items[0]
    assert item.symbols == ("ACME",)
    # Provider published_at preserved exactly (never re-stamped).
    assert item.published_at == datetime.fromtimestamp(inside, tz=UTC)


def test_news_range_paces_through_the_injected_bucket() -> None:
    acquires: list[float] = []

    class CountingBucket(TokenBucket):
        def acquire(self, tokens: float = 1.0) -> None:
            acquires.append(tokens)

    client = FinnhubClient(
        "tok",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
        ),
        bucket=CountingBucket(rate=1.0, capacity=5.0),
    )
    list(client.news_range(["ACME"], date(2026, 1, 1), date(2026, 3, 31)))
    assert len(acquires) == 3  # one paced request per monthly chunk


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
