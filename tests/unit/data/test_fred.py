"""FRED adapter parsing tests (vintage-aware via realtime_start)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from dynamo.data.fred import FredClient
from dynamo.data.ratelimit import TokenBucket

OBSERVATIONS = {
    "observations": [
        {"date": "2026-06-01", "value": "3.2", "realtime_start": "2026-06-15"},
        {"date": "2026-07-01", "value": ".", "realtime_start": "2026-07-15"},  # missing
        {"date": "2026-07-01", "value": "3.4", "realtime_start": "2026-07-16"},
    ]
}


def make_client(handler: httpx.MockTransport) -> FredClient:
    return FredClient(
        "key",
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_series_parses_observations_with_vintages() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS)

    points = list(make_client(httpx.MockTransport(handler)).series("CPIAUCSL", date(2026, 1, 1)))
    assert seen["api_key"] == "key"
    assert seen["series_id"] == "CPIAUCSL"
    assert seen["observation_start"] == "2026-01-01"
    assert len(points) == 2  # "." missing value skipped
    assert points[0].series_id == "CPIAUCSL"
    assert points[0].ts == date(2026, 6, 1)
    assert points[0].value == 3.2
    assert points[0].vintage_ts == datetime(2026, 6, 15, tzinfo=UTC)


def test_series_vintages_widens_realtime_window() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS)

    list(
        make_client(httpx.MockTransport(handler)).series_vintages(
            "UNRATE", date(2026, 1, 1), date(2025, 1, 1)
        )
    )
    assert seen["realtime_start"] == "2025-01-01"
    assert seen["realtime_end"] == "9999-12-31"
