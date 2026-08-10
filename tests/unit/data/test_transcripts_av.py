"""Alpha Vantage transcript adapter: parsing, daily cap, graceful skips."""

from __future__ import annotations

from datetime import date

import httpx

from dynamo.data.ratelimit import TokenBucket
from dynamo.data.transcripts_av import AlphaVantageTranscripts, reported_quarter

PAYLOAD = {
    "symbol": "ACME",
    "quarter": "2026Q2",
    "transcript": [
        {
            "speaker": "Jane Roe",
            "title": "CEO",
            "content": "Welcome to the call.",
            "sentiment": "0.6",
        },
        {
            "speaker": "John Doe",
            "title": "CFO",
            "content": "Revenue grew 12%.",
            "sentiment": "0.7",
        },
        {"speaker": "Jane Roe", "title": "CEO", "content": "Thank you."},
        {"speaker": "Ghost", "title": "", "content": ""},  # empty content dropped
    ],
}


def make_client(
    handler, *, api_key: str = "tok", daily_cap: int = 20, today=lambda: date(2026, 8, 9)
) -> AlphaVantageTranscripts:
    return AlphaVantageTranscripts(
        api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
        daily_cap=daily_cap,
        today=today,
    )


def test_fetch_parses_content_and_speakers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["function"] == "EARNINGS_CALL_TRANSCRIPT"
        assert request.url.params["symbol"] == "ACME"
        assert request.url.params["quarter"] == "2026Q2"
        assert request.url.params["apikey"] == "tok"
        return httpx.Response(200, json=PAYLOAD)

    transcript = make_client(handler).fetch("ACME", date(2026, 8, 5), period="2026Q2")
    assert transcript is not None
    assert transcript.symbol == "ACME"
    assert transcript.event_date == date(2026, 8, 5)
    assert transcript.source == "alphavantage"
    assert transcript.period == "2026Q2"
    assert transcript.content.startswith("Jane Roe (CEO): Welcome to the call.")
    assert "John Doe (CFO): Revenue grew 12%." in transcript.content
    assert transcript.speakers == (
        {"speaker": "Jane Roe", "title": "CEO"},
        {"speaker": "John Doe", "title": "CFO"},
    )
    assert transcript.ref == "transcripts:ACME:2026-08-05:alphavantage"


def test_fetch_derives_quarter_from_event_date() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["quarter"] = request.url.params["quarter"]
        return httpx.Response(200, json=PAYLOAD)

    # August call reports the previous calendar quarter; January reports Q4.
    make_client(handler).fetch("ACME", date(2026, 8, 5))
    assert seen["quarter"] == "2026Q2"
    assert reported_quarter(date(2026, 1, 20)) == "2025Q4"


def test_fetch_returns_none_on_empty_payload() -> None:
    client = make_client(lambda req: httpx.Response(200, json={"Information": "rate limit"}))
    assert client.fetch("ACME", date(2026, 8, 5)) is None


def test_unset_key_skips_without_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no key -> no request")

    client = make_client(handler, api_key="")
    assert not client.configured
    assert client.fetch("ACME", date(2026, 8, 5)) is None


def test_daily_cap_blocks_further_requests_until_next_day() -> None:
    calls = {"n": 0}
    day = {"value": date(2026, 8, 9)}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=PAYLOAD)

    client = make_client(handler, daily_cap=2, today=lambda: day["value"])
    assert client.fetch("ACME", date(2026, 8, 5)) is not None
    assert client.fetch("ACME", date(2026, 8, 5)) is not None
    assert client.fetch("ACME", date(2026, 8, 5)) is None  # capped, no request
    assert calls["n"] == 2
    day["value"] = date(2026, 8, 10)  # counter resets with the (UTC) day
    assert client.fetch("ACME", date(2026, 8, 5)) is not None
    assert calls["n"] == 3
