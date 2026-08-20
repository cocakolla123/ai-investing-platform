"""Finnhub adapter — implements the ``NewsProvider`` port.

Design: doc 02 §1 (company news headlines + earnings calendar, free tier,
60 req/min) and doc 02 §4 (news every 30 min for Tier 1, EOD sweep Tier 2;
earnings calendar daily).

Endpoints used:

- ``/api/v1/company-news?symbol=&from=&to=`` → :class:`NewsItem`
- ``/api/v1/calendar/earnings?from=&to=``   → :class:`EarningsEvent`
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import EarningsEvent, NewsItem

log = structlog.get_logger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubClient:
    """``NewsProvider`` implementation over the Finnhub REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = FINNHUB_BASE,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        # Free tier is 60 req/min → pace at 1 req/s (doc 02 §8 budget).
        self._bucket = bucket or TokenBucket(rate=1.0, capacity=5.0)
        self._base_url = base_url
        self._headers = {"X-Finnhub-Token": api_key}

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        return get_json(
            self._client,
            f"{self._base_url}{path}",
            provider="finnhub",
            bucket=self._bucket,
            params=params,
            headers=self._headers,
        )

    # -- NewsProvider ------------------------------------------------------

    def news(self, symbols: Sequence[str], since: datetime) -> Iterable[NewsItem]:
        """Company news headlines per symbol since ``since`` (UTC)."""
        to = datetime.now(UTC).date().isoformat()
        for symbol in symbols:
            payload = self._get(
                "/company-news",
                params={"symbol": symbol, "from": since.date().isoformat(), "to": to},
            )
            for entry in payload or []:
                entry_id = entry.get("id")
                headline = entry.get("headline")
                published = entry.get("datetime")
                if entry_id is None or not headline or published is None:
                    continue
                published_at = datetime.fromtimestamp(int(published), tz=UTC)
                if published_at < since:
                    continue
                yield NewsItem(
                    id=f"finnhub:{entry_id}",
                    symbols=(symbol,),
                    headline=str(headline),
                    summary=entry.get("summary") or None,
                    source=str(entry.get("source") or "finnhub"),
                    published_at=published_at,
                    url=entry.get("url") or None,
                )

    def news_range(self, symbols: Sequence[str], start: date, end: date) -> Iterable[NewsItem]:
        """Historical company news over ``[start, end]`` (backfill path).

        Same ``/company-news`` endpoint as :meth:`news` but with an explicit
        ``to`` bound (``news()`` hardcodes ``to=today`` for the live poll and
        is deliberately left untouched). One request per (symbol, calendar
        month) — Finnhub caps response sizes, so monthly chunks keep long
        ranges complete. Pacing comes from the injected :class:`TokenBucket`
        (free tier ~60 req/min; the default bucket paces at 1 req/s and tests
        inject a non-sleeping bucket).

        Items whose ``published_at`` falls outside ``[start, end]`` (provider
        slop at chunk edges) are dropped; duplicates across chunk boundaries
        are harmless anyway (ids are stable and the upsert path dedupes).
        """
        range_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
        range_end = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC)
        for symbol in symbols:
            for chunk_start, chunk_end in month_chunks(start, end):
                payload = self._get(
                    "/company-news",
                    params={
                        "symbol": symbol,
                        "from": chunk_start.isoformat(),
                        "to": chunk_end.isoformat(),
                    },
                )
                for entry in payload or []:
                    item = _news_item(symbol, entry)
                    if item is None:
                        continue
                    if item.published_at < range_start or item.published_at > range_end:
                        continue
                    yield item

    def earnings_calendar(self, start: date, end: date) -> Iterable[EarningsEvent]:
        """Earnings calendar with estimates/actuals (doc 02 §2 earnings_surprise)."""
        payload = self._get(
            "/calendar/earnings",
            params={"from": start.isoformat(), "to": end.isoformat()},
        )
        for entry in (payload or {}).get("earningsCalendar", []):
            symbol = entry.get("symbol")
            event_date = entry.get("date")
            if not symbol or not event_date:
                continue
            year = entry.get("year")
            quarter = entry.get("quarter")
            period = f"{year}Q{quarter}" if year and quarter else None
            yield EarningsEvent(
                symbol=str(symbol),
                event_date=date.fromisoformat(event_date),
                period=period,
                eps_est=_opt_float(entry.get("epsEstimate")),
                eps_act=_opt_float(entry.get("epsActual")),
                revenue_est=_opt_float(entry.get("revenueEstimate")),
                revenue_act=_opt_float(entry.get("revenueActual")),
            )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _news_item(symbol: str, entry: Any) -> NewsItem | None:
    """One ``/company-news`` payload entry -> :class:`NewsItem` (or ``None``
    when required fields are missing). Field mapping mirrors :meth:`news`."""
    entry_id = entry.get("id")
    headline = entry.get("headline")
    published = entry.get("datetime")
    if entry_id is None or not headline or published is None:
        return None
    return NewsItem(
        id=f"finnhub:{entry_id}",
        symbols=(symbol,),
        headline=str(headline),
        summary=entry.get("summary") or None,
        source=str(entry.get("source") or "finnhub"),
        published_at=datetime.fromtimestamp(int(published), tz=UTC),
        url=entry.get("url") or None,
    )


def month_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Calendar-month chunks covering ``[start, end]`` inclusive.

    The first chunk starts at ``start``, the last ends at ``end``; interior
    chunks span whole calendar months. ``start > end`` yields nothing.
    """
    if start > end:
        return
    chunk_start = start
    while chunk_start <= end:
        if chunk_start.month == 12:
            next_month = date(chunk_start.year + 1, 1, 1)
        else:
            next_month = date(chunk_start.year, chunk_start.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        yield chunk_start, chunk_end
        chunk_start = next_month
