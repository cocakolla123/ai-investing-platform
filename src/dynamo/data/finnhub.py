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
from datetime import UTC, date, datetime
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
