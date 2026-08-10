"""Alpha Vantage earnings-call transcript adapter.

Design: doc 02 §1 adapter conventions. Endpoint::

    GET /query?function=EARNINGS_CALL_TRANSCRIPT&symbol=IBM&quarter=2024Q1&apikey=...

Free tier is 25 requests/DAY, so pacing is a per-day counter (default cap 20,
``Settings.transcript_daily_cap``) on top of a very gentle token bucket —
exhausting the daily budget is a logged skip, never an error. No key set ->
every fetch is a graceful skip (the platform runs partially configured).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import Transcript

log = structlog.get_logger(__name__)

AV_BASE = "https://www.alphavantage.co"
DEFAULT_DAILY_CAP = 20  # stay under the 25/day free quota

SOURCE = "alphavantage"


def _utc_today() -> date:
    return datetime.now(UTC).date()


def reported_quarter(event_date: date) -> str:
    """The fiscal quarter an earnings call held on ``event_date`` reports on.

    Approximation used only when the calendar row has no ``period``: a call is
    assumed to cover the *previous* calendar quarter (e.g. 2026-08-05 ->
    ``2026Q2``).
    """
    quarter = (event_date.month - 1) // 3 + 1
    if quarter == 1:
        return f"{event_date.year - 1}Q4"
    return f"{event_date.year}Q{quarter - 1}"


class AlphaVantageTranscripts:
    """Fetch one transcript per (symbol, quarter) under a hard daily budget."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = AV_BASE,
        daily_cap: int = DEFAULT_DAILY_CAP,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)
        # 25/day needs no throughput; the bucket only prevents accidental bursts.
        self._bucket = bucket or TokenBucket(rate=0.5, capacity=1.0)
        self._base_url = base_url
        self._daily_cap = daily_cap
        self._today = today
        self._counter_day: date | None = None
        self._calls_today = 0

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _budget_ok(self) -> bool:
        day = self._today()
        if day != self._counter_day:
            self._counter_day = day
            self._calls_today = 0
        return self._calls_today < self._daily_cap

    def fetch(self, symbol: str, event_date: date, period: str | None = None) -> Transcript | None:
        """One transcript, or ``None`` (no key / budget spent / not available)."""
        if not self.configured:
            log.info("alphavantage_skipped", reason="ALPHAVANTAGE_API_KEY unset", symbol=symbol)
            return None
        if not self._budget_ok():
            log.info(
                "alphavantage_daily_cap_reached", symbol=symbol, daily_cap=self._daily_cap
            )
            return None
        quarter = period or reported_quarter(event_date)
        self._calls_today += 1
        payload = get_json(
            self._client,
            f"{self._base_url}/query",
            provider="alphavantage",
            bucket=self._bucket,
            params={
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": symbol,
                "quarter": quarter,
                "apikey": self._api_key,
            },
        )
        entries = (payload or {}).get("transcript") or []
        parts: list[str] = []
        speakers: list[dict[str, str]] = []
        seen_speakers: set[tuple[str, str]] = set()
        for entry in entries:
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            speaker = str(entry.get("speaker") or "unknown").strip()
            title = str(entry.get("title") or "").strip()
            label = f"{speaker} ({title})" if title else speaker
            parts.append(f"{label}: {content}")
            if (speaker, title) not in seen_speakers:
                seen_speakers.add((speaker, title))
                speakers.append({"speaker": speaker, "title": title})
        if not parts:
            log.info("alphavantage_no_transcript", symbol=symbol, quarter=quarter)
            return None
        return Transcript(
            symbol=symbol,
            event_date=event_date,
            source=SOURCE,
            period=str((payload or {}).get("quarter") or quarter),
            content="\n\n".join(parts),
            speakers=tuple(speakers),
        )
