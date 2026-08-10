"""FRED adapter — implements the ``MacroProvider`` port.

Design: doc 02 §1 (macro series, daily cadence) and doc 02 §3
(``macro_series(series_id, ts, value, vintage_ts)`` — vintages kept for
point-in-time macro).

Vintage awareness: FRED/ALFRED exposes ``realtime_start``/``realtime_end`` on
every observation. ``series`` fetches the *current* vintage; each point's
``vintage_ts`` records when that value became known, so downstream features
can be built as-of any date without look-ahead. ``series_vintages`` widens
``realtime_start`` to recover historical revisions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import MacroPoint

log = structlog.get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"


class FredClient:
    """``MacroProvider`` implementation over api.stlouisfed.org."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = FRED_BASE,
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=30.0)
        # ~50 requests/day steady state (doc 02 §8); 2 req/s is generous.
        self._bucket = bucket or TokenBucket(rate=2.0, capacity=4.0)
        self._base_url = base_url

    def _observations(self, params: dict[str, Any]) -> Iterator[MacroPoint]:
        payload = get_json(
            self._client,
            f"{self._base_url}/series/observations",
            provider="fred",
            bucket=self._bucket,
            params={**params, "api_key": self._api_key, "file_type": "json"},
        )
        series_id = str(params["series_id"])
        for obs in payload.get("observations", []):
            raw = obs.get("value")
            if raw in (None, "", "."):  # FRED encodes missing as "."
                continue
            vintage = date.fromisoformat(obs["realtime_start"])
            yield MacroPoint(
                series_id=series_id,
                ts=date.fromisoformat(obs["date"]),
                value=float(raw),
                vintage_ts=datetime(vintage.year, vintage.month, vintage.day, tzinfo=UTC),
            )

    # -- MacroProvider -----------------------------------------------------

    def series(self, series_id: str, start: date) -> Iterable[MacroPoint]:
        """Current-vintage observations from ``start``; vintage_ts = realtime_start."""
        return self._observations(
            {"series_id": series_id, "observation_start": start.isoformat()}
        )

    # -- extras ------------------------------------------------------------

    def series_vintages(
        self, series_id: str, start: date, realtime_start: date
    ) -> Iterable[MacroPoint]:
        """All vintages since ``realtime_start`` (ALFRED revision history)."""
        return self._observations(
            {
                "series_id": series_id,
                "observation_start": start.isoformat(),
                "realtime_start": realtime_start.isoformat(),
                "realtime_end": "9999-12-31",
            }
        )
