"""OddsPipe adapter — cross-venue ENRICHMENT, never a primary source.

OddsPipe (https://oddspipe.com) aggregates prediction-market prices across
Polymarket and Kalshi and exposes matched cross-platform pairs. The direct
Kalshi/Polymarket adapters stay primary; this adapter only adds:

- :meth:`OddsPipeProvider.spreads` — matched pairs with both venues' YES
  prices (``/v1/spreads``), used as a corroboration signal for the Kalshi
  sleeve (ADR-0009 addendum) and as research context;
- :meth:`OddsPipeProvider.candlesticks` — optional OHLCV backfill
  (``/v1/markets/{id}/candlesticks``).

Graceful degradation is the contract: no API key, provider down, or a
malformed payload each produce one log line and an empty result — nothing
downstream ever breaks because enrichment is missing. Network failures are
normalized by :func:`dynamo.data.http.get_json` into the typed provider
errors (``RateLimited`` / ``ProviderDown``) and swallowed here; malformed
rows are skipped individually.

Response schemas are not published (the OpenAPI spec leaves them empty), so
every parser is defensive: unknown shapes and missing keys skip the row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.errors import DataProviderError
from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.pm import CrossVenueSpread

log = structlog.get_logger(__name__)

ODDSPIPE_BASE = "https://oddspipe.com"
CANDLE_INTERVALS = ("1m", "5m", "1h", "1d")

# Keys a venue block may use for its platform-native market identifier.
_ID_KEYS = ("ticker", "market_ticker", "market_id", "external_id", "slug", "id")


@dataclass(frozen=True)
class OddsPipeCandle:
    """One OHLCV candle in YES-price terms (adapter-local; backfill only)."""

    market_id: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _as_float(value: Any) -> float | None:
    """Finite float or None (never raises)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_ts(value: Any) -> datetime | None:
    """ISO string or epoch seconds -> aware UTC datetime, else None."""
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    number = _as_float(value)
    if number is None:
        return None
    try:
        return datetime.fromtimestamp(number, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _venue_market_id(source: Mapping[str, Any], url: Any) -> str | None:
    """Platform-native id from a venue block, else the URL's last path segment."""
    for key in _ID_KEYS:
        raw = source.get(key)
        if raw is not None and str(raw):
            return str(raw)
    if isinstance(url, str) and url:
        tail = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
        if tail:
            return tail
    return None


class OddsPipeProvider:
    """Read-only client for the OddsPipe aggregation API (X-API-Key auth)."""

    def __init__(
        self,
        api_key: str,
        *,
        min_spread: float = 0.03,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = ODDSPIPE_BASE,
    ) -> None:
        self._api_key = api_key
        self._min_spread = min_spread
        self._client = client or httpx.Client(timeout=30.0)
        # Conservative ~30 req/min: enrichment must never crowd the primaries.
        self._bucket = bucket or TokenBucket(rate=0.5, capacity=5.0)
        self._base_url = base_url

    @property
    def provider(self) -> str:
        return "oddspipe"

    @property
    def configured(self) -> bool:
        """True when an API key is present; otherwise every call is a no-op."""
        return bool(self._api_key)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._client,
            f"{self._base_url}{path}",
            provider="oddspipe",
            bucket=self._bucket,
            params=params,
            headers={"X-API-Key": self._api_key},
        )

    # -- mapping -----------------------------------------------------------

    def _to_spread(self, item: Any, as_of: datetime) -> CrossVenueSpread | None:
        """One /v1/spreads item -> CrossVenueSpread; anything malformed -> None."""
        if not isinstance(item, Mapping):
            return None
        question = item.get("title") or item.get("question")
        sources = item.get("sources")
        if not question or not isinstance(sources, Mapping):
            return None
        kalshi = sources.get("kalshi")
        polymarket = sources.get("polymarket")
        if not isinstance(kalshi, Mapping) or not isinstance(polymarket, Mapping):
            return None
        kalshi_yes = _as_float(kalshi.get("yes_price"))
        polymarket_yes = _as_float(polymarket.get("yes_price"))
        if kalshi_yes is None or polymarket_yes is None:
            return None
        if not (0.0 <= kalshi_yes <= 1.0 and 0.0 <= polymarket_yes <= 1.0):
            return None
        kalshi_id = _venue_market_id(kalshi, item.get("kalshi_url"))
        polymarket_id = _venue_market_id(polymarket, item.get("polymarket_url"))
        if not kalshi_id or not polymarket_id:
            return None
        spread = item.get("spread")
        yes_diff = _as_float(spread.get("yes_diff")) if isinstance(spread, Mapping) else None
        if yes_diff is None:
            yes_diff = kalshi_yes - polymarket_yes
        return CrossVenueSpread(
            question=str(question),
            # Kalshi tickers are canonically uppercase; URL slugs are not.
            kalshi_market_id=str(kalshi_id).upper(),
            kalshi_yes=kalshi_yes,
            polymarket_id=str(polymarket_id),
            polymarket_yes=polymarket_yes,
            yes_diff=yes_diff,
            as_of=as_of,
        )

    def _to_candle(self, market_id: str, row: Any) -> OddsPipeCandle | None:
        if not isinstance(row, Mapping):
            return None
        ts = None
        for key in ("ts", "timestamp", "time", "bucket_start", "start"):
            ts = _parse_ts(row.get(key))
            if ts is not None:
                break
        if ts is None:
            return None
        ohlc = [_as_float(row.get(k)) for k in ("open", "high", "low", "close")]
        if any(v is None for v in ohlc):
            return None
        o, h, lo, c = ohlc
        assert o is not None and h is not None and lo is not None and c is not None
        return OddsPipeCandle(
            market_id=market_id,
            ts=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=_as_float(row.get("volume")) or 0.0,
        )

    # -- public surface ----------------------------------------------------

    def spreads(self, min_spread: float | None = None) -> list[CrossVenueSpread]:
        """All matched cross-venue pairs at/above the YES-spread floor.

        Empty list (one log line) when the key is unset or the API is down;
        malformed items are skipped row by row.
        """
        if not self.configured:
            log.info("oddspipe_not_configured_skipping")
            return []
        floor = self._min_spread if min_spread is None else min_spread
        try:
            payload = self._get("/v1/spreads", params={"min_spread": floor, "limit": 200})
        except DataProviderError as exc:
            log.warning("oddspipe_spreads_unavailable_skipping", error=str(exc))
            return []
        items = payload.get("items") if isinstance(payload, Mapping) else payload
        if not isinstance(items, list):
            return []
        as_of = datetime.now(UTC)
        out: list[CrossVenueSpread] = []
        for item in items:
            spread = self._to_spread(item, as_of)
            if spread is not None:
                out.append(spread)
        return out

    def candlesticks(
        self,
        market_id: str,
        interval: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OddsPipeCandle]:
        """OHLCV candles for one OddsPipe market id (historical backfill).

        Empty list (one log line) when the key is unset or the API is down;
        malformed rows are skipped.
        """
        if interval not in CANDLE_INTERVALS:
            raise ValueError(f"interval must be one of {CANDLE_INTERVALS}, got {interval!r}")
        if not self.configured:
            log.info("oddspipe_not_configured_skipping")
            return []
        params: dict[str, Any] = {"interval": interval, "limit": 5000}
        if start is not None:
            params["start"] = start.astimezone(UTC).isoformat()
        if end is not None:
            params["end"] = end.astimezone(UTC).isoformat()
        try:
            payload = self._get(f"/v1/markets/{market_id}/candlesticks", params=params)
        except DataProviderError as exc:
            log.warning("oddspipe_candlesticks_unavailable_skipping", error=str(exc))
            return []
        rows: Any = payload
        if isinstance(payload, Mapping):
            for key in ("candles", "candlesticks", "items", "data"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
        if not isinstance(rows, list):
            return []
        out: list[OddsPipeCandle] = []
        for row in rows:
            candle = self._to_candle(market_id, row)
            if candle is not None:
                out.append(candle)
        return out
