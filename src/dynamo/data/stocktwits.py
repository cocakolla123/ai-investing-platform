"""StockTwits social-mention adapter — awareness input for Tier-2 surveillance.

Design: doc 02 §1 (adapter conventions) + owner-approved social layer design:
mentions feed the ``social_burst`` detector as an awareness trigger, never a
buy signal. Message text is UNTRUSTED and only ever counted.

Public per-symbol streams (``api.stocktwits.com/api/2/streams/symbol/{sym}.json``,
messages already ticker-tagged). The API has had auth-policy changes, so a 403
is treated as ``ProviderDown`` with one clear log and an empty result — the
scan degrades, it never crashes. Requests are capped client-side (~200/hr
bucket) and symbols are fetched in caller priority order.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.errors import DataProviderError
from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import NewsItem

log = structlog.get_logger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
SUMMARY_MAX_CHARS = 500
HEADLINE_MAX_CHARS = 200


class StockTwitsProvider:
    """Fetch per-symbol message streams as ``NewsItem`` rows."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = STOCKTWITS_BASE,
        max_symbols_per_scan: int = 30,
    ) -> None:
        self._enabled = enabled
        self._client = client or httpx.Client(timeout=30.0)
        # Unauthenticated tier ≈200 req/hr → pace client-side, small burst.
        self._bucket = bucket or TokenBucket(rate=200.0 / 3600.0, capacity=10.0)
        self._base_url = base_url
        self._max_symbols = max_symbols_per_scan

    def mentions(self, symbols: Sequence[str]) -> list[NewsItem]:
        """Messages for the first ``max_symbols_per_scan`` symbols, in order.

        Callers should pass symbols highest-priority first (the request cap
        means the tail may not be fetched). Any provider failure (incl. 403
        from their auth changes) logs once and returns what was collected.
        """
        if not self._enabled:
            log.info("stocktwits_skipped", reason="disabled by settings")
            return []
        items: list[NewsItem] = []
        for symbol in list(symbols)[: self._max_symbols]:
            try:
                payload = get_json(
                    self._client,
                    f"{self._base_url}/streams/symbol/{symbol}.json",
                    provider="stocktwits",
                    bucket=self._bucket,
                )
            except DataProviderError as exc:
                log.warning(
                    "stocktwits_unavailable",
                    symbol=symbol,
                    error=str(exc),
                    note="treating provider as down for this scan",
                )
                break
            for message in (payload or {}).get("messages") or []:
                item = self._to_news_item(message, symbol)
                if item is not None:
                    items.append(item)
        return items

    @staticmethod
    def _to_news_item(message: dict[str, Any], queried_symbol: str) -> NewsItem | None:
        msg_id = message.get("id")
        body = message.get("body")
        created = message.get("created_at")
        if msg_id is None or not body or not created:
            return None
        published_at = _parse_ts(str(created))
        if published_at is None:
            return None
        tagged = {
            str(s.get("symbol")).upper()
            for s in message.get("symbols") or []
            if s.get("symbol")
        }
        tagged.add(queried_symbol.upper())
        sentiment = (((message.get("entities") or {}).get("sentiment")) or {}).get("basic")
        prefix = f"[{str(sentiment).lower()}] " if sentiment else ""
        body_text = str(body)
        return NewsItem(
            id=f"stocktwits:{msg_id}",
            symbols=tuple(sorted(tagged)),
            headline=body_text[:HEADLINE_MAX_CHARS],
            summary=(prefix + body_text)[:SUMMARY_MAX_CHARS],
            source="stocktwits",
            published_at=published_at,
        )


def _parse_ts(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
