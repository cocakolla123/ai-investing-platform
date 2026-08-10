"""TickerTick adapter — second news source beside Finnhub (NewsProvider-shaped).

Design: doc 02 §1 adapter conventions. Free API, no key
(https://github.com/hczhu/TickerTick-API)::

    GET https://api.tickertick.com/feed?q=z:<ticker>&n=<1..200>

Rate limit is 10 requests/minute per IP, enforced client-side with a token
bucket well under that; one request per symbol means a run covers at most
``symbol_cap`` symbols (callers pass Tier-1 symbols in priority order).
Stories emit as ``NewsItem(source='tickertick', id='tickertick:<story_id>')``
— ids never collide with Finnhub's, so the shared news upsert just works.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import NewsItem

log = structlog.get_logger(__name__)

TICKERTICK_BASE = "https://api.tickertick.com"
DEFAULT_SYMBOL_CAP = 100  # one request per symbol -> ~10 minutes of budget
STORIES_PER_SYMBOL = 50


class TickerTickClient:
    """``NewsProvider`` implementation over the TickerTick /feed endpoint."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = TICKERTICK_BASE,
        symbol_cap: int = DEFAULT_SYMBOL_CAP,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        # 10 req/min hard limit -> pace at 9/min with a tiny burst allowance.
        self._bucket = bucket or TokenBucket(rate=9.0 / 60.0, capacity=3.0)
        self._base_url = base_url
        self._symbol_cap = symbol_cap

    def news(self, symbols: Sequence[str], since: datetime) -> Iterable[NewsItem]:
        """Stories per symbol since ``since`` (UTC), first ``symbol_cap`` symbols.

        Callers pass symbols in priority order (Tier 1 + promoted first); the
        per-run cap keeps a full sweep inside the 10 req/min budget.
        """
        wanted = [s for s in symbols][: self._symbol_cap]
        if len(symbols) > len(wanted):
            log.info(
                "tickertick_symbol_cap", requested=len(symbols), fetched=len(wanted)
            )
        universe = {s.upper() for s in symbols}
        seen: set[str] = set()
        for symbol in wanted:
            payload = self._get_feed(symbol)
            for story in (payload or {}).get("stories") or []:
                item = self._to_news_item(story, symbol, universe, since)
                if item is not None and item.id not in seen:
                    seen.add(item.id)
                    yield item

    def _get_feed(self, symbol: str) -> dict | None:
        return get_json(
            self._client,
            f"{self._base_url}/feed",
            provider="tickertick",
            bucket=self._bucket,
            params={"q": f"z:{symbol.lower()}", "n": STORIES_PER_SYMBOL},
        )

    @staticmethod
    def _to_news_item(
        story: dict, symbol: str, universe: set[str], since: datetime
    ) -> NewsItem | None:
        story_id = story.get("id")
        title = story.get("title")
        time_ms = story.get("time")
        if not story_id or not title or time_ms is None:
            return None
        published_at = datetime.fromtimestamp(int(time_ms) / 1000.0, tz=UTC)
        if published_at < since:
            return None
        tagged = {str(t).upper() for t in story.get("tickers") or []}
        matched = tuple(sorted(tagged & universe)) or (symbol.upper(),)
        return NewsItem(
            id=f"tickertick:{story_id}",
            symbols=matched,
            headline=str(title),
            summary=str(story.get("description")).strip() or None
            if story.get("description")
            else None,
            source="tickertick",
            published_at=published_at,
            url=story.get("url") or None,
        )
