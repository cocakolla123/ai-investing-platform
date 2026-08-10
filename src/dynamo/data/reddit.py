"""Reddit social-mention adapter — awareness input for Tier-2 surveillance.

Design: doc 02 §1 (adapter conventions) + owner-approved social layer design:
social activity is a DETECTOR INPUT (feeds the ``social_burst`` materiality
check), never a direct buy signal. Post text is UNTRUSTED — it is stored as
``NewsItem`` rows (source='reddit') and only ever counted, not scored.

Flow: OAuth2 client-credentials against ``www.reddit.com/api/v1/access_token``,
then listing reads against ``oauth.reddit.com`` (``/r/<sub>/hot`` + ``/new``).
Free tier allows ~100 req/min with OAuth; we pace at 90/min client-side.

Ticker extraction is defensive: cashtags (``$SYM``) always count; bare
uppercase tokens count only when the symbol is in the caller-provided universe
AND is not a common English word (the ambiguous set below requires a cashtag).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.errors import ProviderDown
from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import NewsItem

log = structlog.get_logger(__name__)

REDDIT_AUTH_BASE = "https://www.reddit.com"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
SUMMARY_MAX_CHARS = 500

# Symbols that collide with common English/Reddit words: a bare uppercase
# token match is meaningless for these, so they require a $ cashtag.
AMBIGUOUS_SYMBOLS: frozenset[str] = frozenset(
    {
        "A", "ALL", "ANY", "ARE", "BE", "BY", "CAN", "COST", "DD", "EDIT",
        "FOR", "GO", "IT", "KEY", "LOVE", "NEW", "NOW", "ON", "ONE", "OR",
        "OUT", "PLAY", "REAL", "RH", "SO", "SEE", "TELL", "WELL", "YOLO",
    }
)

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_UPPER_WORD_RE = re.compile(r"\b[A-Z]{1,5}\b")


def extract_symbols(text: str, universe: Collection[str]) -> tuple[str, ...]:
    """Extract ticker mentions from ``text`` against a known symbol universe.

    - ``$SYM`` cashtags match case-insensitively and always count (if in the
      universe).
    - Bare uppercase tokens count only for non-ambiguous universe symbols
      (word-boundary match, so "IT department" never matches symbol ITX).
    """
    matched: set[str] = set()
    universe_set = {s.upper() for s in universe}
    for m in _CASHTAG_RE.finditer(text):
        sym = m.group(1).upper()
        if sym in universe_set:
            matched.add(sym)
    for m in _UPPER_WORD_RE.finditer(text):
        sym = m.group(0)
        if sym in universe_set and sym not in AMBIGUOUS_SYMBOLS:
            matched.add(sym)
    return tuple(sorted(matched))


class RedditSocialProvider:
    """Fetch hot+new posts from configured subreddits as ``NewsItem`` rows."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        user_agent: str = "dynamo-research/0.1",
        subreddits: Sequence[str] = ("stocks", "investing", "wallstreetbets", "StockMarket"),
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        auth_base_url: str = REDDIT_AUTH_BASE,
        oauth_base_url: str = REDDIT_OAUTH_BASE,
        posts_per_listing: int = 100,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._subreddits = tuple(subreddits)
        self._client = client or httpx.Client(timeout=30.0)
        # OAuth free tier ~100 req/min → pace at 90/min (1.5 req/s).
        self._bucket = bucket or TokenBucket(rate=1.5, capacity=10.0)
        self._auth_base = auth_base_url
        self._oauth_base = oauth_base_url
        self._posts_per_listing = posts_per_listing
        self._token: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def _ensure_token(self) -> str:
        """OAuth2 client-credentials flow; token cached for the provider's life."""
        if self._token is not None:
            return self._token
        self._bucket.acquire()
        try:
            resp = self._client.post(
                f"{self._auth_base}/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
                headers={"User-Agent": self._user_agent},
            )
        except httpx.HTTPError as exc:
            raise ProviderDown("reddit", f"token request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ProviderDown("reddit", f"token request returned {resp.status_code}")
        try:
            token = resp.json().get("access_token")
        except ValueError as exc:
            raise ProviderDown("reddit", f"invalid token JSON: {exc}") from exc
        if not token:
            raise ProviderDown("reddit", "token response missing access_token")
        self._token = str(token)
        return self._token

    def _listing(self, subreddit: str, listing: str) -> list[dict[str, Any]]:
        token = self._ensure_token()
        payload = get_json(
            self._client,
            f"{self._oauth_base}/r/{subreddit}/{listing}",
            provider="reddit",
            bucket=self._bucket,
            params={"limit": self._posts_per_listing},
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": self._user_agent,
            },
        )
        children = ((payload or {}).get("data") or {}).get("children") or []
        return [c.get("data") or {} for c in children]

    def mentions(self, universe: Collection[str]) -> list[NewsItem]:
        """Posts from hot+new across configured subreddits with ticker matches.

        Gracefully returns empty (with one log line) when credentials are
        unset — social awareness is optional, never load-bearing.
        """
        if not self.configured:
            log.info("reddit_skipped", reason="credentials not configured")
            return []
        items: list[NewsItem] = []
        seen_ids: set[str] = set()
        for subreddit in self._subreddits:
            for listing in ("hot", "new"):
                for post in self._listing(subreddit, listing):
                    item = self._to_news_item(post, universe)
                    if item is not None and item.id not in seen_ids:
                        seen_ids.add(item.id)
                        items.append(item)
        return items

    @staticmethod
    def _to_news_item(post: dict[str, Any], universe: Collection[str]) -> NewsItem | None:
        post_id = post.get("id")
        title = post.get("title")
        created = post.get("created_utc")
        if not post_id or not title or created is None:
            return None
        selftext = str(post.get("selftext") or "")
        symbols = extract_symbols(f"{title}\n{selftext}", universe)
        if not symbols:
            return None
        permalink = post.get("permalink")
        return NewsItem(
            id=f"reddit:{post_id}",
            symbols=symbols,
            headline=str(title),
            summary=selftext[:SUMMARY_MAX_CHARS] or None,
            source="reddit",
            published_at=datetime.fromtimestamp(float(created), tz=UTC),
            url=f"https://www.reddit.com{permalink}" if permalink else None,
        )
