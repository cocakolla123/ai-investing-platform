"""Reddit adapter tests: OAuth flow, ticker extraction, graceful skip, pacing."""

from __future__ import annotations

import base64
from datetime import UTC, datetime

import httpx

from dynamo.data.ratelimit import TokenBucket
from dynamo.data.reddit import RedditSocialProvider, extract_symbols

UNIVERSE = {"ACME", "TSLA", "NOW", "COST", "GME", "ITX"}

POST_HOT = {
    "id": "h1",
    "title": "DD on $ACME after earnings",
    "selftext": "Long thesis. " + "x" * 600,
    "created_utc": 1785400000,
    "permalink": "/r/stocks/comments/h1/dd/",
}
POST_NEW = {
    "id": "n1",
    "title": "TSLA delivery numbers discussion",
    "selftext": "",
    "created_utc": 1785400100,
    "permalink": "/r/stocks/comments/n1/tsla/",
}
POST_NO_MATCH = {
    "id": "n2",
    "title": "What broker do you all use?",
    "selftext": "no tickers here",
    "created_utc": 1785400200,
}


def _listing(posts: list[dict]) -> dict:
    return {"data": {"children": [{"data": p} for p in posts]}}


def make_handler(request_log: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        if request.url.path == "/api/v1/access_token":
            expected = base64.b64encode(b"cid:csec").decode()
            assert request.headers["Authorization"] == f"Basic {expected}"
            assert b"grant_type=client_credentials" in request.content
            assert request.headers["User-Agent"] == "dynamo-research/0.1"
            return httpx.Response(200, json={"access_token": "tok-123"})
        assert request.headers["Authorization"] == "Bearer tok-123"
        assert request.headers["User-Agent"] == "dynamo-research/0.1"
        if request.url.path.endswith("/hot"):
            return httpx.Response(200, json=_listing([POST_HOT, POST_NO_MATCH]))
        return httpx.Response(200, json=_listing([POST_NEW, POST_HOT]))  # dup h1

    return handler


def make_provider(
    handler, *, subreddits=("stocks",), bucket: TokenBucket | None = None
) -> RedditSocialProvider:
    return RedditSocialProvider(
        "cid",
        "csec",
        subreddits=subreddits,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=bucket or TokenBucket(rate=10_000.0, capacity=10_000.0),
        auth_base_url="https://auth.test",
        oauth_base_url="https://oauth.test",
    )


def test_oauth_flow_and_parsing() -> None:
    requests: list[httpx.Request] = []
    items = make_provider(make_handler(requests)).mentions(UNIVERSE)

    # 1 token request + hot + new
    assert [r.url.path for r in requests[:1]] == ["/api/v1/access_token"]
    assert len(requests) == 3

    by_id = {i.id: i for i in items}
    assert set(by_id) == {"reddit:h1", "reddit:n1"}  # dedup across listings, no-match dropped
    hot = by_id["reddit:h1"]
    assert hot.symbols == ("ACME",)
    assert hot.headline == "DD on $ACME after earnings"
    assert hot.summary is not None and len(hot.summary) == 500  # truncated
    assert hot.source == "reddit"
    assert hot.published_at == datetime.fromtimestamp(1785400000, tz=UTC)
    assert hot.url == "https://www.reddit.com/r/stocks/comments/h1/dd/"
    assert by_id["reddit:n1"].symbols == ("TSLA",)  # bare uppercase, unambiguous


def test_token_fetched_once_across_subreddits() -> None:
    requests: list[httpx.Request] = []
    make_provider(make_handler(requests), subreddits=("stocks", "investing")).mentions(UNIVERSE)
    token_calls = [r for r in requests if r.url.path == "/api/v1/access_token"]
    assert len(token_calls) == 1
    assert len(requests) == 5  # 1 token + 2 subreddits x 2 listings


def test_graceful_skip_without_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made without credentials")

    provider = RedditSocialProvider(
        "",
        "",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=TokenBucket(rate=1.0, capacity=1.0),
    )
    assert provider.mentions(UNIVERSE) == []


def test_rate_limit_bucket_used() -> None:
    acquired: list[float] = []

    class CountingBucket(TokenBucket):
        def acquire(self, tokens: float = 1.0) -> None:
            acquired.append(tokens)
            super().acquire(tokens)

    requests: list[httpx.Request] = []
    bucket = CountingBucket(rate=10_000.0, capacity=10_000.0)
    make_provider(make_handler(requests), bucket=bucket).mentions(UNIVERSE)
    assert len(acquired) == len(requests) == 3  # every request paced


# -- ticker extraction --------------------------------------------------------


def test_extract_cashtags_case_insensitive() -> None:
    assert extract_symbols("buying $acme and $TSLA today", UNIVERSE) == ("ACME", "TSLA")


def test_extract_ambiguous_symbols_require_cashtag() -> None:
    # NOW and COST are real universe symbols but common words: bare use ignored.
    assert extract_symbols("COST of living is high right NOW", UNIVERSE) == ()
    assert extract_symbols("adding $COST and $NOW here", UNIVERSE) == ("COST", "NOW")


def test_extract_word_boundary_no_substring_match() -> None:
    # "IT" inside "ITX department"? ITX matches only as a whole token.
    assert extract_symbols("The ITXX rollout", UNIVERSE) == ()
    assert extract_symbols("ITX earnings beat", UNIVERSE) == ("ITX",)


def test_extract_filters_to_universe() -> None:
    assert extract_symbols("$ZZZZ to the moon, also GME", UNIVERSE) == ("GME",)
    assert extract_symbols("lowercase gme is not a mention", UNIVERSE) == ()
