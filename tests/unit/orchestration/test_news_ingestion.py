"""A provider rate limit truncates the news sweep — it does not void it.

Finnhub's free tier allows 60 req/min and the asset sweeps the tier-1 +
promoted universe one request per symbol, so a 429 partway through is an
expected operating condition, not an incident. ``get_json`` deliberately does
not retry a 429 (doc 01 §6, "no retry storm"), so the exception reached the
asset and failed the whole run: every symbol after the limit was lost, AND so
were TickerTick's headlines, even though TickerTick is a separate keyless API
that had not been rate-limited at all. Three ``news_job`` failures in
production (2026-09-15/16) were exactly this.

No network, no database.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast

from dagster import MaterializeResult, build_asset_context
from orchestration.assets.ingestion import news

from dynamo.data.errors import RateLimited
from dynamo.domain.models.market import NewsItem

PARTITION = "2026-09-16"


def _item(source: str, n: int) -> NewsItem:
    return NewsItem(
        id=f"{source}:{n}",
        symbols=("ACME",),
        headline=f"{source} headline {n}",
        summary=None,
        source=source,
        published_at=datetime(2026, 9, 16, 12, tzinfo=UTC),
        url=None,
    )


class _Provider:
    """Yields ``before`` items, then optionally rate-limits mid-generator."""

    def __init__(self, source: str, *, before: int, then_429: bool) -> None:
        self._source = source
        self._before = before
        self._then_429 = then_429

    def news(self, symbols: Any, since: Any) -> Iterator[NewsItem]:
        for n in range(self._before):
            yield _item(self._source, n)
        if self._then_429:
            raise RateLimited(self._source, "429 from /company-news")


class _Repo:
    def __init__(self) -> None:
        self.persisted: list[NewsItem] = []

    def upsert_news(self, items: Any) -> int:
        rows = list(items)
        self.persisted.extend(rows)
        return len(rows)


class _Dynamo:
    def __init__(self, finnhub: Any, tickertick: Any) -> None:
        self._finnhub = finnhub
        self._tickertick = tickertick
        self.repo = _Repo()

    def finnhub(self) -> Any:
        return self._finnhub

    def tickertick(self) -> Any:
        return self._tickertick

    def universe_symbols(self, as_of: Any, tiers: Any = None) -> list[str]:
        return ["ACME", "PEER"]

    def market_repo(self) -> Any:
        return self.repo


def _ctx() -> Any:
    return build_asset_context(partition_key=PARTITION)


def _run(finnhub: Any, tickertick: Any) -> tuple[dict[str, Any], _Repo]:
    dynamo = _Dynamo(finnhub, tickertick)
    result = cast(MaterializeResult, news(_ctx(), dynamo))  # type: ignore[arg-type]
    return dict(result.metadata or {}), dynamo.repo


def test_headlines_fetched_before_the_429_are_persisted() -> None:
    metadata, repo = _run(
        _Provider("finnhub", before=3, then_429=True),
        _Provider("tickertick", before=0, then_429=False),
    )
    assert metadata["finnhub"] == 3
    assert [i.id for i in repo.persisted] == ["finnhub:0", "finnhub:1", "finnhub:2"]


def test_a_finnhub_429_does_not_cost_us_tickertick() -> None:
    """The regression: two independent APIs, one shared generator chain."""
    metadata, repo = _run(
        _Provider("finnhub", before=1, then_429=True),
        _Provider("tickertick", before=4, then_429=False),
    )
    assert metadata["tickertick"] == 4
    assert metadata["rows"] == 5
    assert sum(1 for i in repo.persisted if i.source == "tickertick") == 4


def test_the_truncation_is_reported_not_swallowed() -> None:
    metadata, _ = _run(
        _Provider("finnhub", before=1, then_429=True),
        _Provider("tickertick", before=1, then_429=True),
    )
    assert metadata["rate_limited"] == ["finnhub", "tickertick"]


def test_a_clean_sweep_reports_no_rate_limiting() -> None:
    metadata, _ = _run(
        _Provider("finnhub", before=2, then_429=False),
        _Provider("tickertick", before=2, then_429=False),
    )
    assert metadata["rate_limited"] == "none"
    assert metadata["rows"] == 4


def test_finnhub_unconfigured_is_a_skip_not_a_failure() -> None:
    metadata, _ = _run(None, _Provider("tickertick", before=2, then_429=False))
    assert metadata["rows"] == 0
    assert metadata["skipped"] == "finnhub not configured"
