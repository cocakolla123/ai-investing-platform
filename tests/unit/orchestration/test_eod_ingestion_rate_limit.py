"""A Tiingo 429 must not turn the nightly price ingest red.

Tiingo's free tier allows 500 unique symbols a month. A research backfill
spent September's allowance on 2026-09-16, after which the nightly Tiingo
leg 429s on every request. The Alpaca leg runs first and is what every
symbol relies on, so its rows must persist and the run must succeed,
reporting the truncation instead of failing on it.

No network, no database.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast

from dagster import MaterializeResult, build_asset_context
from orchestration.assets.ingestion import eod_bars

from dynamo.data.errors import RateLimited
from dynamo.domain.models.market import Bar


class _Alpaca:
    def daily_bars(self, symbols: Any, start: Any, end: Any) -> Iterator[Bar]:
        for sym in symbols:
            yield Bar(
                symbol=sym,
                ts=datetime(2026, 9, 23, tzinfo=UTC),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=10,
                source="alpaca",
            )


class _Tiingo429:
    def daily_bars(self, symbols: Any, start: Any, end: Any) -> Iterator[Bar]:
        raise RateLimited("tiingo", "429 from /tiingo/daily/aapl/prices")
        yield  # pragma: no cover - makes this a generator, like the real client


class _Repo:
    def __init__(self) -> None:
        self.rows: list[Bar] = []

    def upsert_bars(self, bars: Any, timeframe: str) -> int:
        got = list(bars)
        self.rows.extend(got)
        return len(got)


class _Dynamo:
    def __init__(self) -> None:
        self.repo = _Repo()

    def universe_symbols(self, as_of: Any, tiers: Any = None) -> list[str]:
        if tiers == (1,):
            return ["AAPL", "BTC/USD"]  # tier-1 core (incl. a crypto pair)
        return ["AAPL", "BTC/USD", "MSFT", "TINY"]

    def market_repo(self) -> _Repo:
        return self.repo

    def alpaca(self) -> _Alpaca:
        return _Alpaca()

    def tiingo(self) -> _Tiingo429:
        return _Tiingo429()


def test_tiingo_quota_exhaustion_keeps_alpaca_rows_and_succeeds() -> None:
    dynamo = _Dynamo()
    ctx = build_asset_context(partition_key="2026-09-23")
    result = cast(MaterializeResult, eod_bars(ctx, dynamo))  # type: ignore[arg-type]
    meta = dict(result.metadata or {})
    assert meta["alpaca"] == 3
    assert meta["tiingo"] == 0
    assert meta["rate_limited"] == ["tiingo"]
    assert {b.symbol for b in dynamo.repo.rows} == {"AAPL", "MSFT", "TINY"}


def test_crypto_pairs_never_reach_the_equity_endpoints() -> None:
    """One "AVAX/USD" in a stock-bars batch is a 400 for the whole batch;
    crypto has its own ingest (crypto_bars)."""
    seen: list[str] = []

    class _Spy(_Alpaca):
        def daily_bars(self, symbols: Any, start: Any, end: Any) -> Iterator[Bar]:
            seen.extend(symbols)
            yield from super().daily_bars(symbols, start, end)

    dynamo = _Dynamo()
    dynamo.alpaca = lambda: _Spy()  # type: ignore[method-assign]
    eod_bars(build_asset_context(partition_key="2026-09-23"), dynamo)  # type: ignore[arg-type]
    assert seen == ["AAPL", "MSFT", "TINY"]


def test_tiingo_is_asked_only_for_the_tier1_core() -> None:
    """One request per symbol at 50/hour: the whole universe can never work."""
    asked: list[str] = []

    class _TiingoSpy:
        def daily_bars(self, symbols: Any, start: Any, end: Any) -> Iterator[Bar]:
            asked.extend(symbols)
            return iter(())

    dynamo = _Dynamo()
    dynamo.tiingo = lambda: _TiingoSpy()  # type: ignore[method-assign,assignment]
    eod_bars(build_asset_context(partition_key="2026-09-23"), dynamo)  # type: ignore[arg-type]
    assert asked == ["AAPL"]
