"""Alpaca crypto adapter mapping tests with fake SDK clients (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient

from dynamo.data.alpaca_crypto import AlpacaCryptoData
from dynamo.data.errors import ProviderDown, RateLimited
from dynamo.data.ratelimit import TokenBucket


def _bar(ts: datetime, close: float, volume: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=ts,
        open=close - 100.0,
        high=close + 200.0,
        low=close - 200.0,
        close=close,
        volume=volume,
    )


class FakeCryptoClient:
    def get_crypto_bars(self, request: Any) -> Any:
        # 24/7 market: Fri/Sat/Sun/Mon bars all exist and are all returned.
        days = [date(2026, 8, 14), date(2026, 8, 15), date(2026, 8, 16), date(2026, 8, 17)]
        bars = [
            _bar(datetime(d.year, d.month, d.day, 5, 0, tzinfo=UTC), 60_000.0 + i, 12.345)
            for i, d in enumerate(days)
        ]
        return SimpleNamespace(data={"BTC/USD": bars})

    def get_crypto_latest_bar(self, request: Any) -> Any:
        return {
            # Slashless key exercises edge normalization.
            "ETHUSD": _bar(datetime(2026, 8, 17, 5, 0, tzinfo=UTC), 4_000.0, 3.5),
        }


def make_adapter(client: Any | None = None) -> AlpacaCryptoData:
    return AlpacaCryptoData(
        data_client=cast(CryptoHistoricalDataClient, client or FakeCryptoClient()),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_constructs_without_keys() -> None:
    # Crypto data requires no API keys — the anonymous client works.
    adapter = AlpacaCryptoData(bucket=TokenBucket(rate=10_000.0, capacity=10_000.0))
    assert adapter.source == "alpaca_crypto"


def test_daily_bars_returns_every_calendar_day() -> None:
    adapter = make_adapter()
    bars = list(adapter.daily_bars(["BTC/USD"], date(2026, 8, 14), date(2026, 8, 17)))
    assert len(bars) == 4  # weekend bars included — downstream decides
    assert {b.symbol for b in bars} == {"BTC/USD"}
    assert all(b.source == "alpaca_crypto" for b in bars)
    assert bars[0].close == 60_000.0
    assert bars[0].adj_close == 60_000.0  # no splits/dividends: close IS adjusted
    assert bars[0].volume == 12  # fractional coin volume truncates to int


def test_latest_daily_normalizes_slashless_symbols() -> None:
    adapter = make_adapter()
    bars = list(adapter.latest_daily(["ETH/USD"]))
    assert [b.symbol for b in bars] == ["ETH/USD"]
    assert bars[0].close == 4_000.0


class _Failing:
    def __init__(self, status: int) -> None:
        self._status = status

    def get_crypto_bars(self, request: Any) -> Any:
        http_error = SimpleNamespace(
            response=SimpleNamespace(status_code=self._status), request=None
        )
        raise APIError("boom", http_error)


def test_rate_limit_maps_to_typed_error() -> None:
    adapter = make_adapter(_Failing(429))
    with pytest.raises(RateLimited):
        list(adapter.daily_bars(["BTC/USD"], date(2026, 8, 1), date(2026, 8, 2)))


def test_client_error_maps_to_provider_down() -> None:
    adapter = make_adapter(_Failing(400))
    with pytest.raises(ProviderDown):
        list(adapter.daily_bars(["BTC/USD"], date(2026, 8, 1), date(2026, 8, 2)))
