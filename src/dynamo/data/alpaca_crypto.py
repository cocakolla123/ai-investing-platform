"""Alpaca crypto market-data adapter (ADR-0013 crypto sleeve).

Daily crypto bars from Alpaca's free crypto feed via ``alpaca-py``'s
``CryptoHistoricalDataClient``. No API keys are required for crypto data;
keys are passed through when present for higher rate limits.

Symbol convention: the canonical internal representation is the slash pair
form Alpaca uses for trading and crypto data — ``"BTC/USD"`` — everywhere in
the platform (bars_1d, instruments, universe, positions, orders). Alpaca's
crypto data API accepts and returns this form directly; any slashless
variant (``"BTCUSD"``) is normalized back to canonical at this edge via
:func:`dynamo.domain.symbols.normalize_crypto_symbol`. ``source="alpaca_crypto"``.

Crypto trades 24/7: bars exist for every calendar day and are all returned —
downstream consumers decide which days matter (the live loop ingests daily;
replay trades/marks on equity sessions only in v1). History is available
from ~2021, so arbitrary historical ranges work for backfills (the Phase 1
backfill CLI calls ``daily_bars`` directly).

Error mapping and retry style mirror ``dynamo.data.alpaca_data``: pacing via
a token bucket, one jittered retry on transient failures, typed
``RateLimited`` / ``ProviderDown`` exceptions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Any

import structlog
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from dynamo.data.errors import ProviderDown, RateLimited
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import Bar
from dynamo.domain.symbols import normalize_crypto_symbol

log = structlog.get_logger(__name__)

_DAILY = TimeFrame(1, TimeFrameUnit.Day)  # type: ignore[arg-type]


class _Transient(Exception):
    """Internal retryable marker (5xx / connection issues)."""


class AlpacaCryptoData:
    """Daily crypto bars over ``alpaca-py`` (free feed, keys optional)."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        *,
        data_client: CryptoHistoricalDataClient | None = None,
        bucket: TokenBucket | None = None,
    ) -> None:
        if data_client is not None:
            self._data = data_client
        elif api_key and api_secret:
            self._data = CryptoHistoricalDataClient(api_key, api_secret)
        else:
            # Crypto data needs no keys — anonymous client, lower rate limit.
            self._data = CryptoHistoricalDataClient()
        # Same free-tier pacing as the equities adapter (200 req/min).
        self._bucket = bucket or TokenBucket(rate=3.0, capacity=10.0)

    @property
    def source(self) -> str:
        return "alpaca_crypto"

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """SDK call with pacing, one jittered retry, typed exceptions."""

        @retry(
            retry=retry_if_exception_type(_Transient),
            stop=stop_after_attempt(2),  # 1 retry (doc 01 §6)
            wait=wait_random(0.2, 1.0),
            reraise=True,
        )
        def _do() -> Any:
            self._bucket.acquire()
            try:
                return fn(*args, **kwargs)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status == 429:
                    log.warning("provider_rate_limited", provider="alpaca_crypto")
                    raise RateLimited("alpaca_crypto", str(exc)) from exc
                if status is not None and int(status) >= 500:
                    raise _Transient(str(exc)) from exc
                raise ProviderDown("alpaca_crypto", str(exc)) from exc
            except (ConnectionError, TimeoutError) as exc:
                raise _Transient(str(exc)) from exc

        try:
            return _do()
        except _Transient as exc:
            raise ProviderDown("alpaca_crypto", str(exc)) from exc

    def _map_bar(self, symbol: str, b: Any) -> Bar:
        return Bar(
            symbol=normalize_crypto_symbol(symbol),
            ts=b.timestamp.astimezone(UTC),
            open=float(b.open),
            high=float(b.high),
            low=float(b.low),
            # Crypto volume is fractional coins; bars_1d stores integer volume,
            # so it is truncated. Dollar-volume consumers (ADV) multiply by
            # close, so the sub-coin truncation is negligible.
            close=float(b.close),
            volume=int(b.volume),
            adj_close=float(b.close),  # no splits/dividends: close IS adjusted
            source=self.source,
        )

    # -- data surface ------------------------------------------------------

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> Iterable[Bar]:
        """Daily OHLCV for every calendar day in [start, end] (24/7 market)."""
        request = CryptoBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=_DAILY,
            start=datetime(start.year, start.month, start.day, tzinfo=UTC),
            end=datetime(end.year, end.month, end.day, 23, 59, tzinfo=UTC),
        )
        barset = self._call(self._data.get_crypto_bars, request)
        data: dict[str, list[Any]] = getattr(barset, "data", {}) or {}
        for symbol, bars in data.items():
            for b in bars:
                yield self._map_bar(symbol, b)

    def latest_daily(self, symbols: Sequence[str]) -> Iterable[Bar]:
        """The latest (possibly in-progress) daily bar per symbol — the
        nightly-ingest path; idempotent upserts refresh it next run."""
        request = CryptoLatestBarRequest(symbol_or_symbols=list(symbols))
        latest: dict[str, Any] = self._call(self._data.get_crypto_latest_bar, request) or {}
        for symbol, b in latest.items():
            yield self._map_bar(symbol, b)
