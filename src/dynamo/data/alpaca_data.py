"""Alpaca market-data adapter — implements the ``MarketDataProvider`` port.

Design: doc 02 §1 (daily + minute bars on the free IEX feed, quotes,
corporate actions, tradable-asset list; ``source="alpaca"``).

Uses the official ``alpaca-py`` SDK:

- ``StockHistoricalDataClient`` for bars and latest quotes,
- ``TradingClient`` for the asset list and corporate-action announcements.

Bars are fetched SPLIT-adjusted. Tiingo remains the *fully* adjusted source
(doc 02 §5) and still owns ``adj_close``; what changed is that Alpaca's raw
closes were reaching detectors as prices, and a split is not a price move.
See :data:`DAILY_BAR_ADJUSTMENT` for why split-only and not ``ALL``, and for
the migration this implies for rows already stored.

The corporate-announcements endpoint only accepts ~90-day windows, so
requests are chunked.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus, CorporateActionType
from alpaca.trading.requests import GetAssetsRequest, GetCorporateAnnouncementsRequest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from dynamo.data.errors import ProviderDown, RateLimited
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import Bar, CorporateAction, Instrument

log = structlog.get_logger(__name__)

_CA_WINDOW_DAYS = 89  # announcements endpoint rejects ranges > 90 days
_DAILY = TimeFrame(1, TimeFrameUnit.Day)  # type: ignore[arg-type]

# Daily bars are requested SPLIT-adjusted, not RAW.
#
# WHY THIS CHANGED. These bars were fetched RAW on the reasoning that "Tiingo
# owns adjustment" (doc 02 §5). That holds for the ~55 names Tiingo covers.
# It does not hold for the other ~11,200, which have no Tiingo row at all, so
# `MarketDataRepository.bars_frame` falls back to the raw Alpaca close and the
# canonical series contains unadjusted prices. Corporate actions then read as
# price moves: on 2026-08-17/18 RCON printed a 1-for-141 reverse split as
# +14,015% and CXAI a 1-for-47 as +4,610%, both of which reached the
# surveillance detectors as trading signals.
#
# WHY SPLIT AND NOT ALL. `Adjustment.ALL` would also back out dividends,
# producing a total-return series. Two reasons not to:
#   - splits are the failure mode. A split rewrites the price by 141x; a
#     dividend moves it by a fraction of a percent, which is inside the noise
#     these detectors already tolerate.
#   - `quality.dual_source_close_check` compares the Alpaca and Tiingo *raw*
#     close columns for the overlap names and alerts past 50bps. Dividend
#     adjustment would make that check fire on every historical ex-date for a
#     difference that is not an error. Split adjustment affects only actual
#     split dates, where the two feeds agree once both are adjusted.
#
# MIGRATION — IMPORTANT. This changes what NEW rows mean, not old ones. The
# ~6.0M bars already stored are RAW, so until the affected symbols are
# re-fetched their series carry raw history followed by adjusted history, and
# a symbol that split in the past gains an artificial jump at the changeover.
# `upsert_bars` is keyed on (symbol, ts, source), so re-running the backfill
# over the full range rewrites them in place:
#
#     python -m dynamo.cli.backfill bars --provider alpaca --universe \
#         --tiers 2 --start 2024-01-01 --end <today> --no-skip-covered
#
# Until that has run, `price_zscore`'s sanity ceiling is what stops an
# unadjusted action from reaching a trading detector (see
# dynamo.data.detectors.PRICE_ZSCORE_MAX_PRICE_RATIO). The ceiling stays in
# place afterwards regardless: it is the backstop for any feed that silently
# stops adjusting.
DAILY_BAR_ADJUSTMENT = Adjustment.SPLIT


#: Symbols per bars request. The backfill CLI's measured-safe default; the
#: whole universe in one request exceeds the server's URI limit (HTTP 414).
DAILY_BARS_SYMBOLS_PER_REQUEST = 200


class _Transient(Exception):
    """Internal retryable marker (5xx / connection issues)."""


class AlpacaMarketData:
    """``MarketDataProvider`` implementation over alpaca-py (free IEX feed)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        data_client: StockHistoricalDataClient | None = None,
        trading_client: TradingClient | None = None,
        bucket: TokenBucket | None = None,
        paper: bool = True,
    ) -> None:
        self._data = data_client or StockHistoricalDataClient(api_key, api_secret)
        self._trading = trading_client or TradingClient(api_key, api_secret, paper=paper)
        # Free tier: 200 req/min → ~3.3 req/s.
        self._bucket = bucket or TokenBucket(rate=3.0, capacity=10.0)

    @property
    def source(self) -> str:
        return "alpaca"

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
                    log.warning("provider_rate_limited", provider="alpaca")
                    raise RateLimited("alpaca", str(exc)) from exc
                if status is not None and int(status) >= 500:
                    raise _Transient(str(exc)) from exc
                raise ProviderDown("alpaca", str(exc)) from exc
            except (ConnectionError, TimeoutError) as exc:
                raise _Transient(str(exc)) from exc

        try:
            return _do()
        except _Transient as exc:
            raise ProviderDown("alpaca", str(exc)) from exc

    # -- MarketDataProvider ------------------------------------------------

    def daily_bars(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        *,
        adjustment: Adjustment = DAILY_BAR_ADJUSTMENT,
    ) -> Iterable[Bar]:
        """Split-adjusted daily OHLCV from the IEX feed.

        ``adjustment`` is an argument so the raw series stays reachable (the
        dual-source close check and any corporate-action audit want it), but
        the default is split-adjusted: see :data:`DAILY_BAR_ADJUSTMENT`.
        """
        # Symbols travel in the query string. The nightly ingest passes the
        # whole universe (~11,000 names) and the server rejected that single
        # request with 414 Request-URI Too Large (2026-09-23) — every night,
        # so live EOD prices never landed. Batch it, as the backfill does.
        wanted = list(symbols)
        for offset in range(0, len(wanted), DAILY_BARS_SYMBOLS_PER_REQUEST):
            request = StockBarsRequest(
                symbol_or_symbols=wanted[offset : offset + DAILY_BARS_SYMBOLS_PER_REQUEST],
                timeframe=_DAILY,
                start=datetime(start.year, start.month, start.day, tzinfo=UTC),
                end=datetime(end.year, end.month, end.day, 23, 59, tzinfo=UTC),
                adjustment=adjustment,
                feed=DataFeed.IEX,
            )
            barset = self._call(self._data.get_stock_bars, request)
            data: dict[str, list[Any]] = getattr(barset, "data", {}) or {}
            for symbol, bars in data.items():
                for b in bars:
                    yield Bar(
                        symbol=symbol,
                        ts=b.timestamp.astimezone(UTC),
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=int(b.volume),
                        # Tiingo owns the FULLY adjusted close (doc 02 §5); these
                        # closes are split-adjusted only, so they are not that.
                        adj_close=None,
                        source=self.source,
                    )

    def latest_quotes(self, symbols: Sequence[str]) -> dict[str, float]:
        """Latest quote midpoint per symbol (falls back to the ask/bid side)."""
        request = StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)
        quotes: dict[str, Any] = self._call(self._data.get_stock_latest_quote, request)
        out: dict[str, float] = {}
        for symbol, q in quotes.items():
            bid = float(q.bid_price or 0.0)
            ask = float(q.ask_price or 0.0)
            if bid > 0 and ask > 0:
                out[symbol] = (bid + ask) / 2.0
            elif ask > 0 or bid > 0:
                out[symbol] = max(ask, bid)
        return out

    def corporate_actions(self, start: date, end: date) -> Iterable[CorporateAction]:
        """Splits/dividends/mergers/spinoffs, chunked into <=90d windows."""
        window_start = start
        while window_start <= end:
            window_end = min(window_start + timedelta(days=_CA_WINDOW_DAYS), end)
            request = GetCorporateAnnouncementsRequest(
                ca_types=list(CorporateActionType),
                since=window_start,
                until=window_end,
            )
            announcements = self._call(self._trading.get_corporate_announcements, request)
            yield from self._map_announcements(announcements)
            window_start = window_end + timedelta(days=1)

    def _map_announcements(self, announcements: Iterable[Any]) -> Iterator[CorporateAction]:
        for a in announcements:
            symbol = a.initiating_symbol or a.target_symbol
            ex_date = a.ex_date
            if not symbol or ex_date is None:
                continue
            ca_type = str(getattr(a.ca_type, "value", a.ca_type))
            ratio: float | None = None
            amount: float | None = None
            if ca_type == "split" and a.old_rate and a.new_rate:
                ratio = float(a.new_rate) / float(a.old_rate)
            if ca_type == "dividend" and a.cash is not None:
                amount = float(a.cash)
            yield CorporateAction(
                symbol=str(symbol),
                ex_date=ex_date,
                action_type=ca_type,
                ratio=ratio,
                amount=amount,
                source=self.source,
            )

    def tradable_instruments(self) -> Iterable[Instrument]:
        """Active US-equity assets (the canonical tradable list, doc 02 §1)."""
        request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = self._call(self._trading.get_all_assets, request)
        for a in assets:
            if not a.tradable:
                continue
            exchange = str(getattr(a.exchange, "value", a.exchange))
            yield Instrument(
                symbol=a.symbol,
                name=a.name or a.symbol,
                exchange=exchange,
            )
