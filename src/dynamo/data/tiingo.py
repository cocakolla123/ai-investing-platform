"""Tiingo adapter — implements the ``MarketDataProvider`` port.

Design: doc 02 §1 (second price source, adjusted EOD history *including
delisted names* — survivorship-clean) and doc 02 §5 (``adj_close`` follows
Tiingo's adjustment method).

Division of labour between the two market-data adapters:

- Tiingo is the canonical *adjusted close* source (``adj_close``) and covers
  delisted history; ``source="tiingo"``.
- Alpaca is the canonical source for the tradable-asset list and corporate
  actions. ``corporate_actions``/``tradable_instruments`` here therefore
  return derived/empty results (documented below) rather than duplicating
  Alpaca's role.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, date, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.errors import DataGap
from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import Bar, CorporateAction, Instrument

log = structlog.get_logger(__name__)

TIINGO_BASE = "https://api.tiingo.com"


def _ts(value: str) -> datetime:
    """Tiingo dates like ``2026-07-31T00:00:00.000Z`` → aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class TiingoClient:
    """``MarketDataProvider`` implementation over the Tiingo REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = TIINGO_BASE,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        # Paid plan allows generous RPM; pace to ~5 req/s (doc 02 §8 budget).
        self._bucket = bucket or TokenBucket(rate=5.0, capacity=10.0)
        self._base_url = base_url
        self._headers = {"Authorization": f"Token {api_key}"}

    @property
    def source(self) -> str:
        return "tiingo"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._client,
            f"{self._base_url}{path}",
            provider="tiingo",
            bucket=self._bucket,
            params=params,
            headers=self._headers,
        )

    def _prices(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        # Tiingo URLs use hyphenated share classes (BRK-B), our canon uses dots (BRK.B).
        payload = self._get(
            f"/tiingo/daily/{symbol.lower().replace('.', '-')}/prices",
            params={
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "format": "json",
            },
        )
        if not isinstance(payload, list):
            raise DataGap("tiingo", f"unexpected prices payload for {symbol}")
        return payload

    # -- MarketDataProvider ------------------------------------------------

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> Iterable[Bar]:
        """Raw OHLCV plus Tiingo-adjusted close; works for delisted symbols."""
        for symbol in symbols:
            for row in self._prices(symbol, start, end):
                yield Bar(
                    symbol=symbol,
                    ts=_ts(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row.get("volume") or 0),
                    adj_close=float(row["adjClose"]) if row.get("adjClose") is not None else None,
                    source=self.source,
                )

    def latest_quotes(self, symbols: Sequence[str]) -> dict[str, float]:
        """Last trade price via the IEX endpoint (``tngoLast``)."""
        payload = self._get("/iex/", params={"tickers": ",".join(symbols)})
        quotes: dict[str, float] = {}
        for row in payload or []:
            price = row.get("tngoLast") or row.get("last")
            ticker = row.get("ticker")
            if ticker and price is not None:
                quotes[str(ticker).upper()] = float(price)
        return quotes

    def corporate_actions(self, start: date, end: date) -> Iterable[CorporateAction]:
        """Market-wide corporate actions are sourced from Alpaca (doc 02 §5).

        Tiingo has no free market-wide corp-action feed; use
        :meth:`corporate_actions_for` to derive per-symbol actions from the
        ``splitFactor``/``divCash`` fields of the daily price series.
        """
        log.info("tiingo_corporate_actions_noop", start=str(start), end=str(end))
        return ()

    def tradable_instruments(self) -> Iterable[Instrument]:
        """The tradable-asset list is sourced from Alpaca (doc 02 §1)."""
        log.info("tiingo_tradable_instruments_noop")
        return ()

    # -- extras ------------------------------------------------------------

    def corporate_actions_for(
        self, symbols: Sequence[str], start: date, end: date
    ) -> Iterator[CorporateAction]:
        """Derive splits/dividends from daily ``splitFactor``/``divCash``."""
        for symbol in symbols:
            for row in self._prices(symbol, start, end):
                ex_date = _ts(row["date"]).date()
                split = float(row.get("splitFactor") or 1.0)
                div = float(row.get("divCash") or 0.0)
                if split != 1.0:
                    yield CorporateAction(
                        symbol=symbol,
                        ex_date=ex_date,
                        action_type="split",
                        ratio=split,
                        amount=None,
                        source=self.source,
                    )
                if div > 0.0:
                    yield CorporateAction(
                        symbol=symbol,
                        ex_date=ex_date,
                        action_type="dividend",
                        ratio=None,
                        amount=div,
                        source=self.source,
                    )
