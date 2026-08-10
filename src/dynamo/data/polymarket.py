"""Polymarket adapter — READ-ONLY ``PredictionMarketProvider``. Design: doc 02 §7.

Uses the public Gamma API (``gamma-api.polymarket.com``) which requires no key
for reads. Markets and prices only; no order/CLOB-trading endpoints are wired
anywhere.

Thin-market policy (doc 02 §7): implied probabilities are stored but marked
``trusted=False`` unless ``volumeNum`` ≥ ``pm_min_volume_usd`` and the quoted
``spread`` ≤ ``pm_max_spread``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice

log = structlog.get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _yes_price(market: dict[str, Any]) -> float | None:
    """First outcome price ('Yes' by convention); gamma encodes it as a JSON string."""
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, list) and raw:
        try:
            return float(raw[0])
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class GammaQuote:
    """One ``/markets/{id}`` read: the current YES price + lifecycle status.

    Read-only input for the ADR-0011 shadow-trading sleeve — the simulated
    fill price and the settlement signal both come from here. ``resolved_yes``
    is inferred exactly like ``PolymarketClient._to_market``: a closed market
    quoting >= 0.99 resolved YES, <= 0.01 resolved NO, anything else unknown.
    """

    market_id: str
    yes_price: float | None
    closed: bool
    resolved_yes: bool | None
    ts: datetime


class PolymarketClient:
    """Read-only ``PredictionMarketProvider`` over the Polymarket Gamma API."""

    def __init__(
        self,
        *,
        min_volume_usd: float,
        max_spread: float,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = GAMMA_BASE,
        max_pages: int = 10,
    ) -> None:
        self._min_volume = min_volume_usd
        self._max_spread = max_spread
        self._client = client or httpx.Client(timeout=30.0)
        # ~200 req/day budget (doc 02 §8); 2 req/s is polite for a keyless API.
        self._bucket = bucket or TokenBucket(rate=2.0, capacity=4.0)
        self._base_url = base_url
        self._max_pages = max_pages

    @property
    def venue(self) -> str:
        return "polymarket"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._client,
            f"{self._base_url}{path}",
            provider="polymarket",
            bucket=self._bucket,
            params=params,
        )

    # -- mapping -----------------------------------------------------------

    def _to_market(self, m: dict[str, Any]) -> PredictionMarket | None:
        market_id = m.get("id")
        question = m.get("question")
        if market_id is None or not question:
            return None
        resolved: bool | None = None
        if m.get("closed"):
            price = _yes_price(m)
            if price is not None and price >= 0.99:
                resolved = True
            elif price is not None and price <= 0.01:
                resolved = False
        return PredictionMarket(
            venue=self.venue,
            market_id=str(market_id),
            question=str(question),
            category=m.get("category") or None,
            close_ts=_parse_ts(m.get("endDate")),
            resolved_outcome=resolved,
        )

    def _to_price(self, m: dict[str, Any], ts: datetime) -> PredictionMarketPrice | None:
        market_id = m.get("id")
        implied = _yes_price(m)
        if market_id is None or implied is None:
            return None
        volume = float(m.get("volumeNum") or 0.0)
        spread_raw = m.get("spread")
        spread = float(spread_raw) if spread_raw is not None else 1.0
        return PredictionMarketPrice(
            venue=self.venue,
            market_id=str(market_id),
            ts=ts,
            implied_prob=implied,
            volume=volume,
            liquidity=float(m.get("liquidityNum") or 0.0),
            trusted=volume >= self._min_volume and spread <= self._max_spread,
        )

    # -- PredictionMarketProvider ------------------------------------------

    def markets(self) -> Iterable[PredictionMarket]:
        """Open (not closed) markets snapshot, offset-paginated and capped."""
        page_size = 100
        for page in range(self._max_pages):
            payload = self._get(
                "/markets",
                params={"closed": "false", "limit": page_size, "offset": page * page_size},
            )
            rows = payload or []
            for m in rows:
                market = self._to_market(m)
                if market is not None:
                    yield market
            if len(rows) < page_size:
                return

    def prices(self, market_ids: Sequence[str]) -> Iterator[PredictionMarketPrice]:
        """Per-market snapshot with the thin-market trust flag applied."""
        now = datetime.now(UTC)
        for market_id in market_ids:
            payload = self._get(f"/markets/{market_id}")
            price = self._to_price(payload or {}, now)
            if price is not None:
                yield price

    def quotes(self, market_ids: Sequence[str]) -> Iterator[GammaQuote]:
        """Fresh per-market YES price + closed/resolved status (reads only).

        The ADR-0011 shadow sleeve fills and settles from these; a market the
        API does not know is simply skipped (the sim broker fails closed on
        the missing quote).
        """
        now = datetime.now(UTC)
        for market_id in market_ids:
            payload = self._get(f"/markets/{market_id}")
            m: dict[str, Any] = payload or {}
            if m.get("id") is None:
                continue
            price = _yes_price(m)
            closed = bool(m.get("closed"))
            resolved_yes: bool | None = None
            if closed and price is not None:
                if price >= 0.99:
                    resolved_yes = True
                elif price <= 0.01:
                    resolved_yes = False
            yield GammaQuote(
                market_id=str(m["id"]),
                yes_price=price,
                closed=closed,
                resolved_yes=resolved_yes,
                ts=now,
            )
