"""Polymarket adapter — READ-ONLY ``PredictionMarketProvider``. Design: doc 02 §7.

Uses the public Gamma API (``gamma-api.polymarket.com``) which requires no key
for reads. Markets, prices, and CLOB token-id lookups only; order placement
lives solely in :mod:`dynamo.execution.polymarket_broker` (ADR-0012), behind
the live gate — never here.

Thin-market policy (doc 02 §7): implied probabilities are stored but marked
``trusted=False`` unless ``volumeNum`` ≥ ``pm_min_volume_usd`` and the quoted
``spread`` ≤ ``pm_max_spread``.
"""

from __future__ import annotations

import json
import math
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
class PolymarketResolution:
    """One RESOLVED Gamma market: the outcome plus a derived pre-final price.

    Feedstock for the favorite-longshot calibration study
    (``dynamo.cli.pm_calibration``). Gamma's closed-market payload carries no
    stored pre-settlement snapshot — ``outcomePrices`` is the final 0/1 — so
    ``prior_price`` is DERIVED as ``final_price - oneDayPriceChange``, i.e.
    the YES price roughly 24 hours before the final tick. Two caveats the
    caller must surface: Gamma does not document whether that change field
    freezes at close, and a price one day from resolution has already
    converged toward the outcome. None when the field is absent or the
    derived value falls outside (0, 1).
    """

    market_id: str
    question: str
    resolved_yes: bool
    final_price: float  # outcomePrices[0] at resolution (0/1 by then)
    prior_price: float | None  # final_price - oneDayPriceChange, in (0, 1)
    volume: float


@dataclass(frozen=True)
class GammaEvent:
    """One Gamma /events row: an event grouping its markets with YES prices.

    Dutch-book feedstock (:func:`dynamo.risk.pm_arbitrage.find_dutch_books`
    with a Polymarket event key). ``neg_risk`` marks events whose outcomes
    are mutually exclusive BY CONSTRUCTION (Polymarket's negative-risk
    conversion applies); only those are safe to dutch-book. A nested market's
    price is None when the market is closed or unpriceable, so a stale leg
    poisons its book fail-closed downstream instead of pricing it.
    """

    event_id: str
    title: str
    neg_risk: bool
    markets: tuple[tuple[PredictionMarket, PredictionMarketPrice | None], ...]


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

    def open_markets_with_yes(
        self, limit: int = 500
    ) -> list[tuple[PredictionMarket, float | None]]:
        """ONE page of open markets with their current YES price (keyless read).

        Feedstock for the cross-venue title matcher
        (:mod:`dynamo.research.pm_matching`): a single ``/markets`` request —
        deliberately not paginated, so a matcher run costs the Gamma API one
        call — returning each market alongside the YES price already present
        in the listing payload (no per-market follow-up reads).
        """
        payload = self._get("/markets", params={"closed": "false", "limit": limit, "offset": 0})
        out: list[tuple[PredictionMarket, float | None]] = []
        for m in payload or []:
            market = self._to_market(m)
            if market is not None:
                out.append((market, _yes_price(m)))
        return out

    def open_events(self, limit: int = 200) -> list[GammaEvent]:
        """ONE page of open events with nested markets + YES prices (keyless read).

        Dutch-book feedstock: a single ``/events`` request — deliberately not
        paginated, mirroring :meth:`open_markets_with_yes`, so a scan costs the
        Gamma API one call — returning each event with its nested markets and
        their :class:`PredictionMarketPrice` (thin-market trust flag applied
        from the nested payload's ``volumeNum``/``spread``). Closed nested
        markets keep their :class:`PredictionMarket` row but carry price None,
        so downstream book-building fails closed on the stale leg. Malformed
        events and markets are skipped, never guessed at.
        """
        payload = self._get("/events", params={"closed": "false", "limit": limit, "offset": 0})
        now = datetime.now(UTC)
        events: list[GammaEvent] = []
        for e in payload or []:
            if not isinstance(e, dict):
                continue
            event_id = e.get("id")
            title = e.get("title")
            raw_markets = e.get("markets")
            if event_id is None or not title or not isinstance(raw_markets, list):
                continue
            rows: list[tuple[PredictionMarket, PredictionMarketPrice | None]] = []
            for m in raw_markets:
                if not isinstance(m, dict):
                    continue
                market = self._to_market(m)
                if market is None:
                    continue
                price = None if m.get("closed") else self._to_price(m, now)
                rows.append((market, price))
            events.append(
                GammaEvent(
                    event_id=str(event_id),
                    title=str(title),
                    neg_risk=bool(e.get("negRisk")),
                    markets=tuple(rows),
                )
            )
        return events

    @staticmethod
    def _to_resolution(m: dict[str, Any]) -> PolymarketResolution | None:
        """Defensive mapping of one closed-market payload; None when unusable."""
        market_id = m.get("id")
        question = m.get("question")
        if market_id is None or not question or not m.get("closed"):
            return None
        status = m.get("umaResolutionStatus")
        if status is not None and status != "resolved":
            return None  # disputed/unresolved closes are not outcomes
        final = _yes_price(m)
        if final is None:
            return None
        if final >= 0.99:
            resolved_yes = True
        elif final <= 0.01:
            resolved_yes = False
        else:
            return None  # closed but not clearly resolved — not calibration data
        prior: float | None = None
        change = m.get("oneDayPriceChange")
        if isinstance(change, int | float) and math.isfinite(float(change)):
            derived = final - float(change)
            if 0.0 < derived < 1.0:
                prior = derived
        return PolymarketResolution(
            market_id=str(market_id),
            question=str(question),
            resolved_yes=resolved_yes,
            final_price=final,
            prior_price=prior,
            volume=float(m.get("volumeNum") or 0.0),
        )

    def resolved_markets(self, *, max_pages: int | None = None) -> Iterator[PolymarketResolution]:
        """RESOLVED markets via ``/markets?closed=true``, offset-paginated.

        Read-only calibration feedstock; capped at ``max_pages`` (default: the
        client-wide cap) of 100 rows each. Rows that are not clearly resolved
        — disputed UMA status, a final price away from 0/1 — are dropped, not
        guessed at. See :class:`PolymarketResolution` for the derived-price
        caveats the caller must print.
        """
        pages = max_pages if max_pages is not None else self._max_pages
        page_size = 100
        for page in range(pages):
            payload = self._get(
                "/markets",
                params={"closed": "true", "limit": page_size, "offset": page * page_size},
            )
            rows = payload or []
            for m in rows:
                if not isinstance(m, dict):
                    continue
                resolution = self._to_resolution(m)
                if resolution is not None:
                    yield resolution
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

    def clob_token_ids(self, market_id: str) -> tuple[str, str] | None:
        """(yes_token_id, no_token_id) for one market, from Gamma (keyless read).

        Gamma encodes ``clobTokenIds`` as a JSON string in outcome order
        ('Yes' first by convention, matching :func:`_yes_price`). Returns None
        when the market is unknown or the field is missing/malformed — the
        live broker (ADR-0012) rejects the order fail-closed in that case.
        """
        payload = self._get(f"/markets/{market_id}")
        m: dict[str, Any] = payload or {}
        raw = m.get("clobTokenIds")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
        if isinstance(raw, list) and len(raw) >= 2 and all(raw[:2]):
            return str(raw[0]), str(raw[1])
        return None

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
