"""Kalshi adapter — READ-ONLY ``PredictionMarketProvider``. Design: doc 02 §7.

Markets and prices only; **no order endpoints are wired anywhere** (ToS-compliant
research use). Authentication uses Kalshi's RSA-PSS request signing when an API
key id + private key path are configured; when they are not (or the optional
``cryptography`` dependency is absent) the adapter *skips gracefully* — it
returns no data and logs, so the nightly snapshot degrades per-asset instead of
failing (doc 01 §6).

Thin-market policy (doc 02 §7): implied probabilities are stored but marked
``trusted=False`` unless volume ≥ ``pm_min_volume_usd`` and the yes bid/ask
spread ≤ ``pm_max_spread``. Kalshi volume is reported in contracts (each pays
out $1), so contract volume is used as the dollar-volume proxy — a documented
upper bound.
"""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog

from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice

log = structlog.get_logger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _dollar_price(m: dict[str, Any], cents_key: str, dollars_key: str) -> float | None:
    """A price in dollars from either payload schema; None when absent/malformed.

    Kalshi serves two market schemas: the classic one with integer-cent fields
    (``last_price``) and a newer one with decimal-string ``*_dollars`` fields
    (``last_price_dollars``) — settled listings have been observed serving
    ONLY the ``*_dollars`` form. Prefer the cent field when present, fall back
    to the dollars field, and return None (never a guess) for anything
    missing, malformed, or non-finite.
    """
    raw = m.get(cents_key)
    if raw is not None:
        try:
            value = float(raw) / 100.0
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            return value
    raw = m.get(dollars_key)
    if raw is not None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isfinite(value):
            return value
    return None


@dataclass(frozen=True)
class KalshiSettlement:
    """One SETTLED Kalshi market: the resolved outcome plus its final prices.

    Feedstock for the favorite-longshot calibration study
    (``dynamo.cli.pm_calibration``). ``last_price`` is the last trade before
    settlement (heavily converged toward the outcome); ``previous_price`` is
    Kalshi's previous-period reference price — further from settlement, so
    the study prefers it. Either can be None when the payload lacks a usable
    value; the CALLER decides what a usable study price is.
    """

    market_id: str
    result_yes: bool
    last_price: float | None  # last trade before settlement, dollars in [0, 1]
    previous_price: float | None  # previous-period reference price, dollars in [0, 1]
    volume: float  # contracts traded (≈ $ upper bound, as elsewhere)
    settled_ts: datetime | None


class KalshiClient:
    """Read-only ``PredictionMarketProvider`` over the Kalshi public REST API."""

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        *,
        min_volume_usd: float,
        max_spread: float,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = KALSHI_BASE,
        max_pages: int = 10,
    ) -> None:
        self._api_key_id = api_key_id
        self._private_key_path = private_key_path
        self._min_volume = min_volume_usd
        self._max_spread = max_spread
        self._client = client or httpx.Client(timeout=30.0)
        # ~200 req/day budget (doc 02 §8); 2 req/s is far below Kalshi limits.
        self._bucket = bucket or TokenBucket(rate=2.0, capacity=4.0)
        self._base_url = base_url
        self._max_pages = max_pages
        self._signer = self._load_signer()

    @property
    def venue(self) -> str:
        return "kalshi"

    @property
    def configured(self) -> bool:
        """True when key auth is available; otherwise the adapter is a no-op."""
        return self._signer is not None

    # -- auth --------------------------------------------------------------

    def _load_signer(self) -> Any | None:
        """Load the RSA private key, or None → skip gracefully (doc 02 §7)."""
        if not self._api_key_id or not self._private_key_path:
            log.info("kalshi_not_configured_skipping")
            return None
        try:
            from cryptography.hazmat.primitives import (  # pyright: ignore[reportMissingImports]
                serialization,
            )
        except ImportError:
            log.warning("kalshi_cryptography_missing_skipping")
            return None
        try:
            pem = Path(self._private_key_path).read_bytes()
            return serialization.load_pem_private_key(pem, password=None)
        except (OSError, ValueError) as exc:
            log.warning("kalshi_key_load_failed_skipping", error=str(exc))
            return None

    def _auth_headers(self, path: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import (  # pyright: ignore[reportMissingImports]
            hashes,
        )
        from cryptography.hazmat.primitives.asymmetric import (  # pyright: ignore[reportMissingImports]
            padding,
        )

        assert self._signer is not None
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}GET{path}".encode()
        signature = self._signer.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._client,
            f"{self._base_url}{API_PREFIX}{path}",
            provider="kalshi",
            bucket=self._bucket,
            params=params,
            headers=self._auth_headers(f"{API_PREFIX}{path}"),
        )

    # -- mapping -----------------------------------------------------------

    @staticmethod
    def _resolved(market: dict[str, Any]) -> bool | None:
        result = market.get("result")
        if result == "yes":
            return True
        if result == "no":
            return False
        return None

    def _to_market(self, m: dict[str, Any]) -> PredictionMarket | None:
        ticker = m.get("ticker")
        title = m.get("title")
        if not ticker or not title:
            return None
        return PredictionMarket(
            venue=self.venue,
            market_id=str(ticker),
            question=str(title),
            category=m.get("category") or None,
            close_ts=_parse_ts(m.get("close_time")),
            resolved_outcome=self._resolved(m),
        )

    def _to_price(self, m: dict[str, Any], ts: datetime) -> PredictionMarketPrice | None:
        ticker = m.get("ticker")
        if not ticker:
            return None
        yes_bid = float(m.get("yes_bid") or 0.0) / 100.0
        yes_ask = float(m.get("yes_ask") or 0.0) / 100.0
        last = float(m.get("last_price") or 0.0) / 100.0
        implied = last if last > 0 else (yes_bid + yes_ask) / 2.0
        volume = float(m.get("volume") or 0.0)  # contracts ≈ $ upper bound
        spread = (yes_ask - yes_bid) if yes_ask > 0 else 1.0
        return PredictionMarketPrice(
            venue=self.venue,
            market_id=str(ticker),
            ts=ts,
            implied_prob=implied,
            volume=volume,
            liquidity=float(m.get("liquidity") or 0.0) / 100.0,
            trusted=volume >= self._min_volume and spread <= self._max_spread,
        )

    # -- PredictionMarketProvider ------------------------------------------

    def markets(self) -> Iterable[PredictionMarket]:
        """Open markets snapshot (cursor-paginated, capped); empty if unconfigured."""
        if not self.configured:
            return
        cursor: str | None = None
        for _ in range(self._max_pages):
            params: dict[str, Any] = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/markets", params=params)
            for m in payload.get("markets", []):
                market = self._to_market(m)
                if market is not None:
                    yield market
            cursor = payload.get("cursor")
            if not cursor:
                return

    @staticmethod
    def _to_settlement(m: dict[str, Any]) -> KalshiSettlement | None:
        """Defensive mapping of one settled-market payload; None when unusable."""
        ticker = m.get("ticker")
        result = m.get("result")
        if not ticker or result not in ("yes", "no"):
            return None  # void/scalar/unsettled results are not YES/NO outcomes
        volume = 0.0
        for key in ("volume", "volume_fp"):  # integer-cent schema, then decimal-string
            raw = m.get(key)
            if raw is None:
                continue
            try:
                volume = float(raw)
                break
            except (TypeError, ValueError):
                continue
        return KalshiSettlement(
            market_id=str(ticker),
            result_yes=result == "yes",
            last_price=_dollar_price(m, "last_price", "last_price_dollars"),
            previous_price=_dollar_price(m, "previous_price", "previous_price_dollars"),
            volume=volume if math.isfinite(volume) else 0.0,
            settled_ts=_parse_ts(m.get("settlement_ts")),
        )

    def settled_markets(self, *, max_pages: int | None = None) -> Iterator[KalshiSettlement]:
        """SETTLED markets, newest first (``GET /markets?status=settled``).

        Read-only calibration feedstock; empty when unconfigured, exactly like
        :meth:`markets`. Cursor-paginated and capped at ``max_pages`` (default:
        the client-wide cap) — 200 markets a page, so the default cap costs at
        most ``max_pages`` requests through the shared token bucket. Rows the
        payload cannot support (no ticker, result not yes/no) are dropped, not
        guessed at.
        """
        if not self.configured:
            return
        pages = max_pages if max_pages is not None else self._max_pages
        cursor: str | None = None
        for _ in range(pages):
            params: dict[str, Any] = {"limit": 200, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/markets", params=params)
            for m in payload.get("markets", []):
                settlement = self._to_settlement(m)
                if settlement is not None:
                    yield settlement
            cursor = payload.get("cursor")
            if not cursor:
                return

    def prices(self, market_ids: Sequence[str]) -> Iterator[PredictionMarketPrice]:
        """Per-market price snapshot with the thin-market trust flag applied."""
        if not self.configured:
            return
        now = datetime.now(UTC)
        for market_id in market_ids:
            payload = self._get(f"/markets/{market_id}")
            price = self._to_price(payload.get("market", {}), now)
            if price is not None:
                yield price
