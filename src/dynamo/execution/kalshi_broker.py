"""Kalshi implementation of the ``PMBroker`` port (ADR-0009).

Demo vs production is decided solely by ``DYNAMO_ENV``: anything other than
LIVE talks to the demo exchange (``kalshi_demo_base_url``); production is only
reachable in LIVE — and every submission still passes the doc 09 §7 live gate
before it gets here. Requests are RSA-PSS signed exactly like the read-only
adapter (:mod:`dynamo.data.kalshi`); the signer is injectable so tests run
without the optional ``cryptography`` dependency.

Error discipline mirrors :mod:`dynamo.execution.alpaca_broker`: rejects (4xx)
are terminal and never auto-retried, duplicate ``client_order_id`` surfaces as
``DuplicateClientOrderId`` (fetch state, do not resubmit), 429 is a logged
incident, and a transient 5xx/transport fault gets exactly one jittered retry
before escalating to ``HaltSignal`` (fail-closed).
"""

from __future__ import annotations

import base64
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx
import structlog

from dynamo.config.settings import DynamoEnv, Settings, get_settings
from dynamo.domain.models.pm import PMOrder, PMSide
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)

log = structlog.get_logger(__name__)

API_PREFIX = "/trade-api/v2"
_DUPLICATE_MARKERS = ("duplicate", "already exists")

# Kalshi order statuses → the normalized vocabulary the order manager reads.
_STATUS_MAP = {
    "resting": "new",
    "pending": "new",
    "executed": "filled",
    "canceled": "canceled",
    "cancelled": "canceled",
    "expired": "expired",
    "rejected": "rejected",
}


class RequestSigner(Protocol):
    """Signs ``{timestamp_ms}{METHOD}{path}``; injectable for tests."""

    def sign(self, message: bytes) -> bytes: ...


def load_rsa_signer(private_key_path: str) -> RequestSigner | None:
    """RSA-PSS signer from a PEM file, or None → unconfigured (skip gracefully)."""
    if not private_key_path:
        return None
    try:
        from cryptography.hazmat.primitives import (  # pyright: ignore[reportMissingImports]
            hashes,
            serialization,
        )
        from cryptography.hazmat.primitives.asymmetric import (  # pyright: ignore[reportMissingImports]
            padding,
        )
    except ImportError:
        log.warning("kalshi_broker_cryptography_missing")
        return None
    try:
        pem = Path(private_key_path).read_bytes()
        key = serialization.load_pem_private_key(pem, password=None)
    except (OSError, ValueError) as exc:
        log.warning("kalshi_broker_key_load_failed", error=str(exc))
        return None

    class _RsaPss:
        def sign(self, message: bytes) -> bytes:
            return key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256(),
            )

    return _RsaPss()


class KalshiBroker:
    """``PMBroker`` port over the Kalshi trade API (demo by default)."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        signer: RequestSigner | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = settings or get_settings()
        self._api_key_id = cfg.kalshi_api_key_id
        self._signer = signer if signer is not None else load_rsa_signer(
            cfg.kalshi_private_key_path
        )
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        # Production ONLY in LIVE; every other environment gets the demo exchange.
        self._base_url = (
            cfg.kalshi_prod_base_url
            if cfg.dynamo_env == DynamoEnv.LIVE
            else cfg.kalshi_demo_base_url
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def configured(self) -> bool:
        return bool(self._api_key_id) and self._signer is not None

    # --- PMBroker port ---

    def submit(self, order: PMOrder) -> str:
        """Submit a limit order; ``client_order_id`` is the idempotency key."""
        body: dict[str, Any] = {
            "ticker": order.market_id,
            "client_order_id": order.client_order_id,
            "action": "buy",
            "side": order.side.value,
            "count": order.contracts,
            "type": "limit",
        }
        if order.side is PMSide.YES:
            body["yes_price"] = order.limit_price_cents
        else:
            body["no_price"] = order.limit_price_cents
        payload = self._request(
            "POST", "/portfolio/orders", body=body, client_order_id=order.client_order_id
        )
        return str((payload.get("order") or {}).get("order_id", ""))

    def cancel(self, client_order_id: str) -> None:
        remote = self.get_order(client_order_id)
        broker_id = str(remote.get("broker_order_id", ""))
        if not broker_id:
            raise BrokerRejection(client_order_id, "unknown order at kalshi")
        self._request(
            "DELETE", f"/portfolio/orders/{broker_id}", client_order_id=client_order_id
        )

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/portfolio/orders",
            params={"client_order_id": client_order_id},
            client_order_id=client_order_id,
        )
        orders = payload.get("orders") or []
        if not orders:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        return self._normalize_order(orders[0])

    def open_orders(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/portfolio/orders", params={"status": "resting"})
        return [self._normalize_order(o) for o in payload.get("orders") or []]

    def positions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/portfolio/positions")
        out: list[dict[str, Any]] = []
        for p in payload.get("market_positions") or []:
            signed = int(p.get("position") or 0)
            if signed == 0:
                continue
            out.append(
                {
                    "market_id": str(p.get("ticker", "")),
                    "side": PMSide.YES.value if signed > 0 else PMSide.NO.value,
                    "contracts": abs(signed),
                    "exposure_cents": float(p.get("market_exposure") or 0.0),
                }
            )
        return out

    def balance(self) -> dict[str, Any]:
        payload = self._request("GET", "/portfolio/balance")
        return {"balance_cents": float(payload.get("balance") or 0.0)}

    # --- signing (same scheme as data/kalshi.py, method-aware) ---

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if self._signer is None:  # pragma: no cover - guarded by ``configured``
            raise BrokerUnavailable("kalshi broker not configured (no signer)")
        timestamp_ms = str(int(time.time() * 1000))
        signature = self._signer.sign(f"{timestamp_ms}{method}{path}".encode())
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    # --- error discipline (doc 09 §8) ---

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        client_order_id: str = "",
    ) -> dict[str, Any]:
        full_path = f"{API_PREFIX}{path}"
        url = f"{self._base_url}{full_path}"
        retried = False
        while True:
            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    content=json.dumps(body) if body is not None else None,
                    headers={
                        **self._auth_headers(method, full_path),
                        "Content-Type": "application/json",
                    },
                )
            except httpx.HTTPError as exc:
                if not retried:
                    retried = True
                    self._sleep(0.5 + random.random())
                    continue
                raise BrokerUnavailable(f"kalshi connection failed after retry: {exc}") from exc

            status = response.status_code
            if status == 429:
                raise RateLimited(f"kalshi rate limit: {response.text}")
            if 400 <= status < 500:
                message = response.text
                if any(m in message.lower() for m in _DUPLICATE_MARKERS):
                    raise DuplicateClientOrderId(client_order_id)
                raise BrokerRejection(client_order_id, f"{status}: {message}")
            if status >= 500:
                if not retried:
                    retried = True
                    self._sleep(0.5 + random.random())
                    continue
                raise HaltSignal(f"kalshi unavailable after retry: {status} {response.text}")
            if not response.content:
                return {}
            payload = response.json()
            return payload if isinstance(payload, dict) else {}

    # --- shaping ---

    @staticmethod
    def _normalize_order(d: dict[str, Any]) -> dict[str, Any]:
        initial = int(d.get("initial_count") or d.get("count") or 0)
        remaining = int(d.get("remaining_count") or 0)
        filled = int(d.get("filled_count") or max(initial - remaining, 0))
        status = str(d.get("status", ""))
        avg_price = d.get("filled_avg_price") or d.get("yes_price") or d.get("no_price")
        return {
            "client_order_id": str(d.get("client_order_id", "")),
            "broker_order_id": str(d.get("order_id", "")),
            "market_id": str(d.get("ticker", "")),
            "side": str(d.get("side", "")),
            "contracts": initial,
            "filled_contracts": filled,
            "filled_avg_price_cents": float(avg_price) if avg_price is not None else None,
            "status": _STATUS_MAP.get(status, status),
        }
