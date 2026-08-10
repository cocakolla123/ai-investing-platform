"""Polymarket LIVE implementation of the ``PMBroker`` port (ADR-0012).

Real CLOB order placement — reachable ONLY through the wiring factory
(:func:`orchestration.assets.pm_trading.polymarket_broker_for`), which
requires ``DYNAMO_ENV=LIVE`` + ``polymarket_live_enabled`` + full credentials
+ a passing doc 09 §7 live gate and falls back to the ADR-0011 sim broker on
any missing precondition. Polymarket has NO sandbox: every order this broker
submits is real money (USDC on Polygon).

Transport is the official ``py-clob-client`` (optional ``polymarket`` extra;
this module imports without it and reports ``configured == False``). Requests
are EIP-712 signed with the Polygon wallet private key read from
``polymarket_wallet_key_path`` — the key is loaded lazily at client
construction and never logged, printed, or included in reprs.

Error discipline mirrors :mod:`dynamo.execution.kalshi_broker`: rejects are
terminal and never auto-retried, 429 is a logged incident, and a transient
5xx/transport fault gets exactly one jittered retry before escalating
(``HaltSignal`` for 5xx, ``BrokerUnavailable`` for transport) — fail-closed.

The CLOB has no client-order-id concept, so idempotency is enforced on our
side: an injected ``broker_order_lookup`` (DB truth from ``pm_orders``)
resolves ``client_order_id`` → CLOB order id, and a submit whose id is
already known raises ``DuplicateClientOrderId`` instead of re-ordering.
Belt-and-suspenders beyond risk sizing, ``polymarket_live_max_order_usd``
caps the notional of every single order inside the broker itself.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import structlog

from dynamo.config.settings import Settings, get_settings
from dynamo.domain.models.pm import PMOrder, PMSide
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)

log = structlog.get_logger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137
DATA_API_BASE = "https://data-api.polymarket.com"

REJECT_LIVE_CAP = "order_exceeds_polymarket_live_max_order_usd"
REJECT_NO_TOKEN = "no_clob_token_id_for_market"

# CLOB order statuses → the normalized vocabulary the order manager reads.
_STATUS_MAP = {
    "live": "new",
    "delayed": "new",
    "matched": "filled",
    "unmatched": "canceled",
    "canceled": "canceled",
    "cancelled": "canceled",
}

# market_id -> (yes_token_id, no_token_id), from the Gamma read adapter.
TokenResolver = Callable[[str], tuple[str, str] | None]
# client_order_id -> CLOB order id (DB truth from pm_orders), or None.
BrokerOrderLookup = Callable[[str], str | None]


def _clob_import_error() -> str | None:
    """None when the optional ``polymarket`` extra is importable."""
    try:
        import py_clob_client  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        return str(exc)
    return None


def _order_args(token_id: str, price: float, size: float) -> Any:
    """py-clob-client ``OrderArgs`` (BUY), or a plain mapping for injected
    test clients when the optional extra is absent."""
    try:
        from py_clob_client.clob_types import (  # pyright: ignore[reportMissingImports]
            OrderArgs,
        )
        from py_clob_client.order_builder.constants import (  # pyright: ignore[reportMissingImports]
            BUY,
        )
    except ImportError:
        return {"token_id": token_id, "price": price, "size": size, "side": "BUY"}
    return OrderArgs(token_id=token_id, price=price, size=size, side=BUY)


def _order_type_gtc() -> Any:
    try:
        from py_clob_client.clob_types import (  # pyright: ignore[reportMissingImports]
            OrderType,
        )
    except ImportError:
        return "GTC"
    return OrderType.GTC


def _open_order_params() -> Any:
    try:
        from py_clob_client.clob_types import (  # pyright: ignore[reportMissingImports]
            OpenOrderParams,
        )
    except ImportError:
        return None
    return OpenOrderParams()


def _collateral_params() -> Any:
    try:
        from py_clob_client.clob_types import (  # pyright: ignore[reportMissingImports]
            AssetType,
            BalanceAllowanceParams,
        )
    except ImportError:
        return None
    return BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)


class PolymarketBroker:
    """``PMBroker`` port over the Polymarket CLOB — LIVE ONLY (ADR-0012)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        token_resolver: TokenResolver | None = None,
        broker_order_lookup: BrokerOrderLookup | None = None,
        market_ids_hint: Sequence[str] = (),
        client: Any | None = None,
        positions_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = settings or get_settings()
        self._api_key = cfg.polymarket_api_key
        self._api_secret = cfg.polymarket_api_secret
        self._passphrase = cfg.polymarket_passphrase
        self._wallet_key_path = cfg.polymarket_wallet_key_path
        self._funder = cfg.polymarket_funder_address
        self._signature_type = cfg.polymarket_signature_type
        self._max_order_usd = cfg.polymarket_live_max_order_usd
        self._token_resolver = token_resolver
        self._lookup = broker_order_lookup
        # Markets whose token ids we may need to reverse-map (open positions
        # plus today's decided markets); positions() resolves lazily from it.
        self._market_ids_hint = tuple(market_ids_hint)
        self._client = client  # injected in tests; built lazily otherwise
        self._positions_client = positions_client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        # In-run map: client_order_id -> CLOB order id (backs the DB lookup).
        self._order_ids: dict[str, str] = {}

    def __repr__(self) -> str:  # NEVER include credentials or the wallet key
        return f"PolymarketBroker(configured={self.configured}, live_cap_usd={self._max_order_usd})"

    # --- configuration -----------------------------------------------------

    @property
    def configured(self) -> bool:
        return self.configured_detail() is None

    def configured_detail(self) -> str | None:
        """None when fully configured, else a loggable (secret-free) reason."""
        if self._client is None:
            error = _clob_import_error()
            if error is not None:
                return "py-clob-client not installed (install the 'polymarket' extra)"
        if not (self._api_key and self._api_secret and self._passphrase):
            return "missing CLOB api credentials"
        if not self._wallet_key_path:
            return "polymarket_wallet_key_path unset"
        if self._client is None and not Path(self._wallet_key_path).is_file():
            return "wallet key file missing"
        if self._signature_type in (1, 2) and not self._funder:
            return "polymarket_funder_address required for proxy signature types"
        return None

    def _clob(self) -> Any:
        """The authenticated CLOB client, built lazily (never at import)."""
        if self._client is None:
            detail = self.configured_detail()
            if detail is not None:
                raise BrokerUnavailable(f"polymarket broker not configured: {detail}")
            from py_clob_client.client import (  # pyright: ignore[reportMissingImports]
                ClobClient,
            )
            from py_clob_client.clob_types import (  # pyright: ignore[reportMissingImports]
                ApiCreds,
            )

            # The wallet private key signs EIP-712 orders. Read it here, hand
            # it straight to the client, and never keep another reference.
            wallet_key = Path(self._wallet_key_path).read_text(encoding="utf-8").strip()
            self._client = ClobClient(
                CLOB_HOST,
                chain_id=POLYGON_CHAIN_ID,
                key=wallet_key,
                creds=ApiCreds(
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    api_passphrase=self._passphrase,
                ),
                signature_type=self._signature_type,
                funder=self._funder or None,
            )
        return self._client

    # --- PMBroker port -----------------------------------------------------

    def submit(self, order: PMOrder) -> str:
        """Place one real CLOB limit order; enforces the per-order USD cap."""
        known = self._broker_order_id(order.client_order_id)
        if known:
            # The CLOB has no client_order_id: resubmitting would place a
            # SECOND real order. Surface the duplicate; the manager fetches
            # state instead of resubmitting (doc 09 §8).
            raise DuplicateClientOrderId(order.client_order_id)

        notional_usd = order.contracts * order.limit_price_cents / 100.0
        if notional_usd > self._max_order_usd:
            raise BrokerRejection(
                order.client_order_id,
                f"{REJECT_LIVE_CAP}: ${notional_usd:.2f} > ${self._max_order_usd:.2f}",
            )

        token_id = self._token_id(order)
        client = self._clob()
        args = _order_args(
            token_id=token_id,
            price=order.limit_price_cents / 100.0,  # cents → CLOB 0-1 price
            size=float(order.contracts),  # 1 contract = 1 share paying $1
        )  # the sleeve only ever buys YES/NO tokens (like Kalshi's action=buy)
        signed = self._call(lambda: client.create_order(args), order.client_order_id)
        payload = self._call(
            lambda: client.post_order(signed, _order_type_gtc()), order.client_order_id
        )
        if not isinstance(payload, dict) or not payload.get("success", False):
            reason = ""
            if isinstance(payload, dict):
                reason = str(payload.get("errorMsg") or payload.get("error") or payload)
            raise BrokerRejection(order.client_order_id, f"clob rejected: {reason}")
        broker_order_id = str(payload.get("orderID") or payload.get("orderId") or "")
        self._order_ids[order.client_order_id] = broker_order_id
        log.info(
            "polymarket_live_order_submitted",
            client_order_id=order.client_order_id,
            market_id=order.market_id,
            side=order.side.value,
            contracts=order.contracts,
            limit_price_cents=order.limit_price_cents,
            broker_order_id=broker_order_id,
        )
        return broker_order_id

    def cancel(self, client_order_id: str) -> None:
        broker_id = self._broker_order_id(client_order_id)
        if not broker_id:
            raise BrokerRejection(client_order_id, "unknown order at polymarket")
        client = self._clob()
        self._call(lambda: client.cancel(order_id=broker_id), client_order_id)

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        broker_id = self._broker_order_id(client_order_id)
        if not broker_id:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        client = self._clob()
        payload = self._call(lambda: client.get_order(broker_id), client_order_id)
        if not isinstance(payload, dict) or not payload:
            raise BrokerRejection(client_order_id, "unknown order at polymarket")
        return self._normalize_order(payload, client_order_id=client_order_id)

    def open_orders(self) -> list[dict[str, Any]]:
        client = self._clob()
        payload = self._call(lambda: client.get_orders(_open_order_params()))
        rows = payload if isinstance(payload, list) else []
        return [self._normalize_order(o) for o in rows if isinstance(o, dict)]

    def positions(self) -> list[dict[str, Any]]:
        """Real position truth from the keyless Data API, keyed by funder.

        Token ids reverse-map to our market ids through the Gamma resolver;
        a position in a token we cannot map is surfaced as
        ``token:<id>`` so reconciliation flags it (fail-closed) instead of
        silently dropping it.
        """
        reverse = self._reverse_token_map()

        def fetch() -> Any:
            response = self._positions_client.get(
                f"{DATA_API_BASE}/positions", params={"user": self._funder}
            )
            response.raise_for_status()
            return response.json()

        rows = self._call(fetch)
        out: list[dict[str, Any]] = []
        for p in rows if isinstance(rows, list) else []:
            size = float(p.get("size") or 0.0)
            if size <= 0:
                continue
            token = str(p.get("asset", ""))
            mapped = reverse.get(token)
            if mapped is not None:
                market_id, side = mapped
            else:
                market_id = f"token:{token}"
                side = (
                    PMSide.NO.value
                    if str(p.get("outcome", "")).lower() == "no"
                    else PMSide.YES.value
                )
            avg_price = float(p.get("avgPrice") or 0.0)  # 0-1
            out.append(
                {
                    "market_id": market_id,
                    "side": side,
                    "contracts": round(size),
                    "exposure_cents": size * avg_price * 100.0,
                }
            )
        return out

    def balance(self) -> dict[str, Any]:
        """Available USDC collateral, in cents (best effort)."""
        client = self._clob()
        payload = self._call(lambda: client.get_balance_allowance(_collateral_params()))
        raw = payload.get("balance", 0) if isinstance(payload, dict) else 0
        # USDC has 6 decimals: base units / 1e6 = dollars; * 100 = cents.
        return {"balance_cents": float(raw) / 1e6 * 100.0}

    # --- error discipline (doc 09 §8, mirroring kalshi_broker) -------------

    def _call(self, fn: Callable[[], Any], client_order_id: str = "") -> Any:
        """One CLOB call: rejects terminal, one jittered retry on transient."""
        retried = False
        while True:
            try:
                return fn()
            except Exception as exc:  # PolyApiException or transport error
                status = getattr(exc, "status_code", None)
                if status == 429:
                    raise RateLimited(f"polymarket rate limit: {exc}") from exc
                if status is not None and 400 <= int(status) < 500:
                    raise BrokerRejection(client_order_id, f"{status}: {exc}") from exc
                if not retried:
                    retried = True
                    self._sleep(0.5 + random.random())
                    continue
                if status is not None and int(status) >= 500:
                    raise HaltSignal(f"polymarket unavailable after retry: {exc}") from exc
                raise BrokerUnavailable(
                    f"polymarket connection failed after retry: {exc}"
                ) from exc

    # --- internals ---------------------------------------------------------

    def _broker_order_id(self, client_order_id: str) -> str | None:
        known = self._order_ids.get(client_order_id)
        if known:
            return known
        if self._lookup is not None:
            from_store = self._lookup(client_order_id)
            if from_store:
                self._order_ids[client_order_id] = from_store
                return from_store
        return None

    def _token_id(self, order: PMOrder) -> str:
        if self._token_resolver is None:
            raise BrokerRejection(order.client_order_id, REJECT_NO_TOKEN)
        tokens = self._token_resolver(order.market_id)
        if tokens is None:
            raise BrokerRejection(order.client_order_id, REJECT_NO_TOKEN)
        yes_token, no_token = tokens
        return yes_token if order.side is PMSide.YES else no_token

    def _reverse_token_map(self) -> dict[str, tuple[str, str]]:
        """token_id -> (market_id, side) over the hinted markets."""
        out: dict[str, tuple[str, str]] = {}
        if self._token_resolver is None:
            return out
        for market_id in self._market_ids_hint:
            tokens = self._token_resolver(market_id)
            if tokens is None:
                continue
            yes_token, no_token = tokens
            out[yes_token] = (market_id, PMSide.YES.value)
            out[no_token] = (market_id, PMSide.NO.value)
        return out

    def _normalize_order(
        self, d: dict[str, Any], client_order_id: str = ""
    ) -> dict[str, Any]:
        original = int(float(d.get("original_size") or d.get("size") or 0))
        matched = int(float(d.get("size_matched") or 0))
        status = str(d.get("status", "")).lower()
        normalized = _STATUS_MAP.get(status, status)
        if normalized == "new" and matched >= original and original > 0:
            normalized = "filled"
        price = d.get("price")
        return {
            "client_order_id": client_order_id,
            "broker_order_id": str(d.get("id", "")),
            "market_id": str(d.get("market", "")),  # condition id, informational
            "side": str(d.get("side", "")).lower(),
            "contracts": original,
            "filled_contracts": matched,
            # CLOB prices are 0-1; the order manager speaks cents.
            "filled_avg_price_cents": float(price) * 100.0 if price is not None else None,
            "status": normalized,
        }
