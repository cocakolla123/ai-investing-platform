"""Alpaca implementation of the ``Broker`` port (doc 09 §1, §3, §8).

Paper vs live is decided solely by ``settings.alpaca_base_url`` — the code
path is identical. US equities/ETFs plus Alpaca crypto pairs (ADR-0013);
typed errors; rejects are never auto-retried; transient 5xx gets exactly one
jittered retry then ``HaltSignal``.

Crypto orders (ADR-0013) go through the SAME request classes with three
differences, all handled in ``_build_request``:

- ``time_in_force=GTC`` always — Alpaca rejects DAY/OPG for crypto;
- no opening auction exists, so ``MARKET_ON_OPEN`` maps to a plain GTC
  market order (24/7 market: it fills immediately);
- plain stop orders are not supported for crypto — ``STOP`` maps to a GTC
  stop-limit with the limit 1% through the stop (marketable once triggered).

Fractional quantities are native for crypto and PDT does not apply (the PDT
guard excludes crypto at the call site, orchestration/assets/portfolio_exec).
Symbols use the canonical "BTC/USD" pair form on the way in; position and
order payloads coming back may use the slashless "BTCUSD" form and are
normalized at this edge (``dynamo.domain.symbols``).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)

from dynamo.config.settings import Settings, get_settings
from dynamo.domain.models.portfolio import Order, OrderSide, OrderType
from dynamo.domain.symbols import is_crypto_symbol, normalize_crypto_symbol
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)

_DUPLICATE_MARKERS = ("client_order_id must be unique", "duplicate client_order_id")


def _status_code(exc: APIError) -> int:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return 0


class AlpacaBroker:
    """``Broker`` port over ``alpaca-py`` ``TradingClient``."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: TradingClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = settings or get_settings()
        self._sleep = sleep
        if client is not None:
            self._client = client
        else:
            paper = "paper" in cfg.alpaca_base_url
            self._client = TradingClient(
                api_key=cfg.alpaca_api_key,
                secret_key=cfg.alpaca_api_secret,
                paper=paper,
                url_override=cfg.alpaca_base_url or None,
                raw_data=True,
            )

    # --- Broker port ---

    def submit(self, order: Order) -> str:
        request = self._build_request(order)
        raw = self._call(lambda: self._client.submit_order(order_data=request),
                         client_order_id=order.client_order_id)
        return str(self._as_dict(raw).get("id", ""))

    def cancel(self, client_order_id: str) -> None:
        raw = self._call(lambda: self._client.get_order_by_client_id(client_order_id),
                         client_order_id=client_order_id)
        broker_id = str(self._as_dict(raw).get("id", ""))
        self._call(lambda: self._client.cancel_order_by_id(broker_id),
                   client_order_id=client_order_id)

    def replace(self, client_order_id: str, stop_price: float) -> str:
        raw = self._call(lambda: self._client.get_order_by_client_id(client_order_id),
                         client_order_id=client_order_id)
        broker_id = str(self._as_dict(raw).get("id", ""))
        req = ReplaceOrderRequest(stop_price=round(stop_price, 2))
        replaced = self._call(
            lambda: self._client.replace_order_by_id(broker_id, order_data=req),
            client_order_id=client_order_id,
        )
        return str(self._as_dict(replaced).get("id", ""))

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        raw = self._call(lambda: self._client.get_order_by_client_id(client_order_id),
                         client_order_id=client_order_id)
        return self._normalize_order(self._as_dict(raw))

    def open_orders(self) -> list[dict[str, Any]]:
        raw = self._call(self._client.get_orders)
        items = raw if isinstance(raw, list) else []
        return [self._normalize_order(self._as_dict(o)) for o in items]

    def positions(self) -> list[dict[str, Any]]:
        raw = self._call(self._client.get_all_positions)
        items = raw if isinstance(raw, list) else []
        out: list[dict[str, Any]] = []
        for p in items:
            d = self._as_dict(p)
            out.append(
                {
                    # Crypto positions come back slashless ("BTCUSD") —
                    # normalize to the canonical "BTC/USD" so reconciliation
                    # matches internal state (ADR-0013).
                    "symbol": self._canonical_symbol(d),
                    "qty": float(d.get("qty", 0.0)),
                    "avg_entry_price": float(d.get("avg_entry_price", 0.0)),
                }
            )
        return out

    def account(self) -> dict[str, Any]:
        d = self._as_dict(self._call(self._client.get_account))
        return {
            "cash": float(d.get("cash", 0.0)),
            "equity": float(d.get("equity", 0.0)),
            "buying_power": float(d.get("buying_power", 0.0)),
        }

    # --- request mapping (doc 09 §3) ---

    # Crypto stop-limit band (module docstring): limit 1% through the stop so
    # the triggered order is marketable but never fills at an absurd print.
    _CRYPTO_STOP_LIMIT_BAND = 0.01

    def _build_request(self, order: Order) -> Any:
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        crypto = is_crypto_symbol(order.symbol)
        # Alpaca crypto rejects DAY/OPG; GTC is the only tif the sleeve uses.
        day_tif = TimeInForce.GTC if crypto else TimeInForce.DAY
        common: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": order.qty,
            "side": side,
            "client_order_id": order.client_order_id,
        }
        if order.order_type == OrderType.MARKET_ON_OPEN:
            if crypto:
                # 24/7 market — there is no opening auction; a plain GTC
                # market order fills immediately (module docstring).
                return MarketOrderRequest(time_in_force=TimeInForce.GTC, **common)
            # Opening auction: market order with OPG time-in-force.
            return MarketOrderRequest(time_in_force=TimeInForce.OPG, **common)
        if order.order_type in (OrderType.MARKETABLE_LIMIT, OrderType.SYNTHETIC_STOP_EXIT):
            # Marketable limit applies to crypto too (spreads exist there as
            # well) — only the time-in-force differs (GTC).
            if order.limit_price is None:
                raise BrokerRejection(order.client_order_id, "marketable limit needs limit_price")
            return LimitOrderRequest(
                time_in_force=day_tif,
                limit_price=round(order.limit_price, 2),
                **common,
            )
        if order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise BrokerRejection(order.client_order_id, "stop order needs stop_price")
            stop_price = round(order.stop_price, 2)
            if crypto:
                # Alpaca crypto supports market/limit/stop-limit only — plain
                # stops are rejected. Map to a stop-limit 1% through the stop.
                band = self._CRYPTO_STOP_LIMIT_BAND
                factor = (1.0 - band) if side is AlpacaSide.SELL else (1.0 + band)
                return StopLimitOrderRequest(
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                    limit_price=round(order.stop_price * factor, 2),
                    **common,
                )
            return StopOrderRequest(
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price,
                **common,
            )
        raise BrokerRejection(order.client_order_id, f"unsupported order type {order.order_type}")

    # --- error discipline (doc 09 §8) ---

    def _call(self, fn: Callable[[], Any], client_order_id: str = "") -> Any:
        """Single jittered retry on transient 5xx, then ``HaltSignal``.

        Rejects (4xx) are terminal and never retried; duplicates surface as
        ``DuplicateClientOrderId`` so the order manager fetches state instead.
        """
        retried = False
        while True:
            try:
                return fn()
            except APIError as exc:
                status = _status_code(exc)
                message = str(exc)
                if status == 429:
                    raise RateLimited(f"alpaca rate limit: {message}") from exc
                if 400 <= status < 500:
                    if any(m in message.lower() for m in _DUPLICATE_MARKERS):
                        raise DuplicateClientOrderId(client_order_id) from exc
                    raise BrokerRejection(client_order_id, message) from exc
                if not retried:
                    retried = True
                    self._sleep(0.5 + random.random())
                    continue
                raise HaltSignal(f"alpaca unavailable after retry: {message}") from exc
            except (ConnectionError, TimeoutError) as exc:
                if not retried:
                    retried = True
                    self._sleep(0.5 + random.random())
                    continue
                raise BrokerUnavailable(f"alpaca connection failed after retry: {exc}") from exc

    # --- shaping ---

    @staticmethod
    def _as_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        dump = getattr(raw, "model_dump", None)
        if callable(dump):
            result = dump()
            if isinstance(result, dict):
                return result
        return {}

    @staticmethod
    def _canonical_symbol(d: dict[str, Any]) -> str:
        """Payload symbol in canonical form: crypto rows (``asset_class ==
        "crypto"``) may arrive slashless and are normalized; everything else
        passes through untouched (ADR-0013)."""
        symbol = str(d.get("symbol", ""))
        if str(d.get("asset_class", "")) == "crypto":
            return normalize_crypto_symbol(symbol)
        return symbol

    @classmethod
    def _normalize_order(cls, d: dict[str, Any]) -> dict[str, Any]:
        filled_avg = d.get("filled_avg_price")
        return {
            "client_order_id": str(d.get("client_order_id", "")),
            "broker_order_id": str(d.get("id", "")),
            "symbol": cls._canonical_symbol(d),
            "side": str(d.get("side", "")),
            "qty": float(d.get("qty") or 0.0),
            "filled_qty": float(d.get("filled_qty") or 0.0),
            "filled_avg_price": float(filled_avg) if filled_avg is not None else None,
            "status": str(d.get("status", "")),
            "limit_price": float(d["limit_price"]) if d.get("limit_price") else None,
            "stop_price": float(d["stop_price"]) if d.get("stop_price") else None,
        }
