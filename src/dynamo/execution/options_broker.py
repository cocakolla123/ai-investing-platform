"""Alpaca implementation of the ``OptionsBroker`` port (ADR-0010).

Alpaca supports options trading in PAPER accounts; paper vs live is decided
solely by ``settings.alpaca_base_url`` — the code path is identical, and every
submission still passes the doc 09 §7 gate before it gets here. LIMIT ORDERS
ONLY: the broker refuses any order without a limit price, by construction.

Error discipline mirrors :mod:`dynamo.execution.alpaca_broker`: rejects (4xx)
are terminal and never auto-retried, duplicate ``client_order_id`` surfaces as
``DuplicateClientOrderId`` (fetch state, do not resubmit), 429 is a logged
incident, and a transient 5xx/transport fault gets exactly one jittered retry
before escalating (fail-closed).
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
from alpaca.trading.requests import LimitOrderRequest

from dynamo.config.settings import Settings, get_settings
from dynamo.domain.models.options import OptionOrder
from dynamo.domain.models.portfolio import OrderSide
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)

_DUPLICATE_MARKERS = ("client_order_id must be unique", "duplicate client_order_id")
OPTION_ASSET_CLASS = "us_option"


def _status_code(exc: APIError) -> int:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return 0


class AlpacaOptionsBroker:
    """``OptionsBroker`` port over ``alpaca-py`` ``TradingClient`` (paper first)."""

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

    # --- OptionsBroker port ---

    def submit(self, order: OptionOrder) -> str:
        """Submit a DAY limit order on the OCC symbol; the id is idempotent."""
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        request = LimitOrderRequest(
            symbol=order.occ_symbol,
            qty=order.contracts,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(order.limit_price, 2),
            client_order_id=order.client_order_id,
        )
        raw = self._call(
            lambda: self._client.submit_order(order_data=request),
            client_order_id=order.client_order_id,
        )
        return str(self._as_dict(raw).get("id", ""))

    def cancel(self, client_order_id: str) -> None:
        raw = self._call(
            lambda: self._client.get_order_by_client_id(client_order_id),
            client_order_id=client_order_id,
        )
        broker_id = str(self._as_dict(raw).get("id", ""))
        self._call(
            lambda: self._client.cancel_order_by_id(broker_id),
            client_order_id=client_order_id,
        )

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        raw = self._call(
            lambda: self._client.get_order_by_client_id(client_order_id),
            client_order_id=client_order_id,
        )
        return self._normalize_order(self._as_dict(raw))

    def open_orders(self) -> list[dict[str, Any]]:
        raw = self._call(self._client.get_orders)
        items = raw if isinstance(raw, list) else []
        normalized = [self._normalize_order(self._as_dict(o)) for o in items]
        return [o for o in normalized if o["asset_class"] == OPTION_ASSET_CLASS]

    def positions(self) -> list[dict[str, Any]]:
        """Open OPTION positions only (equities are the equity reconciler's)."""
        raw = self._call(self._client.get_all_positions)
        items = raw if isinstance(raw, list) else []
        out: list[dict[str, Any]] = []
        for p in items:
            d = self._as_dict(p)
            if str(d.get("asset_class", "")) != OPTION_ASSET_CLASS:
                continue
            out.append(
                {
                    "occ_symbol": str(d.get("symbol", "")),
                    "qty": float(d.get("qty", 0.0)),  # signed: negative = short
                    "avg_entry_price": float(d.get("avg_entry_price", 0.0)),
                }
            )
        return out

    # --- error discipline (doc 09 §8, verbatim from the equity broker) ---

    def _call(self, fn: Callable[[], Any], client_order_id: str = "") -> Any:
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
    def _normalize_order(d: dict[str, Any]) -> dict[str, Any]:
        filled_avg = d.get("filled_avg_price")
        return {
            "client_order_id": str(d.get("client_order_id", "")),
            "broker_order_id": str(d.get("id", "")),
            "occ_symbol": str(d.get("symbol", "")),
            "side": str(d.get("side", "")),
            "contracts": int(float(d.get("qty") or 0.0)),
            "filled_contracts": int(float(d.get("filled_qty") or 0.0)),
            "filled_avg_price": float(filled_avg) if filled_avg is not None else None,
            "status": str(d.get("status", "")),
            "limit_price": float(d["limit_price"]) if d.get("limit_price") else None,
            "asset_class": str(d.get("asset_class", "")),
        }
