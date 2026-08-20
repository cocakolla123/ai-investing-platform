"""MockBroker — Broker port implementation for TEST / backtest-final-validation.

Supports scripted fault injection (fills, partials, rejects, disconnects) so
every failure row of doc 09 §8 has a corresponding test. Duplicate
``client_order_id`` submissions return the existing broker order (Alpaca
semantics: the intent is already known — fetch its state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from dynamo.domain.models.portfolio import Order, OrderSide
from dynamo.execution.errors import BrokerDisconnect, BrokerRejection

Behavior = Literal["accept", "fill", "partial", "reject", "disconnect", "disconnect_after_accept"]


@dataclass
class ScriptStep:
    behavior: Behavior = "accept"
    fill_qty: float | None = None  # for "partial"
    fill_price: float | None = None
    reason: str = "scripted"


@dataclass
class _BrokerOrder:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_price: float | None
    stop_price: float | None
    status: str  # new | partially_filled | filled | canceled | rejected | replaced
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
        }


@dataclass
class MockBroker:
    """In-memory broker with scripted behaviors and position/cash tracking."""

    cash: float = 100_000.0
    script: list[ScriptStep] = field(default_factory=list)
    default_fill_price: float = 100.0

    def __post_init__(self) -> None:
        self._orders: dict[str, _BrokerOrder] = {}
        self._positions: dict[str, dict[str, float]] = {}  # symbol -> {qty, avg_entry_price}
        self._seq = 0
        self.submit_calls = 0

    # --- scripting ---

    def push(self, step: ScriptStep) -> None:
        self.script.append(step)

    def _next_step(self) -> ScriptStep:
        return self.script.pop(0) if self.script else ScriptStep(behavior="fill")

    # --- Broker port ---

    def submit(self, order: Order) -> str:
        self.submit_calls += 1
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            # Alpaca semantics: duplicate client order id → the intent already
            # exists; return the existing broker order id, no new order.
            return existing.broker_order_id

        step = self._next_step()
        if step.behavior == "disconnect":
            # Connection dies before the broker records anything.
            raise BrokerDisconnect("connection lost before ack")
        if step.behavior == "reject":
            raise BrokerRejection(order.client_order_id, step.reason)

        self._seq += 1
        broker_order = _BrokerOrder(
            client_order_id=order.client_order_id,
            broker_order_id=f"mock-{self._seq}",
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            order_type=order.order_type.value,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            status="new",
        )
        self._orders[order.client_order_id] = broker_order

        if step.behavior == "disconnect_after_accept":
            # The broker recorded (and here, filled) the order but the ack was
            # lost — the classic ambiguous mid-order disconnect (doc 09 §8).
            self._apply_fill(broker_order, order.qty, step.fill_price or self.default_fill_price)
            raise BrokerDisconnect("connection lost after broker accepted the order")
        if step.behavior == "fill":
            self._apply_fill(broker_order, order.qty, step.fill_price or self.default_fill_price)
        elif step.behavior == "partial":
            qty = step.fill_qty if step.fill_qty is not None else order.qty / 2
            self._apply_fill(broker_order, qty, step.fill_price or self.default_fill_price)
        return broker_order.broker_order_id

    def cancel(self, client_order_id: str) -> None:
        bo = self._require(client_order_id)
        if bo.status in ("filled", "canceled", "rejected"):
            return
        bo.status = "canceled"

    def replace(self, client_order_id: str, stop_price: float) -> str:
        step = self._next_step() if self.script else ScriptStep(behavior="accept")
        if step.behavior == "reject":
            raise BrokerRejection(client_order_id, f"replace rejected: {step.reason}")
        if step.behavior == "disconnect":
            raise BrokerDisconnect("connection lost during replace")
        bo = self._require(client_order_id)
        bo.stop_price = stop_price
        return bo.broker_order_id

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        return self._require(client_order_id).as_dict()

    def open_orders(self) -> list[dict[str, Any]]:
        return [
            bo.as_dict() for bo in self._orders.values() if bo.status in ("new", "partially_filled")
        ]

    def positions(self) -> list[dict[str, Any]]:
        return [
            {"symbol": s, "qty": p["qty"], "avg_entry_price": p["avg_entry_price"]}
            for s, p in self._positions.items()
            if p["qty"] != 0
        ]

    def account(self) -> dict[str, Any]:
        equity = self.cash + sum(p["qty"] * p["avg_entry_price"] for p in self._positions.values())
        return {"cash": self.cash, "equity": equity, "buying_power": self.cash}

    # --- test helpers ---

    def force_fill(
        self, client_order_id: str, qty: float | None = None, price: float | None = None
    ) -> None:
        """Fill an open order out-of-band (simulates fills during downtime)."""
        bo = self._require(client_order_id)
        remaining = bo.qty - bo.filled_qty
        self._apply_fill(
            bo, qty if qty is not None else remaining, price or self.default_fill_price
        )

    def knows(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    # --- internals ---

    def _require(self, client_order_id: str) -> _BrokerOrder:
        bo = self._orders.get(client_order_id)
        if bo is None:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        return bo

    def _apply_fill(self, bo: _BrokerOrder, qty: float, price: float) -> None:
        qty = min(qty, bo.qty - bo.filled_qty)
        if qty <= 0:
            return
        prev = bo.filled_qty
        bo.filled_qty = prev + qty
        if bo.filled_avg_price is None:
            bo.filled_avg_price = price
        else:
            bo.filled_avg_price = (bo.filled_avg_price * prev + price * qty) / bo.filled_qty
        bo.status = "filled" if bo.filled_qty >= bo.qty else "partially_filled"

        signed = qty if bo.side == OrderSide.BUY.value else -qty
        pos = self._positions.setdefault(bo.symbol, {"qty": 0.0, "avg_entry_price": 0.0})
        new_qty = pos["qty"] + signed
        if signed > 0 and new_qty > 0:
            pos["avg_entry_price"] = (
                (pos["avg_entry_price"] * pos["qty"] + price * signed) / new_qty
                if new_qty
                else price
            )
        pos["qty"] = new_qty
        self.cash -= signed * price
