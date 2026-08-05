"""MockOptionsBroker — ``OptionsBroker`` port implementation for TEST.

Mirrors :mod:`dynamo.execution.mock_pm_broker`: scripted fault injection
(fills, partials, rejects, disconnects) plus idempotency semantics — a
duplicate ``client_order_id`` submission returns the existing broker order id
instead of creating a new order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dynamo.domain.models.options import OptionOrder
from dynamo.domain.models.portfolio import OrderSide
from dynamo.execution.errors import BrokerDisconnect, BrokerRejection

Behavior = Literal["accept", "fill", "partial", "reject", "disconnect", "disconnect_after_accept"]


@dataclass
class OptionsScriptStep:
    behavior: Behavior = "accept"
    fill_contracts: int | None = None  # for "partial"
    fill_price: float | None = None  # dollars per share
    reason: str = "scripted"


@dataclass
class _BrokerOptionOrder:
    client_order_id: str
    broker_order_id: str
    occ_symbol: str
    side: str
    contracts: int
    limit_price: float
    status: str  # new | partially_filled | filled | canceled | rejected
    filled_contracts: int = 0
    filled_avg_price: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "occ_symbol": self.occ_symbol,
            "side": self.side,
            "contracts": self.contracts,
            "filled_contracts": self.filled_contracts,
            "filled_avg_price": self.filled_avg_price,
            "status": self.status,
            "asset_class": "us_option",
        }


@dataclass
class MockOptionsBroker:
    """In-memory options broker stand-in with scripted behaviors."""

    script: list[OptionsScriptStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._orders: dict[str, _BrokerOptionOrder] = {}
        self._positions: dict[str, dict[str, Any]] = {}  # occ -> qty (signed) / avg
        self._seq = 0
        self.submit_calls = 0

    # --- scripting ---

    def push(self, step: OptionsScriptStep) -> None:
        self.script.append(step)

    def _next_step(self) -> OptionsScriptStep:
        return self.script.pop(0) if self.script else OptionsScriptStep(behavior="fill")

    # --- OptionsBroker port ---

    def submit(self, order: OptionOrder) -> str:
        self.submit_calls += 1
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            # Idempotency: the intent is already known — same order back.
            return existing.broker_order_id

        step = self._next_step()
        if step.behavior == "disconnect":
            raise BrokerDisconnect("connection lost before ack")
        if step.behavior == "reject":
            raise BrokerRejection(order.client_order_id, step.reason)

        self._seq += 1
        broker_order = _BrokerOptionOrder(
            client_order_id=order.client_order_id,
            broker_order_id=f"alpaca-opt-mock-{self._seq}",
            occ_symbol=order.occ_symbol,
            side=order.side.value,
            contracts=order.contracts,
            limit_price=order.limit_price,
            status="new",
        )
        self._orders[order.client_order_id] = broker_order

        price = step.fill_price if step.fill_price is not None else order.limit_price
        if step.behavior == "disconnect_after_accept":
            self._apply_fill(broker_order, order.contracts, price)
            raise BrokerDisconnect("connection lost after the broker accepted the order")
        if step.behavior == "fill":
            self._apply_fill(broker_order, order.contracts, price)
        elif step.behavior == "partial":
            contracts = (
                step.fill_contracts if step.fill_contracts is not None else order.contracts // 2
            )
            self._apply_fill(broker_order, contracts, price)
        return broker_order.broker_order_id

    def cancel(self, client_order_id: str) -> None:
        bo = self._require(client_order_id)
        if bo.status in ("filled", "canceled", "rejected"):
            return
        bo.status = "canceled"

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        return self._require(client_order_id).as_dict()

    def open_orders(self) -> list[dict[str, Any]]:
        return [
            bo.as_dict()
            for bo in self._orders.values()
            if bo.status in ("new", "partially_filled")
        ]

    def positions(self) -> list[dict[str, Any]]:
        return [
            {
                "occ_symbol": occ,
                "qty": p["qty"],
                "avg_entry_price": p["avg_entry_price"],
            }
            for occ, p in self._positions.items()
            if p["qty"] != 0.0
        ]

    # --- test helpers ---

    def force_fill(
        self, client_order_id: str, contracts: int | None = None, price: float | None = None
    ) -> None:
        """Fill an open order out-of-band (simulates fills during downtime)."""
        bo = self._require(client_order_id)
        remaining = bo.contracts - bo.filled_contracts
        self._apply_fill(
            bo,
            contracts if contracts is not None else remaining,
            price if price is not None else bo.limit_price,
        )

    def set_position(self, occ_symbol: str, qty: float, avg_price: float) -> None:
        """Force broker-side state (for reconciliation divergence tests)."""
        self._positions[occ_symbol] = {"qty": qty, "avg_entry_price": avg_price}

    def knows(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    # --- internals ---

    def _require(self, client_order_id: str) -> _BrokerOptionOrder:
        bo = self._orders.get(client_order_id)
        if bo is None:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        return bo

    def _apply_fill(self, bo: _BrokerOptionOrder, contracts: int, price: float) -> None:
        contracts = min(contracts, bo.contracts - bo.filled_contracts)
        if contracts <= 0:
            return
        prev = bo.filled_contracts
        bo.filled_contracts = prev + contracts
        if bo.filled_avg_price is None:
            bo.filled_avg_price = price
        else:
            bo.filled_avg_price = (
                bo.filled_avg_price * prev + price * contracts
            ) / bo.filled_contracts
        bo.status = "filled" if bo.filled_contracts >= bo.contracts else "partially_filled"

        signed = contracts if bo.side == OrderSide.BUY.value else -contracts
        pos = self._positions.setdefault(
            bo.occ_symbol, {"qty": 0.0, "avg_entry_price": 0.0}
        )
        new_qty = pos["qty"] + signed
        pos["avg_entry_price"] = price if new_qty != 0.0 else 0.0
        pos["qty"] = new_qty
