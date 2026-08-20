"""MockPMBroker — ``PMBroker`` port implementation for TEST.

Mirrors :mod:`dynamo.execution.mock_broker`: scripted fault injection (fills,
partials, rejects, disconnects) plus Kalshi idempotency semantics — a
duplicate ``client_order_id`` submission returns the existing broker order id
instead of creating a new order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dynamo.domain.models.pm import PMOrder, PMSide
from dynamo.execution.errors import BrokerDisconnect, BrokerRejection

Behavior = Literal["accept", "fill", "partial", "reject", "disconnect", "disconnect_after_accept"]


@dataclass
class PMScriptStep:
    behavior: Behavior = "accept"
    fill_contracts: int | None = None  # for "partial"
    fill_price_cents: float | None = None
    reason: str = "scripted"


@dataclass
class _PMBrokerOrder:
    client_order_id: str
    broker_order_id: str
    market_id: str
    side: str
    contracts: int
    limit_price_cents: int
    status: str  # new | partially_filled | filled | canceled | rejected
    filled_contracts: int = 0
    filled_avg_price_cents: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "market_id": self.market_id,
            "side": self.side,
            "contracts": self.contracts,
            "filled_contracts": self.filled_contracts,
            "filled_avg_price_cents": self.filled_avg_price_cents,
            "status": self.status,
        }


@dataclass
class MockPMBroker:
    """In-memory Kalshi stand-in with scripted behaviors and cents accounting."""

    balance_cents: float = 1_000_000.0
    script: list[PMScriptStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._orders: dict[str, _PMBrokerOrder] = {}
        self._positions: dict[str, dict[str, Any]] = {}  # market_id -> side/contracts/avg
        self._seq = 0
        self.submit_calls = 0

    # --- scripting ---

    def push(self, step: PMScriptStep) -> None:
        self.script.append(step)

    def _next_step(self) -> PMScriptStep:
        return self.script.pop(0) if self.script else PMScriptStep(behavior="fill")

    # --- PMBroker port ---

    def submit(self, order: PMOrder) -> str:
        self.submit_calls += 1
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            # Kalshi idempotency: the intent is already known — same order back.
            return existing.broker_order_id

        step = self._next_step()
        if step.behavior == "disconnect":
            raise BrokerDisconnect("connection lost before ack")
        if step.behavior == "reject":
            raise BrokerRejection(order.client_order_id, step.reason)

        self._seq += 1
        broker_order = _PMBrokerOrder(
            client_order_id=order.client_order_id,
            broker_order_id=f"kalshi-mock-{self._seq}",
            market_id=order.market_id,
            side=order.side.value,
            contracts=order.contracts,
            limit_price_cents=order.limit_price_cents,
            status="new",
        )
        self._orders[order.client_order_id] = broker_order

        price = (
            step.fill_price_cents
            if step.fill_price_cents is not None
            else float(order.limit_price_cents)
        )
        if step.behavior == "disconnect_after_accept":
            self._apply_fill(broker_order, order.contracts, price)
            raise BrokerDisconnect("connection lost after kalshi accepted the order")
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
            bo.as_dict() for bo in self._orders.values() if bo.status in ("new", "partially_filled")
        ]

    def positions(self) -> list[dict[str, Any]]:
        return [
            {
                "market_id": market_id,
                "side": p["side"],
                "contracts": p["contracts"],
                "exposure_cents": p["contracts"] * p["avg_price_cents"],
            }
            for market_id, p in self._positions.items()
            if p["contracts"] > 0
        ]

    def balance(self) -> dict[str, Any]:
        return {"balance_cents": self.balance_cents}

    # --- test helpers ---

    def force_fill(
        self, client_order_id: str, contracts: int | None = None, price_cents: float | None = None
    ) -> None:
        """Fill an open order out-of-band (simulates fills during downtime)."""
        bo = self._require(client_order_id)
        remaining = bo.contracts - bo.filled_contracts
        self._apply_fill(
            bo,
            contracts if contracts is not None else remaining,
            price_cents if price_cents is not None else float(bo.limit_price_cents),
        )

    def set_position(self, market_id: str, side: PMSide, contracts: int, avg_cents: float) -> None:
        """Force broker-side state (for reconciliation divergence tests)."""
        self._positions[market_id] = {
            "side": side.value,
            "contracts": contracts,
            "avg_price_cents": avg_cents,
        }

    def knows(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    # --- internals ---

    def _require(self, client_order_id: str) -> _PMBrokerOrder:
        bo = self._orders.get(client_order_id)
        if bo is None:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        return bo

    def _apply_fill(self, bo: _PMBrokerOrder, contracts: int, price_cents: float) -> None:
        contracts = min(contracts, bo.contracts - bo.filled_contracts)
        if contracts <= 0:
            return
        prev = bo.filled_contracts
        bo.filled_contracts = prev + contracts
        if bo.filled_avg_price_cents is None:
            bo.filled_avg_price_cents = price_cents
        else:
            bo.filled_avg_price_cents = (
                bo.filled_avg_price_cents * prev + price_cents * contracts
            ) / bo.filled_contracts
        bo.status = "filled" if bo.filled_contracts >= bo.contracts else "partially_filled"

        pos = self._positions.setdefault(
            bo.market_id, {"side": bo.side, "contracts": 0, "avg_price_cents": 0.0}
        )
        new_contracts = pos["contracts"] + contracts
        pos["avg_price_cents"] = (
            pos["avg_price_cents"] * pos["contracts"] + price_cents * contracts
        ) / new_contracts
        pos["contracts"] = new_contracts
        self.balance_cents -= contracts * price_cents
