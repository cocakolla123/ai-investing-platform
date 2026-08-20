"""In-memory fakes for order-manager unit tests (no database)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from dynamo.domain.models.portfolio import (
    TERMINAL_ORDER_STATES,
    Order,
    OrderState,
    Trade,
)


@dataclass
class FakeClock:
    current: datetime = field(default_factory=lambda: datetime(2026, 8, 3, 14, 0, tzinfo=UTC))

    def now(self) -> datetime:
        return self.current

    def today(self) -> date:
        return self.current.date()

    def advance(self, **kwargs: float) -> None:
        self.current += timedelta(**kwargs)


@dataclass
class EventRow:
    client_order_id: str
    from_state: str | None
    to_state: str
    reason: str
    ts: datetime
    payload: Mapping[str, Any] | None = None


class FakeOrderStore:
    """Satisfies the OrderStore protocol; records one event row per transition."""

    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}
        self.events: list[EventRow] = []

    def insert(self, order: Order) -> None:
        if order.client_order_id in self.orders:
            raise ValueError(f"duplicate insert {order.client_order_id}")
        self.orders[order.client_order_id] = order
        self.events.append(
            EventRow(
                order.client_order_id,
                None,
                order.state.value,
                "created",
                order.created_at or datetime.now(UTC),
            )
        )

    def get(self, client_order_id: str) -> Order | None:
        return self.orders.get(client_order_id)

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.state not in TERMINAL_ORDER_STATES]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        order = self.orders[client_order_id]
        self.orders[client_order_id] = order.model_copy(update={"broker_order_id": broker_order_id})

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None:
        order = self.orders[client_order_id]
        self.orders[client_order_id] = order.model_copy(
            update={"decision_context_id": decision_context_id}
        )

    def transition_with_event(
        self,
        client_order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str,
        ts: datetime,
        payload: Mapping[str, Any] | None = None,
    ) -> Order:
        order = self.orders[client_order_id]
        updated = order.model_copy(update={"state": to_state})
        self.orders[client_order_id] = updated
        self.events.append(
            EventRow(client_order_id, from_state.value, to_state.value, reason, ts, payload)
        )
        return updated

    def state_entered_at(self, client_order_id: str) -> datetime | None:
        for ev in reversed(self.events):
            if ev.client_order_id == client_order_id:
                return ev.ts
        return None

    def transition_events(self, client_order_id: str) -> list[EventRow]:
        return [
            e
            for e in self.events
            if e.client_order_id == client_order_id and e.from_state is not None
        ]


class FakeTradeStore:
    def __init__(self) -> None:
        self.trades: list[Trade] = []

    def insert(self, trade: Trade) -> None:
        self.trades.append(trade)

    def filled_qty(self, client_order_id: str) -> float:
        return sum(t.qty for t in self.trades if t.client_order_id == client_order_id)
