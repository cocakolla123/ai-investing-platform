"""Shared builders and in-memory fakes for PM sleeve unit tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import PMOrder, PMPosition, PMSide, PMTrade
from dynamo.domain.models.portfolio import TERMINAL_ORDER_STATES, OrderState
from dynamo.domain.models.research import PredictionMarketView
from dynamo.execution.mock_pm_broker import MockPMBroker
from dynamo.execution.pm_order_manager import PMOrderManager
from tests.unit.execution.fakes import EventRow, FakeClock

NOW = datetime(2026, 8, 4, 20, 30, tzinfo=UTC)
AS_OF = date(2026, 8, 4)


def make_view(
    market_id: str = "FED-26DEC-T4.00",
    estimated: float = 0.60,
    implied: float = 0.43,
    confidence: float = 0.8,
    venue: str = "kalshi",
) -> PredictionMarketView:
    return PredictionMarketView(
        venue=venue,
        market_id=market_id,
        question=f"Question for {market_id}?",
        as_of=AS_OF,
        market_implied_prob=implied,
        estimated_prob=estimated,
        edge=estimated - implied,
        confidence=confidence,
        uncertainties=("data gap",),
    )


def make_price(
    market_id: str = "FED-26DEC-T4.00", implied: float = 0.43, trusted: bool = True
) -> PredictionMarketPrice:
    return PredictionMarketPrice(
        venue="kalshi",
        market_id=market_id,
        ts=NOW,
        implied_prob=implied,
        volume=50_000.0,
        liquidity=100_000.0,
        trusted=trusted,
    )


def make_market(
    market_id: str = "FED-26DEC-T4.00",
    close_ts: datetime | None = datetime(2026, 9, 18, 21, 0, tzinfo=UTC),
    resolved: bool | None = None,
) -> PredictionMarket:
    return PredictionMarket(
        venue="kalshi",
        market_id=market_id,
        question=f"Question for {market_id}?",
        close_ts=close_ts,
        resolved_outcome=resolved,
    )


def make_pm_order(
    coid: str = "pm-1",
    market_id: str = "FED-26DEC-T4.00",
    side: PMSide = PMSide.YES,
    contracts: int = 10,
    limit_price_cents: int = 43,
    state: OrderState = OrderState.PROPOSED,
    decision_context_id: str | None = None,
) -> PMOrder:
    return PMOrder(
        client_order_id=coid,
        market_id=market_id,
        side=side,
        contracts=contracts,
        limit_price_cents=limit_price_cents,
        state=state,
        decision_context_id=decision_context_id,
    )


def make_position(
    market_id: str = "FED-26DEC-T4.00",
    side: PMSide = PMSide.YES,
    contracts: int = 10,
    avg_price_cents: float = 43.0,
) -> PMPosition:
    return PMPosition(
        market_id=market_id, side=side, contracts=contracts, avg_price_cents=avg_price_cents
    )


class FakePMOrderStore:
    """Satisfies the PMOrderStore protocol; one event row per transition."""

    def __init__(self) -> None:
        self.orders: dict[str, PMOrder] = {}
        self.events: list[EventRow] = []

    def insert(self, order: PMOrder) -> None:
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

    def get(self, client_order_id: str) -> PMOrder | None:
        return self.orders.get(client_order_id)

    def open_orders(self) -> list[PMOrder]:
        return [o for o in self.orders.values() if o.state not in TERMINAL_ORDER_STATES]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        order = self.orders[client_order_id]
        self.orders[client_order_id] = order.model_copy(
            update={"broker_order_id": broker_order_id}
        )

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
    ) -> PMOrder:
        order = self.orders[client_order_id]
        updated = order.model_copy(update={"state": to_state})
        self.orders[client_order_id] = updated
        self.events.append(
            EventRow(client_order_id, from_state.value, to_state.value, reason, ts, payload)
        )
        return updated


class FakePMTradeStore:
    def __init__(self) -> None:
        self.trades: list[PMTrade] = []

    def insert(self, trade: PMTrade) -> None:
        self.trades.append(trade)

    def filled_contracts(self, client_order_id: str) -> int:
        return sum(
            t.contracts
            for t in self.trades
            if t.client_order_id == client_order_id and t.kind == "fill"
        )


class PermissiveGate:
    permitted = True
    reason = "paper_permitted"


class RefusingGate:
    permitted = False
    reason = "approval_missing"


def make_pm_manager(
    broker: MockPMBroker | None = None,
    *,
    gate: Any = None,
) -> tuple[PMOrderManager, FakePMOrderStore, FakePMTradeStore, FakeClock, MockPMBroker]:
    broker = broker if broker is not None else MockPMBroker()
    store = FakePMOrderStore()
    trade_store = FakePMTradeStore()
    clock = FakeClock()
    manager = PMOrderManager(
        broker,
        store,
        trade_store,
        clock,
        gate=(lambda _order: gate) if gate is not None else None,
    )
    return manager, store, trade_store, clock, broker
