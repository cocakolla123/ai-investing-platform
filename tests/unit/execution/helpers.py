"""Shared builders for execution unit tests."""

from __future__ import annotations

from dynamo.domain.models.portfolio import Order, OrderSide, OrderState, OrderType
from dynamo.execution.mock_broker import MockBroker
from dynamo.execution.order_manager import OrderManager
from tests.unit.execution.fakes import FakeClock, FakeOrderStore, FakeTradeStore


def make_order(
    coid: str = "c1",
    symbol: str = "AAPL",
    state: OrderState = OrderState.PROPOSED,
    qty: float = 10.0,
    decision_context_id: str | None = None,
    side: OrderSide = OrderSide.BUY,
) -> Order:
    return Order(
        client_order_id=coid,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=OrderType.MARKETABLE_LIMIT,
        limit_price=100.05,
        state=state,
        decision_context_id=decision_context_id,
    )


def make_manager(
    broker: MockBroker | None = None,
    *,
    auto_approve: bool = True,
    reconciled: bool = True,
    **kwargs: object,
) -> tuple[OrderManager, FakeOrderStore, FakeTradeStore, FakeClock, MockBroker]:
    broker = broker if broker is not None else MockBroker()
    store = FakeOrderStore()
    trade_store = FakeTradeStore()
    clock = FakeClock()
    manager = OrderManager(
        broker,
        store,
        trade_store,
        clock,
        auto_approve=auto_approve,
        **kwargs,  # type: ignore[arg-type]
    )
    if reconciled:
        manager.startup_reconciliation()
    return manager, store, trade_store, clock, broker


def drive_to_pending(manager: OrderManager, store: FakeOrderStore, coid: str = "c1") -> None:
    manager.transition(coid, OrderState.RISK_CHECKED, "risk ok")
    manager.request_approval(coid)
