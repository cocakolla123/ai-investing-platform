"""Idempotency (doc 09 §4): the same intent can never duplicate at the broker."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dynamo.domain.models.portfolio import (
    OrderState,
    PortfolioState,
    TargetPortfolio,
    TargetWeight,
)
from dynamo.domain.models.research import Sleeve
from dynamo.execution.client_order_id import client_order_id
from dynamo.execution.errors import DuplicateClientOrderId
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from tests.unit.execution.helpers import drive_to_pending, make_manager, make_order

AS_OF = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)


def _target() -> TargetPortfolio:
    return TargetPortfolio(
        id="tp-1",
        as_of=AS_OF,
        risk_verdict_ref="rv-1",
        weights=(
            TargetWeight(symbol="AAPL", weight=0.10, sleeve=Sleeve.LONG_TERM),
            TargetWeight(symbol="MSFT", weight=0.001, sleeve=Sleeve.LONG_TERM),  # < 0.25%
        ),
        cash=0.899,
    )


def test_client_order_id_is_deterministic_and_24_chars() -> None:
    a = client_order_id("tp-1", "AAPL", "buy", "2026-08-03")
    b = client_order_id("tp-1", "AAPL", "buy", "2026-08-03")
    assert a == b
    assert len(a) == 24
    assert a != client_order_id("tp-1", "AAPL", "sell", "2026-08-03")
    assert a != client_order_id("tp-1", "AAPL", "buy", "2026-08-04")


def test_propose_orders_skips_sub_quarter_percent_deltas() -> None:
    manager, _store, _, _, _ = make_manager()
    current = PortfolioState(as_of=AS_OF, equity=100_000.0, cash=100_000.0)
    proposed = manager.propose_orders(_target(), current, {"AAPL": 200.0, "MSFT": 400.0})
    assert [o.symbol for o in proposed] == ["AAPL"]  # MSFT delta 0.1% skipped
    assert proposed[0].qty == 50.0  # 10% of 100k at $200


def test_reproposing_same_target_yields_same_orders_no_duplicates() -> None:
    manager, store, _, _, _ = make_manager()
    current = PortfolioState(as_of=AS_OF, equity=100_000.0, cash=100_000.0)
    prices = {"AAPL": 200.0, "MSFT": 400.0}
    first = manager.propose_orders(_target(), current, prices)
    second = manager.propose_orders(_target(), current, prices)  # crash-restart replay
    assert [o.client_order_id for o in first] == [o.client_order_id for o in second]
    assert len(store.orders) == 1


def test_same_intent_twice_yields_one_broker_order() -> None:
    """Crash-restart simulation: a second process re-derives the same intent
    and submits; the broker ends up with exactly one order."""
    broker = MockBroker(script=[ScriptStep(behavior="accept"), ScriptStep(behavior="accept")])
    coid = client_order_id("tp-1", "AAPL", "buy", "2026-08-03")

    manager1, store1, _, _, _ = make_manager(broker)
    store1.insert(make_order(coid=coid, decision_context_id="dc-1"))
    drive_to_pending(manager1, store1, coid)
    order1 = manager1.submit(coid)

    # "Restart": fresh local state, same deterministic intent.
    manager2, store2, _, _, _ = make_manager(broker)
    store2.insert(make_order(coid=coid, decision_context_id="dc-1"))
    drive_to_pending(manager2, store2, coid)
    order2 = manager2.submit(coid)

    assert broker.submit_calls == 2
    assert len(broker.open_orders()) == 1  # exactly one broker order
    assert order1.broker_order_id == order2.broker_order_id


class _DuplicateRaisingBroker(MockBroker):
    """Alpaca-style broker that rejects duplicate ids instead of echoing."""

    def submit(self, order: Any) -> str:
        if self.knows(order.client_order_id):
            self.submit_calls += 1
            raise DuplicateClientOrderId(order.client_order_id)
        return super().submit(order)


def test_duplicate_id_response_handled_as_state_fetch() -> None:
    broker = _DuplicateRaisingBroker(script=[ScriptStep(behavior="accept")])
    coid = client_order_id("tp-1", "AAPL", "buy", "2026-08-03")

    manager1, store1, _, _, _ = make_manager(broker)
    store1.insert(make_order(coid=coid, decision_context_id="dc-1"))
    drive_to_pending(manager1, store1, coid)
    manager1.submit(coid)

    manager2, store2, _, _, _ = make_manager(broker)
    store2.insert(make_order(coid=coid, decision_context_id="dc-1"))
    drive_to_pending(manager2, store2, coid)
    order = manager2.submit(coid)  # duplicate → fetch state, not an error

    assert order.state == OrderState.SUBMITTED
    assert order.broker_order_id  # recovered from the broker
    assert len(broker.open_orders()) == 1
