"""Chaos test (doc 09 §10): broker connection dies mid-order.

The order manager halts (no new submissions), and startup reconciliation then
converges local state to broker truth with zero duplicate orders.
"""

from __future__ import annotations

import pytest

from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.errors import BrokerDisconnect, ReconciliationPending
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from tests.unit.execution.helpers import drive_to_pending, make_manager, make_order


def test_disconnect_after_broker_accept_converges_via_reconciliation() -> None:
    # The broker records AND fills the order, but the ack is lost.
    broker = MockBroker(script=[ScriptStep(behavior="disconnect_after_accept", fill_price=99.5)])
    manager, store, trade_store, _, _ = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1", qty=10))
    drive_to_pending(manager, store)

    with pytest.raises(BrokerDisconnect):
        manager.submit("c1")

    # Ambiguous state: locally submitted, and the manager has halted.
    assert store.orders["c1"].state == OrderState.SUBMITTED
    assert not manager.reconciled
    store.insert(make_order(coid="c2", decision_context_id="dc-2"))
    drive_to_pending(manager, store, "c2")
    with pytest.raises(ReconciliationPending):
        manager.submit("c2")

    # Startup reconciliation converges local state to broker truth.
    manager.startup_reconciliation()
    assert store.orders["c1"].state == OrderState.FILLED
    assert store.orders["c1"].broker_order_id  # repaired from broker
    assert trade_store.filled_qty("c1") == pytest.approx(10)
    assert broker.submit_calls == 1  # zero duplicates

    # And normal operation resumes.
    assert manager.submit("c2").state == OrderState.SUBMITTED


def test_disconnect_before_broker_accept_cancels_locally() -> None:
    # The connection dies before the broker records anything.
    broker = MockBroker(script=[ScriptStep(behavior="disconnect")])
    manager, store, trade_store, _, _ = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)

    with pytest.raises(BrokerDisconnect):
        manager.submit("c1")
    assert store.orders["c1"].state == OrderState.SUBMITTED  # ambiguous until resolved

    manager.startup_reconciliation()
    # Broker truth: order never existed → local cancel, intent re-enters the
    # next scheduling bucket naturally.
    assert store.orders["c1"].state == OrderState.CANCELLED
    assert store.transition_events("c1")[-1].reason == "not_found_at_broker"
    assert trade_store.trades == []
    assert not broker.knows("c1")


def test_reconciliation_replay_is_idempotent() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="disconnect_after_accept")])
    manager, store, trade_store, _, _ = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1", qty=10))
    drive_to_pending(manager, store)
    with pytest.raises(BrokerDisconnect):
        manager.submit("c1")

    manager.startup_reconciliation()
    manager.startup_reconciliation()  # run twice — no double-recorded fills
    assert trade_store.filled_qty("c1") == pytest.approx(10)
    assert store.orders["c1"].state == OrderState.FILLED
