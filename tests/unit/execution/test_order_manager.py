"""Order-manager behaviors: decision-context requirement, approval TTL,
fills + slippage, day-end partial cancel, submission blocking."""

from __future__ import annotations

import pytest

from dynamo.domain.models.portfolio import OrderSide, OrderState
from dynamo.execution.errors import (
    BrokerRejection,
    MissingDecisionContext,
    ReconciliationPending,
)
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from dynamo.execution.order_manager import (
    APPROVAL_EXPIRED_REASON,
    GateRefused,
    marketable_limit_price,
    slippage_bps,
)
from tests.unit.execution.helpers import drive_to_pending, make_manager, make_order


def test_submit_requires_decision_context() -> None:
    manager, store, _, _, broker = make_manager()
    store.insert(make_order())  # no decision_context_id
    drive_to_pending(manager, store)
    with pytest.raises(MissingDecisionContext):
        manager.submit("c1")
    assert broker.submit_calls == 0
    assert store.orders["c1"].state == OrderState.PENDING_APPROVAL


def test_submit_blocked_until_startup_reconciliation() -> None:
    manager, store, _, _, _ = make_manager(reconciled=False)
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)
    with pytest.raises(ReconciliationPending):
        manager.submit("c1")
    manager.startup_reconciliation()
    assert manager.submit("c1").state == OrderState.SUBMITTED


def test_approval_ttl_expiry_cancels_with_reason() -> None:
    manager, store, _, clock, broker = make_manager(auto_approve=False)
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)
    clock.advance(minutes=31)  # past the 30-min TTL
    order = manager.submit("c1")
    assert order.state == OrderState.CANCELLED
    assert broker.submit_calls == 0
    assert store.transition_events("c1")[-1].reason == APPROVAL_EXPIRED_REASON


def test_expire_stale_approvals_sweep() -> None:
    manager, store, _, clock, _ = make_manager(auto_approve=False)
    store.insert(make_order(coid="old", decision_context_id="dc-1"))
    drive_to_pending(manager, store, "old")
    clock.advance(minutes=45)
    store.insert(make_order(coid="new", decision_context_id="dc-2"))
    drive_to_pending(manager, store, "new")
    expired = manager.expire_stale_approvals()
    assert [o.client_order_id for o in expired] == ["old"]
    assert store.orders["new"].state == OrderState.PENDING_APPROVAL


def test_broker_reject_is_terminal_with_reason() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="reject", reason="insufficient buying power")])
    manager, store, _, _, _ = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)
    order = manager.submit("c1")
    assert order.state == OrderState.REJECTED
    assert "buying power" in store.transition_events("c1")[-1].reason
    # Never auto-retried: still exactly one submit attempt.
    assert broker.submit_calls == 1


def test_gate_refusal_blocks_submission() -> None:
    class Refuse:
        permitted = False
        reason = "kill_switch_engaged"

    manager, store, _, _, broker = make_manager(gate=lambda _order: Refuse())
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)
    with pytest.raises(GateRefused):
        manager.submit("c1")
    assert broker.submit_calls == 0


def test_fills_partial_then_full_with_slippage() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    manager, store, trade_store, clock, _ = make_manager(
        broker, decision_price_lookup=lambda _dc, _sym: 100.0
    )
    store.insert(make_order(decision_context_id="dc-1", qty=10))
    drive_to_pending(manager, store)
    manager.submit("c1")

    t1 = manager.record_fill("c1", 4, 100.2, clock.now())
    assert store.orders["c1"].state == OrderState.PARTIALLY_FILLED
    assert t1.decision_price == 100.0
    assert t1.slippage_bps == pytest.approx(20.0)  # buy filled 20 bps above decision

    manager.record_fill("c1", 6, 100.1, clock.now())
    assert store.orders["c1"].state == OrderState.FILLED
    assert trade_store.filled_qty("c1") == 10


def test_day_end_cancels_partial_remainder() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="partial", fill_qty=3)])
    manager, store, _, clock, _ = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1", qty=10))
    drive_to_pending(manager, store)
    manager.submit("c1")
    manager.record_fill("c1", 3, 100.0, clock.now())
    cancelled = manager.day_end()
    assert [o.client_order_id for o in cancelled] == ["c1"]
    assert store.orders["c1"].state == OrderState.CANCELLED
    assert store.transition_events("c1")[-1].reason == "day_end_partial_cancel"
    with pytest.raises(BrokerRejection):  # broker order gone too
        broker._require("missing")


def test_marketable_limit_price_offsets() -> None:
    # Offset = max(2 bps, 0.25 x spread); buys above, sells below.
    assert marketable_limit_price(OrderSide.BUY, 100.0, 0.0) == pytest.approx(100.02)
    assert marketable_limit_price(OrderSide.SELL, 100.0, 0.0) == pytest.approx(99.98)
    assert marketable_limit_price(OrderSide.BUY, 100.0, 0.40) == pytest.approx(100.10)


def test_slippage_sign_convention() -> None:
    assert slippage_bps(OrderSide.BUY, 101.0, 100.0) == pytest.approx(100.0)
    assert slippage_bps(OrderSide.SELL, 99.0, 100.0) == pytest.approx(100.0)
    assert slippage_bps(OrderSide.SELL, 101.0, 100.0) == pytest.approx(-100.0)
