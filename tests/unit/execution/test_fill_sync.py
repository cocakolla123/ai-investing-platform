"""Equity fill polling (doc 09 §5): broker truth -> trades -> positions.

Nothing polled a live equity order after submission, so a real Alpaca fill
never reached ``trades`` and the periodic reconciler halted the sleeve for a
divergence it had itself created. These tests pin the poller's contract:
partials top up, terminal broker states are mirrored, and re-polling an
unchanged order writes nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dynamo.domain.models.portfolio import OrderSide, OrderState, Trade
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from dynamo.persistence.repositories.execution import closed_trades_from_fills
from tests.unit.execution.helpers import drive_to_pending, make_manager, make_order


def _submitted(broker: MockBroker | None = None, qty: float = 10.0):  # type: ignore[no-untyped-def]
    """One order sitting at the broker in ``new`` — accepted, not yet filled."""
    broker = broker or MockBroker(script=[ScriptStep(behavior="accept")])
    manager, store, trades, clock, broker = make_manager(broker)
    store.insert(make_order(decision_context_id="dc-1", qty=qty))
    drive_to_pending(manager, store)
    manager.submit("c1")
    return manager, store, trades, clock, broker


def test_sync_orders_records_a_fill_into_trades() -> None:
    manager, store, trades, _clock, broker = _submitted()
    assert trades.trades == []

    broker.force_fill("c1", 10.0, 101.5)
    recorded = manager.sync_orders()

    assert [t.qty for t in recorded] == [10.0]
    assert recorded[0].fill_price == 101.5
    assert recorded[0].symbol == "AAPL"
    assert len(trades.trades) == 1
    assert store.get("c1") is not None
    assert store.orders["c1"].state == OrderState.FILLED


def test_sync_orders_handles_a_partial_then_tops_it_up() -> None:
    manager, store, trades, _clock, broker = _submitted()

    broker.force_fill("c1", 4.0, 100.0)
    first = manager.sync_orders()
    assert [t.qty for t in first] == [4.0]
    assert store.orders["c1"].state == OrderState.PARTIALLY_FILLED

    broker.force_fill("c1", 6.0, 102.0)
    second = manager.sync_orders()
    # Only the INCREMENT is written — cumulative broker qty minus what we hold.
    assert [t.qty for t in second] == [6.0]
    assert store.orders["c1"].state == OrderState.FILLED
    assert sum(t.qty for t in trades.trades) == 10.0


def test_re_polling_an_unchanged_order_is_idempotent() -> None:
    manager, _store, trades, _clock, broker = _submitted()
    broker.force_fill("c1", 3.0, 100.0)

    manager.sync_orders()
    before = len(trades.trades)
    assert manager.sync_orders() == []
    assert manager.sync_orders() == []
    assert len(trades.trades) == before


def test_sync_orders_mirrors_terminal_broker_states() -> None:
    for status, expected in (
        ("cancel", OrderState.CANCELLED),
        ("reject", OrderState.REJECTED),
    ):
        manager, store, _trades, _clock, broker = _submitted()
        if status == "cancel":
            broker.cancel("c1")
        else:
            broker._require("c1").status = "rejected"
        manager.sync_orders()
        assert store.orders["c1"].state == expected


def test_sync_orders_cancels_an_order_the_broker_never_saw() -> None:
    """A lost submit ack: the intent re-enters the next scheduling bucket."""
    manager, store, _trades, _clock, broker = _submitted()
    broker._orders.pop("c1")

    assert manager.sync_orders() == []
    assert store.orders["c1"].state == OrderState.CANCELLED


def test_sync_orders_ignores_orders_that_never_reached_the_broker() -> None:
    manager, store, _trades, _clock, _broker = make_manager()[:5]
    store.insert(make_order(coid="c9", decision_context_id="dc-9"))

    assert manager.sync_orders() == []
    assert store.orders["c9"].state == OrderState.PROPOSED


def test_startup_reconciliation_still_repairs_and_unblocks_submission() -> None:
    """The refactor kept startup_reconciliation's job: sync, then unblock."""
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    manager, _store, _trades, _clock, _broker = make_manager(broker, reconciled=False)
    assert not manager.reconciled

    manager.startup_reconciliation()
    assert manager.reconciled


def test_recorded_fills_round_trip_through_position_math() -> None:
    """The trades the poller writes are the ones P&L is computed from."""
    manager, _store, trades, _clock, broker = _submitted()
    broker.force_fill("c1", 10.0, 100.0)
    manager.sync_orders()

    exit_fill = Trade(
        client_order_id="c2",
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=10.0,
        fill_price=110.0,
        filled_at=datetime(2026, 8, 4, 15, 0, tzinfo=UTC),
    )
    closed = closed_trades_from_fills([*trades.trades, exit_fill])
    assert len(closed) == 1
    assert closed[0].realized_pnl == 100.0
