"""Stop maintenance (doc 09 §3): broker-held replace / cancel+resubmit, and
synthetic stop breach → exit order through the FULL state machine."""

from __future__ import annotations

from dynamo.domain.models.portfolio import OrderState, OrderType, Position
from dynamo.domain.models.risk import StopKind, StopOrderSpec
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from dynamo.execution.stops_maintenance import StopsMaintainer
from tests.unit.execution.fakes import FakeOrderStore
from tests.unit.execution.helpers import make_manager

POSITIONS = {"AAPL": Position(symbol="AAPL", qty=10, avg_entry_price=100.0)}
DC_IDS = {"AAPL": "dc-stop-1"}


def _maintainer(broker: MockBroker) -> tuple[StopsMaintainer, FakeOrderStore, MockBroker]:
    manager, store, _, clock, _ = make_manager(broker)
    return StopsMaintainer(broker=broker, manager=manager, clock=clock), store, broker


def _spec(price: float, kind: StopKind = StopKind.TRAILING) -> StopOrderSpec:
    return StopOrderSpec(symbol="AAPL", kind=kind, stop_price=price, reason="risk engine")


def test_new_broker_held_stop_goes_through_full_state_machine() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    maintainer, store, _ = _maintainer(broker)
    created = maintainer.apply_stops([_spec(92.0)], POSITIONS, DC_IDS)
    assert len(created) == 1
    order = created[0]
    assert order.order_type == OrderType.STOP
    assert order.state == OrderState.SUBMITTED
    events = [(e.from_state, e.to_state) for e in store.transition_events(order.client_order_id)]
    assert events == [
        ("proposed", "risk_checked"),
        ("risk_checked", "pending_approval"),
        ("pending_approval", "submitted"),
    ]
    assert maintainer.broker_held["AAPL"] == order.client_order_id


def test_stop_refresh_uses_replace_in_place() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    maintainer, _, _ = _maintainer(broker)
    first = maintainer.apply_stops([_spec(92.0)], POSITIONS, DC_IDS)[0]
    # Trailing ratchet: replace, same client-order lineage, no new order.
    updated = maintainer.apply_stops([_spec(95.0)], POSITIONS, DC_IDS)
    assert updated == []
    assert maintainer.broker_held["AAPL"] == first.client_order_id
    assert broker.get_order(first.client_order_id)["stop_price"] == 95.0
    assert broker.submit_calls == 1


def test_failed_replace_cancels_then_resubmits_single_stop() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    maintainer, store, _ = _maintainer(broker)
    first = maintainer.apply_stops([_spec(92.0)], POSITIONS, DC_IDS)[0]

    broker.push(ScriptStep(behavior="reject", reason="replace unsupported"))  # replace fails
    broker.push(ScriptStep(behavior="accept"))  # resubmission succeeds
    replacement = maintainer.apply_stops([_spec(95.0)], POSITIONS, DC_IDS)[0]

    assert replacement.client_order_id != first.client_order_id
    assert store.orders[first.client_order_id].state == OrderState.CANCELLED
    assert replacement.state == OrderState.SUBMITTED
    assert maintainer.broker_held["AAPL"] == replacement.client_order_id
    # Never two live stops for one position.
    open_stops = [o for o in broker.open_orders() if o["order_type"] == "stop"]
    assert len(open_stops) == 1
    assert open_stops[0]["stop_price"] == 95.0


def test_synthetic_stop_breach_creates_exit_through_full_machine() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    maintainer, store, _ = _maintainer(broker)
    exits = maintainer.evaluate_synthetic_stops(
        [_spec(95.0, StopKind.THESIS)], {"AAPL": 94.5}, POSITIONS, DC_IDS
    )
    assert len(exits) == 1
    order = exits[0]
    assert order.order_type == OrderType.SYNTHETIC_STOP_EXIT
    assert order.qty == 10  # full position exit
    assert order.limit_price is not None and order.limit_price < 94.5  # marketable sell
    assert order.state == OrderState.SUBMITTED
    events = [(e.from_state, e.to_state) for e in store.transition_events(order.client_order_id)]
    assert events == [
        ("proposed", "risk_checked"),
        ("risk_checked", "pending_approval"),
        ("pending_approval", "submitted"),
    ]


def test_synthetic_stop_no_breach_no_order() -> None:
    broker = MockBroker()
    maintainer, _store, _ = _maintainer(broker)
    exits = maintainer.evaluate_synthetic_stops(
        [_spec(95.0, StopKind.THESIS)], {"AAPL": 96.0}, POSITIONS, DC_IDS
    )
    assert exits == []
    assert broker.submit_calls == 0


def test_stop_for_symbol_without_position_is_skipped() -> None:
    broker = MockBroker()
    maintainer, _, _ = _maintainer(broker)
    spec = StopOrderSpec(symbol="TSLA", kind=StopKind.ATR, stop_price=200.0)
    assert maintainer.apply_stops([spec], POSITIONS, DC_IDS) == []
    assert broker.submit_calls == 0
