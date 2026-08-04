"""Order state machine properties (doc 09 §2, §10).

No sequence of events reaches an illegal transition; terminal states are
absorbing; every applied transition writes exactly one order_events row.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dynamo.domain.models.portfolio import (
    ORDER_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    OrderState,
)
from dynamo.execution.errors import IllegalTransition
from tests.unit.execution.helpers import drive_to_pending, make_manager, make_order

ALL_STATES = list(OrderState)


@given(st.lists(st.sampled_from(ALL_STATES), min_size=1, max_size=12))
def test_random_sequences_never_reach_illegal_state(targets: list[OrderState]) -> None:
    manager, store, _, _, _ = make_manager()
    store.insert(make_order())
    applied = 0
    for target in targets:
        current = store.orders["c1"].state
        if target in ORDER_TRANSITIONS[current]:
            manager.transition("c1", target, "test")
            applied += 1
        else:
            with pytest.raises(IllegalTransition):
                manager.transition("c1", target, "test")
            # The failed attempt must not have moved the order.
            assert store.orders["c1"].state == current
    # Exactly one event row per applied transition (plus the creation row).
    assert len(store.transition_events("c1")) == applied


@pytest.mark.parametrize("from_state", ALL_STATES)
@pytest.mark.parametrize("to_state", ALL_STATES)
def test_exhaustive_transition_matrix(from_state: OrderState, to_state: OrderState) -> None:
    manager, store, _, _, _ = make_manager()
    store.insert(make_order(state=from_state))
    if to_state in ORDER_TRANSITIONS[from_state]:
        updated = manager.transition("c1", to_state, "test")
        assert updated.state == to_state
    else:
        with pytest.raises(IllegalTransition):
            manager.transition("c1", to_state, "test")


@pytest.mark.parametrize("terminal", sorted(TERMINAL_ORDER_STATES, key=str))
def test_terminal_states_are_absorbing(terminal: OrderState) -> None:
    manager, store, _, _, _ = make_manager()
    store.insert(make_order(state=terminal))
    for target in ALL_STATES:
        with pytest.raises(IllegalTransition):
            manager.transition("c1", target, "test")
    assert store.orders["c1"].state == terminal
    assert store.transition_events("c1") == []


def test_happy_path_writes_one_event_per_transition() -> None:
    manager, store, _, _, _ = make_manager()
    store.insert(make_order(decision_context_id="dc-1"))
    drive_to_pending(manager, store)
    manager.submit("c1")
    events = [(e.from_state, e.to_state) for e in store.transition_events("c1")]
    assert events == [
        ("proposed", "risk_checked"),
        ("risk_checked", "pending_approval"),
        ("pending_approval", "submitted"),
    ]
