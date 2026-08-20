"""The live gate and the sticky halt, consulted per submission (doc 09 §7).

The equity manager used to be built without a gate at all, so ``submit()``
skipped the check entirely and the sleeve relied on a single kill-switch read
per job — every order after the first went out unchecked. These tests pin the
opposite: one gate call per order, evaluated at submission time, on live state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.errors import ReconciliationHalt
from dynamo.execution.gate import (
    GateReason,
    GateResult,
    Readiness,
    kill_switch_engaged,
    live_orders_permitted,
)
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from dynamo.execution.order_manager import GateRefused
from tests.unit.execution.helpers import make_manager, make_order

PERMITTED = GateResult(True, GateReason.PAPER_PERMITTED, "env=PAPER")


def _accepting_broker(n: int) -> MockBroker:
    return MockBroker(script=[ScriptStep(behavior="accept") for _ in range(n)])


def _stage(manager: Any, store: Any, coid: str, symbol: str = "AAPL") -> None:
    store.insert(make_order(coid=coid, symbol=symbol, decision_context_id=f"dc-{coid}"))
    manager.transition(coid, OrderState.RISK_CHECKED, "risk ok")
    manager.request_approval(coid)


def test_gate_is_consulted_once_per_order_not_once_per_job() -> None:
    seen: list[str] = []

    def gate(order: Any) -> GateResult:
        seen.append(order.client_order_id)
        return PERMITTED

    manager, store, _trades, _clock, _broker = make_manager(_accepting_broker(3), gate=gate)
    for coid in ("c1", "c2", "c3"):
        _stage(manager, store, coid)
        manager.submit(coid)

    assert seen == ["c1", "c2", "c3"]


def test_live_submission_without_an_approval_is_refused() -> None:
    """LIVE + no signed approval artifact -> fail closed, nothing reaches the broker."""

    def gate(order: Any) -> GateResult:
        return live_orders_permitted(
            DynamoEnv.LIVE,
            None,  # no approval artifact on disk
            kill_switch_present=False,
            readiness=Readiness(),
            scope_target=order.symbol,
        )

    broker = _accepting_broker(1)
    manager, store, _trades, _clock, broker = make_manager(broker, gate=gate)
    _stage(manager, store, "c1")

    with pytest.raises(GateRefused) as exc:
        manager.submit("c1")

    assert GateReason.APPROVAL_MISSING in str(exc.value)
    assert broker.submit_calls == 0
    # Refused, not mangled: the order is still awaiting a human, not "submitted".
    assert store.orders["c1"].state == OrderState.PENDING_APPROVAL


def test_kill_switch_appearing_mid_job_blocks_every_later_order(tmp_path: Path) -> None:
    """The switch is re-read per submission, so it bites within the same run."""
    switch = tmp_path / "KILL_SWITCH"

    def gate(order: Any) -> GateResult:
        return live_orders_permitted(
            DynamoEnv.PAPER,
            None,
            kill_switch_engaged(switch),
            Readiness(),
            scope_target=order.symbol,
        )

    manager, store, _trades, _clock, broker = make_manager(_accepting_broker(3), gate=gate)
    for coid in ("c1", "c2", "c3"):
        _stage(manager, store, coid)

    manager.submit("c1")
    switch.write_text("engaged_by=owner\n", encoding="utf-8")  # thrown mid-batch

    for coid in ("c2", "c3"):
        with pytest.raises(GateRefused) as exc:
            manager.submit(coid)
        assert GateReason.KILL_SWITCH_ENGAGED in str(exc.value)

    assert broker.submit_calls == 1
    assert store.orders["c1"].state == OrderState.SUBMITTED
    assert store.orders["c2"].state == OrderState.PENDING_APPROVAL


def test_a_paper_submission_still_passes_through_the_same_gate() -> None:
    """PAPER short-circuits to permitted through the SAME function, not around it."""
    calls: list[GateResult] = []

    def gate(order: Any) -> GateResult:
        result = live_orders_permitted(
            DynamoEnv.PAPER, None, False, Readiness(), scope_target=order.symbol
        )
        calls.append(result)
        return result

    manager, store, _trades, _clock, broker = make_manager(_accepting_broker(1), gate=gate)
    _stage(manager, store, "c1")
    manager.submit("c1")

    assert [c.reason for c in calls] == [GateReason.PAPER_PERMITTED]
    assert broker.submit_calls == 1


def test_a_halt_arriving_mid_job_refuses_the_next_order() -> None:
    halted = {"active": False}
    manager, store, _trades, _clock, broker = make_manager(
        _accepting_broker(2), halt_check=lambda _order: halted["active"]
    )
    for coid in ("c1", "c2"):
        _stage(manager, store, coid)

    manager.submit("c1")
    halted["active"] = True

    with pytest.raises(ReconciliationHalt):
        manager.submit("c2")
    assert broker.submit_calls == 1
    assert store.orders["c2"].state == OrderState.PENDING_APPROVAL


def test_halt_is_checked_before_the_gate_and_before_any_transition() -> None:
    """A halted sleeve does not even ask the gate — it simply does not trade."""
    gate_calls: list[str] = []

    def gate(order: Any) -> GateResult:
        gate_calls.append(order.client_order_id)
        return PERMITTED

    manager, store, _trades, _clock, broker = make_manager(
        _accepting_broker(1), gate=gate, halt_check=lambda _order: True
    )
    _stage(manager, store, "c1")

    with pytest.raises(ReconciliationHalt):
        manager.submit("c1")
    assert gate_calls == []
    assert broker.submit_calls == 0
