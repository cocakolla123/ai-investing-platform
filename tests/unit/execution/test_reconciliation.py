"""Reconciliation divergence (doc 09 §5): correct halt scope (symbol vs
global), sticky until explicit acknowledgment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from dynamo.domain.models.portfolio import OrderState, Position
from dynamo.domain.ports import Broker
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from dynamo.execution.reconciliation import GLOBAL_SCOPE, Reconciler
from tests.unit.execution.helpers import make_order

NOW = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


class FakeInternal:
    def __init__(
        self,
        positions: list[Position] | None = None,
        open_orders: list[tuple[str, OrderState, str]] | None = None,
        cash: float = 100_000.0,
        equity: float = 100_000.0,
    ) -> None:
        self._positions = positions or []
        self._open = open_orders or []
        self._cash = cash
        self._equity = equity

    def positions(self) -> list[Position]:
        return self._positions

    def open_order_ids(self) -> list[tuple[str, OrderState, str]]:
        return self._open

    def cash(self) -> float:
        return self._cash

    def equity(self) -> float:
        return self._equity


class FakeReconStore:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []

    def record_run(self, ts, scope, result, divergences) -> int:  # type: ignore[no-untyped-def]
        self.runs.append({"ts": ts, "scope": scope, "result": result, "div": list(divergences)})
        return len(self.runs)

    def record_ack(self, run_id, acked_by, acked_at, note) -> None:  # type: ignore[no-untyped-def]
        self.acks.append({"run_id": run_id, "by": acked_by, "at": acked_at, "note": note})


def _reconciler(broker: Broker, internal: FakeInternal) -> tuple[Reconciler, FakeReconStore, list]:
    store = FakeReconStore()
    alerts: list[tuple[str, str]] = []
    rec = Reconciler(
        broker, internal, store, lambda n, m: alerts.append((n, m)), cash_tolerance=1.0
    )
    return rec, store, alerts


def _broker_with_position(qty: float = 10, price: float = 100.0) -> MockBroker:
    broker = MockBroker(cash=100_000.0 + qty * price, script=[ScriptStep(behavior="fill",
                                                                         fill_price=price)])
    broker.submit(make_order(coid="seed", qty=qty))
    return broker


def test_clean_reconciliation_no_halt() -> None:
    broker = _broker_with_position()
    internal = FakeInternal(
        positions=[Position(symbol="AAPL", qty=10, avg_entry_price=100.0)],
        cash=broker.account()["cash"],
        equity=broker.account()["equity"],
    )
    rec, store, alerts = _reconciler(broker, internal)
    result = rec.reconcile(NOW)
    assert result.clean
    assert store.runs[0]["result"] == "clean"
    assert not rec.halts.is_halted()
    assert not rec.halts.is_halted("AAPL")
    assert alerts == []


def test_position_mismatch_halts_symbol_not_globally() -> None:
    broker = _broker_with_position(qty=10)
    internal = FakeInternal(
        positions=[Position(symbol="AAPL", qty=5, avg_entry_price=100.0)],  # wrong qty
        cash=broker.account()["cash"],
        equity=broker.account()["equity"],
    )
    rec, store, alerts = _reconciler(broker, internal)
    result = rec.reconcile(NOW)
    assert not result.clean
    assert rec.halts.is_halted("AAPL")
    assert not rec.halts.is_halted("MSFT")
    assert not rec.halts.is_halted()  # no global halt
    assert store.runs[0]["result"] == "divergent"
    assert alerts and alerts[0][0] == "reconciliation_divergence"


def test_cash_mismatch_halts_globally() -> None:
    broker = MockBroker(cash=100_000.0)
    internal = FakeInternal(cash=99_000.0, equity=100_000.0)  # $1000 off, beyond tolerance
    rec, _, _ = _reconciler(broker, internal)
    result = rec.reconcile(NOW)
    assert not result.clean
    assert GLOBAL_SCOPE in result.halted_scopes
    assert rec.halts.is_halted()  # everything halted
    assert rec.halts.is_halted("ANY")


def test_cash_within_tolerance_is_clean() -> None:
    broker = MockBroker(cash=100_000.0)
    internal = FakeInternal(cash=100_000.50, equity=100_000.50)  # fee rounding
    rec, _, _ = _reconciler(broker, internal)
    assert rec.reconcile(NOW).clean


def test_halt_is_sticky_until_ack() -> None:
    broker = _broker_with_position(qty=10)
    ok_account = broker.account()
    internal = FakeInternal(
        positions=[Position(symbol="AAPL", qty=5, avg_entry_price=100.0)],
        cash=ok_account["cash"],
        equity=ok_account["equity"],
    )
    rec, store, _ = _reconciler(broker, internal)
    run = rec.reconcile(NOW)
    assert rec.halts.is_halted("AAPL")

    # A later clean run does NOT quietly clear the halt.
    internal._positions = [Position(symbol="AAPL", qty=10, avg_entry_price=100.0)]
    assert rec.reconcile(NOW).clean
    assert rec.halts.is_halted("AAPL")

    # An ack without a note is refused; a real ack clears exactly this run.
    with pytest.raises(ValueError, match="note"):
        rec.acknowledge(run.run_id, "owner", " ", NOW)
    cleared = rec.acknowledge(run.run_id, "owner", "verified with broker statement", NOW)
    assert cleared == ["AAPL"]
    assert not rec.halts.is_halted("AAPL")
    assert store.acks and store.acks[0]["run_id"] == run.run_id


def test_unknown_broker_order_is_a_divergence() -> None:
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    broker.submit(make_order(coid="ghost", qty=5))  # broker knows it; we don't
    internal = FakeInternal(cash=broker.account()["cash"], equity=broker.account()["equity"])
    rec, _, _ = _reconciler(broker, internal)
    result = rec.reconcile(NOW)
    assert not result.clean
    assert any(d.kind == "unknown_order" for d in result.divergences)
    assert rec.halts.is_halted("AAPL")
