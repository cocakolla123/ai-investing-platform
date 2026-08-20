"""Halts survive the process that found them, and only a human clears them.

``HaltRegistry`` is an in-memory dict and the Dagster op builds a fresh
``Reconciler`` every 15 minutes, so an in-memory halt evaporates before the
next order is ever proposed. The durable halt is the unacknowledged
``divergent`` row; these tests pin both halves of that, plus the ack that is
the only way out.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from dynamo.domain.models.portfolio import OrderState, Position
from dynamo.execution.reconciliation import (
    GLOBAL_SCOPE,
    HaltRegistry,
    Reconciler,
    acknowledge_reconciliation,
)

NOW = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


class RunStore:
    """Stands in for ``reconciliation_runs`` + ``reconciliation_acks``.

    Deliberately shared across Reconciler instances: that is exactly what the
    real table is, and what makes a halt sticky.
    """

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []

    def record_run(
        self, ts: datetime, scope: str, result: str, divergences: Sequence[dict[str, Any]]
    ) -> int:
        run_id = len(self.runs) + 1
        self.runs.append(
            {"id": run_id, "ts": ts, "scope": scope, "result": result, "div": list(divergences)}
        )
        return run_id

    def record_ack(self, run_id: int, acked_by: str, acked_at: datetime, note: str) -> None:
        self.acks.append({"run_id": run_id, "by": acked_by, "at": acked_at, "note": note})

    def unacked_divergent_runs(self) -> list[dict[str, Any]]:
        acked = {a["run_id"] for a in self.acks}
        return [r for r in self.runs if r["result"] == "divergent" and r["id"] not in acked]


class DivergentBroker:
    """Broker truth that disagrees with internal state on one symbol."""

    def positions(self) -> list[dict[str, Any]]:
        return [{"symbol": "AAPL", "qty": 7.0, "avg_entry_price": 100.0}]

    def open_orders(self) -> list[dict[str, Any]]:
        return []

    def get_order(self, coid: str) -> dict[str, Any]:  # pragma: no cover - unused here
        return {"status": "filled"}

    def account(self) -> dict[str, Any]:
        # Cash and equity agree — the ONLY divergence here is the AAPL qty, so
        # the run is scoped to symbols rather than globally.
        return {"cash": 1_000.0, "equity": 2_000.0}

    def submit(self, order: Any) -> str:  # pragma: no cover - unused here
        raise AssertionError("reconciliation never submits")

    def cancel(self, coid: str) -> None:  # pragma: no cover - unused here
        pass

    def replace(self, coid: str, stop_price: float) -> str:  # pragma: no cover - unused
        raise AssertionError("reconciliation never replaces")


class InternalState:
    def positions(self) -> list[Position]:
        return [Position(symbol="AAPL", qty=10.0, avg_entry_price=100.0)]

    def open_order_ids(self) -> list[tuple[str, OrderState, str]]:
        return []

    def cash(self) -> float:
        return 1_000.0

    def equity(self) -> float:
        return 2_000.0


def _reconciler(store: RunStore) -> Reconciler:
    return Reconciler(
        DivergentBroker(),  # type: ignore[arg-type]
        InternalState(),
        store,
        lambda _name, _msg: None,
    )


def test_a_halt_written_by_one_reconciler_is_visible_to_a_fresh_one() -> None:
    store = RunStore()

    first = _reconciler(store)
    result = first.reconcile(NOW)
    assert not result.clean
    assert "AAPL" in result.halted_scopes

    # 15 minutes later the op builds a brand-new Reconciler: its in-memory
    # registry is empty, but the durable halt is still on the table.
    second = _reconciler(store)
    assert second.halts.active == {}
    assert [r["id"] for r in store.unacked_divergent_runs()] == [result.run_id]


def test_equity_runs_are_scope_labelled_so_sleeves_do_not_share_halts() -> None:
    store = RunStore()
    _reconciler(store).reconcile(NOW)

    scopes = [r["scope"] for r in store.unacked_divergent_runs()]
    assert scopes == ["equities:symbols"]
    assert not any(s.startswith(("pm", "options")) for s in scopes)


def test_a_global_divergence_is_labelled_global() -> None:
    class WideDivergence(DivergentBroker):
        def account(self) -> dict[str, Any]:
            return {"cash": 9_999.0, "equity": 9_999.0}

    store = RunStore()
    reconciler = Reconciler(
        WideDivergence(),  # type: ignore[arg-type]
        InternalState(),
        store,
        lambda _name, _msg: None,
    )
    reconciler.reconcile(NOW)
    assert store.unacked_divergent_runs()[0]["scope"] == "equities:global"


def test_the_ack_is_what_clears_the_halt_and_it_needs_a_note() -> None:
    store = RunStore()
    reconciler = _reconciler(store)
    result = reconciler.reconcile(NOW)
    assert store.unacked_divergent_runs()

    with pytest.raises(ValueError, match="note"):
        acknowledge_reconciliation(store, result.run_id, "owner", "   ", NOW)
    assert store.unacked_divergent_runs(), "a blank note must not clear anything"

    acknowledge_reconciliation(store, result.run_id, "owner", "manually squared off", NOW)
    assert store.unacked_divergent_runs() == []
    assert store.acks[0]["by"] == "owner"


def test_reconciler_acknowledge_clears_only_that_runs_scopes() -> None:
    store = RunStore()
    reconciler = _reconciler(store)
    result = reconciler.reconcile(NOW)
    reconciler.halts.engage("MSFT", 999)  # an older, unrelated halt

    cleared = reconciler.acknowledge(result.run_id, "owner", "reviewed", NOW)
    assert cleared == ["AAPL"]
    assert reconciler.halts.is_halted("MSFT")


def test_a_clean_run_never_clears_an_outstanding_halt() -> None:
    """Nothing "re-checks and quietly resumes" — the divergent row stays unacked."""
    store = RunStore()
    _reconciler(store).reconcile(NOW)

    class Agreeing(DivergentBroker):
        def positions(self) -> list[dict[str, Any]]:
            return [{"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0}]

    clean = Reconciler(
        Agreeing(),  # type: ignore[arg-type]
        InternalState(),
        store,
        lambda _name, _msg: None,
    ).reconcile(NOW)

    assert clean.clean
    assert len(store.unacked_divergent_runs()) == 1


def test_halt_registry_alone_is_not_sticky_across_instances() -> None:
    """The regression this whole design exists to prevent."""
    a = HaltRegistry()
    a.engage(GLOBAL_SCOPE, 1)
    assert a.is_halted()
    assert not HaltRegistry().is_halted()
