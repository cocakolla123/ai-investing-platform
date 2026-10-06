"""The nightly equity path, driven end to end on fakes (docs 07/09).

Before this suite the orchestration tests only asserted that asset *names*
existed, and the equity op had never been executed by anything. It would have
raised ``IllegalTransition`` on its first order of the night, every night.

Nothing here touches a database, a network, or a real broker: the repositories
are swapped for in-memory stores and the broker is ``MockBroker``. The headline
test is :func:`test_nightly_equity_path_target_to_position` — target portfolio
in, submitted order, fill, position out.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orchestration.assets.portfolio_exec as pexec
import pandas as pd
import pytest
from dagster import build_op_context
from orchestration.assets.portfolio_exec import (
    _equities_sticky_halt_active,
    _ladder_halt_active,
    execute_paper_orders,
    poll_equity_fills,
)

from dynamo.config.settings import DynamoEnv, Settings
from dynamo.domain.models.portfolio import (
    ORDER_TRANSITIONS,
    OrderSide,
    OrderState,
    Position,
    TargetPortfolio,
    TargetWeight,
    Trade,
)
from dynamo.domain.models.research import Sleeve
from dynamo.execution.gate import GateReason
from dynamo.execution.mock_broker import MockBroker, ScriptStep
from tests.unit.orchestration.conftest import (
    _STATE,
    FakeDynamo,
    FakeRepo,
    FakeState,
    make_bars,
)

TODAY = datetime.now(UTC).date()
EQUITY = 100_000.0


# --------------------------------------------------------------------------
# In-memory stands-in for the persistence layer
# --------------------------------------------------------------------------


class MemoryOrders:
    """``OrderRepository`` in a dict — one event row per transition, as SQL does."""

    def __init__(self) -> None:
        self.orders: dict[str, Any] = {}
        self.events: list[tuple[str, str | None, str, str]] = []

    def insert(self, order: Any) -> None:
        self.orders[order.client_order_id] = order
        self.events.append((order.client_order_id, None, order.state.value, "created"))

    def get(self, coid: str) -> Any | None:
        return self.orders.get(coid)

    def open_orders(self) -> list[Any]:
        from dynamo.domain.models.portfolio import TERMINAL_ORDER_STATES

        return [o for o in self.orders.values() if o.state not in TERMINAL_ORDER_STATES]

    def set_broker_order_id(self, coid: str, broker_order_id: str) -> None:
        self.orders[coid] = self.orders[coid].model_copy(
            update={"broker_order_id": broker_order_id}
        )

    def set_decision_context(self, coid: str, decision_context_id: str) -> None:
        self.orders[coid] = self.orders[coid].model_copy(
            update={"decision_context_id": decision_context_id}
        )

    def transition_with_event(
        self,
        coid: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str,
        ts: datetime,
        payload: Any = None,
    ) -> Any:
        self.orders[coid] = self.orders[coid].model_copy(update={"state": to_state})
        self.events.append((coid, from_state.value, to_state.value, reason))
        return self.orders[coid]

    def state_entered_at(self, coid: str) -> datetime | None:
        return None

    def path(self, coid: str) -> list[str]:
        """The states this order actually walked through, in order."""
        return [to for c, _from, to, _reason in self.events if c == coid]


class MemoryTrades:
    def __init__(self) -> None:
        self.trades: list[Trade] = []

    def insert(self, trade: Trade) -> None:
        self.trades.append(trade)

    def filled_qty(self, coid: str) -> float:
        return sum(t.qty for t in self.trades if t.client_order_id == coid)


class MemoryCashLedger:
    """CashLedgerRepository over MemoryTrades: opening cash EQUITY plus every
    recorded fill, the same arithmetic as the real ledger."""

    def __init__(self, trades: MemoryTrades, opening_cash: float = EQUITY) -> None:
        self._trades = trades
        self.opening_cash = opening_cash

    def cash(self, as_of: datetime | None = None) -> float:
        from dynamo.execution.cash_ledger import Fill, ledger_cash

        return ledger_cash(
            self.opening_cash,
            (
                Fill(t.side.value, t.qty, t.fill_price, t.fees)
                for t in self._trades.trades
                if as_of is None or t.filled_at <= as_of
            ),
        )


class MemoryPositions:
    """``PositionRepository`` — the same upsert/close arithmetic, in memory."""

    def __init__(self, seed: Sequence[Position] = ()) -> None:
        self.rows: dict[str, Position] = {p.symbol: p for p in seed}
        self.applied: list[Trade] = []

    def apply_fill(self, trade: Trade, now: datetime) -> None:
        self.applied.append(trade)
        signed = trade.qty if trade.side == OrderSide.BUY else -trade.qty
        current = self.rows.get(trade.symbol)
        if current is None:
            if signed > 0:
                self.rows[trade.symbol] = Position(
                    symbol=trade.symbol, qty=signed, avg_entry_price=trade.fill_price
                )
            return
        new_qty = current.qty + signed
        if new_qty <= 1e-9:
            del self.rows[trade.symbol]
            return
        avg = (
            (current.avg_entry_price * current.qty + trade.fill_price * signed) / new_qty
            if signed > 0
            else current.avg_entry_price
        )
        self.rows[trade.symbol] = current.model_copy(
            update={"qty": new_qty, "avg_entry_price": avg}
        )

    def all_positions(self) -> list[Position]:
        return list(self.rows.values())


class MemoryDecisionContexts:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def append(self, dc: Any) -> str:
        self.rows.append(dc)
        return dc.id


class MemoryReconciliation:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.acks: list[int] = []

    def record_run(
        self, ts: datetime, scope: str, result: str, divergences: Sequence[dict[str, Any]]
    ) -> int:
        run_id = len(self.runs) + 1
        self.runs.append({"id": run_id, "scope": scope, "result": result})
        return run_id

    def record_ack(self, run_id: int, acked_by: str, acked_at: datetime, note: str) -> None:
        self.acks.append(run_id)

    def unacked_divergent_runs(self) -> list[dict[str, Any]]:
        return [r for r in self.runs if r["result"] == "divergent" and r["id"] not in self.acks]

    def halt(self, scope: str = "equities:symbols") -> int:
        return self.record_run(datetime.now(UTC), scope, "divergent", [])


class MemoryLadderEvents:
    """In-memory ``ladder_events``: an unacked ack-requiring row is a halt."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> int:
        event_id = len(self.events) + 1
        self.events.append({"id": event_id, "acked_at": None, **kwargs})
        return event_id

    def unacked_blocking(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("requires_ack") and e.get("acked_at") is None]

    def is_halted(self) -> bool:
        return bool(self.unacked_blocking())

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(reversed(self.events))[:limit]

    def acknowledge(self, event_id: int, acked_by: str, note: str, now: Any = None) -> bool:
        for e in self.events:
            if e["id"] == event_id and e["acked_at"] is None:
                e.update(acked_at=now or datetime.now(UTC), acked_by=acked_by, ack_note=note)
                return True
        return False

    def halt(self, rung: str = "dd8") -> int:
        return self.record(
            as_of=TODAY,
            from_rung="normal",
            to_rung=rung,
            drawdown=-0.09,
            requires_ack=True,
            kill_switch_engaged=False,
        )


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

_SETTINGS: dict[str, Settings] = {}


class ExecDynamo(FakeDynamo):
    """FakeDynamo with an engine sentinel and per-test execution Settings."""

    settings_key: str = ""

    def settings(self) -> Settings:
        return _SETTINGS[self.settings_key]

    def engine(self) -> Any:
        return "engine-sentinel"  # every repository is faked; nothing dereferences it


class Wiring:
    """Everything the op writes through, plus the broker it talks to."""

    def __init__(self) -> None:
        self.orders = MemoryOrders()
        self.trades = MemoryTrades()
        self.cash_ledger = MemoryCashLedger(self.trades)
        self.positions = MemoryPositions()
        self.decision_contexts = MemoryDecisionContexts()
        self.reconciliation = MemoryReconciliation()
        self.ladder_events = MemoryLadderEvents()
        self.broker = MockBroker(script=[ScriptStep(behavior="accept") for _ in range(20)])
        self.dynamo: Any = None
        self.settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]


def _target(*weights: tuple[str, float]) -> TargetPortfolio:
    return TargetPortfolio(
        id=f"tp-{TODAY.isoformat()}",
        as_of=datetime.now(UTC),
        risk_verdict_ref=f"risk-{TODAY.isoformat()}",
        weights=tuple(
            TargetWeight(symbol=s, weight=w, sleeve=Sleeve.LONG_TERM, verdict_id=f"v-{s}")
            for s, w in weights
        ),
        cash=1.0 - sum(w for _s, w in weights),
    )


def _fake_read_df(target: TargetPortfolio) -> Any:
    """``_read_df`` for the three tables the nightly op reads."""

    def read_df(_dynamo: Any, stmt: Any) -> pd.DataFrame:
        sql = str(stmt)
        if "target_portfolios" in sql:
            return pd.DataFrame(
                [
                    {
                        "id": target.id,
                        "as_of": target.as_of,
                        "payload": target.model_dump(mode="json"),
                    }
                ]
            )
        if "equity_curve" in sql:
            return pd.DataFrame([{"as_of": TODAY, "equity": EQUITY, "cash": EQUITY}])
        cols = [c.key for c in stmt.selected_columns]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    return read_df


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    import dynamo.persistence.repositories.execution as exec_repos
    import dynamo.persistence.repositories.learning as learn_repos
    import dynamo.persistence.repositories.risk as risk_repos

    w = Wiring()
    w.settings = Settings(
        DYNAMO_ENV="PAPER",
        ALPACA_API_KEY="paper-key",
        ALPACA_API_SECRET="paper-secret",
        DYNAMO_KILL_SWITCH_PATH=str(tmp_path / "KILL_SWITCH"),
        DYNAMO_APPROVALS_DIR=str(tmp_path / "approvals"),
        _env_file=None,  # type: ignore[call-arg]
    )

    monkeypatch.setattr(exec_repos, "OrderRepository", lambda _e: w.orders)
    monkeypatch.setattr(exec_repos, "TradeRepository", lambda _e: w.trades)
    monkeypatch.setattr(exec_repos, "PositionRepository", lambda _e: w.positions)
    monkeypatch.setattr(exec_repos, "CashLedgerRepository", lambda _e: w.cash_ledger)
    monkeypatch.setattr(exec_repos, "ReconciliationRepository", lambda _e: w.reconciliation)
    monkeypatch.setattr(learn_repos, "DecisionContextRepository", lambda _e: w.decision_contexts)
    monkeypatch.setattr(risk_repos, "LadderEventRepository", lambda _e: w.ladder_events)
    monkeypatch.setattr(pexec, "_broker", lambda _d: w.broker)

    key = uuid.uuid4().hex
    _STATE[key] = FakeState(repo=FakeRepo(make_bars(["AAA", "BBB"], TODAY), []))
    _SETTINGS[key] = w.settings
    w.dynamo = ExecDynamo(state_key=key, settings_key=key)
    yield w
    _STATE.pop(key, None)
    _SETTINGS.pop(key, None)


def _run(w: Any, target: TargetPortfolio, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(pexec, "_read_df", _fake_read_df(target))
    ctx = build_op_context()
    execute_paper_orders(ctx, w.dynamo)
    return ctx


# --------------------------------------------------------------------------
# D1 — the illegal transition that killed order #1
# --------------------------------------------------------------------------


def test_proposed_cannot_jump_straight_to_pending_approval() -> None:
    """The transition the op used to attempt simply does not exist."""
    assert OrderState.PENDING_APPROVAL not in ORDER_TRANSITIONS[OrderState.PROPOSED]
    assert ORDER_TRANSITIONS[OrderState.PROPOSED] == frozenset(
        {OrderState.RISK_CHECKED, OrderState.REJECTED}
    )


def test_the_nightly_op_walks_the_full_legal_state_path(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)

    assert len(wiring.orders.orders) == 1
    coid = next(iter(wiring.orders.orders))
    assert wiring.orders.path(coid) == [
        OrderState.PROPOSED.value,
        OrderState.RISK_CHECKED.value,
        OrderState.PENDING_APPROVAL.value,
        OrderState.SUBMITTED.value,
    ]


def test_every_order_in_a_multi_name_batch_is_submitted(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old bug aborted the whole batch on the first order."""
    _run(wiring, _target(("AAA", 0.10), ("BBB", 0.08)), monkeypatch)

    assert len(wiring.orders.orders) == 2
    assert all(o.state == OrderState.SUBMITTED for o in wiring.orders.orders.values())
    assert wiring.broker.submit_calls == 2


# --------------------------------------------------------------------------
# D2 — startup reconciliation
# --------------------------------------------------------------------------


def test_submission_without_startup_reconciliation_is_blocked() -> None:
    """The guard the nightly path used to walk straight into."""
    from dynamo.execution.errors import ReconciliationPending
    from dynamo.execution.order_manager import OrderManager
    from tests.unit.execution.fakes import FakeClock
    from tests.unit.execution.helpers import make_order

    orders, trades = MemoryOrders(), MemoryTrades()
    manager = OrderManager(MockBroker(), orders, trades, FakeClock())  # type: ignore[arg-type]
    orders.insert(make_order(decision_context_id="dc-1"))
    manager.transition("c1", OrderState.RISK_CHECKED, "risk ok")
    manager.request_approval("c1")

    with pytest.raises(ReconciliationPending):
        manager.submit("c1")

    manager.startup_reconciliation()
    assert manager.submit("c1").state == OrderState.SUBMITTED


def test_the_nightly_op_reconciles_before_it_submits(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fill that landed while we were down is recorded before new orders go out."""
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    coid = next(iter(wiring.orders.orders))
    wiring.broker.force_fill(coid, None, 100.0)

    # Second night: the same intent is already terminal, and startup
    # reconciliation picks the downtime fill up rather than tripping on it.
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.trades.trades, "the downtime fill was recovered"
    assert wiring.orders.orders[coid].state == OrderState.FILLED


# --------------------------------------------------------------------------
# D3 — the live gate, per submission
# --------------------------------------------------------------------------


def test_live_mode_without_an_approval_submits_nothing(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent refusals: LIVE never auto-approves, and the gate says no."""
    live = wiring.settings.model_copy(update={"dynamo_env": DynamoEnv.LIVE})
    _SETTINGS[wiring.dynamo.settings_key] = live
    assert not live.auto_approve, "LIVE must never self-approve"

    _run(wiring, _target(("AAA", 0.10), ("BBB", 0.08)), monkeypatch)

    assert wiring.broker.submit_calls == 0
    # Refused, not silently dropped: both orders wait for a human.
    assert all(o.state == OrderState.PENDING_APPROVAL for o in wiring.orders.orders.values())
    # And had one been approved, the wired gate would still refuse it: no
    # approval artifact exists on disk for today.
    gate = pexec._equity_gate(live)
    result = gate(next(iter(wiring.orders.orders.values())))
    assert not result.permitted
    assert result.reason == GateReason.APPROVAL_MISSING


def test_the_wired_gate_permits_paper_and_refuses_under_the_kill_switch(
    wiring: Any,
) -> None:
    """The production closure, not a stand-in: same function, both answers."""
    from tests.unit.execution.helpers import make_order

    order = make_order(symbol="AAA")
    gate = pexec._equity_gate(wiring.settings)
    assert gate(order).reason == GateReason.PAPER_PERMITTED

    Path(wiring.settings.kill_switch_path).write_text("engaged\n", encoding="utf-8")
    # Same closure, re-read from disk — never cached at construction.
    assert gate(order).reason == GateReason.KILL_SWITCH_ENGAGED


def test_a_kill_switch_thrown_mid_job_blocks_the_rest_of_the_batch(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proof the gate is per submission, not once per job."""
    switch = Path(wiring.settings.kill_switch_path)
    real_submit = wiring.broker.submit

    def submit_then_engage(order: Any) -> str:
        broker_order_id = real_submit(order)
        switch.write_text("engaged_by=owner\n", encoding="utf-8")
        return broker_order_id

    monkeypatch.setattr(wiring.broker, "submit", submit_then_engage)
    _run(wiring, _target(("AAA", 0.10), ("BBB", 0.08)), monkeypatch)

    assert wiring.broker.submit_calls == 1, "only the first order got through"
    states = sorted(o.state.value for o in wiring.orders.orders.values())
    assert states == [OrderState.PENDING_APPROVAL.value, OrderState.SUBMITTED.value]


def test_a_kill_switch_engaged_up_front_stops_the_job(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    Path(wiring.settings.kill_switch_path).write_text("engaged\n", encoding="utf-8")
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)

    assert wiring.orders.orders == {}
    assert wiring.broker.submit_calls == 0


# --------------------------------------------------------------------------
# D4 — fills reach trades AND positions
# --------------------------------------------------------------------------


def test_poll_equity_fills_writes_trades_and_positions(wiring: Any) -> None:
    from tests.unit.execution.helpers import make_order

    order = make_order(coid="c1", symbol="AAA", qty=10.0, decision_context_id="dc-1")
    wiring.orders.insert(order)
    wiring.orders.orders["c1"] = order.model_copy(update={"state": OrderState.SUBMITTED})
    wiring.broker.submit(order)
    wiring.broker.force_fill("c1", 10.0, 250.0)

    ctx = build_op_context()
    poll_equity_fills(ctx, wiring.dynamo)

    assert [t.qty for t in wiring.trades.trades] == [10.0]
    assert wiring.positions.rows["AAA"].qty == 10.0
    assert wiring.positions.rows["AAA"].avg_entry_price == 250.0


def test_poll_equity_fills_is_idempotent_across_polls(wiring: Any) -> None:
    from tests.unit.execution.helpers import make_order

    order = make_order(coid="c1", symbol="AAA", qty=10.0, decision_context_id="dc-1")
    wiring.orders.insert(order)
    wiring.orders.orders["c1"] = order.model_copy(update={"state": OrderState.SUBMITTED})
    wiring.broker.submit(order)
    wiring.broker.force_fill("c1", 4.0, 100.0)

    ctx = build_op_context()
    poll_equity_fills(ctx, wiring.dynamo)
    poll_equity_fills(ctx, wiring.dynamo)
    assert [t.qty for t in wiring.trades.trades] == [4.0]
    assert wiring.positions.rows["AAA"].qty == 4.0

    wiring.broker.force_fill("c1", 6.0, 100.0)
    poll_equity_fills(ctx, wiring.dynamo)
    assert sum(t.qty for t in wiring.trades.trades) == 10.0
    assert wiring.positions.rows["AAA"].qty == 10.0
    assert wiring.orders.orders["c1"].state == OrderState.FILLED


def test_poll_equity_fills_skips_without_broker_keys(wiring: Any) -> None:
    _SETTINGS[wiring.dynamo.settings_key] = wiring.settings.model_copy(
        update={"alpaca_api_key": "", "alpaca_api_secret": ""}
    )
    ctx = build_op_context()
    poll_equity_fills(ctx, wiring.dynamo)
    assert wiring.trades.trades == []


# --------------------------------------------------------------------------
# D5 — the sticky halt is read, and only an ack clears it
# --------------------------------------------------------------------------


def test_an_unacked_divergent_run_halts_the_nightly_job(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiring.reconciliation.halt()
    assert _equities_sticky_halt_active(wiring.dynamo)

    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.orders.orders == {}
    assert wiring.broker.submit_calls == 0


def test_acknowledging_the_run_lets_trading_resume(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dynamo.execution.reconciliation import acknowledge_reconciliation

    run_id = wiring.reconciliation.halt()
    acknowledge_reconciliation(
        wiring.reconciliation, run_id, "owner", "positions squared by hand", datetime.now(UTC)
    )

    assert not _equities_sticky_halt_active(wiring.dynamo)
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.broker.submit_calls == 1


def test_a_pm_or_options_halt_does_not_stop_the_equity_sleeve(wiring: Any) -> None:
    wiring.reconciliation.halt("pm:positions")
    wiring.reconciliation.halt("options:positions")
    assert not _equities_sticky_halt_active(wiring.dynamo)


# --------------------------------------------------------------------------
# The deliverable: target portfolio -> order -> fill -> position
# --------------------------------------------------------------------------


def test_nightly_equity_path_target_to_position(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One night, start to finish, with nothing stubbed but the edges.

    Target portfolio in; a marketable-limit order proposed, risk-checked,
    approved and submitted; the broker fills it; the poller records the trade
    and applies it to positions. Every one of the five defects sat somewhere on
    this line.
    """
    target = _target(("AAA", 0.10))
    _run(wiring, target, monkeypatch)

    # --- proposal through submission ---
    assert len(wiring.orders.orders) == 1
    coid, order = next(iter(wiring.orders.orders.items()))
    assert order.symbol == "AAA"
    assert order.side == OrderSide.BUY
    assert order.state == OrderState.SUBMITTED
    assert order.limit_price is not None and order.limit_price > 0
    assert order.broker_order_id, "the broker acknowledged it"

    # --- the decision context that must exist before any order ---
    assert [dc.symbol for dc in wiring.decision_contexts.rows] == ["AAA"]
    dc = wiring.decision_contexts.rows[0]
    assert dc.target_portfolio_id == target.id
    assert dc.payload["risk_verdict_ref"] == target.risk_verdict_ref
    assert order.decision_context_id == dc.id

    # --- the order is sized to the target weight at the decision price ---
    price = dc.payload["decision_price"]
    assert order.qty == pytest.approx(0.10 * EQUITY / price, rel=1e-3)

    # --- the fill comes back and lands where the platform can see it ---
    wiring.broker.force_fill(coid, None, price)
    ctx = build_op_context()
    poll_equity_fills(ctx, wiring.dynamo)

    assert wiring.orders.orders[coid].state == OrderState.FILLED
    assert [t.symbol for t in wiring.trades.trades] == ["AAA"]
    assert wiring.trades.trades[0].qty == pytest.approx(order.qty)
    assert wiring.positions.rows["AAA"].qty == pytest.approx(order.qty)

    # --- and a re-poll of the same night changes nothing ---
    poll_equity_fills(ctx, wiring.dynamo)
    assert len(wiring.trades.trades) == 1


def test_a_buy_fill_moves_cash_not_equity_and_reconciles_clean(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the 2026-09-24 halt.

    Internal cash used to be the latest ``equity_curve`` row, which the nightly
    snapshot wrote back unchanged, so the first paper fill (SPY) left cash at
    $100,000 while the broker showed $5,409. Equity counted the position on top
    of untouched cash, every reconciliation diverged, and the sticky halt
    stopped all trading. Cash now comes from the ledger: opening cash plus
    fills.
    """
    from orchestration.assets.portfolio_exec import _portfolio_state, reconcile_broker_state

    target = _target(("AAA", 0.10))
    _run(wiring, target, monkeypatch)
    coid, order = next(iter(wiring.orders.orders.items()))
    price = wiring.decision_contexts.rows[0].payload["decision_price"]
    wiring.broker.force_fill(coid, None, price)
    poll_equity_fills(build_op_context(), wiring.dynamo)
    # Mark at the fill price, as the broker does, so equity is comparable.
    monkeypatch.setattr(pexec, "_latest_prices", lambda _d, _s, _a: {"AAA": price})

    state = _portfolio_state(wiring.dynamo)
    assert state is not None
    notional = order.qty * price
    assert state.cash == pytest.approx(EQUITY - notional)
    assert state.equity == pytest.approx(EQUITY)
    assert state.cash == pytest.approx(wiring.broker.account()["cash"])

    reconcile_broker_state(build_op_context(), wiring.dynamo)
    assert [r["result"] for r in wiring.reconciliation.runs][-1] != "divergent"
    assert wiring.reconciliation.unacked_divergent_runs() == []
    assert not _equities_sticky_halt_active(wiring.dynamo)


def test_portfolio_state_carries_the_high_water_mark(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, the risk engine used today's equity as the HWM: drawdown was 0
    every night and the ladder could never move."""
    from orchestration.assets.portfolio_exec import _portfolio_state

    def read_df(peak: float | None) -> Any:
        def _read(_dynamo: Any, _stmt: Any) -> pd.DataFrame:
            rows = [] if peak is None else [{"equity": peak}]
            return pd.DataFrame(rows, columns=["equity"])

        return _read

    monkeypatch.setattr(pexec, "_read_df", read_df(110_000.0))
    state = _portfolio_state(wiring.dynamo)
    assert state is not None and state.high_water_mark == pytest.approx(110_000.0)

    monkeypatch.setattr(pexec, "_read_df", read_df(90_000.0))  # today is the high
    state = _portfolio_state(wiring.dynamo)
    assert state is not None and state.high_water_mark == pytest.approx(EQUITY)

    monkeypatch.setattr(pexec, "_read_df", read_df(None))  # no snapshot yet
    state = _portfolio_state(wiring.dynamo)
    assert state is not None and state.high_water_mark == pytest.approx(EQUITY)


# --------------------------------------------------------------------------
# Drawdown-ladder acks (doc 06 §3) — RungEffects.requires_human_ack had no
# reader at all before this: DD8/DD15 changed nothing in the live path.
# --------------------------------------------------------------------------


def test_an_unacked_ack_requiring_rung_blocks_every_order(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    wiring.ladder_events.halt("dd8")
    assert _ladder_halt_active(wiring.dynamo) is True
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.broker.submit_calls == 0


def test_acknowledging_the_rung_lets_trading_resume(
    wiring: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_id = wiring.ladder_events.halt("dd15")
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.broker.submit_calls == 0

    wiring.ladder_events.acknowledge(event_id, "owner", "reviewed the drawdown")
    assert _ladder_halt_active(wiring.dynamo) is False
    _run(wiring, _target(("AAA", 0.10)), monkeypatch)
    assert wiring.broker.submit_calls == 1


def test_a_rung_that_needs_no_ack_does_not_halt(wiring: Any) -> None:
    """DD5 warns; it does not require a human before the next order."""
    wiring.ladder_events.record(
        as_of=TODAY,
        from_rung="normal",
        to_rung="dd5",
        drawdown=-0.06,
        requires_ack=False,
        kill_switch_engaged=False,
    )
    assert _ladder_halt_active(wiring.dynamo) is False
