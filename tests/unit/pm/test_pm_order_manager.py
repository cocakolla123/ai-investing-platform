"""PM order manager: state machine reuse, gate integration, broker faults,
sync-from-broker-truth, and sticky reconciliation halts (ADR-0009 / doc 09)."""

from __future__ import annotations

import pytest

from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.pm import PMSide, PMTradeDecision
from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.errors import (
    BrokerUnavailable,
    DuplicateClientOrderId,
    IllegalTransition,
    MissingDecisionContext,
    ReconciliationHalt,
)
from dynamo.execution.gate import Readiness, live_orders_permitted
from dynamo.execution.mock_pm_broker import MockPMBroker, PMScriptStep
from dynamo.execution.order_manager import GateRefused
from dynamo.execution.pm_order_manager import PMReconciler, pm_scope
from tests.unit.pm.helpers import (
    AS_OF,
    NOW,
    PermissiveGate,
    RefusingGate,
    make_pm_manager,
    make_position,
)


def decision(market_id: str = "FED-26DEC-T4.00", contracts: int = 100) -> PMTradeDecision:
    return PMTradeDecision(
        market_id=market_id,
        side=PMSide.YES,
        edge=0.17,
        estimated_prob=0.60,
        implied_prob=0.43,
        confidence=0.8,
        kelly_fraction=0.0746,
        contracts=contracts,
        price_cents=43,
        rationale="test decision",
        view_ref=f"kalshi:{market_id}:{AS_OF.isoformat()}",
    )


def ready_order(manager, store, coid=None):
    orders = manager.propose([decision()], AS_OF)
    order = orders[0]
    manager.attach_decision_context(order.client_order_id, "dc-hash-1")
    manager.request_approval(order.client_order_id)
    return order.client_order_id


class TestProposal:
    def test_deterministic_ids_survive_rerun(self) -> None:
        manager, store, _trades, _clock, _broker = make_pm_manager()
        first = manager.propose([decision()], AS_OF)
        second = manager.propose([decision()], AS_OF)
        assert first[0].client_order_id == second[0].client_order_id
        assert len(store.orders) == 1  # re-derived, not duplicated

    def test_proposal_writes_creation_event(self) -> None:
        manager, store, _trades, _clock, _broker = make_pm_manager()
        manager.propose([decision()], AS_OF)
        assert [e.to_state for e in store.events] == ["proposed"]


class TestStateMachine:
    def test_full_lifecycle_events(self) -> None:
        manager, store, _trades, _clock, _broker = make_pm_manager(
            MockPMBroker(script=[PMScriptStep(behavior="fill")]), gate=PermissiveGate()
        )
        coid = ready_order(manager, store)
        manager.submit(coid)
        manager.record_fill(coid, 100, 43.0, NOW)
        assert store.orders[coid].state == OrderState.FILLED
        assert [e.to_state for e in store.events] == [
            "proposed",
            "risk_checked",
            "pending_approval",
            "submitted",
            "filled",
        ]

    def test_illegal_transition_raises(self) -> None:
        manager, _store, _trades, _clock, _broker = make_pm_manager()
        manager.propose([decision()], AS_OF)
        coid = manager.propose([decision()], AS_OF)[0].client_order_id
        with pytest.raises(IllegalTransition):
            manager.transition(coid, OrderState.FILLED)

    def test_terminal_state_is_absorbing(self) -> None:
        manager, store, _trades, _clock, _broker = make_pm_manager(gate=PermissiveGate())
        coid = ready_order(manager, store)
        manager.cancel(coid, "operator cancel")
        with pytest.raises(IllegalTransition):
            manager.transition(coid, OrderState.SUBMITTED)


class TestSubmission:
    def test_no_submission_without_decision_context(self) -> None:
        manager, _store, _trades, _clock, broker = make_pm_manager(gate=PermissiveGate())
        order = manager.propose([decision()], AS_OF)[0]
        manager.request_approval(order.client_order_id)
        with pytest.raises(MissingDecisionContext):
            manager.submit(order.client_order_id)
        assert not broker.knows(order.client_order_id)

    def test_gate_refusal_blocks_submission(self) -> None:
        manager, store, _trades, _clock, broker = make_pm_manager(gate=RefusingGate())
        coid = ready_order(manager, store)
        with pytest.raises(GateRefused):
            manager.submit(coid)
        assert not broker.knows(coid)

    def test_reject_is_terminal_never_retried(self) -> None:
        broker = MockPMBroker(script=[PMScriptStep(behavior="reject", reason="market closed")])
        manager, store, _trades, _clock, broker = make_pm_manager(broker, gate=PermissiveGate())
        coid = ready_order(manager, store)
        manager.submit(coid)
        assert store.orders[coid].state == OrderState.REJECTED
        assert broker.submit_calls == 1

    def test_duplicate_submission_fetches_state(self) -> None:
        class DuplicateBroker(MockPMBroker):
            def submit(self, order):  # type: ignore[override]
                self.submit_calls += 1
                raise DuplicateClientOrderId(order.client_order_id)

            def get_order(self, client_order_id):  # type: ignore[override]
                return {
                    "broker_order_id": "kalshi-known-1",
                    "status": "new",
                    "filled_contracts": 0,
                }

        manager, store, _trades, _clock, _broker = make_pm_manager(
            DuplicateBroker(), gate=PermissiveGate()
        )
        coid = ready_order(manager, store)
        manager.submit(coid)  # "already submitted — fetch state", never an error
        assert store.orders[coid].broker_order_id == "kalshi-known-1"
        assert store.orders[coid].state == OrderState.SUBMITTED


class TestGateIntegration:
    """The real doc 09 §7 gate over PM scopes: demo permitted, prod fail-closed."""

    def test_demo_env_short_circuits_like_paper(self) -> None:
        result = live_orders_permitted(
            DynamoEnv.PAPER, None, False, Readiness(), scope_target=pm_scope("FED-26DEC")
        )
        assert result.permitted

    def test_kill_switch_halts_pm_orders_in_every_env(self) -> None:
        for env in (DynamoEnv.PAPER, DynamoEnv.LIVE):
            result = live_orders_permitted(
                env, None, True, Readiness(), scope_target=pm_scope("FED-26DEC")
            )
            assert not result.permitted
            assert result.reason.value == "kill_switch_engaged"

    def test_live_refused_without_approval_and_readiness(self) -> None:
        result = live_orders_permitted(
            DynamoEnv.LIVE, None, False, Readiness(), scope_target=pm_scope("FED-26DEC")
        )
        assert not result.permitted
        assert result.reason.value == "approval_missing"


class TestSyncOrders:
    def test_fills_during_downtime_are_recorded_once(self) -> None:
        broker = MockPMBroker(script=[PMScriptStep(behavior="accept")])
        manager, store, trades, _clock, broker = make_pm_manager(broker, gate=PermissiveGate())
        coid = ready_order(manager, store)
        manager.submit(coid)
        broker.force_fill(coid, price_cents=43.0)  # filled while we were away
        recorded = manager.sync_orders()
        assert len(recorded) == 1
        assert store.orders[coid].state == OrderState.FILLED
        assert manager.sync_orders() == []  # idempotent: nothing new
        assert trades.filled_contracts(coid) == 100

    def test_order_unknown_at_broker_is_cancelled_locally(self) -> None:
        broker = MockPMBroker(script=[PMScriptStep(behavior="disconnect")])
        manager, store, _trades, _clock, broker = make_pm_manager(broker, gate=PermissiveGate())
        coid = ready_order(manager, store)
        with pytest.raises(BrokerUnavailable):
            manager.submit(coid)
        assert store.orders[coid].state == OrderState.SUBMITTED  # ambiguous, held
        manager.sync_orders()
        assert store.orders[coid].state == OrderState.CANCELLED


class RecordingStore:
    def __init__(self) -> None:
        self.runs: list[tuple] = []
        self.acks: list[tuple] = []

    def record_run(self, ts, scope, result, divergences) -> int:
        self.runs.append((ts, scope, result, list(divergences)))
        return len(self.runs)

    def record_ack(self, run_id, acked_by, acked_at, note) -> None:
        self.acks.append((run_id, acked_by, acked_at, note))


class TestReconciliation:
    def _reconciler(self, broker: MockPMBroker):
        alerts: list[tuple[str, str]] = []
        store = RecordingStore()
        reconciler = PMReconciler(
            broker, store, lambda name, msg: alerts.append((name, msg))
        )
        return reconciler, store, alerts

    def test_clean_when_positions_match(self) -> None:
        broker = MockPMBroker()
        broker.set_position("FED-26DEC-T4.00", PMSide.YES, 100, 43.0)
        reconciler, _store, alerts = self._reconciler(broker)
        result = reconciler.reconcile(NOW, [make_position(contracts=100)])
        assert result.clean
        assert alerts == []
        assert not reconciler.halts.is_halted(pm_scope("FED-26DEC-T4.00"))

    def test_divergence_engages_sticky_halt_and_blocks_submission(self) -> None:
        broker = MockPMBroker()
        broker.set_position("FED-26DEC-T4.00", PMSide.YES, 150, 43.0)  # we think 100
        reconciler, _store, alerts = self._reconciler(broker)
        result = reconciler.reconcile(NOW, [make_position(contracts=100)])
        assert not result.clean
        assert alerts and alerts[0][0] == "pm_reconciliation_divergence"
        assert reconciler.halts.is_halted(pm_scope("FED-26DEC-T4.00"))

        # A manager sharing the halt registry refuses new submissions for the scope.
        manager, order_store, _trades, _clock, _b = make_pm_manager(gate=PermissiveGate())
        manager.halts = reconciler.halts
        coid = ready_order(manager, order_store)
        with pytest.raises(ReconciliationHalt):
            manager.submit(coid)

    def test_balance_divergence_halts_globally(self) -> None:
        broker = MockPMBroker(balance_cents=90_000.0)
        reconciler, _store, _alerts = self._reconciler(broker)
        result = reconciler.reconcile(NOW, [], internal_balance_cents=100_000.0)
        assert not result.clean
        assert reconciler.halts.is_halted("ANY-MARKET")  # global halt covers all

    def test_halt_is_sticky_until_no_ack_no_clear(self) -> None:
        broker = MockPMBroker()
        broker.set_position("X", PMSide.NO, 5, 60.0)
        reconciler, _store, _alerts = self._reconciler(broker)
        reconciler.reconcile(NOW, [])
        # Re-running clean does NOT clear the halt — only an ack does.
        broker._positions.clear()
        result = reconciler.reconcile(NOW, [])
        assert result.clean
        assert reconciler.halts.is_halted(pm_scope("X"))
