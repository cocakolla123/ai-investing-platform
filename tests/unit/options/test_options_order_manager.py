"""Options order manager: state machine reuse, gate integration, collateral
re-checks at submission, broker faults, sync-from-broker-truth, and sticky
reconciliation halts (ADR-0010 / doc 09)."""

from __future__ import annotations

import pytest

from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.options import CollateralShortfall, OptionStructure
from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.errors import (
    BrokerUnavailable,
    DuplicateClientOrderId,
    IllegalTransition,
    MissingDecisionContext,
    ReconciliationHalt,
)
from dynamo.execution.gate import Readiness, live_orders_permitted
from dynamo.execution.mock_options_broker import MockOptionsBroker, OptionsScriptStep
from dynamo.execution.options_order_manager import OptionsReconciler, opt_scope
from dynamo.execution.order_manager import GateRefused
from tests.unit.options.helpers import (
    AS_OF,
    NOW,
    PermissiveGate,
    RefusingGate,
    make_decision,
    make_option_position,
    make_options_manager,
)


def ready_order(manager, decision=None):
    orders = manager.propose([decision if decision is not None else make_decision()], AS_OF)
    order = orders[0]
    manager.attach_decision_context(order.client_order_id, "dc-hash-1")
    manager.request_approval(order.client_order_id)
    return order.client_order_id


class TestProposal:
    def test_deterministic_ids_survive_rerun(self) -> None:
        manager, store, _trades, _clock, _broker = make_options_manager()
        first = manager.propose([make_decision()], AS_OF)
        second = manager.propose([make_decision()], AS_OF)
        assert first[0].client_order_id == second[0].client_order_id
        assert len(store.orders) == 1  # re-derived, not duplicated

    def test_proposal_writes_creation_event(self) -> None:
        manager, store, _trades, _clock, _broker = make_options_manager()
        manager.propose([make_decision()], AS_OF)
        assert [e.to_state for e in store.events] == ["proposed"]


class TestStateMachine:
    def test_full_lifecycle_events(self) -> None:
        manager, store, _trades, _clock, _broker = make_options_manager(
            MockOptionsBroker(script=[OptionsScriptStep(behavior="fill")]),
            gate=PermissiveGate(),
        )
        coid = ready_order(manager)
        manager.submit(coid)
        manager.record_fill(coid, 4, 2.50, NOW)
        assert store.orders[coid].state == OrderState.FILLED
        assert [e.to_state for e in store.events] == [
            "proposed",
            "risk_checked",
            "pending_approval",
            "submitted",
            "filled",
        ]

    def test_illegal_transition_raises(self) -> None:
        manager, _store, _trades, _clock, _broker = make_options_manager()
        coid = manager.propose([make_decision()], AS_OF)[0].client_order_id
        with pytest.raises(IllegalTransition):
            manager.transition(coid, OrderState.FILLED)

    def test_terminal_state_is_absorbing(self) -> None:
        manager, _store, _trades, _clock, _broker = make_options_manager(
            gate=PermissiveGate()
        )
        coid = ready_order(manager)
        manager.cancel(coid, "operator cancel")
        with pytest.raises(IllegalTransition):
            manager.transition(coid, OrderState.SUBMITTED)


class TestSubmission:
    def test_no_submission_without_decision_context(self) -> None:
        manager, _store, _trades, _clock, broker = make_options_manager(
            gate=PermissiveGate()
        )
        order = manager.propose([make_decision()], AS_OF)[0]
        manager.request_approval(order.client_order_id)
        with pytest.raises(MissingDecisionContext):
            manager.submit(order.client_order_id)
        assert not broker.knows(order.client_order_id)

    def test_gate_refusal_blocks_submission(self) -> None:
        manager, _store, _trades, _clock, broker = make_options_manager(gate=RefusingGate())
        coid = ready_order(manager)
        with pytest.raises(GateRefused):
            manager.submit(coid)
        assert not broker.knows(coid)

    def test_reject_is_terminal_never_retried(self) -> None:
        broker = MockOptionsBroker(
            script=[OptionsScriptStep(behavior="reject", reason="not permissioned")]
        )
        manager, store, _trades, _clock, broker = make_options_manager(
            broker, gate=PermissiveGate()
        )
        coid = ready_order(manager)
        manager.submit(coid)
        assert store.orders[coid].state == OrderState.REJECTED
        assert broker.submit_calls == 1

    def test_duplicate_submission_fetches_state(self) -> None:
        class DuplicateBroker(MockOptionsBroker):
            def submit(self, order):  # type: ignore[override]
                self.submit_calls += 1
                raise DuplicateClientOrderId(order.client_order_id)

            def get_order(self, client_order_id):  # type: ignore[override]
                return {
                    "broker_order_id": "alpaca-opt-known-1",
                    "status": "new",
                    "filled_contracts": 0,
                }

        manager, store, _trades, _clock, _broker = make_options_manager(
            DuplicateBroker(), gate=PermissiveGate()
        )
        coid = ready_order(manager)
        manager.submit(coid)  # "already submitted — fetch state", never an error
        assert store.orders[coid].broker_order_id == "alpaca-opt-known-1"
        assert store.orders[coid].state == OrderState.SUBMITTED


class TestCollateralAtSubmission:
    """Proposal-time validation lives on the decision; submission re-checks
    against CURRENT share/cash state — collateral that has since disappeared
    (or cannot be verified) refuses the order."""

    def covered_call(self):
        return make_decision(
            OptionStructure.COVERED_CALL, contracts=2, collateral_shares=200.0
        )

    def csp(self):
        return make_decision(
            OptionStructure.CASH_SECURED_PUT,
            strike=95.0,
            contracts=1,
            collateral_cash=9_500.0,
        )

    def test_covered_call_shares_gone_by_submission(self) -> None:
        manager, _s, _t, _c, broker = make_options_manager(
            gate=PermissiveGate(), shares=150.0  # was 200 at proposal
        )
        coid = ready_order(manager, self.covered_call())
        with pytest.raises(CollateralShortfall):
            manager.submit(coid)
        assert not broker.knows(coid)

    def test_covered_call_with_shares_submits(self) -> None:
        manager, store, _t, _c, _b = make_options_manager(
            gate=PermissiveGate(), shares=200.0
        )
        coid = ready_order(manager, self.covered_call())
        manager.submit(coid)
        assert store.orders[coid].state == OrderState.SUBMITTED

    def test_csp_cash_gone_by_submission(self) -> None:
        manager, _s, _t, _c, broker = make_options_manager(
            gate=PermissiveGate(), cash=9_000.0
        )
        coid = ready_order(manager, self.csp())
        with pytest.raises(CollateralShortfall):
            manager.submit(coid)
        assert not broker.knows(coid)

    def test_csp_with_cash_submits(self) -> None:
        manager, store, _t, _c, _b = make_options_manager(
            gate=PermissiveGate(), cash=10_000.0
        )
        coid = ready_order(manager, self.csp())
        manager.submit(coid)
        assert store.orders[coid].state == OrderState.SUBMITTED

    def test_unverifiable_collateral_fails_closed(self) -> None:
        # No lookups wired at all → short structures refuse to submit.
        manager, _s, _t, _c, broker = make_options_manager(gate=PermissiveGate())
        coid = ready_order(manager, self.covered_call())
        with pytest.raises(CollateralShortfall):
            manager.submit(coid)
        assert not broker.knows(coid)

    def test_long_structures_need_no_collateral_lookup(self) -> None:
        manager, store, _t, _c, _b = make_options_manager(gate=PermissiveGate())
        coid = ready_order(manager, make_decision())  # long call
        manager.submit(coid)
        assert store.orders[coid].state == OrderState.SUBMITTED


class TestGateIntegration:
    """The real doc 09 §7 gate over OPT scopes: paper permitted, live fail-closed."""

    def test_paper_env_short_circuits(self) -> None:
        result = live_orders_permitted(
            DynamoEnv.PAPER, None, False, Readiness(), scope_target=opt_scope("AAPL260918C")
        )
        assert result.permitted

    def test_kill_switch_halts_options_in_every_env(self) -> None:
        for env in (DynamoEnv.PAPER, DynamoEnv.LIVE):
            result = live_orders_permitted(
                env, None, True, Readiness(), scope_target=opt_scope("AAPL260918C")
            )
            assert not result.permitted
            assert result.reason.value == "kill_switch_engaged"

    def test_live_refused_without_approval_and_readiness(self) -> None:
        result = live_orders_permitted(
            DynamoEnv.LIVE, None, False, Readiness(), scope_target=opt_scope("AAPL260918C")
        )
        assert not result.permitted
        assert result.reason.value == "approval_missing"


class TestSyncOrders:
    def test_fills_during_downtime_are_recorded_once(self) -> None:
        broker = MockOptionsBroker(script=[OptionsScriptStep(behavior="accept")])
        manager, store, trades, _clock, broker = make_options_manager(
            broker, gate=PermissiveGate()
        )
        coid = ready_order(manager)
        manager.submit(coid)
        broker.force_fill(coid, price=2.50)  # filled while we were away
        recorded = manager.sync_orders()
        assert len(recorded) == 1
        assert store.orders[coid].state == OrderState.FILLED
        assert manager.sync_orders() == []  # idempotent: nothing new
        assert trades.filled_contracts(coid) == 4

    def test_order_unknown_at_broker_is_cancelled_locally(self) -> None:
        broker = MockOptionsBroker(script=[OptionsScriptStep(behavior="disconnect")])
        manager, store, _trades, _clock, broker = make_options_manager(
            broker, gate=PermissiveGate()
        )
        coid = ready_order(manager)
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
    def _reconciler(self, broker: MockOptionsBroker):
        alerts: list[tuple[str, str]] = []
        store = RecordingStore()
        reconciler = OptionsReconciler(
            broker, store, lambda name, msg: alerts.append((name, msg))
        )
        return reconciler, store, alerts

    def test_clean_when_positions_match(self) -> None:
        position = make_option_position(contracts=4)
        broker = MockOptionsBroker()
        broker.set_position(position.occ_symbol, 4.0, 2.50)
        reconciler, _store, alerts = self._reconciler(broker)
        result = reconciler.reconcile(NOW, [position])
        assert result.clean
        assert alerts == []
        assert not reconciler.halts.is_halted(opt_scope(position.occ_symbol))

    def test_short_positions_compare_signed(self) -> None:
        position = make_option_position(
            OptionStructure.COVERED_CALL, strike=105.0, contracts=2, avg_price=2.40
        )
        broker = MockOptionsBroker()
        broker.set_position(position.occ_symbol, -2.0, 2.40)  # short at the broker
        reconciler, _store, _alerts = self._reconciler(broker)
        assert reconciler.reconcile(NOW, [position]).clean

    def test_divergence_engages_sticky_halt_and_blocks_submission(self) -> None:
        position = make_option_position(contracts=4)
        broker = MockOptionsBroker()
        broker.set_position(position.occ_symbol, 6.0, 2.50)  # we think 4
        reconciler, _store, alerts = self._reconciler(broker)
        result = reconciler.reconcile(NOW, [position])
        assert not result.clean
        assert alerts and alerts[0][0] == "options_reconciliation_divergence"
        assert reconciler.halts.is_halted(opt_scope(position.occ_symbol))

        # A manager sharing the halt registry refuses new submissions for the scope.
        manager, _os, _t, _c, _b = make_options_manager(gate=PermissiveGate())
        manager.halts = reconciler.halts
        coid = ready_order(manager)  # same occ symbol as the position
        with pytest.raises(ReconciliationHalt):
            manager.submit(coid)

    def test_halt_is_sticky_no_ack_no_clear(self) -> None:
        broker = MockOptionsBroker()
        broker.set_position("AAPL260918C00095000", 1.0, 2.0)
        reconciler, _store, _alerts = self._reconciler(broker)
        reconciler.reconcile(NOW, [])
        # Re-running clean does NOT clear the halt — only an ack does.
        broker._positions.clear()
        result = reconciler.reconcile(NOW, [])
        assert result.clean
        assert reconciler.halts.is_halted(opt_scope("AAPL260918C00095000"))
