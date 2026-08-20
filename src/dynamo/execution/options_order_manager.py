"""Options order manager — the doc 09 discipline for the defined-risk sleeve
(ADR-0010).

Same contract as :mod:`dynamo.execution.pm_order_manager`, over ``OptionOrder``:

- one state machine (``ORDER_TRANSITIONS`` reused verbatim), one event row per
  transition (``options_order_events``);
- deterministic client order ids, so a crash/retry re-derives the same intents;
- NO submission without a ``decision_context_id`` (hash-chained snapshot);
- the gate callable is consulted on EVERY submission (kill-switch halts every
  environment; LIVE additionally needs the signed approval + readiness);
- collateral for covered calls / cash-secured puts is re-validated AT
  SUBMISSION through an injectable check (proposal-time validation lives in
  the ``OptionTradeDecision`` domain validators);
- broker rejects are terminal, duplicates mean "fetch state", a disconnect
  leaves the order ``submitted`` until ``sync_orders`` repairs it from broker
  truth;
- reconciliation vs broker option positions engages a STICKY halt that only
  an explicit human acknowledgment clears.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Protocol

from dynamo.domain.models.options import (
    CollateralShortfall,
    OptionOrder,
    OptionPosition,
    OptionStructure,
    OptionTrade,
    OptionTradeDecision,
    validate_covered_call_collateral,
    validate_csp_collateral,
)
from dynamo.domain.models.portfolio import (
    ORDER_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    OrderState,
)
from dynamo.domain.ports import Clock, OptionsBroker
from dynamo.execution.client_order_id import client_order_id
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    ExecutionError,
    IllegalTransition,
    MissingDecisionContext,
    ReconciliationHalt,
)
from dynamo.execution.order_manager import GateRefused
from dynamo.execution.reconciliation import (
    AlertCallback,
    Divergence,
    HaltRegistry,
    ReconciliationResult,
    ReconciliationStore,
)

OPT_SCOPE_PREFIX = "OPT:"


def opt_scope(occ_symbol: str) -> str:
    """Halt/approval scope for one contract (also the DecisionContext symbol)."""
    return f"{OPT_SCOPE_PREFIX}{occ_symbol}"


# Injected at wiring time: current share qty / free cash for collateral checks.
SharesLookup = Callable[[str], float]  # underlying -> shares held
CashLookup = Callable[[], float]  # free cash in dollars


class OptionOrderStore(Protocol):
    """OptionOrder persistence surface required by the manager."""

    def insert(self, order: OptionOrder) -> None: ...  # also writes the creation event

    def get(self, client_order_id: str) -> OptionOrder | None: ...

    def open_orders(self) -> list[OptionOrder]: ...  # non-terminal states

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None: ...

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None: ...

    def transition_with_event(
        self,
        client_order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str,
        ts: datetime,
        payload: Mapping[str, Any] | None = None,
    ) -> OptionOrder: ...


class OptionTradeStore(Protocol):
    def insert(self, trade: OptionTrade) -> None: ...

    def filled_contracts(self, client_order_id: str) -> int: ...


class OptionsOrderManager:
    def __init__(
        self,
        broker: OptionsBroker,
        orders: OptionOrderStore,
        trades: OptionTradeStore,
        clock: Clock,
        *,
        gate: Callable[[OptionOrder], Any] | None = None,  # returns GateResult-like
        halts: HaltRegistry | None = None,
        shares_lookup: SharesLookup | None = None,
        cash_lookup: CashLookup | None = None,
    ) -> None:
        self._broker = broker
        self._orders = orders
        self._trades = trades
        self._clock = clock
        self._gate = gate
        self._shares_lookup = shares_lookup
        self._cash_lookup = cash_lookup
        self.halts = halts if halts is not None else HaltRegistry()

    # --- proposal (risk.options_sizing → orders) ---

    def propose(
        self, decisions: Sequence[OptionTradeDecision], run_date: date
    ) -> list[OptionOrder]:
        """One ``proposed`` order per sized decision, deterministic ids.

        ``sha256(opt-<structure>|occ|side|run-date)`` — re-running after a
        crash re-derives the same intents instead of new ones.
        """
        out: list[OptionOrder] = []
        for decision in decisions:
            coid = client_order_id(
                f"opt-{decision.structure.value}",
                decision.occ_symbol,
                decision.side.value,
                run_date.isoformat(),
            )
            existing = self._orders.get(coid)
            if existing is not None:
                out.append(existing)
                continue
            order = OptionOrder(
                client_order_id=coid,
                structure=decision.structure,
                side=decision.side,
                right=decision.right,
                occ_symbol=decision.occ_symbol,
                underlying=decision.underlying,
                expiry=decision.expiry,
                strike=decision.strike,
                contracts=decision.contracts,
                limit_price=decision.limit_price,
                state=OrderState.PROPOSED,
                created_at=self._clock.now(),
            )
            self._orders.insert(order)
            out.append(order)
        return out

    # --- state machine (doc 09 §2, reused verbatim) ---

    def transition(
        self,
        coid: str,
        to_state: OrderState,
        reason: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> OptionOrder:
        """Apply one transition; illegal transitions raise, never coerce."""
        order = self._require(coid)
        allowed = ORDER_TRANSITIONS.get(order.state, frozenset())
        if to_state not in allowed:
            raise IllegalTransition(coid, order.state.value, to_state.value)
        return self._orders.transition_with_event(
            coid, order.state, to_state, reason, self._clock.now(), payload
        )

    def attach_decision_context(self, coid: str, decision_context_id: str) -> None:
        if not decision_context_id:
            raise MissingDecisionContext(coid)
        self._require(coid)
        self._orders.set_decision_context(coid, decision_context_id)

    def request_approval(self, coid: str) -> OptionOrder:
        self.transition(coid, OrderState.RISK_CHECKED, "options sizing rules passed")
        return self.transition(coid, OrderState.PENDING_APPROVAL, "awaiting approval")

    # --- submission (doc 09 §4, §6, §7) ---

    def submit(self, coid: str) -> OptionOrder:
        """Submit a ``pending_approval`` order to the broker.

        Requires no active reconciliation halt for the contract, a decision
        context id, a permitting gate, and — for short structures — collateral
        STILL present at submission time. Duplicate-id responses fetch state;
        rejects are terminal; a disconnect leaves the order ``submitted`` for
        ``sync_orders`` to repair from broker truth.
        """
        order = self._require(coid)
        if self.halts.is_halted(opt_scope(order.occ_symbol)):
            raise ReconciliationHalt(opt_scope(order.occ_symbol))
        if order.state != OrderState.PENDING_APPROVAL:
            raise IllegalTransition(coid, order.state.value, OrderState.SUBMITTED.value)
        if not order.decision_context_id:
            raise MissingDecisionContext(coid)
        self._check_collateral(order)

        if self._gate is not None:
            result = self._gate(order)
            if not getattr(result, "permitted", False):
                raise GateRefused(f"gate refused {coid}: {getattr(result, 'reason', '?')}")

        order = self.transition(coid, OrderState.SUBMITTED, "submitted to broker")
        try:
            broker_order_id = self._broker.submit(order)
        except DuplicateClientOrderId:
            state = self._broker.get_order(coid)
            self._orders.set_broker_order_id(coid, str(state.get("broker_order_id", "")))
            return self._require(coid)
        except BrokerRejection as exc:
            return self.transition(coid, OrderState.REJECTED, exc.reason)
        except BrokerUnavailable:
            # Ambiguous: the broker may or may not have the order. Stay
            # ``submitted`` until sync_orders resolves it from broker truth.
            raise
        self._orders.set_broker_order_id(coid, broker_order_id)
        return self._require(coid)

    def cancel(self, coid: str, reason: str = "cancelled") -> OptionOrder:
        order = self._require(coid)
        if order.state in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
            self._broker.cancel(coid)
        return self.transition(coid, OrderState.CANCELLED, reason)

    # --- fills ---

    def record_fill(
        self,
        coid: str,
        contracts: int,
        price: float,
        filled_at: datetime,
        fees: float = 0.0,
    ) -> OptionTrade:
        order = self._require(coid)
        if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
            raise IllegalTransition(coid, order.state.value, OrderState.FILLED.value)
        trade = OptionTrade(
            client_order_id=coid,
            occ_symbol=order.occ_symbol,
            underlying=order.underlying,
            side=order.side,
            right=order.right,
            contracts=contracts,
            price=price,
            fees=fees,
            kind="fill",
            filled_at=filled_at,
        )
        self._trades.insert(trade)

        cumulative = self._trades.filled_contracts(coid)
        if cumulative >= order.contracts:
            self.transition(coid, OrderState.FILLED, "fully filled")
        elif order.state == OrderState.SUBMITTED:
            self.transition(coid, OrderState.PARTIALLY_FILLED, "partial fill")
        return trade

    # --- broker-truth sync (doc 09 §4 startup reconciliation, per run) ---

    def sync_orders(self) -> list[OptionTrade]:
        """Repair local order state from broker truth; returns new fills.

        Fills that happened while we were away are recorded (idempotent by
        cumulative count), terminal broker states are mirrored, and orders the
        broker never received are cancelled locally.
        """
        recorded: list[OptionTrade] = []
        for order in self._orders.open_orders():
            if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
                continue
            coid = order.client_order_id
            try:
                remote = self._broker.get_order(coid)
            except BrokerRejection:
                self.transition(coid, OrderState.CANCELLED, "not_found_at_broker")
                continue

            broker_id = str(remote.get("broker_order_id", ""))
            if broker_id and not order.broker_order_id:
                self._orders.set_broker_order_id(coid, broker_id)

            remote_filled = int(remote.get("filled_contracts") or 0)
            missing = remote_filled - self._trades.filled_contracts(coid)
            if missing > 0:
                price = float(remote.get("filled_avg_price") or 0.0)
                recorded.append(self.record_fill(coid, missing, price, self._clock.now()))
                order = self._require(coid)

            status = str(remote.get("status", ""))
            if status in ("canceled", "cancelled") and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.CANCELLED, "broker: cancelled")
            elif status == "rejected" and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.REJECTED, "broker: rejected")
            elif status == "expired" and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.EXPIRED, "broker: expired")
        return recorded

    # --- internals ---

    def _check_collateral(self, order: OptionOrder) -> None:
        """Re-validate covered-call/CSP collateral AT SUBMISSION (fail closed).

        A missing lookup for a short structure refuses the order — collateral
        that cannot be verified does not exist.
        """
        if order.structure is OptionStructure.COVERED_CALL:
            if self._shares_lookup is None:
                raise CollateralShortfall(
                    f"{order.client_order_id}: no shares lookup — cannot verify coverage"
                )
            validate_covered_call_collateral(order.contracts, self._shares_lookup(order.underlying))
        elif order.structure is OptionStructure.CASH_SECURED_PUT:
            if self._cash_lookup is None:
                raise CollateralShortfall(
                    f"{order.client_order_id}: no cash lookup — cannot verify reserve"
                )
            validate_csp_collateral(order.contracts, order.strike, self._cash_lookup())

    def _require(self, coid: str) -> OptionOrder:
        order = self._orders.get(coid)
        if order is None:
            raise ExecutionError(f"unknown option order {coid}")
        return order


class OptionsReconciler:
    """Positions reconciliation vs broker truth; divergence → sticky halt.

    Same rules as :class:`dynamo.execution.pm_order_manager.PMReconciler`: a
    contract mismatch halts that contract (``OPT:<occ>``), and halts clear
    only through an explicit acknowledgment recorded in ``reconciliation_acks``.
    Options share the equity cash account, so there is no separate balance
    check here — cash is owned by the equity reconciler.
    """

    def __init__(
        self,
        broker: OptionsBroker,
        store: ReconciliationStore,
        alert: AlertCallback,
        *,
        halts: HaltRegistry | None = None,
    ) -> None:
        self._broker = broker
        self._store = store
        self._alert = alert
        self.halts = halts if halts is not None else HaltRegistry()

    def reconcile(
        self, now: datetime, internal_positions: Sequence[OptionPosition]
    ) -> ReconciliationResult:
        divergences: list[Divergence] = []
        internal: dict[str, int] = {}
        for p in internal_positions:
            if p.contracts <= 0:
                continue
            signed = p.contracts if p.side.value == "buy" else -p.contracts
            internal[p.occ_symbol] = internal.get(p.occ_symbol, 0) + signed
        broker = {
            str(p["occ_symbol"]): int(float(p["qty"]))
            for p in self._broker.positions()
            if float(p["qty"]) != 0.0
        }
        for occ in sorted(set(internal) | set(broker)):
            ours, theirs = internal.get(occ), broker.get(occ)
            if ours == theirs:
                continue
            kind = (
                "position_qty" if ours is not None and theirs is not None else ("unknown_position")
            )
            divergences.append(
                Divergence(opt_scope(occ), kind, ours, theirs, "option position mismatch")
            )

        clean = not divergences
        run_id = self._store.record_run(
            now,
            "options" if clean else "options:positions",
            "clean" if clean else "divergent",
            [d.as_dict() for d in divergences],
        )
        if not clean:
            for d in divergences:
                self.halts.engage(d.scope, run_id)
            self._alert(
                "options_reconciliation_divergence",
                f"run {run_id}: {len(divergences)} divergence(s); "
                f"halted scopes: {sorted({d.scope for d in divergences})}",
            )
        return ReconciliationResult(run_id, clean, tuple(divergences))
