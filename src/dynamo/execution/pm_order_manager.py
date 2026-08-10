"""Prediction-market order manager — the doc 09 discipline for the Kalshi
sleeve (ADR-0009).

Same contract as :mod:`dynamo.execution.order_manager`, over ``PMOrder``:

- one state machine (``ORDER_TRANSITIONS`` reused verbatim), one event row per
  transition (``pm_order_events``);
- deterministic client order ids, so a crash/retry re-derives the same intents;
- NO submission without a ``decision_context_id`` (hash-chained snapshot);
- the gate callable is consulted on EVERY submission (demo short-circuits to
  permitted exactly like paper; the kill-switch halts every environment);
- broker rejects are terminal, duplicates mean "fetch state", a disconnect
  leaves the order ``submitted`` until ``sync_orders`` repairs it from broker
  truth;
- reconciliation vs Kalshi positions/balance engages a STICKY halt that only
  an explicit human acknowledgment clears.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any, Protocol

from dynamo.domain.models.pm import PMOrder, PMPosition, PMTrade, PMTradeDecision
from dynamo.domain.models.portfolio import (
    ORDER_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    OrderState,
)
from dynamo.domain.ports import Clock, PMBroker
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
    GLOBAL_SCOPE,
    AlertCallback,
    Divergence,
    HaltRegistry,
    ReconciliationResult,
    ReconciliationStore,
)

PM_SCOPE_PREFIX = "KALSHI:"
BALANCE_TOLERANCE_CENTS = 100.0  # $1, mirroring the equity cash tolerance


def pm_scope(market_id: str) -> str:
    """Halt/approval scope for one Kalshi market (also the DecisionContext symbol)."""
    return f"{PM_SCOPE_PREFIX}{market_id}"


class PMOrderStore(Protocol):
    """PMOrder persistence surface required by the manager."""

    def insert(self, order: PMOrder) -> None: ...  # also writes the creation event

    def get(self, client_order_id: str) -> PMOrder | None: ...

    def open_orders(self) -> list[PMOrder]: ...  # non-terminal states

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
    ) -> PMOrder: ...


class PMTradeStore(Protocol):
    def insert(self, trade: PMTrade) -> None: ...

    def filled_contracts(self, client_order_id: str) -> int: ...


class PMOrderManager:
    def __init__(
        self,
        broker: PMBroker,
        orders: PMOrderStore,
        trades: PMTradeStore,
        clock: Clock,
        *,
        gate: Callable[[PMOrder], Any] | None = None,  # returns GateResult-like
        halts: HaltRegistry | None = None,
    ) -> None:
        self._broker = broker
        self._orders = orders
        self._trades = trades
        self._clock = clock
        self._gate = gate
        self.halts = halts if halts is not None else HaltRegistry()

    # --- proposal (risk.pm_sizing → orders) ---

    def propose(self, decisions: Sequence[PMTradeDecision], run_date: date) -> list[PMOrder]:
        """One ``proposed`` order per sized decision, deterministic ids.

        ``sha256(pm-<venue>|market|side|run-date)`` — re-running after a crash
        re-derives the same intents instead of new ones.
        """
        out: list[PMOrder] = []
        for decision in decisions:
            coid = client_order_id(
                f"pm-{decision.venue}",
                decision.market_id,
                decision.side.value,
                run_date.isoformat(),
            )
            existing = self._orders.get(coid)
            if existing is not None:
                out.append(existing)
                continue
            order = PMOrder(
                client_order_id=coid,
                venue=decision.venue,
                market_id=decision.market_id,
                side=decision.side,
                contracts=decision.contracts,
                limit_price_cents=decision.price_cents,
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
    ) -> PMOrder:
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

    def request_approval(self, coid: str) -> PMOrder:
        self.transition(coid, OrderState.RISK_CHECKED, "pm sizing rules passed")
        return self.transition(coid, OrderState.PENDING_APPROVAL, "awaiting approval")

    # --- submission (doc 09 §4, §6, §7) ---

    def submit(self, coid: str) -> PMOrder:
        """Submit a ``pending_approval`` order to Kalshi.

        Requires no active reconciliation halt for the market, a decision
        context id, and a permitting gate. Duplicate-id responses fetch state;
        rejects are terminal; a disconnect leaves the order ``submitted`` for
        ``sync_orders`` to repair from broker truth.
        """
        order = self._require(coid)
        if self.halts.is_halted(pm_scope(order.market_id)):
            raise ReconciliationHalt(pm_scope(order.market_id))
        if order.state != OrderState.PENDING_APPROVAL:
            raise IllegalTransition(coid, order.state.value, OrderState.SUBMITTED.value)
        if not order.decision_context_id:
            raise MissingDecisionContext(coid)

        if self._gate is not None:
            result = self._gate(order)
            if not getattr(result, "permitted", False):
                raise GateRefused(f"gate refused {coid}: {getattr(result, 'reason', '?')}")

        order = self.transition(coid, OrderState.SUBMITTED, "submitted to kalshi")
        try:
            broker_order_id = self._broker.submit(order)
        except DuplicateClientOrderId:
            state = self._broker.get_order(coid)
            self._orders.set_broker_order_id(coid, str(state.get("broker_order_id", "")))
            return self._require(coid)
        except BrokerRejection as exc:
            return self.transition(coid, OrderState.REJECTED, exc.reason)
        except BrokerUnavailable:
            # Ambiguous: kalshi may or may not have the order. Stay ``submitted``
            # until sync_orders resolves it from broker truth (doc 09 §8).
            raise
        self._orders.set_broker_order_id(coid, broker_order_id)
        return self._require(coid)

    def cancel(self, coid: str, reason: str = "cancelled") -> PMOrder:
        order = self._require(coid)
        if order.state in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
            self._broker.cancel(coid)
        return self.transition(coid, OrderState.CANCELLED, reason)

    # --- fills ---

    def record_fill(
        self,
        coid: str,
        contracts: int,
        price_cents: float,
        filled_at: datetime,
        fees_cents: float = 0.0,
    ) -> PMTrade:
        order = self._require(coid)
        if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
            raise IllegalTransition(coid, order.state.value, OrderState.FILLED.value)
        trade = PMTrade(
            client_order_id=coid,
            venue=order.venue,
            market_id=order.market_id,
            side=order.side,
            contracts=contracts,
            price_cents=price_cents,
            fees_cents=fees_cents,
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

    def sync_orders(self) -> list[PMTrade]:
        """Repair local order state from Kalshi truth; returns new fills.

        Fills that happened while we were away are recorded (idempotent by
        cumulative count), terminal broker states are mirrored, and orders
        kalshi never received are cancelled locally.
        """
        recorded: list[PMTrade] = []
        for order in self._orders.open_orders():
            if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
                continue
            coid = order.client_order_id
            try:
                remote = self._broker.get_order(coid)
            except BrokerRejection:
                self.transition(coid, OrderState.CANCELLED, "not_found_at_kalshi")
                continue

            broker_id = str(remote.get("broker_order_id", ""))
            if broker_id and not order.broker_order_id:
                self._orders.set_broker_order_id(coid, broker_id)

            remote_filled = int(remote.get("filled_contracts") or 0)
            missing = remote_filled - self._trades.filled_contracts(coid)
            if missing > 0:
                price = float(remote.get("filled_avg_price_cents") or 0.0)
                recorded.append(self.record_fill(coid, missing, price, self._clock.now()))
                order = self._require(coid)

            status = str(remote.get("status", ""))
            if status in ("canceled", "cancelled") and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.CANCELLED, "kalshi: cancelled")
            elif status == "rejected" and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.REJECTED, "kalshi: rejected")
            elif status == "expired" and order.state not in TERMINAL_ORDER_STATES:
                self.transition(coid, OrderState.EXPIRED, "kalshi: expired")
        return recorded

    # --- internals ---

    def _require(self, coid: str) -> PMOrder:
        order = self._orders.get(coid)
        if order is None:
            raise ExecutionError(f"unknown pm order {coid}")
        return order


class PMReconciler:
    """Positions/balance reconciliation vs Kalshi truth; divergence → sticky halt.

    Same rules as :class:`dynamo.execution.reconciliation.Reconciler`: a market
    mismatch halts that market (``KALSHI:<market_id>``), a balance mismatch
    beyond tolerance halts the sleeve globally, and halts clear only through an
    explicit acknowledgment recorded in ``reconciliation_acks``.

    ``scope_fn``/``run_scope`` default to the Kalshi sleeve; the live
    Polymarket sleeve (ADR-0012) passes ``pm_sim_scope`` (``POLYMARKET:`` halt
    scopes) and ``run_scope="pm_live"`` so its divergent runs are still caught
    by the sticky ``pm*`` halt check.
    """

    def __init__(
        self,
        broker: PMBroker,
        store: ReconciliationStore,
        alert: AlertCallback,
        *,
        balance_tolerance_cents: float = BALANCE_TOLERANCE_CENTS,
        halts: HaltRegistry | None = None,
        scope_fn: Callable[[str], str] = pm_scope,
        run_scope: str = "pm",
    ) -> None:
        self._broker = broker
        self._store = store
        self._alert = alert
        self._tolerance = balance_tolerance_cents
        self._scope_fn = scope_fn
        self._run_scope = run_scope
        self.halts = halts if halts is not None else HaltRegistry()

    def reconcile(
        self,
        now: datetime,
        internal_positions: Sequence[PMPosition],
        internal_balance_cents: float | None = None,
    ) -> ReconciliationResult:
        divergences: list[Divergence] = []
        internal = {p.market_id: (p.side.value, p.contracts) for p in internal_positions}
        broker = {
            str(p["market_id"]): (str(p["side"]), int(p["contracts"]))
            for p in self._broker.positions()
        }
        for market_id in sorted(set(internal) | set(broker)):
            ours, theirs = internal.get(market_id), broker.get(market_id)
            if ours == theirs:
                continue
            kind = "position_qty" if ours is not None and theirs is not None else (
                "unknown_position"
            )
            divergences.append(
                Divergence(
                    self._scope_fn(market_id),
                    kind,
                    f"{ours[0]}:{ours[1]}" if ours is not None else None,
                    f"{theirs[0]}:{theirs[1]}" if theirs is not None else None,
                    "pm position mismatch",
                )
            )
        if internal_balance_cents is not None:
            broker_balance = float(self._broker.balance().get("balance_cents", 0.0))
            if abs(broker_balance - internal_balance_cents) > self._tolerance:
                divergences.append(
                    Divergence(
                        GLOBAL_SCOPE,
                        "balance",
                        internal_balance_cents,
                        broker_balance,
                        "pm balance mismatch",
                    )
                )

        clean = not divergences
        run_id = self._store.record_run(
            now,
            self._run_scope if clean else (f"{self._run_scope}:global" if any(
                d.scope == GLOBAL_SCOPE for d in divergences
            ) else f"{self._run_scope}:markets"),
            "clean" if clean else "divergent",
            [d.as_dict() for d in divergences],
        )
        if not clean:
            for d in divergences:
                self.halts.engage(d.scope, run_id)
            self._alert(
                "pm_reconciliation_divergence",
                f"run {run_id}: {len(divergences)} divergence(s); "
                f"halted scopes: {sorted({d.scope for d in divergences})}",
            )
        return ReconciliationResult(run_id, clean, tuple(divergences))
