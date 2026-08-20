"""Order manager — lifecycle state machine, idempotent submission, fills.

Design: doc 09 §2 (state machine), §4 (idempotency, startup reconciliation),
§5 (slippage), §6 (DecisionContext required), §8 (failure handling).

Persistence is injected through small structural stores so the manager is
testable with in-memory fakes; production wires the repositories from
``dynamo.persistence.repositories.execution``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Protocol

from dynamo.domain.models.portfolio import (
    ORDER_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    PortfolioState,
    TargetPortfolio,
    Trade,
)
from dynamo.domain.ports import Broker, Clock
from dynamo.domain.symbols import is_crypto_symbol
from dynamo.execution.client_order_id import client_order_id
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    ExecutionError,
    IllegalTransition,
    MissingDecisionContext,
    ReconciliationHalt,
    ReconciliationPending,
)

MIN_WEIGHT_DELTA = 0.0025  # skip rebalances under 0.25% of equity (doc 07/09)
LIMIT_OFFSET_MIN_BPS = 2.0
LIMIT_OFFSET_SPREAD_MULT = 0.25
APPROVAL_EXPIRED_REASON = "approval_expired"


class GateRefused(ExecutionError):
    """The live gate refused this submission (doc 09 §7)."""


class OrderStore(Protocol):
    """Order persistence surface required by the manager."""

    def insert(self, order: Order) -> None: ...  # also writes the creation event

    def get(self, client_order_id: str) -> Order | None: ...

    def open_orders(self) -> list[Order]: ...  # non-terminal states

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
    ) -> Order: ...

    def state_entered_at(self, client_order_id: str) -> datetime | None: ...


class TradeStore(Protocol):
    def insert(self, trade: Trade) -> None: ...

    def filled_qty(self, client_order_id: str) -> float: ...


DecisionPriceLookup = Callable[[str, str], float | None]  # (decision_context_id, symbol)


def marketable_limit_price(side: OrderSide, price: float, spread: float) -> float:
    """Limit = quote ± max(2 bps, 0.25 x spread); buys above, sells below."""
    offset = max(price * LIMIT_OFFSET_MIN_BPS / 10_000.0, LIMIT_OFFSET_SPREAD_MULT * spread)
    return price + offset if side == OrderSide.BUY else price - offset


def slippage_bps(side: OrderSide, fill_price: float, decision_price: float) -> float:
    """``side_sign x (fill - decision) / decision x 10^4`` (doc 09 §5)."""
    sign = 1.0 if side == OrderSide.BUY else -1.0
    return sign * (fill_price - decision_price) / decision_price * 10_000.0


class OrderManager:
    def __init__(
        self,
        broker: Broker,
        orders: OrderStore,
        trades: TradeStore,
        clock: Clock,
        *,
        auto_approve: bool = True,
        approval_ttl_minutes: int = 30,
        decision_price_lookup: DecisionPriceLookup | None = None,
        gate: Callable[[Order], Any] | None = None,  # returns GateResult-like
        halt_check: Callable[[Order], bool] | None = None,  # True -> sticky halt
        min_weight_delta: float = MIN_WEIGHT_DELTA,
    ) -> None:
        self._broker = broker
        self._orders = orders
        self._trades = trades
        self._clock = clock
        self._auto_approve = auto_approve
        self._approval_ttl = timedelta(minutes=approval_ttl_minutes)
        self._decision_price = decision_price_lookup or (lambda _dc, _sym: None)
        self._gate = gate
        self._halt_check = halt_check
        self._min_weight_delta = min_weight_delta
        self._reconciled = False

    @property
    def orders_store(self) -> OrderStore:
        return self._orders

    # --- proposal (doc 07 → doc 09 handoff) ---

    def propose_orders(
        self,
        target: TargetPortfolio,
        current: PortfolioState,
        prices: Mapping[str, float],
        spreads: Mapping[str, float] | None = None,
    ) -> list[Order]:
        """Diff target vs current weights into ``proposed`` orders.

        Deltas under ``min_weight_delta`` (0.25% of equity) are skipped. Ids are
        deterministic per (target, symbol, side, run date), so re-running after
        a crash re-derives the same intents instead of new ones.
        """
        if current.equity <= 0:
            return []
        spreads = spreads or {}
        current_weights: dict[str, float] = {}
        for pos in current.positions:
            px = prices.get(pos.symbol, pos.avg_entry_price)
            current_weights[pos.symbol] = (
                current_weights.get(pos.symbol, 0.0) + pos.qty * px / current.equity
            )

        # A symbol may carry multiple target lots (long-term + tactical, or an
        # active lot + the ballast residual): trade the SUMMED weight in one
        # order, labeled with the first lot's sleeve (constructor priority
        # order — active before ballast).
        target_by_symbol: dict[str, Any] = {}
        target_weight_sum: dict[str, float] = {}
        for tw in target.weights:
            target_by_symbol.setdefault(tw.symbol, tw)
            target_weight_sum[tw.symbol] = target_weight_sum.get(tw.symbol, 0.0) + tw.weight
        ts_bucket = target.as_of.date().isoformat()
        out: list[Order] = []
        for symbol in sorted(set(target_by_symbol) | set(current_weights)):
            tw = target_by_symbol.get(symbol)
            target_weight = target_weight_sum.get(symbol, 0.0)
            delta = target_weight - current_weights.get(symbol, 0.0)
            if abs(delta) < self._min_weight_delta:
                continue
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            coid = client_order_id(target.id, symbol, side.value, ts_bucket)
            existing = self._orders.get(coid)
            if existing is not None:
                out.append(existing)
                continue
            order = Order(
                client_order_id=coid,
                symbol=symbol,
                side=side,
                qty=round(abs(delta) * current.equity / price, 4),
                order_type=OrderType.MARKETABLE_LIMIT,
                limit_price=round(marketable_limit_price(side, price, spreads.get(symbol, 0.0)), 4),
                # Crypto rejects DAY at Alpaca — GTC is the sleeve's only tif
                # (ADR-0013); the broker enforces this again at request build.
                tif="gtc" if is_crypto_symbol(symbol) else "day",
                state=OrderState.PROPOSED,
                sleeve=tw.sleeve if tw is not None else None,
                created_at=self._clock.now(),
            )
            self._orders.insert(order)
            out.append(order)
        return out

    # --- state machine (doc 09 §2) ---

    def transition(
        self,
        coid: str,
        to_state: OrderState,
        reason: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> Order:
        """Apply one transition. Illegal transitions raise, never coerce; a
        terminal state is absorbing. Every transition writes one event row."""
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

    # --- approval + submission (doc 09 §2, §4, §6, §7) ---

    def request_approval(self, coid: str) -> Order:
        order = self.transition(coid, OrderState.PENDING_APPROVAL, "awaiting approval")
        return order

    def submit(self, coid: str) -> Order:
        """Submit a ``pending_approval`` order to the broker.

        Requires startup reconciliation complete, no sticky reconciliation
        halt, a decision context id (no order without a snapshot), an unexpired
        approval window, and a permitting gate. A duplicate-id response is
        "already submitted — fetch state", never an error; broker rejects are
        terminal; a disconnect leaves the order ``submitted`` and blocks new
        submissions until reconciliation.

        Both ``halt_check`` and ``gate`` are consulted HERE, per submission —
        never once per job and never cached — so a kill switch or a halt that
        appears mid-run stops every subsequent order in that run (doc 09 §7).
        """
        if not self._reconciled:
            raise ReconciliationPending("startup reconciliation has not completed")
        order = self._require(coid)
        if self._halt_check is not None and self._halt_check(order):
            raise ReconciliationHalt(order.symbol)
        if order.state != OrderState.PENDING_APPROVAL:
            raise IllegalTransition(coid, order.state.value, OrderState.SUBMITTED.value)
        if not order.decision_context_id:
            raise MissingDecisionContext(coid)

        now = self._clock.now()
        if not self._auto_approve:
            entered = self._orders.state_entered_at(coid)
            if entered is not None and now - entered > self._approval_ttl:
                return self.transition(coid, OrderState.CANCELLED, APPROVAL_EXPIRED_REASON)

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
            # Ambiguous: the broker may or may not have the order. Stay in
            # ``submitted`` and block all new submissions until reconciliation
            # resolves it from broker truth (doc 09 §8).
            self._reconciled = False
            raise
        self._orders.set_broker_order_id(coid, broker_order_id)
        return self._require(coid)

    def expire_stale_approvals(self) -> list[Order]:
        """``pending_approval`` past TTL → cancelled ``approval_expired``."""
        now = self._clock.now()
        expired: list[Order] = []
        for order in self._orders.open_orders():
            if order.state != OrderState.PENDING_APPROVAL:
                continue
            entered = self._orders.state_entered_at(order.client_order_id)
            if entered is not None and now - entered > self._approval_ttl:
                expired.append(
                    self.transition(
                        order.client_order_id, OrderState.CANCELLED, APPROVAL_EXPIRED_REASON
                    )
                )
        return expired

    # --- fills (doc 09 §5) ---

    def record_fill(
        self,
        coid: str,
        qty: float,
        price: float,
        filled_at: datetime,
        fees: float = 0.0,
    ) -> Trade:
        order = self._require(coid)
        if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
            raise IllegalTransition(coid, order.state.value, OrderState.FILLED.value)

        decision_price = None
        if order.decision_context_id:
            decision_price = self._decision_price(order.decision_context_id, order.symbol)
        trade = Trade(
            client_order_id=coid,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            fill_price=price,
            decision_price=decision_price,
            slippage_bps=(
                slippage_bps(order.side, price, decision_price)
                if decision_price is not None and decision_price > 0
                else None
            ),
            fees=fees,
            filled_at=filled_at,
        )
        self._trades.insert(trade)

        cumulative = self._trades.filled_qty(coid)
        if cumulative >= order.qty - 1e-9:
            self.transition(coid, OrderState.FILLED, "fully filled")
        elif order.state == OrderState.SUBMITTED:
            self.transition(coid, OrderState.PARTIALLY_FILLED, "partial fill")
        return trade

    # --- day end (doc 09 §8) ---

    def day_end(self) -> list[Order]:
        """Cancel partial-fill remainders; the unfilled portion re-enters the
        next nightly target diff naturally. Also expires stale approvals."""
        cancelled: list[Order] = []
        for order in self._orders.open_orders():
            if order.state == OrderState.PARTIALLY_FILLED:
                self._broker.cancel(order.client_order_id)
                cancelled.append(
                    self.transition(
                        order.client_order_id, OrderState.CANCELLED, "day_end_partial_cancel"
                    )
                )
        self.expire_stale_approvals()
        return cancelled

    # --- startup reconciliation (doc 09 §4) ---

    @property
    def reconciled(self) -> bool:
        return self._reconciled

    def startup_reconciliation(self) -> None:
        """Repair local state from broker truth; blocks submissions until done.

        An order stuck in ``submitted``/``partially_filled`` locally is matched
        by client_order_id: fills that happened during downtime are recorded
        (idempotency guarantees zero duplicates), terminal broker states are
        mirrored, and orders the broker never received are cancelled locally.
        """
        self.sync_orders()
        self._reconciled = True

    def sync_orders(self) -> list[Trade]:
        """Poll every live order against broker truth; return the NEW fills.

        This is the equity/crypto fill poller (the twin of the options and PM
        sleeves' ``sync_orders``). Without it an Alpaca fill never reaches
        ``trades``/``positions`` and periodic reconciliation correctly halts
        the sleeve for divergence.

        Idempotent by construction: the broker's cumulative ``filled_qty``
        minus what we have already recorded is what gets written, so re-polling
        an unchanged order records nothing. Partial fills leave the order
        ``partially_filled`` and are topped up on the next poll; terminal
        broker states are mirrored into the local state machine.
        """
        recorded: list[Trade] = []
        for order in self._orders.open_orders():
            if order.state not in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED):
                continue
            coid = order.client_order_id
            try:
                remote = self._broker.get_order(coid)
            except BrokerRejection:
                # The broker never saw this order: the submit ack was lost
                # before acceptance. Safe to cancel; the intent re-enters the
                # next scheduling bucket with a fresh deterministic id.
                self.transition(coid, OrderState.CANCELLED, "not_found_at_broker")
                continue

            broker_id = str(remote.get("broker_order_id", ""))
            if broker_id and not order.broker_order_id:
                self._orders.set_broker_order_id(coid, broker_id)

            remote_filled = float(remote.get("filled_qty") or 0.0)
            local_filled = self._trades.filled_qty(coid)
            missing = remote_filled - local_filled
            if missing > 1e-9:
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

    def _require(self, coid: str) -> Order:
        order = self._orders.get(coid)
        if order is None:
            raise ExecutionError(f"unknown order {coid}")
        return order
