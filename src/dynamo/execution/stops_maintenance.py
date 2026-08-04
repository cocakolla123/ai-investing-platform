"""Protective-stop maintenance (doc 09 §3; doc 06 §5).

Broker-held stops are refreshed with ``replace`` (same client-order lineage);
a failed replace becomes cancel + resubmit — never two live stops for one
position. Synthetic stops are evaluated on the 15-minute intraday check and a
breach produces a marketable-limit exit order through the *full* state machine
(no shortcut path exists, including for stop maintenance).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from dynamo.domain.models.portfolio import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Position,
)
from dynamo.domain.models.risk import StopOrderSpec
from dynamo.domain.ports import Broker, Clock
from dynamo.execution.client_order_id import client_order_id
from dynamo.execution.errors import BrokerRejection, ExecutionError
from dynamo.execution.order_manager import OrderManager, marketable_limit_price


def intraday_ts_bucket(clock: Clock) -> str:
    """15-minute scheduling slot for stop-maintenance idempotency (doc 09 §4)."""
    now = clock.now()
    return f"{now.date().isoformat()}T{now.hour:02d}:{(now.minute // 15) * 15:02d}"


@dataclass
class StopsMaintainer:
    broker: Broker
    manager: OrderManager
    clock: Clock
    # symbol -> client_order_id of the single live broker-held stop
    broker_held: dict[str, str] = field(default_factory=dict)

    def apply_stops(
        self,
        specs: list[StopOrderSpec],
        positions: Mapping[str, Position],
        decision_context_ids: Mapping[str, str],
        target_portfolio_id: str = "stops",
    ) -> list[Order]:
        """Apply broker-held stop updates: replace in place; on a failed
        replace, cancel and resubmit through the full state machine."""
        touched: list[Order] = []
        for spec in specs:
            position = positions.get(spec.symbol)
            if position is None or position.qty <= 0:
                continue
            existing = self.broker_held.get(spec.symbol)
            if existing is not None:
                try:
                    self.broker.replace(existing, spec.stop_price)
                    continue
                except ExecutionError:
                    # Failed replace → cancel + resubmit; never two live stops.
                    with contextlib.suppress(BrokerRejection):
                        self.broker.cancel(existing)
                    self.manager.transition(
                        existing, OrderState.CANCELLED, "stop replace failed; resubmitting"
                    )
                    del self.broker_held[spec.symbol]
            order = self._submit_stop(
                spec, position, decision_context_ids.get(spec.symbol, ""), target_portfolio_id
            )
            touched.append(order)
            if order.state == OrderState.SUBMITTED:
                self.broker_held[spec.symbol] = order.client_order_id
        return touched

    def evaluate_synthetic_stops(
        self,
        specs: list[StopOrderSpec],
        quotes: Mapping[str, float],
        positions: Mapping[str, Position],
        decision_context_ids: Mapping[str, str],
        target_portfolio_id: str = "synthetic-stops",
    ) -> list[Order]:
        """Quote at/below a synthetic stop → marketable-limit exit order,
        driven through the full proposed → ... → submitted machine."""
        exits: list[Order] = []
        for spec in specs:
            position = positions.get(spec.symbol)
            quote = quotes.get(spec.symbol)
            if position is None or position.qty <= 0 or quote is None:
                continue
            if quote > spec.stop_price:
                continue
            order = self._create_exit_order(
                spec, position, quote, decision_context_ids.get(spec.symbol, ""),
                target_portfolio_id,
            )
            exits.append(order)
        return exits

    # --- internals: every path goes through the full state machine ---

    def _drive_to_submission(self, order: Order, decision_context_id: str) -> Order:
        coid = order.client_order_id
        self.manager.transition(coid, OrderState.RISK_CHECKED, "stop spec from risk engine")
        self.manager.request_approval(coid)
        if decision_context_id:
            self.manager.attach_decision_context(coid, decision_context_id)
        return self.manager.submit(coid)

    def _submit_stop(
        self,
        spec: StopOrderSpec,
        position: Position,
        decision_context_id: str,
        target_portfolio_id: str,
    ) -> Order:
        # Stop level is part of the intent: a cancel+resubmit at a new level
        # inside the same 15-min bucket must not collide with the old id.
        coid = client_order_id(
            f"{target_portfolio_id}:{spec.stop_price}",
            spec.symbol,
            OrderSide.SELL.value,
            intraday_ts_bucket(self.clock),
        )
        order = Order(
            client_order_id=coid,
            symbol=spec.symbol,
            side=OrderSide.SELL,
            qty=position.qty,
            order_type=OrderType.STOP,
            stop_price=spec.stop_price,
            tif="gtc",
            state=OrderState.PROPOSED,
            created_at=self.clock.now(),
        )
        self.manager.orders_store.insert(order)
        return self._drive_to_submission(order, decision_context_id)

    def _create_exit_order(
        self,
        spec: StopOrderSpec,
        position: Position,
        quote: float,
        decision_context_id: str,
        target_portfolio_id: str,
    ) -> Order:
        coid = client_order_id(
            target_portfolio_id, spec.symbol, OrderSide.SELL.value, intraday_ts_bucket(self.clock)
        )
        order = Order(
            client_order_id=coid,
            symbol=spec.symbol,
            side=OrderSide.SELL,
            qty=position.qty,
            order_type=OrderType.SYNTHETIC_STOP_EXIT,
            limit_price=round(marketable_limit_price(OrderSide.SELL, quote, 0.0), 4),
            tif="day",
            state=OrderState.PROPOSED,
            created_at=self.clock.now(),
        )
        self.manager.orders_store.insert(order)
        return self._drive_to_submission(order, decision_context_id)
