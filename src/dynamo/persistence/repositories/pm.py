"""Prediction-market trading repositories (ADR-0009; ADR-0011).

``PMOrderRepository`` satisfies the ``PMOrderStore`` protocol of the PM order
manager and ``PMTradeRepository`` its ``PMTradeStore``; every state transition
writes exactly one append-only ``pm_order_events`` row in the same transaction
(the doc 09 §9 discipline, verbatim). Both sleeves share these tables: Kalshi
rows are real (demo/prod), ``venue='polymarket'`` rows are SIMULATED
(ADR-0011). ``PMSimLedgerRepository`` holds the sim bankroll's cash ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.pm import (
    PM_VENUE_POLYMARKET,
    PMOrder,
    PMPosition,
    PMSide,
    PMSimLedgerEntry,
    PMTrade,
)
from dynamo.domain.models.portfolio import TERMINAL_ORDER_STATES, OrderState
from dynamo.persistence.schema import (
    pm_order_events,
    pm_orders,
    pm_positions,
    pm_sim_ledger,
    pm_trades,
)

_TERMINAL_VALUES = tuple(s.value for s in TERMINAL_ORDER_STATES)


def _row_to_order(row: Any) -> PMOrder:
    return PMOrder(
        client_order_id=row.client_order_id,
        venue=row.venue,
        market_id=row.market_id,
        side=PMSide(row.side),
        contracts=row.contracts,
        limit_price_cents=row.limit_price_cents,
        state=OrderState(row.state),
        decision_context_id=row.decision_context_id or None,
        broker_order_id=row.broker_order_id,
        created_at=row.created_at,
    )


class PMOrderRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, order: PMOrder) -> None:
        created = order.created_at or datetime.now()
        with self._engine.begin() as cx:
            cx.execute(
                pm_orders.insert().values(
                    client_order_id=order.client_order_id,
                    broker_order_id=order.broker_order_id,
                    decision_context_id=order.decision_context_id or "",
                    venue=order.venue,
                    market_id=order.market_id,
                    side=order.side.value,
                    contracts=order.contracts,
                    limit_price_cents=order.limit_price_cents,
                    state=order.state.value,
                    created_at=created,
                    updated_at=created,
                )
            )
            cx.execute(
                pm_order_events.insert().values(
                    client_order_id=order.client_order_id,
                    from_state=None,
                    to_state=order.state.value,
                    reason="created",
                    payload=None,
                    ts=created,
                )
            )

    def get(self, client_order_id: str) -> PMOrder | None:
        with self._engine.connect() as cx:
            row = cx.execute(
                select(pm_orders).where(pm_orders.c.client_order_id == client_order_id)
            ).first()
        return _row_to_order(row) if row is not None else None

    def open_orders(self) -> list[PMOrder]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(pm_orders).where(pm_orders.c.state.not_in(_TERMINAL_VALUES))
            ).all()
        return [_row_to_order(r) for r in rows]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(pm_orders)
                .where(pm_orders.c.client_order_id == client_order_id)
                .values(broker_order_id=broker_order_id)
            )

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(pm_orders)
                .where(pm_orders.c.client_order_id == client_order_id)
                .values(decision_context_id=decision_context_id)
            )

    def transition_with_event(
        self,
        client_order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str,
        ts: datetime,
        payload: Mapping[str, Any] | None = None,
    ) -> PMOrder:
        """State update + exactly one append-only event row, atomically."""
        with self._engine.begin() as cx:
            cx.execute(
                update(pm_orders)
                .where(pm_orders.c.client_order_id == client_order_id)
                .values(state=to_state.value, updated_at=ts)
            )
            cx.execute(
                pm_order_events.insert().values(
                    client_order_id=client_order_id,
                    from_state=from_state.value,
                    to_state=to_state.value,
                    reason=reason,
                    payload=dict(payload) if payload is not None else None,
                    ts=ts,
                )
            )
        order = self.get(client_order_id)
        if order is None:  # pragma: no cover - row cannot vanish mid-transaction
            raise LookupError(f"pm order {client_order_id} disappeared during transition")
        return order

    def events(self, client_order_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(pm_order_events)
                .where(pm_order_events.c.client_order_id == client_order_id)
                .order_by(pm_order_events.c.id)
            ).all()
        return [dict(r._mapping) for r in rows]


class PMTradeRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, trade: PMTrade) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                pm_trades.insert().values(
                    client_order_id=trade.client_order_id,
                    venue=trade.venue,
                    market_id=trade.market_id,
                    side=trade.side.value,
                    contracts=trade.contracts,
                    price_cents=trade.price_cents,
                    fees_cents=trade.fees_cents,
                    kind=trade.kind,
                    realized_pnl_cents=trade.realized_pnl_cents,
                    filled_at=trade.filled_at,
                )
            )

    def filled_contracts(self, client_order_id: str) -> int:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(pm_trades.c.contracts).where(
                    pm_trades.c.client_order_id == client_order_id,
                    pm_trades.c.kind == "fill",
                )
            ).all()
        return int(sum(r.contracts for r in rows))

    def all_trades(self, market_id: str | None = None) -> list[PMTrade]:
        stmt = select(pm_trades).order_by(pm_trades.c.filled_at)
        if market_id is not None:
            stmt = stmt.where(pm_trades.c.market_id == market_id)
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return [
            PMTrade(
                client_order_id=r.client_order_id,
                venue=r.venue,
                market_id=r.market_id,
                side=PMSide(r.side),
                contracts=r.contracts,
                price_cents=r.price_cents,
                fees_cents=r.fees_cents,
                kind=r.kind,
                realized_pnl_cents=r.realized_pnl_cents,
                filled_at=r.filled_at,
            )
            for r in rows
        ]

    def realized_pnl_cents(self) -> float:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(pm_trades.c.realized_pnl_cents).where(pm_trades.c.kind == "settlement")
            ).all()
        return float(sum(r.realized_pnl_cents or 0.0 for r in rows))


class PMPositionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def apply_fill(self, trade: PMTrade, now: datetime, question: str = "") -> None:
        """Upsert a position from a buy fill (long-only: buys accumulate)."""
        with self._engine.begin() as cx:
            row = cx.execute(
                select(pm_positions).where(
                    pm_positions.c.venue == trade.venue,
                    pm_positions.c.market_id == trade.market_id,
                )
            ).first()
            if row is None:
                if trade.contracts <= 0:
                    return
                cx.execute(
                    pm_positions.insert().values(
                        venue=trade.venue,
                        market_id=trade.market_id,
                        side=trade.side.value,
                        contracts=trade.contracts,
                        avg_price_cents=trade.price_cents,
                        question=question,
                        updated_at=now,
                    )
                )
                return
            new_contracts = row.contracts + trade.contracts
            avg = (
                (row.avg_price_cents * row.contracts + trade.price_cents * trade.contracts)
                / new_contracts
                if new_contracts > 0
                else 0.0
            )
            cx.execute(
                update(pm_positions)
                .where(
                    pm_positions.c.venue == trade.venue,
                    pm_positions.c.market_id == trade.market_id,
                )
                .values(contracts=new_contracts, avg_price_cents=avg, updated_at=now)
            )

    def upsert(self, position: PMPosition, now: datetime) -> None:
        stmt = pg_insert(pm_positions).values(
            venue=position.venue,
            market_id=position.market_id,
            side=position.side.value,
            contracts=position.contracts,
            avg_price_cents=position.avg_price_cents,
            question=position.question,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[pm_positions.c.venue, pm_positions.c.market_id],
            set_={
                "side": position.side.value,
                "contracts": position.contracts,
                "avg_price_cents": position.avg_price_cents,
                "question": position.question,
                "updated_at": now,
            },
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)

    def close(self, venue: str, market_id: str) -> None:
        """Remove a position after resolution settles it."""
        with self._engine.begin() as cx:
            cx.execute(
                pm_positions.delete().where(
                    pm_positions.c.venue == venue, pm_positions.c.market_id == market_id
                )
            )

    def open_positions(self, venue: str | None = None) -> list[PMPosition]:
        """Open positions, optionally for ONE venue (Kalshi call sites must
        pass their venue so SIMULATED Polymarket rows never leak into Kalshi
        sizing or reconciliation)."""
        stmt = select(pm_positions).where(pm_positions.c.contracts > 0)
        if venue is not None:
            stmt = stmt.where(pm_positions.c.venue == venue)
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return [
            PMPosition(
                venue=r.venue,
                market_id=r.market_id,
                side=PMSide(r.side),
                contracts=r.contracts,
                avg_price_cents=r.avg_price_cents,
                question=r.question,
            )
            for r in rows
        ]


class PMSimLedgerRepository:
    """Cash ledger for the SIMULATED Polymarket sleeve's bankroll (ADR-0011).

    Satisfies the ``PMSimLedgerStore`` protocol of
    :mod:`dynamo.execution.polymarket_sim`. Every movement is idempotent via
    the (venue, kind, ref) unique key — ``append`` reports whether the row
    was actually written, so a re-run debits/credits nothing twice.
    """

    def __init__(self, engine: Engine, venue: str = PM_VENUE_POLYMARKET) -> None:
        self._engine = engine
        self._venue = venue

    def ensure_seeded(self, bankroll_cents: float, now: datetime) -> None:
        """Write the one-time seed deposit (no-op once it exists)."""
        self.append(
            PMSimLedgerEntry(
                venue=self._venue,
                kind="deposit",
                ref="initial",
                market_id=None,
                amount_cents=bankroll_cents,
                note="simulated bankroll seed",
                ts=now,
            )
        )

    def append(self, entry: PMSimLedgerEntry) -> bool:
        stmt = (
            pg_insert(pm_sim_ledger)
            .values(
                venue=entry.venue,
                kind=entry.kind,
                ref=entry.ref,
                market_id=entry.market_id,
                amount_cents=entry.amount_cents,
                note=entry.note,
                ts=entry.ts,
            )
            .on_conflict_do_nothing(constraint="uq_pm_sim_ledger_movement")
        )
        with self._engine.begin() as cx:
            result = cx.execute(stmt)
        return bool(result.rowcount)

    def cash_cents(self) -> float:
        with self._engine.connect() as cx:
            value = cx.execute(
                select(func.coalesce(func.sum(pm_sim_ledger.c.amount_cents), 0.0)).where(
                    pm_sim_ledger.c.venue == self._venue
                )
            ).scalar()
        return float(value or 0.0)

    def entries(self) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(pm_sim_ledger)
                .where(pm_sim_ledger.c.venue == self._venue)
                .order_by(pm_sim_ledger.c.id)
            ).all()
        return [dict(r._mapping) for r in rows]
