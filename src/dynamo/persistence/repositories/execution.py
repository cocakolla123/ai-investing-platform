"""Execution repositories: orders, order events, trades, positions,
reconciliation, equity curve (doc 09 §9).

``OrderRepository`` satisfies the ``OrderStore`` protocol of the order manager
and ``TradeRepository`` its ``TradeStore``; every state transition writes
exactly one append-only ``order_events`` row in the same transaction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.portfolio import (
    TERMINAL_ORDER_STATES,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    PortfolioState,
    Position,
    Trade,
)
from dynamo.domain.models.research import Sleeve
from dynamo.persistence.schema import (
    equity_curve,
    order_events,
    orders,
    positions,
    reconciliation_acks,
    reconciliation_runs,
    trades,
)

_TERMINAL_VALUES = tuple(s.value for s in TERMINAL_ORDER_STATES)


def _row_to_order(row: Any) -> Order:
    return Order(
        client_order_id=row.client_order_id,
        symbol=row.symbol,
        side=OrderSide(row.side),
        qty=row.qty,
        order_type=OrderType(row.order_type),
        limit_price=row.limit_price,
        stop_price=row.stop_price,
        tif=row.tif,
        state=OrderState(row.state),
        sleeve=Sleeve(row.sleeve) if row.sleeve else None,
        decision_context_id=row.decision_context_id or None,
        broker_order_id=row.broker_order_id,
        created_at=row.created_at,
    )


class OrderRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, order: Order) -> None:
        created = order.created_at or datetime.now()
        with self._engine.begin() as cx:
            cx.execute(
                orders.insert().values(
                    client_order_id=order.client_order_id,
                    broker_order_id=order.broker_order_id,
                    decision_context_id=order.decision_context_id or "",
                    symbol=order.symbol,
                    side=order.side.value,
                    qty=order.qty,
                    order_type=order.order_type.value,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    tif=order.tif,
                    state=order.state.value,
                    sleeve=order.sleeve.value if order.sleeve else None,
                    created_at=created,
                    updated_at=created,
                )
            )
            cx.execute(
                order_events.insert().values(
                    client_order_id=order.client_order_id,
                    from_state=None,
                    to_state=order.state.value,
                    reason="created",
                    payload=None,
                    ts=created,
                )
            )

    def get(self, client_order_id: str) -> Order | None:
        with self._engine.connect() as cx:
            row = cx.execute(
                select(orders).where(orders.c.client_order_id == client_order_id)
            ).first()
        return _row_to_order(row) if row is not None else None

    def open_orders(self) -> list[Order]:
        with self._engine.connect() as cx:
            rows = cx.execute(select(orders).where(orders.c.state.not_in(_TERMINAL_VALUES))).all()
        return [_row_to_order(r) for r in rows]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(orders)
                .where(orders.c.client_order_id == client_order_id)
                .values(broker_order_id=broker_order_id)
            )

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(orders)
                .where(orders.c.client_order_id == client_order_id)
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
    ) -> Order:
        """State update + exactly one append-only event row, atomically."""
        with self._engine.begin() as cx:
            cx.execute(
                update(orders)
                .where(orders.c.client_order_id == client_order_id)
                .values(state=to_state.value, updated_at=ts)
            )
            cx.execute(
                order_events.insert().values(
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
            raise LookupError(f"order {client_order_id} disappeared during transition")
        return order

    def state_entered_at(self, client_order_id: str) -> datetime | None:
        with self._engine.connect() as cx:
            row = cx.execute(
                select(order_events.c.ts)
                .where(order_events.c.client_order_id == client_order_id)
                .order_by(order_events.c.id.desc())
                .limit(1)
            ).first()
        return row.ts if row is not None else None

    def events(self, client_order_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(order_events)
                .where(order_events.c.client_order_id == client_order_id)
                .order_by(order_events.c.id)
            ).all()
        return [dict(r._mapping) for r in rows]


@dataclass(frozen=True)
class ClosedTrade:
    """A fully exited round trip, realized P&L from FIFO lot matching."""

    symbol: str
    qty: float
    opened_at: datetime
    closed_at: datetime
    entry_avg_price: float
    exit_avg_price: float
    realized_pnl: float

    @property
    def pnl_pct(self) -> float:
        if self.entry_avg_price <= 0:
            return 0.0
        return (self.exit_avg_price - self.entry_avg_price) / self.entry_avg_price * 100.0

    @property
    def holding_days(self) -> int:
        return (self.closed_at - self.opened_at).days


def closed_trades_from_fills(fills: Sequence[Trade]) -> list[ClosedTrade]:
    """Detect closed round trips per symbol: FIFO-lot matching, a trip closes
    when the net position returns to zero (long-only platform, doc 00 §3)."""
    by_symbol: dict[str, list[Trade]] = {}
    for f in sorted(fills, key=lambda t: t.filled_at):
        by_symbol.setdefault(f.symbol, []).append(f)

    out: list[ClosedTrade] = []
    for symbol, symbol_fills in by_symbol.items():
        lots: list[list[float]] = []  # [qty_remaining, price]
        opened_at: datetime | None = None
        bought_qty = bought_cost = sold_qty = sold_proceeds = realized = 0.0
        for f in symbol_fills:
            if f.side == OrderSide.BUY:
                if opened_at is None:
                    opened_at = f.filled_at
                lots.append([f.qty, f.fill_price])
                bought_qty += f.qty
                bought_cost += f.qty * f.fill_price
            else:
                remaining = f.qty
                while remaining > 1e-9 and lots:
                    lot = lots[0]
                    take = min(lot[0], remaining)
                    realized += take * (f.fill_price - lot[1])
                    lot[0] -= take
                    remaining -= take
                    if lot[0] <= 1e-9:
                        lots.pop(0)
                sold_qty += f.qty
                sold_proceeds += f.qty * f.fill_price
                if not lots and opened_at is not None and sold_qty > 0:
                    out.append(
                        ClosedTrade(
                            symbol=symbol,
                            qty=sold_qty,
                            opened_at=opened_at,
                            closed_at=f.filled_at,
                            entry_avg_price=bought_cost / bought_qty if bought_qty else 0.0,
                            exit_avg_price=sold_proceeds / sold_qty,
                            realized_pnl=realized,
                        )
                    )
                    opened_at = None
                    bought_qty = bought_cost = sold_qty = sold_proceeds = realized = 0.0
    return out


class TradeRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, trade: Trade) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                trades.insert().values(
                    client_order_id=trade.client_order_id,
                    symbol=trade.symbol,
                    side=trade.side.value,
                    qty=trade.qty,
                    fill_price=trade.fill_price,
                    decision_price=trade.decision_price,
                    slippage_bps=trade.slippage_bps,
                    fees=trade.fees,
                    filled_at=trade.filled_at,
                )
            )

    def filled_qty(self, client_order_id: str) -> float:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(trades.c.qty).where(trades.c.client_order_id == client_order_id)
            ).all()
        return float(sum(r.qty for r in rows))

    def fills(self, symbol: str | None = None) -> list[Trade]:
        stmt = select(trades).order_by(trades.c.filled_at)
        if symbol is not None:
            stmt = stmt.where(trades.c.symbol == symbol)
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return [
            Trade(
                client_order_id=r.client_order_id,
                symbol=r.symbol,
                side=OrderSide(r.side),
                qty=r.qty,
                fill_price=r.fill_price,
                decision_price=r.decision_price,
                slippage_bps=r.slippage_bps,
                fees=r.fees,
                filled_at=r.filled_at,
            )
            for r in rows
        ]

    def closed_trades(self, symbol: str | None = None) -> list[ClosedTrade]:
        return closed_trades_from_fills(self.fills(symbol))


class PositionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def apply_fill(self, trade: Trade, now: datetime) -> None:
        """Upsert the position from a fill; a position at zero qty is removed."""
        signed = trade.qty if trade.side == OrderSide.BUY else -trade.qty
        with self._engine.begin() as cx:
            row = cx.execute(select(positions).where(positions.c.symbol == trade.symbol)).first()
            if row is None:
                if signed <= 0:
                    return
                cx.execute(
                    positions.insert().values(
                        symbol=trade.symbol,
                        qty=signed,
                        avg_entry_price=trade.fill_price,
                        lots=[],
                        updated_at=now,
                    )
                )
                return
            new_qty = row.qty + signed
            if new_qty <= 1e-9:
                cx.execute(positions.delete().where(positions.c.symbol == trade.symbol))
                return
            if signed > 0:
                avg = (row.avg_entry_price * row.qty + trade.fill_price * signed) / new_qty
            else:
                avg = row.avg_entry_price
            cx.execute(
                update(positions)
                .where(positions.c.symbol == trade.symbol)
                .values(qty=new_qty, avg_entry_price=avg, updated_at=now)
            )

    def all_positions(self) -> list[Position]:
        with self._engine.connect() as cx:
            rows = cx.execute(select(positions)).all()
        return [
            Position(
                symbol=r.symbol,
                qty=r.qty,
                avg_entry_price=r.avg_entry_price,
                current_stop=r.current_stop,
                stop_kind=r.stop_kind,
            )
            for r in rows
        ]

    def snapshot(self, as_of: datetime, equity: float, cash: float) -> PortfolioState:
        return PortfolioState(
            as_of=as_of, equity=equity, cash=cash, positions=tuple(self.all_positions())
        )


class ReconciliationRepository:
    """``reconciliation_runs`` + sticky-halt acks (doc 09 §5)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record_run(
        self, ts: datetime, scope: str, result: str, divergences: Sequence[dict[str, Any]]
    ) -> int:
        with self._engine.begin() as cx:
            run_id = cx.execute(
                reconciliation_runs.insert()
                .values(ts=ts, scope=scope, result=result, divergences=list(divergences))
                .returning(reconciliation_runs.c.id)
            ).scalar_one()
        return int(run_id)

    def record_ack(self, run_id: int, acked_by: str, acked_at: datetime, note: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                reconciliation_acks.insert().values(
                    run_id=run_id, acked_by=acked_by, acked_at=acked_at, note=note
                )
            )

    def unacked_divergent_runs(self) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(reconciliation_runs)
                .where(
                    reconciliation_runs.c.result == "divergent",
                    reconciliation_runs.c.id.not_in(select(reconciliation_acks.c.run_id)),
                )
                .order_by(reconciliation_runs.c.id)
            ).all()
        return [dict(r._mapping) for r in rows]


class EquityCurveRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, as_of: date, equity: float, cash: float, gross_exposure: float = 0.0) -> None:
        stmt = pg_insert(equity_curve).values(
            as_of=as_of, equity=equity, cash=cash, gross_exposure=gross_exposure
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[equity_curve.c.as_of],
            set_={"equity": equity, "cash": cash, "gross_exposure": gross_exposure},
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)


class CashLedgerRepository:
    """Internal cash from the ledger (dynamo.execution.cash_ledger).

    Opening cash is the earliest ``equity_curve`` row: the bootstrap snapshot
    taken from the broker before any fill. Every trade filled on or after that
    date is applied on top. Only the equities/crypto ``trades`` table is
    included; options and prediction-market fills keep their own ledgers, so an
    options fill on the same broker account shows up as a cash divergence
    rather than being silently absorbed.

    ``equity_curve`` has no write timestamp, so the opening row must be taken
    before the account's first fill, on a day with no fills, and never
    rewritten (the nightly snapshot skips it). Fills on the opening date
    itself are counted.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def opening(self) -> tuple[date, float] | None:
        with self._engine.connect() as cx:
            row = cx.execute(
                select(equity_curve.c.as_of, equity_curve.c.cash)
                .order_by(equity_curve.c.as_of)
                .limit(1)
            ).first()
        return None if row is None else (row.as_of, float(row.cash))

    def cash(self, as_of: datetime | None = None) -> float | None:
        """Ledger cash, optionally as of a moment (fills after it excluded).
        None when there is no opening snapshot yet."""
        from datetime import UTC
        from datetime import time as dtime

        from dynamo.execution.cash_ledger import Fill, ledger_cash

        opening = self.opening()
        if opening is None:
            return None
        start = datetime.combine(opening[0], dtime.min, tzinfo=UTC)
        stmt = select(trades.c.side, trades.c.qty, trades.c.fill_price, trades.c.fees).where(
            trades.c.filled_at >= start
        )
        if as_of is not None:
            stmt = stmt.where(trades.c.filled_at <= as_of)
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return ledger_cash(
            opening[1],
            (Fill(r.side, float(r.qty), float(r.fill_price), float(r.fees or 0.0)) for r in rows),
        )
