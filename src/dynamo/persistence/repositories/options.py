"""Options trading repositories — defined-risk sleeve (ADR-0010).

``OptionsOrderRepository`` satisfies the ``OptionOrderStore`` protocol of the
options order manager and ``OptionsTradeRepository`` its ``OptionTradeStore``;
every state transition writes exactly one append-only ``options_order_events``
row in the same transaction (the doc 09 §9 discipline, verbatim).
``OptionsChainRepository`` persists minimal chain snapshots plus the ATM-IV
history behind the IV-rank proxy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.options import (
    OptionChainRow,
    OptionOrder,
    OptionPosition,
    OptionRight,
    OptionStructure,
    OptionTrade,
)
from dynamo.domain.models.portfolio import TERMINAL_ORDER_STATES, OrderSide, OrderState
from dynamo.persistence.schema import (
    options_chain_snapshots,
    options_iv_history,
    options_order_events,
    options_orders,
    options_positions,
    options_trades,
)

_TERMINAL_VALUES = tuple(s.value for s in TERMINAL_ORDER_STATES)


def _row_to_order(row: Any) -> OptionOrder:
    return OptionOrder(
        client_order_id=row.client_order_id,
        occ_symbol=row.occ_symbol,
        underlying=row.underlying,
        structure=OptionStructure(row.structure),
        side=OrderSide(row.side),
        right=OptionRight(row.right),
        expiry=row.expiry,
        strike=row.strike,
        contracts=row.contracts,
        limit_price=row.limit_price,
        state=OrderState(row.state),
        decision_context_id=row.decision_context_id or None,
        broker_order_id=row.broker_order_id,
        created_at=row.created_at,
    )


class OptionsOrderRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, order: OptionOrder) -> None:
        created = order.created_at or datetime.now()
        with self._engine.begin() as cx:
            cx.execute(
                options_orders.insert().values(
                    client_order_id=order.client_order_id,
                    broker_order_id=order.broker_order_id,
                    decision_context_id=order.decision_context_id or "",
                    occ_symbol=order.occ_symbol,
                    underlying=order.underlying,
                    structure=order.structure.value,
                    side=order.side.value,
                    right=order.right.value,
                    expiry=order.expiry,
                    strike=order.strike,
                    contracts=order.contracts,
                    limit_price=order.limit_price,
                    state=order.state.value,
                    created_at=created,
                    updated_at=created,
                )
            )
            cx.execute(
                options_order_events.insert().values(
                    client_order_id=order.client_order_id,
                    from_state=None,
                    to_state=order.state.value,
                    reason="created",
                    payload=None,
                    ts=created,
                )
            )

    def get(self, client_order_id: str) -> OptionOrder | None:
        with self._engine.connect() as cx:
            row = cx.execute(
                select(options_orders).where(options_orders.c.client_order_id == client_order_id)
            ).first()
        return _row_to_order(row) if row is not None else None

    def open_orders(self) -> list[OptionOrder]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(options_orders).where(options_orders.c.state.not_in(_TERMINAL_VALUES))
            ).all()
        return [_row_to_order(r) for r in rows]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(options_orders)
                .where(options_orders.c.client_order_id == client_order_id)
                .values(broker_order_id=broker_order_id)
            )

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(options_orders)
                .where(options_orders.c.client_order_id == client_order_id)
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
    ) -> OptionOrder:
        """State update + exactly one append-only event row, atomically."""
        with self._engine.begin() as cx:
            cx.execute(
                update(options_orders)
                .where(options_orders.c.client_order_id == client_order_id)
                .values(state=to_state.value, updated_at=ts)
            )
            cx.execute(
                options_order_events.insert().values(
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
            raise LookupError(f"option order {client_order_id} disappeared during transition")
        return order

    def events(self, client_order_id: str) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(options_order_events)
                .where(options_order_events.c.client_order_id == client_order_id)
                .order_by(options_order_events.c.id)
            ).all()
        return [dict(r._mapping) for r in rows]


class OptionsTradeRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, trade: OptionTrade) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                options_trades.insert().values(
                    client_order_id=trade.client_order_id,
                    occ_symbol=trade.occ_symbol,
                    underlying=trade.underlying,
                    side=trade.side.value,
                    right=trade.right.value,
                    contracts=trade.contracts,
                    price=trade.price,
                    fees=trade.fees,
                    kind=trade.kind,
                    filled_at=trade.filled_at,
                )
            )

    def filled_contracts(self, client_order_id: str) -> int:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(options_trades.c.contracts).where(
                    options_trades.c.client_order_id == client_order_id,
                    options_trades.c.kind == "fill",
                )
            ).all()
        return int(sum(r.contracts for r in rows))

    def fills_today(self, day: date) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(select(options_trades).where(options_trades.c.kind == "fill")).all()
        return [dict(r._mapping) for r in rows if r.filled_at.date() == day]


class OptionsPositionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def apply_fill(self, order: OptionOrder, trade: OptionTrade, now: datetime) -> None:
        """Upsert a position from a fill on its opening order.

        Opening fills accumulate contracts at a blended average price; the
        structure/side/right come from the order (the fill alone cannot know
        whether a short call is covered).
        """
        with self._engine.begin() as cx:
            row = cx.execute(
                select(options_positions).where(options_positions.c.occ_symbol == order.occ_symbol)
            ).first()
            if row is None:
                if trade.contracts <= 0:
                    return
                cx.execute(
                    options_positions.insert().values(
                        occ_symbol=order.occ_symbol,
                        underlying=order.underlying,
                        structure=order.structure.value,
                        side=order.side.value,
                        right=order.right.value,
                        expiry=order.expiry,
                        strike=order.strike,
                        contracts=trade.contracts,
                        avg_price=trade.price,
                        updated_at=now,
                    )
                )
                return
            new_contracts = row.contracts + trade.contracts
            avg = (
                (row.avg_price * row.contracts + trade.price * trade.contracts) / new_contracts
                if new_contracts > 0
                else 0.0
            )
            cx.execute(
                update(options_positions)
                .where(options_positions.c.occ_symbol == order.occ_symbol)
                .values(contracts=new_contracts, avg_price=avg, updated_at=now)
            )

    def upsert(self, position: OptionPosition, now: datetime) -> None:
        stmt = pg_insert(options_positions).values(
            occ_symbol=position.occ_symbol,
            underlying=position.underlying,
            structure=position.structure.value,
            side=position.side.value,
            right=position.right.value,
            expiry=position.expiry,
            strike=position.strike,
            contracts=position.contracts,
            avg_price=position.avg_price,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[options_positions.c.occ_symbol],
            set_={
                "contracts": position.contracts,
                "avg_price": position.avg_price,
                "updated_at": now,
            },
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)

    def close(self, occ_symbol: str) -> None:
        """Remove a position after expiry/assignment settles it."""
        with self._engine.begin() as cx:
            cx.execute(
                options_positions.delete().where(options_positions.c.occ_symbol == occ_symbol)
            )

    def open_positions(self) -> list[OptionPosition]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(options_positions).where(options_positions.c.contracts > 0)
            ).all()
        return [
            OptionPosition(
                occ_symbol=r.occ_symbol,
                underlying=r.underlying,
                structure=OptionStructure(r.structure),
                side=OrderSide(r.side),
                right=OptionRight(r.right),
                expiry=r.expiry,
                strike=r.strike,
                contracts=r.contracts,
                avg_price=r.avg_price,
            )
            for r in rows
        ]

    def expired_positions(self, today: date) -> list[OptionPosition]:
        return [p for p in self.open_positions() if p.expiry < today]


class OptionsChainRepository:
    """Minimal chain persistence + the ATM-IV history behind IV rank."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert_snapshot(self, rows: Sequence[OptionChainRow]) -> int:
        if not rows:
            return 0
        values = [
            {
                "underlying": r.underlying,
                "as_of": r.as_of,
                "expiry": r.expiry,
                "strike": r.strike,
                "cp": r.right.value,
                "bid": r.bid,
                "ask": r.ask,
                "oi": r.open_interest,
                "iv": r.iv,
                "delta": r.delta,
            }
            for r in rows
        ]
        stmt = pg_insert(options_chain_snapshots).values(values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[
                options_chain_snapshots.c.underlying,
                options_chain_snapshots.c.as_of,
                options_chain_snapshots.c.expiry,
                options_chain_snapshots.c.strike,
                options_chain_snapshots.c.cp,
            ]
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)
        return len(values)

    def append_iv(self, underlying: str, as_of: date, atm_iv: float) -> None:
        stmt = pg_insert(options_iv_history).values(
            underlying=underlying, as_of=as_of, atm_iv=atm_iv
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[options_iv_history.c.underlying, options_iv_history.c.as_of],
            set_={"atm_iv": atm_iv},
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)

    def iv_history(self, underlying: str, limit: int = 252) -> list[float]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(options_iv_history.c.atm_iv)
                .where(options_iv_history.c.underlying == underlying)
                .order_by(options_iv_history.c.as_of.desc())
                .limit(limit)
            ).all()
        return [float(r.atm_iv) for r in rows]
