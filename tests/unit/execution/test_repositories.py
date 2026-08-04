"""Execution repositories: pure closed-trade detection, plus DB round-trips
(integration-marked; skipped unless INTEGRATION_DB_URL is set)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from dynamo.domain.models.portfolio import OrderSide, OrderState, Trade
from dynamo.persistence.repositories.execution import (
    ClosedTrade,
    closed_trades_from_fills,
)
from tests.unit.execution.helpers import make_order

NOW = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)


def _fill(side: OrderSide, qty: float, price: float, day: int, symbol: str = "AAPL") -> Trade:
    return Trade(
        client_order_id=f"c-{side.value}-{day}",
        symbol=symbol,
        side=side,
        qty=qty,
        fill_price=price,
        filled_at=NOW + timedelta(days=day),
    )


class TestClosedTradeDetection:
    def test_simple_round_trip(self) -> None:
        fills = [_fill(OrderSide.BUY, 10, 100.0, 0), _fill(OrderSide.SELL, 10, 110.0, 5)]
        closed = closed_trades_from_fills(fills)
        assert len(closed) == 1
        ct = closed[0]
        assert ct.realized_pnl == pytest.approx(100.0)
        assert ct.pnl_pct == pytest.approx(10.0)
        assert ct.holding_days == 5

    def test_fifo_lot_matching(self) -> None:
        fills = [
            _fill(OrderSide.BUY, 10, 100.0, 0),
            _fill(OrderSide.BUY, 10, 120.0, 1),
            _fill(OrderSide.SELL, 15, 130.0, 2),  # closes lot 1 + half of lot 2
            _fill(OrderSide.SELL, 5, 90.0, 3),  # rest of lot 2 at a loss
        ]
        closed = closed_trades_from_fills(fills)
        assert len(closed) == 1
        # FIFO: 10@(130-100) + 5@(130-120) + 5@(90-120) = 300 + 50 - 150
        assert closed[0].realized_pnl == pytest.approx(200.0)

    def test_open_position_not_closed(self) -> None:
        fills = [_fill(OrderSide.BUY, 10, 100.0, 0), _fill(OrderSide.SELL, 4, 110.0, 1)]
        assert closed_trades_from_fills(fills) == []

    def test_reopened_position_yields_two_round_trips(self) -> None:
        fills = [
            _fill(OrderSide.BUY, 10, 100.0, 0),
            _fill(OrderSide.SELL, 10, 105.0, 1),
            _fill(OrderSide.BUY, 5, 102.0, 2),
            _fill(OrderSide.SELL, 5, 101.0, 4),
        ]
        closed = closed_trades_from_fills(fills)
        assert [c.realized_pnl for c in closed] == [pytest.approx(50.0), pytest.approx(-5.0)]
        assert closed[1].holding_days == 2

    def test_symbols_are_independent(self) -> None:
        fills = [
            _fill(OrderSide.BUY, 10, 100.0, 0, "AAPL"),
            _fill(OrderSide.BUY, 10, 50.0, 0, "MSFT"),
            _fill(OrderSide.SELL, 10, 101.0, 1, "AAPL"),
        ]
        closed = closed_trades_from_fills(fills)
        assert [c.symbol for c in closed] == ["AAPL"]


# --- DB round-trips (require a real Postgres with pgvector) ---

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")


@pytest.fixture()
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine, text

    from dynamo.persistence.schema import metadata

    eng = create_engine(INTEGRATION_URL, future=True)
    with eng.begin() as cx:
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    metadata.create_all(eng)
    yield eng
    metadata.drop_all(eng)
    eng.dispose()


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set")
class TestOrderRepositoryDb:
    def test_insert_transition_event_roundtrip(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.execution import OrderRepository

        repo = OrderRepository(engine)  # type: ignore[arg-type]
        order = make_order(coid="db-1", decision_context_id="dc-1")
        repo.insert(order.model_copy(update={"created_at": NOW}))

        loaded = repo.get("db-1")
        assert loaded is not None and loaded.state == OrderState.PROPOSED
        assert [o.client_order_id for o in repo.open_orders()] == ["db-1"]

        updated = repo.transition_with_event(
            "db-1", OrderState.PROPOSED, OrderState.RISK_CHECKED, "risk ok", NOW
        )
        assert updated.state == OrderState.RISK_CHECKED
        events = repo.events("db-1")
        assert [e["to_state"] for e in events] == ["proposed", "risk_checked"]
        assert repo.state_entered_at("db-1") is not None

    def test_trade_and_position_and_recon(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.execution import (
            EquityCurveRepository,
            PositionRepository,
            ReconciliationRepository,
            TradeRepository,
        )

        trades = TradeRepository(engine)  # type: ignore[arg-type]
        positions = PositionRepository(engine)  # type: ignore[arg-type]
        buy = _fill(OrderSide.BUY, 10, 100.0, 0)
        sell = _fill(OrderSide.SELL, 10, 110.0, 3)
        for t in (buy, sell):
            trades.insert(t)
            positions.apply_fill(t, NOW)
        assert trades.filled_qty(buy.client_order_id) == 10
        assert positions.all_positions() == []  # flat after the round trip
        assert isinstance(trades.closed_trades("AAPL")[0], ClosedTrade)

        recon = ReconciliationRepository(engine)  # type: ignore[arg-type]
        run_id = recon.record_run(NOW, "AAPL", "divergent", [{"kind": "position_qty"}])
        assert recon.unacked_divergent_runs()
        recon.record_ack(run_id, "owner", NOW, "verified")
        assert recon.unacked_divergent_runs() == []

        curve = EquityCurveRepository(engine)  # type: ignore[arg-type]
        curve.upsert(NOW.date(), 100_000.0, 5_000.0)
        curve.upsert(NOW.date(), 100_100.0, 5_000.0)  # idempotent restate
