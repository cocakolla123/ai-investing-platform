"""PM schema presence + repository round-trips.

Integration-marked; skipped unless INTEGRATION_DB_URL points at a Postgres.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from dynamo.domain.models.pm import PMPosition, PMSide, PMTrade
from dynamo.domain.models.portfolio import OrderState
from tests.unit.pm.helpers import make_pm_order

NOW = datetime(2026, 8, 4, 20, 30, tzinfo=UTC)

PM_TABLES = ("pm_orders", "pm_order_events", "pm_positions", "pm_trades")


def test_schema_declares_pm_tables() -> None:
    """Migration 0002 tables exist in the canonical schema module."""
    from dynamo.persistence import schema

    for name in PM_TABLES:
        assert name in schema.metadata.tables, name
    orders = schema.metadata.tables["pm_orders"]
    assert not orders.c.decision_context_id.nullable  # no order without a snapshot


def test_migration_0002_targets_exactly_the_pm_tables() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "infra/migrations/versions/0002_prediction_market_trading.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0002", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == "0002" and module.down_revision == "0001"
    assert tuple(t.name for t in module.PM_TABLES) == PM_TABLES


# --- DB round-trips (require a real Postgres) ---

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
class TestPMRepositoriesDb:
    def test_order_transition_event_roundtrip(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.pm import PMOrderRepository

        repo = PMOrderRepository(engine)  # type: ignore[arg-type]
        order = make_pm_order(coid="pm-db-1", decision_context_id="dc-1")
        repo.insert(order.model_copy(update={"created_at": NOW}))

        loaded = repo.get("pm-db-1")
        assert loaded is not None and loaded.state == OrderState.PROPOSED
        assert [o.client_order_id for o in repo.open_orders()] == ["pm-db-1"]

        updated = repo.transition_with_event(
            "pm-db-1", OrderState.PROPOSED, OrderState.RISK_CHECKED, "pm rules ok", NOW
        )
        assert updated.state == OrderState.RISK_CHECKED
        assert [e["to_state"] for e in repo.events("pm-db-1")] == ["proposed", "risk_checked"]

    def test_trades_positions_and_settlement(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.pm import (
            PMPositionRepository,
            PMTradeRepository,
        )

        trades = PMTradeRepository(engine)  # type: ignore[arg-type]
        positions = PMPositionRepository(engine)  # type: ignore[arg-type]
        fill = PMTrade(
            client_order_id="pm-db-2",
            market_id="FED-26DEC",
            side=PMSide.YES,
            contracts=100,
            price_cents=43.0,
            filled_at=NOW,
        )
        trades.insert(fill)
        positions.apply_fill(fill, NOW, question="Fed above 4%?")
        assert trades.filled_contracts("pm-db-2") == 100
        open_positions = positions.open_positions()
        assert open_positions == [
            PMPosition(
                market_id="FED-26DEC",
                side=PMSide.YES,
                contracts=100,
                avg_price_cents=43.0,
                question="Fed above 4%?",
            )
        ]

        settlement = PMTrade(
            client_order_id=None,
            market_id="FED-26DEC",
            side=PMSide.YES,
            contracts=100,
            price_cents=100.0,
            kind="settlement",
            realized_pnl_cents=5700.0,
            filled_at=NOW,
        )
        trades.insert(settlement)
        positions.close("kalshi", "FED-26DEC")
        assert positions.open_positions() == []
        assert trades.realized_pnl_cents() == 5700.0
