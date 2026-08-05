"""Options schema presence + repository round-trips.

The migration-presence tests run everywhere; the DB round-trips are
integration-marked and skipped unless INTEGRATION_DB_URL points at a Postgres.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from dynamo.domain.models.options import OptionStructure, OptionTrade
from dynamo.domain.models.portfolio import OrderState
from tests.unit.options.helpers import MONTHLY_EXPIRY, make_decision

NOW = datetime(2026, 8, 4, 0, 45, tzinfo=UTC)

OPTIONS_TABLES = (
    "options_orders",
    "options_order_events",
    "options_positions",
    "options_trades",
    "options_chain_snapshots",
    "options_iv_history",
)


def test_schema_declares_options_tables() -> None:
    """Migration 0003 tables exist in the canonical schema module."""
    from dynamo.persistence import schema

    for name in OPTIONS_TABLES:
        assert name in schema.metadata.tables, name
    orders = schema.metadata.tables["options_orders"]
    assert not orders.c.decision_context_id.nullable  # no order without a snapshot


def test_migration_0003_targets_exactly_the_options_tables() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "infra/migrations/versions/0003_options.py"
    spec = importlib.util.spec_from_file_location("migration_0003", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == "0003" and module.down_revision == "0002"
    assert tuple(t.name for t in module.OPTIONS_TABLES) == OPTIONS_TABLES


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
class TestOptionsRepositoriesDb:
    def _order(self, coid: str):
        from dynamo.domain.models.options import OptionOrder

        decision = make_decision()
        return OptionOrder(
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
            decision_context_id="dc-1",
            created_at=NOW,
        )

    def test_order_transition_event_roundtrip(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.options import OptionsOrderRepository

        repo = OptionsOrderRepository(engine)  # type: ignore[arg-type]
        repo.insert(self._order("opt-db-1"))

        loaded = repo.get("opt-db-1")
        assert loaded is not None and loaded.state == OrderState.PROPOSED
        assert loaded.structure is OptionStructure.LONG_CALL
        assert [o.client_order_id for o in repo.open_orders()] == ["opt-db-1"]

        updated = repo.transition_with_event(
            "opt-db-1", OrderState.PROPOSED, OrderState.RISK_CHECKED, "options rules ok", NOW
        )
        assert updated.state == OrderState.RISK_CHECKED
        assert [e["to_state"] for e in repo.events("opt-db-1")] == [
            "proposed",
            "risk_checked",
        ]

    def test_trades_positions_and_expiry(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.options import (
            OptionsPositionRepository,
            OptionsTradeRepository,
        )

        order = self._order("opt-db-2")
        trades = OptionsTradeRepository(engine)  # type: ignore[arg-type]
        positions = OptionsPositionRepository(engine)  # type: ignore[arg-type]
        fill = OptionTrade(
            client_order_id="opt-db-2",
            occ_symbol=order.occ_symbol,
            underlying=order.underlying,
            side=order.side,
            right=order.right,
            contracts=4,
            price=2.50,
            filled_at=NOW,
        )
        trades.insert(fill)
        positions.apply_fill(order, fill, NOW)
        assert trades.filled_contracts("opt-db-2") == 4
        open_positions = positions.open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].occ_symbol == order.occ_symbol
        assert open_positions[0].premium_at_risk == 1_000.0

        assert positions.expired_positions(MONTHLY_EXPIRY) == []
        positions.close(order.occ_symbol)
        assert positions.open_positions() == []

    def test_chain_snapshot_and_iv_history(self, engine) -> None:  # type: ignore[no-untyped-def]
        from dynamo.persistence.repositories.options import OptionsChainRepository
        from tests.unit.options.helpers import default_chain

        repo = OptionsChainRepository(engine)  # type: ignore[arg-type]
        rows = default_chain("AAPL")
        assert repo.insert_snapshot(rows) == len(rows)
        repo.insert_snapshot(rows)  # idempotent on the composite key

        repo.append_iv("AAPL", NOW.date(), 0.31)
        repo.append_iv("AAPL", NOW.date(), 0.32)  # same day upserts
        assert repo.iv_history("AAPL") == [0.32]
