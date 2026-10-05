"""CashLedgerRepository against Postgres: opening snapshot plus fills.

Reproduces the paper account of 2026-09-24 (one SPY buy from $100,000, broker
cash $5,409.29). Skipped unless ``INTEGRATION_DB_URL`` is set.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import Engine, create_engine, delete, insert

from dynamo.persistence import schema
from dynamo.persistence.repositories.execution import CashLedgerRepository, EquityCurveRepository

pytestmark = pytest.mark.integration

TABLES = [schema.equity_curve, schema.trades]


@pytest.fixture()
def engine(integration_db_url: str) -> Iterator[Engine]:
    eng = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(eng, tables=TABLES)
    with eng.begin() as conn:
        for t in TABLES:
            conn.execute(delete(t))
    yield eng
    with eng.begin() as conn:
        for t in TABLES:
            conn.execute(delete(t))
    eng.dispose()


def _fill(
    eng: Engine, side: str, qty: float, price: float, at: datetime, fees: float = 0.0
) -> None:
    with eng.begin() as conn:
        conn.execute(
            insert(schema.trades).values(
                client_order_id=f"c-{at.isoformat()}-{side}",
                symbol="SPY",
                side=side,
                qty=qty,
                fill_price=price,
                fees=fees,
                filled_at=at,
            )
        )


def test_no_opening_snapshot_means_no_ledger(engine: Engine) -> None:
    assert CashLedgerRepository(engine).cash() is None


def test_opening_plus_fills_matches_the_broker(engine: Engine) -> None:
    curve = EquityCurveRepository(engine)
    curve.upsert(date(2026, 9, 14), 100_000.0, 100_000.0)
    _fill(engine, "buy", 123.7092, 764.621455, datetime(2026, 9, 24, 13, 31, tzinfo=UTC))
    # Later snapshots carrying the stale cash must not change the answer.
    curve.upsert(date(2026, 10, 2), 194_500.0, 100_000.0)

    ledger = CashLedgerRepository(engine)
    assert ledger.opening() == (date(2026, 9, 14), 100_000.0)
    assert ledger.cash() == pytest.approx(5_409.29, abs=0.01)
    # Point in time: before the fill, cash is the opening cash.
    assert ledger.cash(datetime(2026, 9, 23, tzinfo=UTC)) == pytest.approx(100_000.0)


def test_fills_before_the_opening_snapshot_are_ignored(engine: Engine) -> None:
    EquityCurveRepository(engine).upsert(date(2026, 9, 14), 100_000.0, 100_000.0)
    _fill(engine, "buy", 1, 500.0, datetime(2026, 9, 13, 15, 0, tzinfo=UTC))
    _fill(engine, "sell", 1, 510.0, datetime(2026, 9, 15, 15, 0, tzinfo=UTC), fees=0.02)
    assert CashLedgerRepository(engine).cash() == pytest.approx(100_509.98)
