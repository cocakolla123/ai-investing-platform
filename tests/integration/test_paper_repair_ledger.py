"""scripts/paper_repair_ledger.py end to end against Postgres, on a copy of
the real paper rows (2026-09-15..10-05). Skipped unless INTEGRATION_DB_URL
is set."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Engine, create_engine, delete, insert, inspect, select, text

from dynamo.persistence import schema

pytestmark = pytest.mark.integration

TABLES = [
    schema.equity_curve,
    schema.trades,
    schema.ladder_states,
    schema.orders,
    schema.order_events,
    schema.reconciliation_runs,
    schema.reconciliation_acks,
]
QTY, PRICE = 123.7092, 764.621455
UTHR = "bbd67a8862331fe741171242"
CURVE = [
    (date(2026, 9, 15), 100_000.0, 100_000.0, 0.0),
    (date(2026, 9, 24), 100_000.0, 100_000.0, 0.0),
    (date(2026, 9, 25), 194_920.832068, 100_000.0, 0.4870),
    (date(2026, 9, 26), 195_423.09142, 100_000.0, 0.4883),
    (date(2026, 10, 5), 195_211.548688, 100_000.0, 0.4877),
]
LADDER = [
    (date(2026, 9, 24), "normal", 0.0, 100_000.0),
    (date(2026, 9, 26), "normal", 0.0, 194_920.832068),
    (date(2026, 10, 5), "normal", 0.0, 195_211.548688),
]


def _script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "paper_repair_ledger.py"
    spec = importlib.util.spec_from_file_location("paper_repair_ledger", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drop_backups(eng: Engine) -> None:
    with eng.begin() as c:
        for name in inspect(eng).get_table_names():
            if "_backup_" in name and name.split("_backup_")[0] in {
                "equity_curve",
                "ladder_states",
                "orders",
            }:
                c.execute(text(f"drop table {name}"))


@pytest.fixture()
def engine(integration_db_url: str) -> Iterator[Engine]:
    eng = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(eng, tables=TABLES)
    _drop_backups(eng)
    with eng.begin() as c:
        for t in reversed(TABLES):
            c.execute(delete(t))
        c.execute(
            insert(schema.equity_curve),
            [dict(as_of=a, equity=e, cash=k, gross_exposure=g) for a, e, k, g in CURVE],
        )
        c.execute(
            insert(schema.trades).values(
                client_order_id="963805526365c4972a3c8374",
                symbol="SPY",
                side="buy",
                qty=QTY,
                fill_price=PRICE,
                fees=0.0,
                filled_at=datetime(2026, 9, 24, 13, 45, 6, tzinfo=UTC),
            )
        )
        c.execute(
            insert(schema.ladder_states),
            [dict(as_of=a, rung=r, drawdown=d, hwm=h) for a, r, d, h in LADDER],
        )
        at = datetime(2026, 10, 1, 13, 0, tzinfo=UTC)
        c.execute(
            insert(schema.orders).values(
                client_order_id=UTHR,
                decision_context_id="dc-uthr",
                symbol="UTHR",
                side="buy",
                qty=14.2684,
                order_type="marketable_limit",
                limit_price=500.0,
                tif="day",
                state="pending_approval",
                created_at=at,
                updated_at=at,
            )
        )
        for i in range(3):
            c.execute(
                insert(schema.reconciliation_runs).values(
                    ts=datetime(2026, 10, 2, 20, i, tzinfo=UTC),
                    scope="equities:global",
                    result="divergent",
                    divergences=[],
                )
            )
    yield eng
    _drop_backups(eng)
    with eng.begin() as c:
        for t in reversed(TABLES):
            c.execute(delete(t))
    eng.dispose()


def test_dry_run_writes_nothing(engine: Engine) -> None:
    report = _script().main([], engine)
    assert report["mode"] == "dry-run"
    assert report["curve"]["rows_to_fix"] == 3
    assert report["ladder"]["rows_to_fix"] == 2
    assert report["ack"]["blocked_by"] == "the latest equities reconciliation run is not clean"
    with engine.connect() as c:
        cash = c.execute(select(schema.equity_curve.c.cash)).scalars().all()
    assert set(cash) == {100_000.0}


def test_apply_needs_an_explicit_step(engine: Engine) -> None:
    with pytest.raises(SystemExit):
        _script().main(["--apply"], engine)


def test_ladder_refuses_before_the_curve_is_repaired(engine: Engine) -> None:
    with pytest.raises(SystemExit, match="not repaired"):
        _script().main(["--step", "ladder", "--apply"], engine)


def test_curve_then_ladder_backs_up_repairs_and_is_idempotent(engine: Engine) -> None:
    script = _script()
    report = script.main(["--step", "curve", "--apply"], engine)
    assert "equity_curve" in report["backup_tables"]
    with engine.connect() as c:
        backup = report["backup_tables"]["equity_curve"]
        assert c.execute(text(f"select count(*) from {backup}")).scalar() == len(CURVE)
        rows = dict(
            c.execute(select(schema.equity_curve.c.as_of, schema.equity_curve.c.cash)).all()
        )
    assert rows[date(2026, 9, 24)] == pytest.approx(100_000.0)
    assert rows[date(2026, 10, 5)] == pytest.approx(5_409.29, abs=0.01)

    report = script.main(["--step", "ladder", "--apply"], engine)
    with engine.connect() as c:
        hwm = dict(
            c.execute(select(schema.ladder_states.c.as_of, schema.ladder_states.c.hwm)).all()
        )
        rungs = set(c.execute(select(schema.ladder_states.c.rung)).scalars())
    assert max(hwm.values()) < 101_000.0
    assert rungs == {"normal"}

    again = script.main([], engine)
    assert again["curve"]["rows_to_fix"] == 0
    assert again["ladder"]["rows_to_fix"] == 0


def test_cancel_uses_the_state_machine_and_refuses_broker_orders(engine: Engine) -> None:
    script = _script()
    with pytest.raises(SystemExit, match="--order-id"):
        script.main(["--step", "cancel", "--apply"], engine)
    script.main(["--step", "cancel", "--apply", "--order-id", UTHR], engine)
    with engine.connect() as c:
        state = c.execute(
            select(schema.orders.c.state).where(schema.orders.c.client_order_id == UTHR)
        ).scalar()
        events = c.execute(
            select(schema.order_events.c.from_state, schema.order_events.c.to_state).where(
                schema.order_events.c.client_order_id == UTHR
            )
        ).all()
    assert state == "cancelled"
    assert [tuple(e) for e in events] == [("pending_approval", "cancelled")]
    with pytest.raises(SystemExit, match="not pending_approval"):
        script.main(["--step", "cancel", "--apply", "--order-id", UTHR], engine)

    with engine.begin() as c:
        c.execute(
            schema.orders.update()
            .where(schema.orders.c.client_order_id == UTHR)
            .values(state="pending_approval", broker_order_id="b-1")
        )
    with pytest.raises(SystemExit, match="broker order id"):
        script.main(["--step", "cancel", "--apply", "--order-id", UTHR], engine)


def test_ack_needs_a_clean_run_after_the_last_divergent_one(engine: Engine) -> None:
    script = _script()
    args = ["--step", "ack", "--apply", "--acked-by", "owner", "--note", "cash ledger fixed"]
    with pytest.raises(SystemExit, match="not clean"):
        script.main(args, engine)
    with engine.begin() as c:
        c.execute(
            insert(schema.reconciliation_runs).values(
                ts=datetime(2026, 10, 6, 14, 0, tzinfo=UTC),
                scope="equities",
                result="clean",
                divergences=[],
            )
        )
    with pytest.raises(SystemExit, match="--acked-by and --note"):
        script.main(["--step", "ack", "--apply"], engine)
    report = script.main(args, engine)
    assert report["acked"] == 3
    with engine.connect() as c:
        assert (
            c.execute(select(schema.reconciliation_acks.c.note)).scalars().all()
            == ["cash ledger fixed"] * 3
        )
    assert script.main([], engine)["ack"]["unacked_divergent_runs"] == 0
