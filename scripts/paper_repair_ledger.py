"""Repair paper bookkeeping written while cash was frozen. DRY RUN by default.

See docs/runbook_paper_cash_ledger_2026-10.md for the order of operations.

    PYTHONPATH=. .venv/bin/python scripts/paper_repair_ledger.py            # dry run, every step
    PYTHONPATH=. .venv/bin/python scripts/paper_repair_ledger.py --step curve --apply

Steps (each applied only when named with ``--step`` and ``--apply``):

- ``curve``: ``equity_curve`` rows after the first fill get ledger cash; each
  row's market value is kept (dynamo.execution.ledger_repair).
- ``ladder``: ``ladder_states`` HWM and drawdown rebuilt from the repaired
  curve. The rung is never changed. Refuses unless the curve is already
  repaired or ``curve`` is applied in the same run.
- ``cancel``: moves the named ``pending_approval`` orders to ``cancelled``
  through the state machine (one ``order_events`` row each). Database only:
  refuses any order that has a broker order id, so nothing at the broker is
  touched. Needs ``--order-id`` for each order.
- ``ack``: acknowledges every unacked divergent reconciliation run with the
  same function the admin API uses. Needs ``--acked-by`` and ``--note``, and
  refuses unless the latest equities reconciliation run is clean and newer
  than the last divergent one.

``curve``, ``ladder`` and ``cancel`` copy the affected table to
``<table>_backup_<UTC timestamp>`` in its own transaction first and refuse to
write if the copy is short. ``ack`` only inserts rows into
``reconciliation_acks``; it changes nothing that needs a backup. Every step is
idempotent: a second run finds nothing to do.

Nothing here talks to the broker.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, text

from dynamo.config.settings import Settings
from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.cash_ledger import Fill
from dynamo.execution.ledger_repair import (
    CurveRow,
    LadderRow,
    TimedFill,
    plan_curve,
    plan_ladder,
    repaired_curve,
)
from dynamo.persistence.db import make_engine

STEPS = ("curve", "ladder", "cancel", "ack")


def _utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _backup(
    engine: Engine, table: str, where: str = "", params: dict[str, Any] | None = None
) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup = f"{table}_backup_{stamp}"
    sql_where = f" where {where}" if where else ""
    # Commits on its own, BEFORE any update, so it survives a failed update.
    with engine.begin() as c:
        c.execute(text(f"create table {backup} as select * from {table}{sql_where}"), params or {})
    with engine.connect() as c:
        copied = c.execute(text(f"select count(*) from {backup}")).scalar()
        expected = c.execute(
            text(f"select count(*) from {table}{sql_where}"), params or {}
        ).scalar()
    if copied != expected:
        raise SystemExit(f"backup {backup} has {copied} rows, expected {expected}; nothing written")
    return backup


def _read(engine: Engine) -> dict[str, Any]:
    with engine.connect() as c:
        curve = [
            CurveRow(r[0], float(r[1]), float(r[2]), float(r[3]))
            for r in c.execute(
                text("select as_of, equity, cash, gross_exposure from equity_curve order by as_of")
            )
        ]
        fills = [
            TimedFill(_utc(r[0]), Fill(str(r[1]), float(r[2]), float(r[3]), float(r[4] or 0.0)))
            for r in c.execute(
                text("select filled_at, side, qty, fill_price, fees from trades order by filled_at")
            )
        ]
        ladder = [
            LadderRow(r[0], str(r[1]), float(r[2]), float(r[3]))
            for r in c.execute(
                text("select as_of, rung, drawdown, hwm from ladder_states order by as_of")
            )
        ]
        pending = [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "select client_order_id, broker_order_id, symbol, side, qty, created_at "
                    "from orders where state = :s order by created_at"
                ),
                {"s": OrderState.PENDING_APPROVAL.value},
            )
        ]
        unacked = [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "select id, ts, scope from reconciliation_runs where result = 'divergent' "
                    "and id not in (select run_id from reconciliation_acks) order by id"
                )
            )
        ]
        latest_equities = c.execute(
            text(
                "select id, ts, scope, result from reconciliation_runs "
                "where scope like 'equities%' order by id desc limit 1"
            )
        ).first()
    return {
        "curve": curve,
        "fills": fills,
        "ladder": ladder,
        "pending": pending,
        "unacked": unacked,
        "latest_equities": None if latest_equities is None else dict(latest_equities._mapping),
    }


def _ack_blocker(state: dict[str, Any]) -> str | None:
    latest = state["latest_equities"]
    if not state["unacked"]:
        return None
    if latest is None or latest["result"] != "clean":
        return "the latest equities reconciliation run is not clean"
    if latest["id"] < max(r["id"] for r in state["unacked"]):
        return "no clean equities run since the last divergent one"
    return None


def main(argv: list[str] | None = None, engine: Engine | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", action="append", choices=STEPS, default=[])
    ap.add_argument("--apply", action="store_true", help="back up, then write (default: dry run)")
    ap.add_argument("--order-id", action="append", default=[], help="order to cancel (repeat)")
    ap.add_argument("--acked-by", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    if args.apply and not args.step:
        raise SystemExit("--apply needs at least one explicit --step")
    steps = args.step or list(STEPS)

    engine = engine or make_engine(Settings().database_url)
    state = _read(engine)
    curve_fixes = plan_curve(state["curve"], state["fills"])
    curve_after = repaired_curve(state["curve"], curve_fixes)
    ladder_fixes = plan_ladder(state["ladder"], curve_after)
    now = datetime.now(UTC)
    report: dict[str, Any] = {"mode": "apply" if args.apply else "dry-run", "steps": steps}

    if "curve" in steps:
        report["curve"] = {
            "rows": len(state["curve"]),
            "rows_to_fix": len(curve_fixes),
            "fixes": [
                {
                    "as_of": str(f.old.as_of),
                    "cash": [round(f.old.cash, 2), round(f.new.cash, 2)],
                    "equity": [round(f.old.equity, 2), round(f.new.equity, 2)],
                    "gross": [round(f.old.gross_exposure, 4), round(f.new.gross_exposure, 4)],
                }
                for f in curve_fixes
            ],
        }
    if "ladder" in steps:
        report["ladder"] = {
            "rows": len(state["ladder"]),
            "rows_to_fix": len(ladder_fixes),
            "fixes": [
                {
                    "as_of": str(f.old.as_of),
                    "hwm": [round(f.old.hwm, 2), round(f.new.hwm, 2)],
                    "drawdown": [round(f.old.drawdown, 5), round(f.new.drawdown, 5)],
                    "rung_kept": f.old.rung,
                    "rung_implied": f.implied_rung,
                }
                for f in ladder_fixes
            ],
        }
    if "cancel" in steps:
        report["cancel"] = {
            "pending_approval": [
                {**o, "created_at": str(o["created_at"])} for o in state["pending"]
            ],
            "requested": args.order_id,
        }
    if "ack" in steps:
        runs = state["unacked"]
        report["ack"] = {
            "unacked_divergent_runs": len(runs),
            "ids": [runs[0]["id"], runs[-1]["id"]] if runs else [],
            "latest_equities_run": {k: str(v) for k, v in (state["latest_equities"] or {}).items()},
            "blocked_by": _ack_blocker(state),
        }

    if args.apply:
        _apply(engine, args, state, curve_fixes, ladder_fixes, now, report)
    print(json.dumps(report, indent=2, default=str))
    return report


def _apply(
    engine: Engine,
    args: argparse.Namespace,
    state: dict[str, Any],
    curve_fixes: list[Any],
    ladder_fixes: list[Any],
    now: datetime,
    report: dict[str, Any],
) -> None:
    from dynamo.execution.reconciliation import acknowledge_reconciliation
    from dynamo.persistence.repositories.execution import (
        OrderRepository,
        ReconciliationRepository,
    )

    # Validate everything before writing anything.
    if "ladder" in args.step and curve_fixes and "curve" not in args.step:
        raise SystemExit("ladder: equity_curve is not repaired yet; apply --step curve first")
    if "cancel" in args.step:
        pending = {o["client_order_id"]: o for o in state["pending"]}
        if not args.order_id:
            raise SystemExit("cancel: name each order with --order-id")
        for coid in args.order_id:
            if coid not in pending:
                raise SystemExit(f"cancel: {coid} is not pending_approval")
            if pending[coid]["broker_order_id"]:
                raise SystemExit(f"cancel: {coid} has a broker order id; cancel at the broker")
    if "ack" in args.step:
        if not (args.acked_by.strip() and args.note.strip()):
            raise SystemExit("ack: --acked-by and --note are required")
        blocker = _ack_blocker(state)
        if blocker:
            raise SystemExit(f"ack: {blocker}")

    backups: dict[str, str] = {}
    if "curve" in args.step and curve_fixes:
        backups["equity_curve"] = _backup(engine, "equity_curve")
        with engine.begin() as c:
            for f in curve_fixes:
                c.execute(
                    text(
                        "update equity_curve set equity = :e, cash = :c, gross_exposure = :g "
                        "where as_of = :d and abs(cash - :old_cash) < 0.005"
                    ),
                    {
                        "e": f.new.equity,
                        "c": f.new.cash,
                        "g": f.new.gross_exposure,
                        "d": f.new.as_of,
                        "old_cash": f.old.cash,
                    },
                )
    if "ladder" in args.step and ladder_fixes:
        backups["ladder_states"] = _backup(engine, "ladder_states")
        with engine.begin() as c:
            for lf in ladder_fixes:
                c.execute(
                    text("update ladder_states set hwm = :h, drawdown = :dd where as_of = :d"),
                    {"h": lf.new.hwm, "dd": lf.new.drawdown, "d": lf.new.as_of},
                )
    if "cancel" in args.step:
        backups["orders"] = _backup(
            engine, "orders", "client_order_id = any(:ids)", {"ids": list(args.order_id)}
        )
        repo = OrderRepository(engine)
        for coid in args.order_id:
            repo.transition_with_event(
                coid,
                OrderState.PENDING_APPROVAL,
                OrderState.CANCELLED,
                "approval never given; expired by the 2026-10 cash-ledger repair",
                now,
                {"repair": "paper_repair_ledger"},
            )
    if "ack" in args.step:
        store = ReconciliationRepository(engine)
        for run in state["unacked"]:
            acknowledge_reconciliation(store, int(run["id"]), args.acked_by, args.note, now)
        report["acked"] = len(state["unacked"])
    report["backup_tables"] = backups


if __name__ == "__main__":
    main()
