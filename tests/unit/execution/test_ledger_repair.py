"""Planning the paper bookkeeping repair (dynamo.execution.ledger_repair).

Fixture rows are the real paper values from 2026-09-15 to 10-05.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from dynamo.execution.cash_ledger import Fill
from dynamo.execution.ledger_repair import (
    CurveRow,
    LadderRow,
    TimedFill,
    plan_curve,
    plan_ladder,
    repaired_curve,
)

QTY, PRICE = 123.7092, 764.621455
BROKER_CASH = 100_000.0 - QTY * PRICE  # 5,409.29
FILLS = [TimedFill(datetime(2026, 9, 24, 13, 45, 6, tzinfo=UTC), Fill("buy", QTY, PRICE))]
CURVE = [
    CurveRow(date(2026, 9, 15), 100_000.0, 100_000.0, 0.0),
    CurveRow(date(2026, 9, 24), 100_000.0, 100_000.0, 0.0),
    CurveRow(date(2026, 9, 25), 194_920.832068, 100_000.0, 0.4870),
    CurveRow(date(2026, 9, 26), 195_423.09142, 100_000.0, 0.4883),
    CurveRow(date(2026, 10, 5), 195_211.548688, 100_000.0, 0.4877),
]


def test_rows_before_the_fill_are_untouched() -> None:
    fixed = {f.old.as_of for f in plan_curve(CURVE, FILLS)}
    assert date(2026, 9, 15) not in fixed  # the opening row, never changed
    assert date(2026, 9, 24) not in fixed  # written at 02:00 ET, before the 09:45 ET fill
    assert fixed == {date(2026, 9, 25), date(2026, 9, 26), date(2026, 10, 5)}


def test_cash_becomes_broker_cash_and_market_value_is_kept() -> None:
    for fix in plan_curve(CURVE, FILLS):
        assert fix.new.cash == pytest.approx(BROKER_CASH)
        assert fix.new.cash == pytest.approx(5_409.29, abs=0.01)
        assert fix.new.equity - fix.new.cash == pytest.approx(fix.old.equity - fix.old.cash)
        assert fix.new.gross_exposure == pytest.approx(
            (fix.old.equity - fix.old.cash) / fix.new.equity
        )
    last = plan_curve(CURVE, FILLS)[-1].new
    # Broker equity on 10-05 was about $100.6k.
    assert last.equity == pytest.approx(100_620.84, abs=0.01)
    assert last.gross_exposure == pytest.approx(0.946, abs=0.001)


def test_the_plan_is_idempotent() -> None:
    fixes = plan_curve(CURVE, FILLS)
    assert plan_curve(repaired_curve(CURVE, fixes), FILLS) == []


def test_ladder_hwm_comes_from_the_curve_the_risk_run_saw() -> None:
    curve = repaired_curve(CURVE, plan_curve(CURVE, FILLS))
    ladder = [
        LadderRow(date(2026, 9, 24), "normal", 0.0, 100_000.0),
        LadderRow(date(2026, 9, 26), "normal", 0.0, 194_920.832068),
        LadderRow(date(2026, 10, 6), "normal", 0.0, 195_211.548688),
    ]
    fixes = {f.old.as_of: f for f in plan_ladder(ladder, curve)}
    # 09-24 saw only the 100k rows: already right.
    assert date(2026, 9, 24) not in fixes
    # 09-26 saw up to the 09-25 row.
    f26 = fixes[date(2026, 9, 26)]
    assert f26.new.hwm == pytest.approx(194_920.832068 - (100_000.0 - BROKER_CASH))
    assert f26.new.drawdown == pytest.approx(0.0)
    assert f26.new.rung == "normal"
    # 10-06 saw the 09-26 high and the lower 10-05 row: a small drawdown.
    f06 = fixes[date(2026, 10, 6)]
    assert f06.new.hwm == pytest.approx(195_423.09142 - (100_000.0 - BROKER_CASH))
    assert f06.new.drawdown == pytest.approx(-0.0021, abs=0.0001)
    assert f06.implied_rung == "normal"


def test_the_unrepaired_curve_would_not_be_used_for_the_ladder() -> None:
    """Guard on intent: planning the ladder from the raw curve keeps the
    inflated HWM, which is why the ladder step runs after the curve step."""
    ladder = [LadderRow(date(2026, 10, 6), "normal", 0.0, 195_211.548688)]
    raw = plan_ladder(ladder, CURVE)
    assert raw and raw[0].new.hwm == pytest.approx(195_423.09142)
