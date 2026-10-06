"""One-off repair of paper bookkeeping written while cash was frozen.

From the first paper fill (2026-09-24) until the cash-ledger fix, every
nightly ``equity_curve`` row carried cash at the opening $100,000, so equity
counted the SPY position on top of cash that had already paid for it. The
marks themselves were right: the positions table was correct and each row's
market value (equity - cash) used that night's prices. Only cash was wrong.

The repair therefore keeps each row's market value and replaces its cash with
ledger cash at the moment the row was written:

    new cash   = opening cash + fills before the snapshot
    new equity = new cash + (old equity - old cash)
    new gross  = (old equity - old cash) / new equity

``ladder_states`` stored that night's (inflated) equity as its high-water
mark. It is rebuilt from the repaired curve as the risk asset would have seen
it: the rows written before that night's risk run. The rung is never changed
here; the implied rung is reported so a human can compare.

Everything here is pure: the script reads rows, these functions plan, and the
script writes only after a backup and an explicit ``--apply``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from dynamo.execution.cash_ledger import Fill, ledger_cash
from dynamo.risk.ladder import rung_for_drawdown

ET = ZoneInfo("America/New_York")
#: nightly_rollup_schedule writes the equity_curve row for ET date D at 02:00
#: ET; risk_portfolio_schedule reads the curve at 00:30 ET (schedules.py).
SNAPSHOT_TIME_ET = time(2, 0)
#: Changes smaller than this (dollars) are rounding, not repairs.
EPS = 0.005


@dataclass(frozen=True)
class CurveRow:
    as_of: date
    equity: float
    cash: float
    gross_exposure: float


@dataclass(frozen=True)
class TimedFill:
    filled_at: datetime
    fill: Fill


@dataclass(frozen=True)
class CurveFix:
    old: CurveRow
    new: CurveRow


@dataclass(frozen=True)
class LadderRow:
    as_of: date
    rung: str
    drawdown: float
    hwm: float


@dataclass(frozen=True)
class LadderFix:
    old: LadderRow
    new: LadderRow
    implied_rung: str  # what the repaired drawdown implies; reported, not written


def snapshot_moment(as_of: date) -> datetime:
    return datetime.combine(as_of, SNAPSHOT_TIME_ET, tzinfo=ET)


def plan_curve(
    rows: Sequence[CurveRow],
    fills: Iterable[TimedFill],
    *,
    moment: Callable[[date], datetime] = snapshot_moment,
) -> list[CurveFix]:
    """Fixes for every row whose cash differs from ledger cash at its snapshot.

    The earliest row is the ledger's opening and is never changed. Running the
    plan again on repaired rows returns nothing (idempotent).
    """
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: r.as_of)
    opening = ordered[0]
    start = datetime.combine(opening.as_of, time.min, tzinfo=ET)
    timed = sorted(fills, key=lambda f: f.filled_at)
    out: list[CurveFix] = []
    for row in ordered[1:]:
        cutoff = moment(row.as_of)
        cash = ledger_cash(opening.cash, (t.fill for t in timed if start <= t.filled_at < cutoff))
        if abs(cash - row.cash) <= EPS:
            continue
        market_value = row.equity - row.cash
        equity = cash + market_value
        gross = market_value / equity if equity > 0 else 0.0
        out.append(CurveFix(row, CurveRow(row.as_of, equity, cash, gross)))
    return out


def repaired_curve(rows: Sequence[CurveRow], fixes: Sequence[CurveFix]) -> list[CurveRow]:
    by_date = {f.new.as_of: f.new for f in fixes}
    return sorted((by_date.get(r.as_of, r) for r in rows), key=lambda r: r.as_of)


def plan_ladder(ladder: Sequence[LadderRow], curve: Sequence[CurveRow]) -> list[LadderFix]:
    """Rebuild each ladder row's HWM and drawdown from the repaired curve.

    The risk asset for ET date D runs before that night's snapshot, so it saw
    the curve rows dated before D. A ladder row with no earlier curve row is
    left alone.
    """
    ordered = sorted(curve, key=lambda r: r.as_of)
    out: list[LadderFix] = []
    for row in ladder:
        seen = [c for c in ordered if c.as_of < row.as_of]
        if not seen:
            continue
        equity = seen[-1].equity
        hwm = max(c.equity for c in seen)
        drawdown = equity / hwm - 1.0 if hwm > 0 else 0.0
        if abs(hwm - row.hwm) <= EPS and abs(drawdown - row.drawdown) <= 1e-9:
            continue
        new = LadderRow(row.as_of, row.rung, drawdown, hwm)
        out.append(LadderFix(row, new, rung_for_drawdown(drawdown).value))
    return out
