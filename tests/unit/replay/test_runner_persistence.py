"""Runner persistence + fill economics (backtest-campaign Phase 5).

Covers: the per-session replay_equity rows matching the in-memory curve,
fill rows matching ``ReplayResult.fills`` (with ``trade_value`` /
``realized_pnl``), the monthly flush cadence, average-cost realized-P&L math
(hand-recomputed ledger AND direct parity with ``BacktestEngine._execute``
on an identical scenario), and ``turnover_total`` accumulating across months
while ``turnover_mtd`` keeps its monthly reset.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytest

from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_runner import (
    END,
    HISTORY_SESSIONS,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    make_bars,
    make_scores,
    nyse_sessions,
)

# Two calendar months (~May 26 -> Jun 30 given END=2026-06-30): exercises the
# month-boundary flush and cross-month turnover accumulation. horizon 3 keeps
# positions churning (exit + re-entry) in both months.
TWO_MONTH_SESSIONS = 25


@pytest.fixture(scope="module")
def result() -> tuple[Any, FakeRunRepo]:
    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    bars = make_bars(sessions)
    scores = make_scores(sessions)
    window = sessions[-TWO_MONTH_SESSIONS:]
    assert len({(d.year, d.month) for d in window}) >= 2, "fixture must span months"
    config = ReplayConfig(start=window[0], end=window[-1], label="persist", horizon_sessions=3)
    run_repo = FakeRunRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=run_repo,
        config=config,
    )
    return runner.run(), run_repo


def test_equity_rows_round_trip_the_in_memory_curve(result: Any) -> None:
    res, run_repo = result
    rows = run_repo.equity_rows
    assert len(rows) == len(res.sessions) == len(res.session_records)
    for row, (day, equity), rec in zip(rows, res.equity_curve, res.session_records, strict=True):
        assert row["run_id"] == res.run_id
        assert row["as_of"] == day == rec.session
        assert row["equity"] == equity == rec.equity
        assert row["cash"] == rec.cash
        assert row["gross"] == rec.gross
        assert 0.0 <= row["gross"] <= 1.5
        assert row["regime"] == rec.regime
        assert row["ladder_rung"] == rec.ladder
        assert row["vetoes"] == len(rec.vetoes)
    # Gross ties the accounting identity together: cash + gross*equity = equity.
    invested = [r for r in rows if r["gross"] > 0.0]
    assert invested, "the run should have held positions"
    for r in invested:
        assert r["cash"] + r["gross"] * r["equity"] == pytest.approx(r["equity"], abs=0.05)


def test_fill_rows_round_trip_result_fills(result: Any) -> None:
    res, run_repo = result
    assert len(run_repo.fill_rows) == len(res.fills) > 0
    for row, f in zip(run_repo.fill_rows, res.fills, strict=True):
        assert row == {
            "run_id": res.run_id,
            "session": f.session,
            "symbol": f.symbol,
            "side": f.side,
            "qty": f.qty,
            "price": f.price,
            "trade_value": f.trade_value,
            "cost": f.cost,
            "realized_pnl": f.realized_pnl,
        }
        assert row["trade_value"] == pytest.approx(round(f.qty * f.price, 2), abs=0.01)
    # At most one fill per (session, symbol): the replay_fills PK is safe.
    keys = [(f.session, f.symbol) for f in res.fills]
    assert len(keys) == len(set(keys))


def test_monthly_flush_cadence(result: Any) -> None:
    """Per-session rows are flushed per calendar month: a two-month window
    produces (at least) the boundary flush plus the final flush, and every
    flush contains only rows from a single month."""
    res, run_repo = result
    assert len(run_repo.equity_flushes) >= 2
    assert sum(run_repo.equity_flushes) == len(res.sessions)
    offset = 0
    for size in run_repo.equity_flushes:
        chunk = run_repo.equity_rows[offset : offset + size]
        assert len({(r["as_of"].year, r["as_of"].month) for r in chunk}) == 1
        offset += size


def test_realized_pnl_matches_average_cost_ledger(result: Any) -> None:
    """Re-walk the fills through an independent average-cost ledger: every
    sell's realized_pnl must equal value - cost - qty x basis (to the cent),
    and every buy carries None."""
    res, _ = result
    qty_held: dict[str, float] = {}
    basis: dict[str, float] = {}
    sells = 0
    for f in res.fills:
        if f.side == "buy":
            assert f.realized_pnl is None
            prev_qty = qty_held.get(f.symbol, 0.0)
            prev_basis = basis.get(f.symbol, 0.0)
            new_qty = prev_qty + f.qty
            basis[f.symbol] = (prev_qty * prev_basis + f.trade_value + f.cost) / new_qty
            qty_held[f.symbol] = new_qty
        else:
            sells += 1
            expected = round(f.trade_value - f.cost - f.qty * basis[f.symbol], 2)
            assert f.realized_pnl == pytest.approx(expected, abs=0.011)
            qty_held[f.symbol] = qty_held.get(f.symbol, 0.0) - f.qty
            if abs(qty_held[f.symbol]) < 1e-9:
                qty_held.pop(f.symbol, None)
                basis.pop(f.symbol, None)
    assert sells > 0, "the fixture must exercise the sell branch"


def test_turnover_total_accumulates_across_monthly_resets(result: Any) -> None:
    """turnover_total is the whole-run accumulator: it equals the hand-derived
    sum over sessions of traded value / decision equity / 2, and it exceeds
    any single month's contribution (proving it survives the monthly reset
    that turnover_mtd performs)."""
    res, _ = result
    equity_by_day = dict(res.equity_curve)
    session_order = list(res.sessions)
    prev_day = {session_order[i]: session_order[i - 1] for i in range(1, len(session_order))}
    per_month: dict[tuple[int, int], float] = {}
    total = 0.0
    traded_by_day: dict[date, float] = {}
    for f in res.fills:
        traded_by_day[f.session] = traded_by_day.get(f.session, 0.0) + f.trade_value
    for day, traded in traded_by_day.items():
        decision_equity = equity_by_day[prev_day[day]]
        increment = traded / decision_equity / 2.0
        total += increment
        key = (day.year, day.month)
        per_month[key] = per_month.get(key, 0.0) + increment
    assert res.turnover_total == pytest.approx(total, rel=1e-9)
    assert len([m for m, v in per_month.items() if v > 0.0]) >= 2, (
        "fixture must trade in both months"
    )
    assert res.turnover_total > max(per_month.values())
    assert res.turnover_total == pytest.approx(sum(per_month.values()), rel=1e-9)


# ---------------------------------------------------------------------------
# Backtest-engine parity: identical scenario through both _execute paths.
# ---------------------------------------------------------------------------


def _replay_runner() -> ReplayRunner:
    return ReplayRunner(
        gateway_factory=lambda d: None,  # type: ignore[arg-type,return-value]
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=ReplayConfig(start=date(2026, 6, 1), end=date(2026, 6, 5), label="parity"),
    )


def _target(day: date, symbol: str, weight: float) -> Any:
    from dynamo.domain.models.portfolio import TargetPortfolio, TargetWeight
    from dynamo.domain.models.research import Sleeve

    weights = (
        (TargetWeight(symbol=symbol, weight=weight, sleeve=Sleeve.LONG_TERM),)
        if weight > 0.0
        else ()
    )
    return TargetPortfolio(
        id=f"tp-{day.isoformat()}",
        as_of=datetime(day.year, day.month, day.day, tzinfo=UTC),
        risk_verdict_ref="rv-parity",
        weights=weights,
        cash=1.0 - weight,
    )


def test_realized_pnl_parity_with_backtest_engine() -> None:
    """A buy-then-full-sell round trip produces bit-identical fills (qty,
    price, trade_value, cost, realized_pnl) through ``ReplayRunner._execute``
    and ``BacktestEngine._execute`` — the replay realized-P&L math mirrors
    engine.py, never forks it."""
    from dynamo.backtest.engine import BacktestConfig, BacktestData, BacktestEngine

    symbol = "XYZ"
    d1, d2 = date(2026, 6, 2), date(2026, 6, 3)
    sigma_annual = 0.30
    adv = 50_000_000.0
    equity = 100_000.0
    closes = {symbol: 100.0}
    opens_1 = {symbol: 101.0}
    opens_2 = {symbol: 103.0}

    frame_index = pd.Index([d1, d2])
    data = BacktestData(
        opens=pd.DataFrame({symbol: [opens_1[symbol], opens_2[symbol]]}, index=frame_index),
        closes=pd.DataFrame({symbol: [102.0, 103.5]}, index=frame_index),
        universe={symbol: ((date(2026, 1, 2), None),)},
        adv_dollar_20d={symbol: adv},
        sigma_annual_override={symbol: sigma_annual},
    )
    engine = BacktestEngine(BacktestConfig())
    runner = _replay_runner()

    r_cash, e_cash = equity, equity
    r_shares: dict[str, float] = {}
    r_basis: dict[str, float] = {}
    e_shares: dict[str, float] = {}
    e_basis: dict[str, float] = {}
    r_fills: list[Any] = []
    e_fills: list[Any] = []
    for day, target_weight, opens in ((d1, 0.10, opens_1), (d2, 0.0, opens_2)):
        target = _target(day, symbol, target_weight)
        r_cash, fills, _ = runner._execute(
            target,
            equity,
            closes,
            opens,
            day,
            r_cash,
            r_shares,
            r_basis,
            {symbol: sigma_annual},
            {symbol: adv},
        )
        r_fills.extend(fills)
        e_cash, fills2, _ = engine._execute(
            target,
            equity,
            closes,
            data.opens.loc[day],
            day,
            e_cash,
            e_shares,
            e_basis,
            data,
        )
        e_fills.extend(fills2)

    assert len(r_fills) == len(e_fills) == 2
    for rf, ef in zip(r_fills, e_fills, strict=True):
        assert (rf.symbol, rf.side, rf.qty, rf.price) == (
            ef.symbol,
            ef.side,
            ef.qty,
            ef.price,
        )
        assert rf.trade_value == ef.trade_value
        assert rf.cost == ef.cost
        assert rf.realized_pnl == ef.realized_pnl
    assert r_cash == e_cash

    # Hand-computed check of the same numbers (average-cost basis).
    buy, sell = r_fills
    assert buy.side == "buy" and sell.side == "sell"
    assert buy.realized_pnl is None
    hand_basis = (buy.trade_value + buy.cost) / buy.qty
    expected = round(sell.trade_value - sell.cost - sell.qty * hand_basis, 2)
    assert sell.realized_pnl == pytest.approx(expected, abs=0.011)
    assert sell.realized_pnl is not None and math.isfinite(sell.realized_pnl)
