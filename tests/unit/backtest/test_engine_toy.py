"""Hand-computed toy backtest (doc 08 section 9): 3 symbols, 10 bars, known
fills and costs -> exact equity curve and per-trade P&L asserted to the cent.

Script: 3 equal-sigma tactical signals appear at session 0 -> the production
constructor targets 30% each (90% invested, 10% cash floor). Fills at session
1 opens with flat 10 bps. Sell verdicts at session 4 -> full exit at session
5 opens. Every number below is hand-derived:

  buys at d1 opens (A 51, B 24, C 20), target $30,000 per name at d0 closes
  (A 50 -> 600 sh, B 25 -> 1200 sh, C 20 -> 1500 sh):
    A: 600*51  = 30,600.00  cost 30.60
    B: 1200*24 = 28,800.00  cost 28.80
    C: 1500*20 = 30,000.00  cost 30.00
    cash = 100,000 - 89,489.40 = 10,510.60
  sells at d5 opens (A 53, B 24, C 19):
    A: 31,800.00 cost 31.80  basis 30,630.60 -> pnl +1,137.60
    B: 28,800.00 cost 28.80  basis 28,828.80 -> pnl    -57.60
    C: 28,500.00 cost 28.50  basis 30,030.00 -> pnl -1,558.50
    cash = 10,510.60 + 89,010.90 = 99,521.50
"""

from __future__ import annotations

import pytest

from dynamo.backtest.engine import BacktestEngine, SessionAdvice
from tests.unit.backtest.builders import (
    ScriptedProvider,
    sell_verdict,
    sessions,
    signal,
    toy_backtest_config,
    toy_construction_config,
    toy_data,
    toy_risk_config,
)

DATES = sessions(10)

OPENS = {
    "AAA": [50.0, 51.0, 51.5, 51.0, 52.0, 53.0, 52.0, 52.0, 52.0, 52.0],
    "BBB": [25.0, 24.0, 25.0, 24.5, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
    "CCC": [20.0, 20.0, 20.5, 20.0, 20.0, 19.0, 19.0, 19.0, 19.0, 19.0],
}
CLOSES = {
    "AAA": [50.0, 52.0, 51.0, 52.0, 52.5, 53.0, 52.0, 52.0, 52.0, 52.0],
    "BBB": [25.0, 25.0, 25.0, 24.0, 24.5, 24.0, 24.0, 24.0, 24.0, 24.0],
    "CCC": [20.0, 21.0, 20.5, 20.0, 19.5, 19.0, 19.0, 19.0, 19.0, 19.0],
}

EXPECTED_EQUITY = [
    100_000.00,  # d0: all cash
    103_210.60,  # d1: 10,510.60 + 31,200 + 30,000 + 31,500
    101_860.60,  # d2: 10,510.60 + 30,600 + 30,000 + 30,750
    100_510.60,  # d3: 10,510.60 + 31,200 + 28,800 + 30,000
    100_660.60,  # d4: 10,510.60 + 31,500 + 29,400 + 29,250
    99_521.50,  # d5-d9: flat, fully in cash
    99_521.50,
    99_521.50,
    99_521.50,
    99_521.50,
]


@pytest.fixture(scope="module")
def result():
    script = {
        DATES[i]: SessionAdvice(
            signals=(
                signal("AAA", DATES[i]),
                signal("BBB", DATES[i]),
                signal("CCC", DATES[i]),
            )
        )
        for i in range(4)
    }
    script[DATES[4]] = SessionAdvice(
        verdicts=(
            sell_verdict("AAA", DATES[4]),
            sell_verdict("BBB", DATES[4]),
            sell_verdict("CCC", DATES[4]),
        )
    )
    engine = BacktestEngine(
        toy_backtest_config(cost_bps=10.0),
        construction_config=toy_construction_config(),
        risk_config=toy_risk_config(),
    )
    return engine.run(toy_data(OPENS, CLOSES, DATES), ScriptedProvider(script))


def test_equity_curve_exact_to_the_cent(result):
    assert list(result.equity_curve) == EXPECTED_EQUITY


def test_fills_exact(result):
    buys = [f for f in result.fills if f.side == "buy"]
    sells = [f for f in result.fills if f.side == "sell"]
    assert [(f.symbol, f.qty, f.price, f.trade_value, f.cost) for f in buys] == [
        ("AAA", 600.0, 51.0, 30_600.00, 30.60),
        ("BBB", 1200.0, 24.0, 28_800.00, 28.80),
        ("CCC", 1500.0, 20.0, 30_000.00, 30.00),
    ]
    assert all(f.session == DATES[1] for f in buys)  # NEXT open, never same-bar
    assert [(f.symbol, f.trade_value, f.cost) for f in sells] == [
        ("AAA", 31_800.00, 31.80),
        ("BBB", 28_800.00, 28.80),
        ("CCC", 28_500.00, 28.50),
    ]
    assert all(f.session == DATES[5] for f in sells)


def test_realized_pnl_to_the_cent(result):
    pnls = {f.symbol: f.realized_pnl for f in result.fills if f.side == "sell"}
    assert pnls == {"AAA": 1_137.60, "BBB": -57.60, "CCC": -1_558.50}


def test_no_intermediate_churn_and_zero_violations(result):
    # Retained holdings are pinned at current weight: no trades d2-d4.
    assert len(result.fills) == 6
    assert result.constraint_violation_count == 0
    # Cash floor honored every session in the target.
    for record in result.sessions:
        assert not record.violations


def test_deterministic_rerun(result):
    engine = BacktestEngine(
        toy_backtest_config(cost_bps=10.0),
        construction_config=toy_construction_config(),
        risk_config=toy_risk_config(),
    )
    script = {
        DATES[i]: SessionAdvice(
            signals=(
                signal("AAA", DATES[i]),
                signal("BBB", DATES[i]),
                signal("CCC", DATES[i]),
            )
        )
        for i in range(4)
    }
    script[DATES[4]] = SessionAdvice(
        verdicts=(
            sell_verdict("AAA", DATES[4]),
            sell_verdict("BBB", DATES[4]),
            sell_verdict("CCC", DATES[4]),
        )
    )
    again = engine.run(toy_data(OPENS, CLOSES, DATES), ScriptedProvider(script))
    assert list(again.equity_curve) == list(result.equity_curve)
    assert again.fills == result.fills
