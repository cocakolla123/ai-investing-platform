"""Survivorship defenses (doc 08 sections 2, 9): the point-in-time universe is
a REQUIRED input (API-level), and out-of-membership symbols are refused."""

from __future__ import annotations

import pytest

from dynamo.backtest.engine import BacktestData, BacktestEngine, SessionAdvice, in_universe
from tests.unit.backtest.builders import (
    ScriptedProvider,
    frame,
    sessions,
    signal,
    toy_backtest_config,
    toy_construction_config,
    toy_data,
    toy_risk_config,
)

DATES = sessions(8)
SYM = "DDD"
PRICES = {SYM: [50.0] * 8}


def test_universe_is_a_required_input_api_level():
    opens = frame(PRICES, DATES)
    closes = frame(PRICES, DATES)
    with pytest.raises(TypeError):
        BacktestData(opens=opens, closes=closes)  # type: ignore[call-arg]


def test_in_universe_windows():
    universe = {SYM: ((DATES[2], DATES[4]),)}
    assert not in_universe(universe, SYM, DATES[1])
    assert in_universe(universe, SYM, DATES[2])
    assert in_universe(universe, SYM, DATES[4])
    assert not in_universe(universe, SYM, DATES[5])
    assert not in_universe(universe, "OTHER", DATES[2])


def test_engine_refuses_symbols_outside_membership():
    # Signal fires every session, but the symbol is a member only for d2-d3:
    # nothing may be bought outside that window, and the refusals are logged.
    universe = {SYM: ((DATES[2], DATES[3]),)}
    script = {d: SessionAdvice(signals=(signal(SYM, d),)) for d in DATES}
    engine = BacktestEngine(
        toy_backtest_config(cost_bps=0.0),
        construction_config=toy_construction_config(),
        risk_config=toy_risk_config(),
    )
    result = engine.run(toy_data(PRICES, PRICES, DATES, universe), ScriptedProvider(script))
    buys = [f for f in result.fills if f.side == "buy"]
    # Only decisions taken on d2/d3 can buy (fills land d3/d4).
    assert all(f.session in (DATES[3], DATES[4]) for f in buys)
    # Out-of-membership advice is recorded as refused.
    refused_days = {d for d, s in result.excluded_out_of_universe if s == SYM}
    assert DATES[0] in refused_days and DATES[6] in refused_days
    assert result.constraint_violation_count == 0
