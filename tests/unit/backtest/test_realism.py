"""Bankroll-realism knobs in the backtest engine (backtest-campaign Phase 3).

Same semantics as the replay runner (shared helpers in ``backtest.costs``):
whole_shares floors desired quantities (crypto exempt), min_order_notional
skips dust orders, commission_per_order adds a flat fee per executed order —
and the defaults reproduce the hand-computed toy backtest to the cent.
"""

from __future__ import annotations

import math

import pytest

from dynamo.backtest.costs import (
    CostParams,
    order_cost,
    order_skipped,
    whole_share_qty,
)
from dynamo.backtest.engine import BacktestConfig, BacktestEngine, SessionAdvice
from tests.unit.backtest.builders import (
    ScriptedProvider,
    sell_verdict,
    sessions,
    signal,
    toy_construction_config,
    toy_data,
    toy_risk_config,
)
from tests.unit.backtest.test_engine_toy import (
    CLOSES,
    EXPECTED_EQUITY,
    OPENS,
)

DATES = sessions(10)


def _script() -> dict:
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
    return script


def _run(**config_overrides: object):
    payload: dict[str, object] = {"initial_equity": 100_000.0, "flat_cost_bps": 10.0}
    payload.update(config_overrides)
    config = BacktestConfig(**payload)  # type: ignore[arg-type]
    engine = BacktestEngine(
        config,
        construction_config=toy_construction_config(),
        risk_config=toy_risk_config(),
    )
    return engine.run(toy_data(OPENS, CLOSES, DATES), ScriptedProvider(_script()))


# -- shared helpers ----------------------------------------------------------


def test_cost_helpers_defaults_are_inert() -> None:
    p = CostParams()
    assert not order_skipped(0.005, p)  # min 0 never skips
    assert order_cost(10_000.0, 0.01, 1e9, p) == pytest.approx(
        10_000.0 * 2.0 / 10_000.0 + 10_000.0 * 0.01 * math.sqrt(10_000.0 / 1e9)
    )
    assert whole_share_qty("AAPL", 12.7, whole_shares=False) == 12.7


def test_cost_helpers_apply_knobs() -> None:
    p = CostParams(commission_per_order=1.5, min_order_notional=25.0)
    assert order_skipped(24.99, p)
    assert not order_skipped(25.0, p)
    base = order_cost(10_000.0, 0.01, 1e9, CostParams())
    assert order_cost(10_000.0, 0.01, 1e9, p) == pytest.approx(base + 1.5)
    assert whole_share_qty("AAPL", 12.7, whole_shares=True) == 12.0
    assert whole_share_qty("BTC/USD", 12.7, whole_shares=True) == 12.7  # exempt


# -- bit-parity: defaults reproduce the pre-change engine --------------------


def test_default_knobs_reproduce_hand_computed_toy_to_the_cent() -> None:
    result = _run()
    assert list(result.equity_curve) == EXPECTED_EQUITY


def test_explicit_zero_knobs_are_bit_identical_to_unset() -> None:
    baseline = _run()
    explicit = _run(whole_shares=False, min_order_notional=0.0, commission_per_order=0.0)
    assert list(explicit.equity_curve) == list(baseline.equity_curve)
    assert explicit.fills == baseline.fills


# -- whole shares ------------------------------------------------------------


def test_whole_shares_floors_desired_quantities() -> None:
    # $100,100 -> $30,030 per name -> fractional desired (600.6 / 1201.2 /
    # 1501.5 shares at d0 closes 50/25/20); flooring buys 600/1201/1501.
    fractional = _run(initial_equity=100_100.0)  # type: ignore[arg-type]
    whole = _run(initial_equity=100_100.0, whole_shares=True)  # type: ignore[arg-type]
    frac_buys = {f.symbol: f.qty for f in fractional.fills if f.side == "buy"}
    whole_buys = {f.symbol: f.qty for f in whole.fills if f.side == "buy"}
    assert frac_buys["AAA"] == pytest.approx(600.6)
    assert whole_buys == {"AAA": 600.0, "BBB": 1201.0, "CCC": 1501.0}
    # Full exits sell the exact (whole) held quantities.
    whole_sells = {f.symbol: f.qty for f in whole.fills if f.side == "sell"}
    assert whole_sells == whole_buys
    # Divergence bounded: sub-share residue on a $100k book.
    assert abs(whole.terminal_equity - fractional.terminal_equity) < 100.0


def test_crypto_symbol_exempt_from_flooring_in_engine() -> None:
    dates = sessions(6)
    opens = {"BTC/USD": [100.0] * 6}
    closes = {"BTC/USD": [100.0] * 6}
    script = {dates[0]: SessionAdvice(signals=(signal("BTC/USD", dates[0]),))}
    config = BacktestConfig(initial_equity=1_000.0, flat_cost_bps=10.0, whole_shares=True)
    engine = BacktestEngine(
        config,
        construction_config=toy_construction_config(),
        risk_config=toy_risk_config(),
    )
    result = engine.run(toy_data(opens, closes, dates), ScriptedProvider(script))
    buys = [f for f in result.fills if f.side == "buy"]
    assert buys and buys[0].qty != math.floor(buys[0].qty)  # fractional fill


# -- min order notional ------------------------------------------------------


def test_min_order_notional_skips_orders_below_the_floor() -> None:
    # $1,000 book -> $300 per name; a $400 floor skips everything, a $100
    # floor fills everything.
    skipped = _run(initial_equity=1_000.0, min_order_notional=400.0)  # type: ignore[arg-type]
    assert skipped.fills == ()
    filled = _run(initial_equity=1_000.0, min_order_notional=100.0)  # type: ignore[arg-type]
    assert {f.symbol for f in filled.fills if f.side == "buy"} == {"AAA", "BBB", "CCC"}


# -- commission --------------------------------------------------------------


def test_commission_added_per_executed_order_and_hits_equity() -> None:
    fee = 2.0
    baseline = _run()
    with_fee = _run(commission_per_order=fee)  # type: ignore[arg-type]
    base_costs = {(f.session, f.symbol, f.side): f.cost for f in baseline.fills}
    for f in with_fee.fills:
        assert f.cost == pytest.approx(base_costs[(f.session, f.symbol, f.side)] + fee)
    # d1 (first fill day): 3 buys -> equity exactly 3 x fee lower.
    assert with_fee.equity_curve.iloc[1] == pytest.approx(
        baseline.equity_curve.iloc[1] - 3 * fee
    )
