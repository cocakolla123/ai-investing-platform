"""Cash ledger arithmetic (dynamo.execution.cash_ledger)."""

from __future__ import annotations

import pytest

from dynamo.domain.models.portfolio import Position
from dynamo.execution.cash_ledger import Fill, cash_flow, ledger_cash, mark_to_market


def test_buy_pays_notional_plus_fees() -> None:
    assert cash_flow(Fill("buy", 10, 100.0, 1.5)) == pytest.approx(-1_001.5)


def test_sell_receives_notional_minus_fees() -> None:
    assert cash_flow(Fill("SELL", 10, 100.0, 1.5)) == pytest.approx(998.5)


def test_unknown_side_is_an_error() -> None:
    with pytest.raises(ValueError):
        cash_flow(Fill("short", 1, 1.0))


def test_the_paper_account_reproduces_broker_cash() -> None:
    """The first paper fill: 123.7092 SPY at 764.621455 from $100,000. The
    broker showed $5,409.29; the old code kept $100,000."""
    cash = ledger_cash(100_000.0, [Fill("buy", 123.7092, 764.621455)])
    assert cash == pytest.approx(5_409.29, abs=0.01)


def test_round_trip_returns_to_opening_less_fees() -> None:
    fills = [Fill("buy", 5, 50.0, 0.5), Fill("sell", 5, 50.0, 0.5)]
    assert ledger_cash(1_000.0, fills) == pytest.approx(999.0)


def test_mark_to_market_equity_and_gross() -> None:
    positions = [Position(symbol="SPY", qty=10, avg_entry_price=90.0)]
    equity, gross = mark_to_market(1_000.0, positions, {"SPY": 100.0})
    assert equity == pytest.approx(2_000.0)
    assert gross == pytest.approx(0.5)


def test_missing_price_falls_back_to_entry_price_not_zero() -> None:
    positions = [Position(symbol="XYZ", qty=10, avg_entry_price=90.0)]
    equity, _ = mark_to_market(0.0, positions, {})
    assert equity == pytest.approx(900.0)
