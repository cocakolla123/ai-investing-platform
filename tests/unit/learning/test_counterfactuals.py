"""Golden tests for the counterfactual path-walkers on synthetic bar paths.

Each walker: a gain path, a loss path, and insufficient data → ``None``
(plan Phase 1). Index 0 of every session array is the entry session.
"""

from __future__ import annotations

import pytest

from dynamo.learning.counterfactuals import (
    delayed_entry_pnl_pct,
    ev_per_decision,
    half_size_pnl_pct,
    hold_pnl_pct,
    hold_to_horizon_pnl_pct,
    obey_quant_veto_pnl_pct,
)

# --- hold-instead: no entry ⇒ exactly zero, always --------------------------


def test_hold_is_always_zero() -> None:
    assert hold_pnl_pct() == 0.0


# --- half size: half the pnl on the original notional -----------------------


def test_half_size_gain_and_loss() -> None:
    assert half_size_pnl_pct(8.0) == 4.0
    assert half_size_pnl_pct(-6.0) == -3.0
    assert half_size_pnl_pct(0.0) == 0.0


def test_half_size_insufficient_data() -> None:
    assert half_size_pnl_pct(None) is None


# --- delayed entry (+1 session open) ----------------------------------------


def test_delayed_entry_gain_path() -> None:
    # Entered at 100; next session opened lower at 95, exit at 110:
    # the delayed entry did BETTER (110/95 - 1 = +15.789...%).
    opens = [100.0, 95.0, 101.0]
    result = delayed_entry_pnl_pct(opens, exit_session_index=2, exit_price=110.0)
    assert result == pytest.approx((110.0 / 95.0 - 1.0) * 100.0)


def test_delayed_entry_loss_path() -> None:
    # Next session gapped UP to 105 and the trade exited at 90 — the delay
    # made the loss deeper: 90/105 - 1 = -14.28...%.
    opens = [100.0, 105.0, 98.0]
    result = delayed_entry_pnl_pct(opens, exit_session_index=2, exit_price=90.0)
    assert result == pytest.approx((90.0 / 105.0 - 1.0) * 100.0)


def test_delayed_entry_same_session_exit_never_enters() -> None:
    # Exit happened on the entry session — the +1 delay means no position: 0.
    assert delayed_entry_pnl_pct([100.0, 101.0], 0, 100.5) == 0.0


def test_delayed_entry_insufficient_data() -> None:
    assert delayed_entry_pnl_pct([100.0], 3, 110.0) is None  # no next-session open
    assert delayed_entry_pnl_pct([100.0, 0.0], 3, 110.0) is None  # degenerate open
    assert delayed_entry_pnl_pct([100.0, 95.0], 3, None) is None  # no exit price


# --- hold to original horizon ------------------------------------------------


def test_hold_to_horizon_gain_path() -> None:
    closes = [100.0, 102.0, 104.0, 108.0, 112.0]
    # horizon = 3 sessions after entry → closes[3] = 108.
    assert hold_to_horizon_pnl_pct(100.0, closes, 3) == pytest.approx(8.0)


def test_hold_to_horizon_loss_path() -> None:
    closes = [100.0, 99.0, 95.0, 91.0]
    assert hold_to_horizon_pnl_pct(100.0, closes, 3) == pytest.approx(-9.0)


def test_hold_to_horizon_insufficient_data() -> None:
    closes = [100.0, 101.0]
    assert hold_to_horizon_pnl_pct(100.0, closes, 5) is None  # horizon close missing
    assert hold_to_horizon_pnl_pct(0.0, closes, 1) is None  # degenerate entry
    assert hold_to_horizon_pnl_pct(100.0, closes, None) is None  # no horizon
    assert hold_to_horizon_pnl_pct(100.0, closes, 0) is None  # zero horizon


# --- obey the quant veto ------------------------------------------------------


def test_obey_quant_veto_when_vetoed_is_flat() -> None:
    # Obeying the veto means no position, regardless of what the trade did.
    assert obey_quant_veto_pnl_pct(True, 12.0) == 0.0
    assert obey_quant_veto_pnl_pct(True, -8.0) == 0.0


def test_obey_quant_veto_when_not_vetoed_is_realized() -> None:
    assert obey_quant_veto_pnl_pct(False, 12.0) == 12.0
    assert obey_quant_veto_pnl_pct(False, -8.0) == -8.0


def test_obey_quant_veto_insufficient_data() -> None:
    assert obey_quant_veto_pnl_pct(None, 5.0) is None
    assert obey_quant_veto_pnl_pct(False, None) is None


# --- EV per decision (v1: the realization; see docstring for Phase 6) --------


def test_ev_per_decision_is_realized_pnl_v1() -> None:
    assert ev_per_decision(7.5) == 7.5
    assert ev_per_decision(-3.0) == -3.0
    assert ev_per_decision(None) is None
