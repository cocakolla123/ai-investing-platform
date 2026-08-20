"""Regime detection: planted-crash labeling and no flapping on noise."""

from __future__ import annotations

import numpy as np
from risk_helpers import make_dates, make_regime_history

from dynamo.domain.models.risk import Regime, RiskConfig
from dynamo.risk.regime import (
    RegimeHistory,
    backfill_regimes,
    classify_composite,
    latest_regime,
)

CFG = RiskConfig()


def test_too_short_history_yields_nothing_fail_closed() -> None:
    history = make_regime_history(sessions=150)
    assert backfill_regimes(history, CFG) == ()
    assert latest_regime(history, CFG) is None


def test_calm_bull_history_is_risk_on() -> None:
    history = make_regime_history(
        sessions=320, drift=0.0008, vol=0.006, oas=3.0, slope=1.0, breadth=0.8
    )
    states = backfill_regimes(history, CFG)
    assert states
    assert states[-1].regime is Regime.RISK_ON
    assert states[-1].composite_score > 0.75


def test_planted_2008_like_crash_labels_risk_off_or_crisis() -> None:
    """Synthetic history with a planted crash: calm ramp, then a -60% slide
    with credit blowout, inverted curve, and collapsed breadth."""
    calm, crash = 280, 60
    n = calm + crash
    rng = np.random.default_rng(42)
    close = np.empty(n)
    close[:calm] = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.006, calm))
    crash_returns = rng.normal(-0.015, 0.03, crash)
    close[calm:] = close[calm - 1] * np.cumprod(1.0 + crash_returns)
    oas = np.concatenate([np.full(calm, 4.0), np.full(crash, 9.0)])
    slope = np.concatenate([np.full(calm, 1.0), np.full(crash, -0.6)])
    breadth = np.concatenate([np.full(calm, 0.65), np.full(crash, 0.08)])
    history = RegimeHistory(make_dates(n), close, oas, slope, breadth)

    states = backfill_regimes(history, CFG)
    assert states
    crash_states = states[-(crash - 20) :]  # after the crash is well established
    assert all(s.regime in (Regime.RISK_OFF, Regime.CRISIS) for s in crash_states)
    assert states[-1].regime is Regime.CRISIS  # full stress: every input red
    assert states[-1].composite_score <= -1.25


def test_no_flapping_on_boundary_noise() -> None:
    """The candidate regime crosses the risk_on boundary every other session;
    hysteresis (5 consecutive sessions) must prevent any switch at all.

    Construction: trend +2 (steady uptrend), curve +1, and credit + breadth
    alternating (+2/-2 each). Whatever the vol score v in [-2, 2] does, even
    sessions score (7 + v) / 5 >= 1.0 (risk_on candidate) and odd sessions
    score (-1 + v) / 5 <= 0.2 (neutral or risk_off candidate), so no candidate
    other than the starting neutral can ever persist 5 sessions.
    """
    n = 320
    even = np.arange(n) % 2 == 0
    close = 100.0 * np.power(1.001, np.arange(n))
    history = RegimeHistory(
        make_dates(n),
        close,
        np.where(even, 3.0, 9.0),  # credit +2 / -2
        np.full(n, 1.0),  # curve +1
        np.where(even, 0.80, 0.05),  # breadth +2 / -2
    )
    states = backfill_regimes(history, CFG)
    assert states
    # The boundary really is crossed on even sessions...
    assert max(s.composite_score for s in states) >= 0.75
    # ...but the labeled regime never moves off the starting state.
    assert {s.regime for s in states} == {Regime.NEUTRAL}


def test_hysteresis_delays_but_does_not_block_persistent_change() -> None:
    n = 320
    rng = np.random.default_rng(9)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.005, n))
    # Persistent risk-on inputs: regime should switch after >= 5 sessions.
    history = RegimeHistory(
        make_dates(n),
        close,
        np.full(n, 3.0),
        np.full(n, 1.5),
        np.full(n, 0.85),
    )
    states = backfill_regimes(history, CFG)
    assert states
    first_on = next(i for i, s in enumerate(states) if s.regime is Regime.RISK_ON)
    assert first_on >= CFG.hysteresis_sessions - 1  # not instant
    assert states[-1].regime is Regime.RISK_ON


def test_classify_composite_thresholds_and_fail_closed() -> None:
    assert classify_composite(1.0) is Regime.RISK_ON
    assert classify_composite(0.0) is Regime.NEUTRAL
    assert classify_composite(-0.75) is Regime.RISK_OFF
    assert classify_composite(-1.5) is Regime.CRISIS
    assert classify_composite(float("nan")) is Regime.RISK_OFF


def test_state_bookkeeping() -> None:
    history = make_regime_history(sessions=320)
    states = backfill_regimes(history, CFG)
    assert states
    for state in states:
        assert -2.0 <= state.composite_score <= 2.0
        assert state.sessions_in_state >= 1
        assert set(state.inputs) == {"trend", "vol", "credit", "curve", "breadth"}
