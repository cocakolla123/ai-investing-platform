"""Drawdown ladder: stickiness and recovery hysteresis on random equity paths."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.ladder import (
    RECOVERY_BAND,
    RUNG_EFFECTS,
    RUNG_TRIGGERS,
    LadderState,
    effects_for,
    rung_for_drawdown,
    severity,
    step,
)

HWM = 1_000_000.0

drawdowns = st.floats(min_value=-0.6, max_value=0.2)
paths = st.lists(
    st.one_of(drawdowns, st.just(float("nan"))), min_size=1, max_size=120
)


@settings(max_examples=100, deadline=None)
@given(path=paths)
def test_ladder_stickiness_and_hysteresis(path: list[float]) -> None:
    state = LadderState()
    for i, dd in enumerate(path):
        prev = state
        equity = HWM * (1.0 + dd)
        state = step(prev, equity, HWM, i)
        assert state.last_session_index == i
        if dd != dd:  # NaN input: hold rung, never recover on bad data
            assert state.rung == prev.rung
            assert state.sessions_above_band == 0
            continue
        # Recompute exactly as the state machine does (same float rounding).
        dd = equity / HWM - 1.0
        implied = rung_for_drawdown(dd)
        # Downgrades are immediate: never end a session better than implied.
        assert severity(state.rung) >= severity(implied)
        if severity(state.rung) < severity(prev.rung):
            # Improvement only via full recovery hysteresis: the counter must
            # have been one short, and this session must clear the band.
            assert prev.sessions_above_band == 4
            assert dd > RUNG_TRIGGERS[prev.rung] + RECOVERY_BAND
            assert state.rung == implied
            assert state.sessions_above_band == 0


def _run(dds: list[float], start: LadderState | None = None) -> LadderState:
    state = start or LadderState()
    for i, dd in enumerate(dds):
        state = step(state, HWM * (1.0 + dd), HWM, i)
    return state


def test_downgrade_is_immediate_and_can_skip_rungs() -> None:
    assert _run([-0.005]).rung == LadderRung.NORMAL
    assert _run([-0.06]).rung == LadderRung.DD5
    assert _run([-0.16]).rung == LadderRung.DD15  # one session, three rungs down
    assert _run([-0.03, -0.21]).rung == LadderRung.DD20


def test_recovery_requires_five_consecutive_sessions_in_band() -> None:
    # Trigger DD5, then sit just inside the recovery band (-4% needs > -0.04).
    path = [-0.06] + [-0.03] * 4
    assert _run(path).rung == LadderRung.DD5  # four sessions: not yet
    path = [-0.06] + [-0.03] * 5
    assert _run(path).rung == LadderRung.NORMAL  # fifth clears it


def test_recovery_counter_resets_on_dip_back_below_band() -> None:
    # 4 clean sessions, one dip below the -4% band, then 4 more: still stuck.
    path = [-0.06] + [-0.03] * 4 + [-0.045] + [-0.03] * 4
    assert _run(path).rung == LadderRung.DD5
    # One more clean session completes a fresh streak of 5.
    assert _run([*path, -0.03]).rung == LadderRung.NORMAL


def test_sticky_between_trigger_and_band() -> None:
    # -4.2% is above the -5% trigger but below the -4% recovery threshold:
    # the rung must never improve, no matter how long it sits there.
    path = [-0.055] + [-0.042] * 50
    assert _run(path).rung == LadderRung.DD5


def test_recovery_rederives_rung_from_actual_drawdown() -> None:
    # From DD8, five sessions at -6% (above -7% band) recover to DD5, not NORMAL.
    path = [-0.09] + [-0.06] * 5
    assert _run(path).rung == LadderRung.DD5


def test_invalid_inputs_hold_state_fail_closed() -> None:
    start = _run([-0.09])
    assert start.rung == LadderRung.DD8
    state = step(start, float("nan"), HWM, 99)
    assert state.rung == LadderRung.DD8 and state.sessions_above_band == 0
    state = step(start, 1.0, 0.0, 100)  # broken HWM
    assert state.rung == LadderRung.DD8 and state.sessions_above_band == 0


def test_rung_effects_table_matches_doc() -> None:
    normal = effects_for(LadderRung.NORMAL)
    assert normal.risk_multiplier == 1.0 and normal.new_positions_allowed
    dd5 = effects_for(LadderRung.DD5)
    assert dd5.risk_multiplier == 0.5 and dd5.tactical_multiplier == 0.5
    dd8 = effects_for(LadderRung.DD8)
    assert not dd8.tactical_allowed and dd8.new_positions_allowed
    assert dd8.requires_human_ack
    dd10 = effects_for(LadderRung.DD10)
    assert not dd10.new_positions_allowed
    dd15 = effects_for(LadderRung.DD15)
    assert dd15.gross_override == 0.5 and dd15.defensive
    dd20 = effects_for(LadderRung.DD20)
    assert dd20.gross_override == 0.0 and dd20.kill_switch
    assert set(RUNG_EFFECTS) == set(LadderRung)


def test_rung_for_drawdown_thresholds() -> None:
    assert rung_for_drawdown(0.02) == LadderRung.NORMAL
    assert rung_for_drawdown(-0.0499) == LadderRung.NORMAL
    assert rung_for_drawdown(-0.05) == LadderRung.DD5
    assert rung_for_drawdown(-0.08) == LadderRung.DD8
    assert rung_for_drawdown(-0.10) == LadderRung.DD10
    assert rung_for_drawdown(-0.15) == LadderRung.DD15
    assert rung_for_drawdown(-0.20) == LadderRung.DD20
    assert rung_for_drawdown(float("nan")) == LadderRung.DD20  # fail closed
