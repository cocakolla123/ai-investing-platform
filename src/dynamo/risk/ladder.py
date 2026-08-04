"""Drawdown ladder: the capital-preservation backbone.

Design: docs/design/06-risk-engine.md section 3. A pure, sticky state machine
over (equity, high-water mark). Downgrades are immediate; recovery requires
5 consecutive sessions with drawdown inside the trigger - 1% band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dynamo.domain.models.risk import LadderRung

RECOVERY_BAND = 0.01
DEFAULT_RECOVERY_SESSIONS = 5

RUNG_TRIGGERS: dict[LadderRung, float] = {
    LadderRung.DD5: -0.05,
    LadderRung.DD8: -0.08,
    LadderRung.DD10: -0.10,
    LadderRung.DD15: -0.15,
    LadderRung.DD20: -0.20,
}

_SEVERITY: dict[LadderRung, int] = {
    LadderRung.NORMAL: 0,
    LadderRung.DD5: 1,
    LadderRung.DD8: 2,
    LadderRung.DD10: 3,
    LadderRung.DD15: 4,
    LadderRung.DD20: 5,
}


@dataclass(frozen=True)
class LadderState:
    """Sticky ladder state; carries the recovery-hysteresis counter."""

    rung: LadderRung = LadderRung.NORMAL
    sessions_above_band: int = 0
    last_session_index: int = -1


@dataclass(frozen=True)
class RungEffects:
    """Cumulative rung effects (doc 06 section 3 table)."""

    risk_multiplier: float  # scales new-position risk budget
    tactical_multiplier: float  # scales the tactical sleeve position cap
    tactical_allowed: bool
    new_positions_allowed: bool
    gross_override: float | None  # de-risk gross ceiling (None = no override)
    defensive: bool
    requires_human_ack: bool
    kill_switch: bool


RUNG_EFFECTS: dict[LadderRung, RungEffects] = {
    LadderRung.NORMAL: RungEffects(1.0, 1.0, True, True, None, False, False, False),
    LadderRung.DD5: RungEffects(0.5, 0.5, True, True, None, False, False, False),
    LadderRung.DD8: RungEffects(0.5, 0.0, False, True, None, False, True, False),
    LadderRung.DD10: RungEffects(0.0, 0.0, False, False, None, False, True, False),
    LadderRung.DD15: RungEffects(0.0, 0.0, False, False, 0.5, True, True, False),
    LadderRung.DD20: RungEffects(0.0, 0.0, False, False, 0.0, True, True, True),
}


def severity(rung: LadderRung) -> int:
    return _SEVERITY[rung]


def drawdown(equity: float, high_water_mark: float) -> float:
    """Fractional drawdown (<= 0 when below HWM); NaN when inputs unusable."""
    if not (math.isfinite(equity) and math.isfinite(high_water_mark)):
        return float("nan")
    if high_water_mark <= 0.0 or equity < 0.0:
        return float("nan")
    return equity / high_water_mark - 1.0


def rung_for_drawdown(dd: float) -> LadderRung:
    """Rung implied by a drawdown level. Non-finite drawdown fails closed (DD20)."""
    if not math.isfinite(dd):
        return LadderRung.DD20
    for rung in (
        LadderRung.DD20,
        LadderRung.DD15,
        LadderRung.DD10,
        LadderRung.DD8,
        LadderRung.DD5,
    ):
        if dd <= RUNG_TRIGGERS[rung]:
            return rung
    return LadderRung.NORMAL


def step(
    prev: LadderState,
    equity: float,
    high_water_mark: float,
    session_index: int,
    recovery_sessions: int = DEFAULT_RECOVERY_SESSIONS,
) -> LadderState:
    """One daily ladder evaluation.

    - Invalid inputs: hold the current rung and reset the recovery counter
      (never recover on bad data).
    - Deterioration past a deeper trigger: move down immediately (sticky).
    - Recovery: only after ``recovery_sessions`` consecutive sessions with
      drawdown above the current rung's trigger + 1% band; then re-derive the
      rung from the actual drawdown.
    """
    dd = drawdown(equity, high_water_mark)
    if not math.isfinite(dd):
        return LadderState(prev.rung, 0, session_index)

    implied = rung_for_drawdown(dd)
    if severity(implied) > severity(prev.rung):
        return LadderState(implied, 0, session_index)
    if prev.rung == LadderRung.NORMAL:
        return LadderState(LadderRung.NORMAL, 0, session_index)

    recovery_threshold = RUNG_TRIGGERS[prev.rung] + RECOVERY_BAND
    if dd > recovery_threshold:
        count = prev.sessions_above_band + 1
        if count >= max(recovery_sessions, 1):
            return LadderState(implied, 0, session_index)
        return LadderState(prev.rung, count, session_index)
    return LadderState(prev.rung, 0, session_index)


def effects_for(rung: LadderRung) -> RungEffects:
    return RUNG_EFFECTS[rung]
