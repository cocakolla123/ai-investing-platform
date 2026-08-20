"""What a drawdown-ladder transition *does* (doc 06 §3, doc 12 §7).

``ladder.RUNG_EFFECTS`` already describes the state each rung implies. Two of
its fields had no consumer outside the backtester:

- ``kill_switch`` (DD20) was read only by ``backtest/engine.py``. In the live
  path a -20% drawdown reduced to ``gross_override = 0.0``: no file, no
  alert, no freeze. Doc 06 §3 says "full stop, freeze platform trading".
- ``requires_human_ack`` (DD8/DD10/DD15/DD20) was read by nothing at all.

This module is the pure decision — *given* a rung transition, what must
happen — so the effects are testable without a broker, a database or a
filesystem. The composition root (``orchestration/assets/portfolio_exec.py``)
performs them.

Ack semantics deliberately mirror reconciliation (doc 09 §5): the blocking
state is a row in the database with no ack row against it, so nothing
in-memory has to survive a restart for the halt to hold, and the ack itself
is an audited human act with a mandatory note.
"""

from __future__ import annotations

from dataclasses import dataclass

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.ladder import RUNG_EFFECTS, severity

#: doc 12 §7 alert catalog: "-10% and below" pages, "-5%, -8%" warns.
PAGE_FROM_RUNG = LadderRung.DD10


@dataclass(frozen=True)
class LadderResponse:
    """Everything a transition into ``to_rung`` obliges the platform to do."""

    from_rung: LadderRung
    to_rung: LadderRung
    deteriorated: bool
    engage_kill_switch: bool
    requires_human_ack: bool
    alert_severity: str  # page | warn | info
    alert_name: str
    message: str

    @property
    def is_noop(self) -> bool:
        return self.from_rung == self.to_rung


def _describe(from_rung: LadderRung, to_rung: LadderRung, drawdown: float) -> str:
    direction = "deteriorated to" if severity(to_rung) > severity(from_rung) else "recovered to"
    return (
        f"drawdown ladder {direction} {to_rung.value} from {from_rung.value} "
        f"(drawdown {drawdown:+.2%})"
    )


def respond_to_transition(
    from_rung: LadderRung, to_rung: LadderRung, drawdown: float
) -> LadderResponse | None:
    """The obligations of one ladder transition; None when the rung is unchanged.

    Only *deterioration* engages the kill switch or requires an ack —
    recovering out of DD15 is news, not an emergency, and it must never write
    a kill-switch file. Recovery still alerts (at ``info``) because a ladder
    move in either direction is something the operator wants to know about.
    """
    if from_rung == to_rung:
        return None

    deteriorated = severity(to_rung) > severity(from_rung)
    effects = RUNG_EFFECTS[to_rung]

    if not deteriorated:
        return LadderResponse(
            from_rung=from_rung,
            to_rung=to_rung,
            deteriorated=False,
            engage_kill_switch=False,
            requires_human_ack=False,
            alert_severity="info",
            alert_name="ladder_recovery",
            message=_describe(from_rung, to_rung, drawdown),
        )

    page = severity(to_rung) >= severity(PAGE_FROM_RUNG)
    message = _describe(from_rung, to_rung, drawdown)
    if effects.kill_switch:
        message += " — KILL SWITCH ENGAGED: platform trading frozen, open orders cancelled"
    if effects.requires_human_ack:
        message += (
            " — new orders are blocked until a human acknowledges this rung "
            "(POST /admin/ladder/{id}/ack)"
        )
    return LadderResponse(
        from_rung=from_rung,
        to_rung=to_rung,
        deteriorated=True,
        engage_kill_switch=effects.kill_switch,
        requires_human_ack=effects.requires_human_ack,
        alert_severity="page" if page else "warn",
        alert_name="ladder_transition",
        message=message,
    )


def ack_note_required(note: str) -> str:
    """Validate a ladder acknowledgment note (mirrors reconciliation acks).

    A blank note is refused: "who/when/why" is the entire value of the audit
    row, and nothing ever clears a ladder halt automatically.
    """
    cleaned = note.strip()
    if not cleaned:
        raise ValueError("ladder ack requires a note describing the review")
    return cleaned
