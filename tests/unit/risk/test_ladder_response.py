"""Ladder transitions actually do something (doc 06 §3).

``RungEffects.kill_switch`` was read only by ``backtest/engine.py`` and
``requires_human_ack`` by nothing at all: in the live path a -20% drawdown
reduced to ``gross_override = 0.0`` — no file, no alert, no freeze.
"""

from __future__ import annotations

import pytest

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.ladder import RUNG_EFFECTS
from dynamo.risk.ladder_response import ack_note_required, respond_to_transition


def test_no_transition_is_no_response() -> None:
    assert respond_to_transition(LadderRung.DD5, LadderRung.DD5, -0.06) is None


def test_dd20_engages_the_kill_switch() -> None:
    """doc 06 §3: -20% = full stop, freeze platform trading."""
    response = respond_to_transition(LadderRung.DD15, LadderRung.DD20, -0.21)
    assert response is not None
    assert response.engage_kill_switch is True
    assert response.requires_human_ack is True
    assert response.alert_severity == "page"
    assert "KILL SWITCH ENGAGED" in response.message


def test_a_straight_drop_to_dd20_still_engages_it() -> None:
    """A gap down skips the intermediate rungs; the freeze must not be skipped."""
    response = respond_to_transition(LadderRung.NORMAL, LadderRung.DD20, -0.35)
    assert response is not None and response.engage_kill_switch is True


@pytest.mark.parametrize(
    "rung", [LadderRung.DD8, LadderRung.DD10, LadderRung.DD15, LadderRung.DD20]
)
def test_every_ack_requiring_rung_asks_for_an_ack(rung: LadderRung) -> None:
    response = respond_to_transition(LadderRung.NORMAL, rung, -0.30)
    assert response is not None
    assert response.requires_human_ack is RUNG_EFFECTS[rung].requires_human_ack is True
    assert "blocked until a human acknowledges" in response.message


def test_dd5_warns_and_needs_no_ack() -> None:
    response = respond_to_transition(LadderRung.NORMAL, LadderRung.DD5, -0.06)
    assert response is not None
    assert (response.alert_severity, response.requires_human_ack) == ("warn", False)
    assert response.engage_kill_switch is False


@pytest.mark.parametrize("rung", [LadderRung.DD5, LadderRung.DD8])
def test_shallow_rungs_warn_and_deep_rungs_page(rung: LadderRung) -> None:
    """doc 12 §7: -5%/-8% warn; -10% and below page."""
    shallow = respond_to_transition(LadderRung.NORMAL, rung, -0.09)
    deep = respond_to_transition(LadderRung.NORMAL, LadderRung.DD10, -0.11)
    assert shallow is not None and shallow.alert_severity == "warn"
    assert deep is not None and deep.alert_severity == "page"


def test_recovery_never_engages_the_kill_switch() -> None:
    """Climbing out of DD15 is news, not an emergency."""
    response = respond_to_transition(LadderRung.DD15, LadderRung.DD5, -0.06)
    assert response is not None
    assert response.deteriorated is False
    assert response.engage_kill_switch is False
    assert response.requires_human_ack is False
    assert response.alert_severity == "info"
    assert response.alert_name == "ladder_recovery"


def test_recovery_out_of_dd20_does_not_rewrite_the_switch_file() -> None:
    response = respond_to_transition(LadderRung.DD20, LadderRung.NORMAL, -0.02)
    assert response is not None and response.engage_kill_switch is False


def test_the_drawdown_is_in_the_message() -> None:
    response = respond_to_transition(LadderRung.NORMAL, LadderRung.DD10, -0.1234)
    assert response is not None and "-12.34%" in response.message


def test_a_blank_ack_note_is_refused() -> None:
    """Same rule as reconciliation acks: who/why is the point of the row."""
    with pytest.raises(ValueError, match="requires a note"):
        ack_note_required("   ")


def test_an_ack_note_is_stripped() -> None:
    assert ack_note_required("  reviewed  ") == "reviewed"
