"""Gate refusal matrix (doc 09 §7): full truth table of
(env, approval validity, kill-switch, readiness) — exactly one permitted LIVE
combination, PAPER permitted, every refusal with the correct typed reason."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.portfolio import Order, OrderSide, OrderState, OrderType
from dynamo.execution.gate import (
    ApprovalArtifact,
    GateReason,
    Readiness,
    disengage_kill_switch,
    engage_kill_switch,
    kill_switch_engaged,
    live_orders_permitted,
    load_approval,
    sign_approval_body,
)
from dynamo.execution.mock_broker import MockBroker, ScriptStep

KEY = b"test-approval-key"
WRONG_KEY = b"wrong-key"
TODAY = date(2026, 8, 3)

READY = Readiness(
    clean_paper_months=4.0, calibration_ok=True, risk_suite_green=True, backup_drill_ok=True
)
UNREADY = Readiness(
    clean_paper_months=1.0, calibration_ok=True, risk_suite_green=True, backup_drill_ok=True
)


def _approval(
    *, expiry: date = date(2026, 8, 31), key: bytes = KEY, scope: tuple[str, ...] = ("all",)
) -> ApprovalArtifact:
    body = {"scope": list(scope), "expiry": expiry.isoformat(), "date": TODAY.isoformat()}
    return ApprovalArtifact(
        scope=scope, expiry=expiry, signature=sign_approval_body(body, key), body=body
    )


VALID = _approval()
EXPIRED = _approval(expiry=date(2026, 7, 1))
BAD_SIG = _approval(key=WRONG_KEY)

APPROVALS = {"none": None, "valid": VALID, "expired": EXPIRED, "bad_sig": BAD_SIG}
READINESS = {"ready": READY, "unready": UNREADY}


def _gate(env: DynamoEnv, approval: ApprovalArtifact | None, kill: bool, readiness: Readiness):
    return live_orders_permitted(env, approval, kill, readiness, approval_key=KEY, today=TODAY)


def test_full_truth_table_exactly_one_live_permitted() -> None:
    permitted_live = []
    for approval_name, approval in APPROVALS.items():
        for kill in (False, True):
            for readiness_name, readiness in READINESS.items():
                result = _gate(DynamoEnv.LIVE, approval, kill, readiness)
                if result.permitted:
                    permitted_live.append((approval_name, kill, readiness_name))
                    assert result.reason == GateReason.LIVE_PERMITTED
    assert permitted_live == [("valid", False, "ready")]


@pytest.mark.parametrize(
    ("approval_name", "kill", "readiness_name", "expected"),
    [
        ("valid", True, "ready", GateReason.KILL_SWITCH_ENGAGED),
        ("none", True, "unready", GateReason.KILL_SWITCH_ENGAGED),
        ("none", False, "ready", GateReason.APPROVAL_MISSING),
        ("bad_sig", False, "ready", GateReason.APPROVAL_BAD_SIGNATURE),
        ("expired", False, "ready", GateReason.APPROVAL_EXPIRED),
        ("valid", False, "unready", GateReason.READINESS_UNMET),
    ],
)
def test_live_refusals_have_typed_reasons(
    approval_name: str, kill: bool, readiness_name: str, expected: GateReason
) -> None:
    result = _gate(DynamoEnv.LIVE, APPROVALS[approval_name], kill, READINESS[readiness_name])
    assert not result.permitted
    assert result.reason == expected
    assert result.detail  # refusals are never silent


@pytest.mark.parametrize("env", [DynamoEnv.PAPER, DynamoEnv.DEV, DynamoEnv.TEST])
def test_paper_permitted_through_same_function(env: DynamoEnv) -> None:
    # Conditions 2-4 don't apply outside LIVE; no approval, unready — still OK.
    result = _gate(env, None, False, UNREADY)
    assert result.permitted
    assert result.reason == GateReason.PAPER_PERMITTED


@pytest.mark.parametrize("env", list(DynamoEnv))
def test_kill_switch_halts_every_environment(env: DynamoEnv) -> None:
    result = _gate(env, VALID, True, READY)
    assert not result.permitted
    assert result.reason == GateReason.KILL_SWITCH_ENGAGED


def test_out_of_scope_symbol_refused() -> None:
    scoped = _approval(scope=("AAPL",))
    ok = live_orders_permitted(
        DynamoEnv.LIVE,
        scoped,
        False,
        READY,
        approval_key=KEY,
        today=TODAY,
        scope_target="AAPL",
    )
    assert ok.permitted
    refused = live_orders_permitted(
        DynamoEnv.LIVE,
        scoped,
        False,
        READY,
        approval_key=KEY,
        today=TODAY,
        scope_target="TSLA",
    )
    assert not refused.permitted
    assert refused.reason == GateReason.APPROVAL_OUT_OF_SCOPE


def test_load_approval_roundtrip(tmp_path: Path) -> None:
    body = {"scope": ["all"], "expiry": "2026-08-31", "date": "2026-08-03"}
    signature = sign_approval_body(body, KEY)
    path = tmp_path / "live-2026-08-03.yaml"
    path.write_text(
        f"scope: [all]\nexpiry: 2026-08-31\ndate: 2026-08-03\nsignature: {signature}\n",
        encoding="utf-8",
    )
    approval = load_approval(path)
    assert approval.scope == ("all",)
    assert approval.expiry == date(2026, 8, 31)
    result = live_orders_permitted(
        DynamoEnv.LIVE, approval, False, READY, approval_key=KEY, today=TODAY
    )
    assert result.permitted


def test_tampered_approval_body_fails_signature(tmp_path: Path) -> None:
    body = {"scope": ["AAPL"], "expiry": "2026-08-31"}
    signature = sign_approval_body(body, KEY)
    path = tmp_path / "live.yaml"
    path.write_text(  # scope widened after signing
        f"scope: [all]\nexpiry: 2026-08-31\nsignature: {signature}\n", encoding="utf-8"
    )
    approval = load_approval(path)
    result = live_orders_permitted(
        DynamoEnv.LIVE, approval, False, READY, approval_key=KEY, today=TODAY
    )
    assert not result.permitted
    assert result.reason == GateReason.APPROVAL_BAD_SIGNATURE


def test_kill_switch_engage_cancels_open_orders_and_alerts(tmp_path: Path) -> None:
    path = tmp_path / "KILL_SWITCH"
    broker = MockBroker(script=[ScriptStep(behavior="accept")])
    broker.submit(
        Order(
            client_order_id="c1",
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=5,
            order_type=OrderType.MARKETABLE_LIMIT,
            limit_price=100.0,
            state=OrderState.PENDING_APPROVAL,
        )
    )
    alerts: list[tuple[str, str]] = []
    assert not kill_switch_engaged(path)
    cancelled = engage_kill_switch(
        path,
        broker,
        lambda name, msg: alerts.append((name, msg)),
        engaged_by="test",
        now=datetime(2026, 8, 3, 15, 0, tzinfo=UTC),
    )
    assert kill_switch_engaged(path)
    assert cancelled == ["c1"]
    assert broker.open_orders() == []
    assert alerts and alerts[0][0] == "kill_switch_engaged"

    with pytest.raises(ValueError, match="acknowledgment"):
        disengage_kill_switch(path, "  ")
    assert kill_switch_engaged(path)
    disengage_kill_switch(path, "reviewed and resolved")
    assert not kill_switch_engaged(path)
