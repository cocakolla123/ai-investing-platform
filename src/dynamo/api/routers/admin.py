"""Admin actions (doc 11 §1.2) — the complete list, nothing more.

Every action: requires ``X-Dynamo-Confirm: <action-name>``, is idempotent,
writes an ``admin_actions`` audit row, and returns the new state. No endpoint
mutates positions, orders, models, or config directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from dynamo.api.deps import Deps, DepsDep

router = APIRouter(tags=["admin"])

ADMIN_ACTOR = "owner"  # single-user platform (doc 11 §1.4)

CONFIRM_HEADER = "X-Dynamo-Confirm"


def _require_confirm(action: str, supplied: str | None) -> None:
    if supplied != action:
        raise HTTPException(
            400,
            f"admin action requires header '{CONFIRM_HEADER}: {action}'",
        )


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_universe_recommendation(message: str) -> tuple[str, str]:
    """Parse ``ADD SYM to core universe: ...`` / ``DEMOTE SYM from ...``."""
    parts = message.split()
    if len(parts) >= 2 and parts[0] in ("ADD", "DEMOTE"):
        return parts[0].lower(), parts[1].strip(":,")
    raise HTTPException(409, f"unparseable universe recommendation: {message!r}")


# --- Universe queue ---


@router.post("/universe/queue/{alert_id}/approve")
def approve_universe_change(
    alert_id: int,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("universe-approve", x_dynamo_confirm)
    alert = deps.admin.get_alert(alert_id)
    if alert is None or alert.get("name") != "universe_change_recommendation":
        raise HTTPException(404, f"no universe recommendation with alert id {alert_id}")
    if alert.get("acked_at") is not None:
        return {"alert_id": alert_id, "status": "already_processed"}
    change, symbol = _parse_universe_recommendation(str(alert.get("message", "")))
    now = _now()
    deps.admin.approve_universe_change(
        symbol=symbol,
        change=change,
        reason=str(alert.get("message", "")),
        approved_by=ADMIN_ACTOR,
        as_of=now.date(),
    )
    deps.admin.ack_alert(alert_id, now)
    deps.admin.audit("universe_approve", {"alert_id": alert_id, "symbol": symbol, "change": change})
    return {"alert_id": alert_id, "status": "approved", "symbol": symbol, "change": change}


@router.post("/universe/queue/{alert_id}/reject")
def reject_universe_change(
    alert_id: int,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("universe-reject", x_dynamo_confirm)
    alert = deps.admin.get_alert(alert_id)
    if alert is None or alert.get("name") != "universe_change_recommendation":
        raise HTTPException(404, f"no universe recommendation with alert id {alert_id}")
    if alert.get("acked_at") is not None:
        return {"alert_id": alert_id, "status": "already_processed"}
    deps.admin.ack_alert(alert_id, _now())
    deps.admin.audit("universe_reject", {"alert_id": alert_id})
    return {"alert_id": alert_id, "status": "rejected"}


# --- Rebalance proposals ---


@router.post("/rebalance/{target_portfolio_id}/approve")
def approve_rebalance(
    target_portfolio_id: str,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("rebalance-approve", x_dynamo_confirm)
    tp = deps.admin.get_target_portfolio(target_portfolio_id)
    if tp is None:
        raise HTTPException(404, f"no target portfolio {target_portfolio_id}")
    if tp.get("approved_at") is not None:
        return {"id": target_portfolio_id, "status": "already_approved"}
    if isinstance(tp.get("payload"), dict) and tp["payload"].get("rejected"):
        raise HTTPException(409, f"target portfolio {target_portfolio_id} was rejected")
    deps.admin.approve_target_portfolio(target_portfolio_id, ADMIN_ACTOR, _now())
    deps.admin.audit("rebalance_approve", {"target_portfolio_id": target_portfolio_id})
    return {"id": target_portfolio_id, "status": "approved", "approved_by": ADMIN_ACTOR}


@router.post("/rebalance/{target_portfolio_id}/reject")
def reject_rebalance(
    target_portfolio_id: str,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("rebalance-reject", x_dynamo_confirm)
    tp = deps.admin.get_target_portfolio(target_portfolio_id)
    if tp is None:
        raise HTTPException(404, f"no target portfolio {target_portfolio_id}")
    if isinstance(tp.get("payload"), dict) and tp["payload"].get("rejected"):
        return {"id": target_portfolio_id, "status": "already_rejected"}
    if tp.get("approved_at") is not None:
        raise HTTPException(409, f"target portfolio {target_portfolio_id} already approved")
    deps.admin.reject_target_portfolio(target_portfolio_id, f"rejected by {ADMIN_ACTOR}", _now())
    deps.admin.audit("rebalance_reject", {"target_portfolio_id": target_portfolio_id})
    return {"id": target_portfolio_id, "status": "rejected"}


# --- Improvement proposals ---


def _improvement_action(
    proposal_id: str, target_status: str, action: str, deps: Deps
) -> dict[str, Any]:
    proposal = deps.admin.get_improvement(proposal_id)
    if proposal is None:
        raise HTTPException(404, f"no improvement proposal {proposal_id}")
    current = str(proposal.get("status", ""))
    if current == target_status:
        return {"id": proposal_id, "status": current, "note": "already_processed"}
    if current != "open":
        raise HTTPException(409, f"proposal {proposal_id} is {current}, not open")
    deps.admin.set_improvement_status(proposal_id, target_status)
    deps.admin.audit(action, {"proposal_id": proposal_id})
    return {"id": proposal_id, "status": target_status}


@router.post("/improvements/{proposal_id}/approve")
def approve_improvement(
    proposal_id: str,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("improvement-approve", x_dynamo_confirm)
    return _improvement_action(proposal_id, "accepted", "improvement_approve", deps)


@router.post("/improvements/{proposal_id}/reject")
def reject_improvement(
    proposal_id: str,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("improvement-reject", x_dynamo_confirm)
    return _improvement_action(proposal_id, "rejected", "improvement_reject", deps)


# --- Alert ack ---


@router.post("/alerts/{alert_id}/ack")
def ack_alert(
    alert_id: int,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("alert-ack", x_dynamo_confirm)
    alert = deps.admin.get_alert(alert_id)
    if alert is None:
        raise HTTPException(404, f"no alert {alert_id}")
    if not alert.get("ackable", False):
        raise HTTPException(
            409, f"alert {alert_id} is not ackable; it clears with the underlying condition"
        )
    if alert.get("acked_at") is not None:
        return {"alert_id": alert_id, "status": "already_acked"}
    deps.admin.ack_alert(alert_id, _now())
    deps.admin.audit("alert_ack", {"alert_id": alert_id})
    return {"alert_id": alert_id, "status": "acked"}


# --- Sticky reconciliation halts (doc 09 §5) ---


class ReconciliationAckBody(BaseModel):
    note: str


@router.post("/reconciliation/{run_id}/ack")
def ack_reconciliation(
    run_id: int,
    body: ReconciliationAckBody,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    """Clear the sticky halt engaged by one divergent reconciliation run.

    A divergent run halts its sleeve until a human says otherwise; this is the
    only thing that says otherwise. Nothing re-checks and quietly resumes, and
    an empty note is refused — the ack row is the record of who resumed
    trading, when, and why.
    """
    _require_confirm("reconciliation-ack", x_dynamo_confirm)
    run = deps.admin.get_reconciliation_run(run_id)
    if run is None:
        raise HTTPException(404, f"no reconciliation run {run_id}")
    if str(run.get("result", "")) != "divergent":
        raise HTTPException(409, f"reconciliation run {run_id} is not divergent; nothing to clear")
    if deps.admin.reconciliation_acked(run_id):
        return {"run_id": run_id, "status": "already_acked"}
    if not body.note.strip():
        raise HTTPException(400, "reconciliation ack requires a non-empty note")
    deps.admin.ack_reconciliation(run_id, ADMIN_ACTOR, body.note, _now())
    deps.admin.audit("reconciliation_ack", {"run_id": run_id, "note": body.note})
    return {
        "run_id": run_id,
        "status": "acked",
        "acked_by": ADMIN_ACTOR,
        "scope": run.get("scope"),
    }


# --- Kill-switch ---


class KillSwitchBody(BaseModel):
    engage: bool
    note: str = ""


@router.post("/killswitch")
def toggle_killswitch(
    body: KillSwitchBody,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_confirm("killswitch", x_dynamo_confirm)
    currently = deps.kill.engaged()
    if body.engage and not currently:
        deps.kill.engage(body.note, ADMIN_ACTOR)
    elif not body.engage and currently:
        deps.kill.disengage(body.note or f"disengaged by {ADMIN_ACTOR}")
    deps.admin.audit("killswitch", {"engage": body.engage, "note": body.note})
    return {"engaged": deps.kill.engaged(), "note": body.note}


# --- Drawdown-ladder acknowledgments (doc 06 §3) ---


class LadderAckBody(BaseModel):
    note: str


@router.post("/ladder/{event_id}/ack")
def ack_ladder_event(
    event_id: int,
    body: LadderAckBody,
    deps: DepsDep,
    x_dynamo_confirm: str | None = Header(default=None),
) -> dict[str, Any]:
    """Clear the halt engaged by one drawdown-ladder transition.

    DD8 ("human ack") and DD15 ("human review") are documented mitigations in
    doc 06 §3. Until a human acks the rung, new orders are refused — this is
    the only thing that resumes them. The mandatory note is the record of who
    reviewed the drawdown, and what they concluded.

    Acking does NOT disengage the kill switch: a DD20 transition writes the
    switch file, and clearing that is a separate, deliberate act
    (``POST /admin/kill-switch/disengage``).
    """
    _require_confirm("ladder-ack", x_dynamo_confirm)
    event = deps.admin.get_ladder_event(event_id)
    if event is None:
        raise HTTPException(404, f"no ladder event {event_id}")
    if not event.get("requires_ack", False):
        raise HTTPException(
            409, f"ladder event {event_id} does not require acknowledgment; nothing to clear"
        )
    if event.get("acked_at") is not None:
        return {"event_id": event_id, "status": "already_acked"}
    if not body.note.strip():
        raise HTTPException(400, "ladder ack requires a non-empty note")
    if not deps.admin.ack_ladder_event(event_id, ADMIN_ACTOR, body.note, _now()):
        return {"event_id": event_id, "status": "already_acked"}
    deps.admin.audit("ladder_ack", {"event_id": event_id, "note": body.note})
    return {
        "event_id": event_id,
        "status": "acked",
        "acked_by": ADMIN_ACTOR,
        "rung": event.get("to_rung"),
        "kill_switch_engaged": event.get("kill_switch_engaged", False),
    }
