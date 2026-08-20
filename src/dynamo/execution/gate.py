"""The live gate + global kill-switch (doc 09 §7).

One pure function, consulted on every submission — not cached, not at startup
only. In PAPER (and DEV/TEST) the gate short-circuits to permitted through the
*same* function, so the code path stays identical. The kill-switch is checked
for every environment: there is exactly one halt pathway to test.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from dynamo.config.settings import DynamoEnv
from dynamo.domain.ports import Broker

APPROVAL_KEY_ENV = "DYNAMO_APPROVAL_KEY"
MIN_CLEAN_PAPER_MONTHS = 3.0


class GateReason(StrEnum):
    PAPER_PERMITTED = "paper_permitted"
    LIVE_PERMITTED = "live_permitted"
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_BAD_SIGNATURE = "approval_bad_signature"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_OUT_OF_SCOPE = "approval_out_of_scope"
    READINESS_UNMET = "readiness_unmet"


@dataclass(frozen=True)
class GateResult:
    permitted: bool
    reason: GateReason
    detail: str = ""


@dataclass(frozen=True)
class Readiness:
    """Live-readiness criteria of doc 00 §5, evaluated from stored metrics."""

    clean_paper_months: float = 0.0
    calibration_ok: bool = False
    risk_suite_green: bool = False
    backup_drill_ok: bool = False

    def unmet(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.clean_paper_months < MIN_CLEAN_PAPER_MONTHS:
            out.append(f"clean_paper_months {self.clean_paper_months} < {MIN_CLEAN_PAPER_MONTHS}")
        if not self.calibration_ok:
            out.append("calibration_ok is false")
        if not self.risk_suite_green:
            out.append("risk_suite_green is false")
        if not self.backup_drill_ok:
            out.append("backup_drill_ok is false")
        return tuple(out)

    @property
    def satisfied(self) -> bool:
        return not self.unmet()


@dataclass(frozen=True)
class ApprovalArtifact:
    """Parsed ``approvals/live-<date>.yaml``: scope, expiry, HMAC signature."""

    scope: tuple[str, ...]  # symbols / sleeves, or ("all",)
    expiry: date
    signature: str
    body: Mapping[str, Any] = field(default_factory=dict)  # canonical signed body

    def covers(self, symbol_or_sleeve: str | None) -> bool:
        if "all" in self.scope:
            return True
        return symbol_or_sleeve is not None and symbol_or_sleeve in self.scope


def canonical_approval_body(body: Mapping[str, Any]) -> bytes:
    """Deterministic byte serialization of the approval body (signature input)."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_approval_body(body: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_approval_body(body), hashlib.sha256).hexdigest()


def approval_signature_valid(approval: ApprovalArtifact, key: bytes) -> bool:
    expected = sign_approval_body(approval.body, key)
    return hmac.compare_digest(expected, approval.signature)


def approval_key_from_env() -> bytes | None:
    raw = os.environ.get(APPROVAL_KEY_ENV, "")
    return raw.encode("utf-8") if raw else None


def load_approval(path: str | Path) -> ApprovalArtifact:
    """Parse ``approvals/live-<date>.yaml``. The signed body is every key
    except ``signature`` itself."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"approval file {path} is not a mapping")
    signature = str(data.get("signature", ""))
    body = {k: v for k, v in data.items() if k != "signature"}
    raw_scope = body.get("scope", [])
    scope = (raw_scope,) if isinstance(raw_scope, str) else tuple(str(s) for s in raw_scope)
    raw_expiry = body.get("expiry")
    if isinstance(raw_expiry, datetime):
        expiry = raw_expiry.date()
    elif isinstance(raw_expiry, date):
        expiry = raw_expiry
    else:
        expiry = date.fromisoformat(str(raw_expiry))
    return ApprovalArtifact(scope=scope, expiry=expiry, signature=signature, body=body)


MISSING_KEY_HELP = (
    f"${APPROVAL_KEY_ENV} is not set, so no approval artifact can be verified.\n"
    "  1. generate a key:   python -m dynamo.cli.approve keygen\n"
    f"  2. store it OUTSIDE the repo and export it:   export {APPROVAL_KEY_ENV}=<key>\n"
    "  3. sign the artifact: python -m dynamo.cli.approve sign --kind promotion \\\n"
    "         --target <id> --expiry YYYY-MM-DD --approver <you>\n"
    "See docs/runbooks/kill-switch.md and docs/design/12-operations.md."
)


def check_approval(
    approval: ApprovalArtifact | None,
    *,
    key: bytes | None,
    today: date,
    scope_target: str | None = None,
) -> GateResult:
    """Presence + HMAC + expiry + scope check for a signed approval artifact.

    This is the ONE verification routine. ``live_orders_permitted`` calls it,
    and so does the promotion CLI (doc 08 §7 stage 7) — a promotion approval
    and a live-trading approval are the same artifact format, the same
    ``hmac.compare_digest`` check and the same refusal reasons.
    """
    if approval is None:
        return GateResult(False, GateReason.APPROVAL_MISSING, "no approval artifact")
    if key is None:
        return GateResult(False, GateReason.APPROVAL_BAD_SIGNATURE, MISSING_KEY_HELP)
    if not approval.signature:
        return GateResult(
            False, GateReason.APPROVAL_BAD_SIGNATURE, "artifact carries no signature field"
        )
    if not approval_signature_valid(approval, key):
        return GateResult(
            False,
            GateReason.APPROVAL_BAD_SIGNATURE,
            f"HMAC mismatch: the artifact was not signed by the current ${APPROVAL_KEY_ENV} "
            "(or its body was edited after signing)",
        )
    if approval.expiry < today:
        return GateResult(
            False, GateReason.APPROVAL_EXPIRED, f"expired {approval.expiry.isoformat()}"
        )
    if not approval.covers(scope_target):
        return GateResult(
            False, GateReason.APPROVAL_OUT_OF_SCOPE, f"{scope_target!r} not in approval scope"
        )
    return GateResult(True, GateReason.LIVE_PERMITTED, "approval artifact verified")


def live_orders_permitted(
    env: DynamoEnv,
    approval: ApprovalArtifact | None,
    kill_switch_present: bool,
    readiness: Readiness,
    *,
    approval_key: bytes | None = None,
    today: date | None = None,
    scope_target: str | None = None,
) -> GateResult:
    """Pure gate decision (doc 09 §7). Consulted on every submission.

    Refusals return a typed reason; the caller logs and alerts — the gate never
    fails silently. ``approval_key`` defaults to ``$DYNAMO_APPROVAL_KEY`` and
    ``today`` to the wall-clock date only when not supplied (pass both in tests).
    """
    # Condition 3 first: the kill-switch halts every environment — one pathway.
    if kill_switch_present:
        return GateResult(False, GateReason.KILL_SWITCH_ENGAGED, "kill-switch file present")

    # Same function for paper: conditions 2-4 don't apply outside LIVE.
    if env != DynamoEnv.LIVE:
        return GateResult(True, GateReason.PAPER_PERMITTED, f"env={env}")

    key = approval_key if approval_key is not None else approval_key_from_env()
    effective_today = today if today is not None else date.today()
    checked = check_approval(approval, key=key, today=effective_today, scope_target=scope_target)
    if not checked.permitted:
        return checked

    if not readiness.satisfied:
        return GateResult(False, GateReason.READINESS_UNMET, "; ".join(readiness.unmet()))

    return GateResult(True, GateReason.LIVE_PERMITTED, "all gate conditions satisfied")


# --- Kill-switch helpers (doc 09 §7) ---

AlertCallback = Callable[[str, str], None]  # (name, message)


def kill_switch_engaged(path: str | Path) -> bool:
    return Path(path).exists()


def engage_kill_switch(
    path: str | Path,
    broker: Broker,
    alert: AlertCallback,
    *,
    engaged_by: str,
    now: datetime,
) -> list[str]:
    """Engage the kill-switch: write the file, cancel all open orders, alert.

    Positions are *not* auto-liquidated — that is a human decision. Returns the
    client order ids whose cancels were attempted.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"engaged_by={engaged_by}\nengaged_at={now.isoformat()}\n", encoding="utf-8")

    cancelled: list[str] = []
    failures: list[str] = []
    for o in broker.open_orders():
        coid = str(o.get("client_order_id", ""))
        try:
            broker.cancel(coid)
            cancelled.append(coid)
        except Exception as exc:
            failures.append(f"{coid}: {exc}")

    msg = f"kill-switch engaged by {engaged_by} at {now.isoformat()}; cancelled={cancelled}"
    if failures:
        msg += f"; cancel_failures={failures}"
    alert("kill_switch_engaged", msg)
    return cancelled


def disengage_kill_switch(path: str | Path, ack: str) -> None:
    """Reset requires deleting the file *and* an explicit acknowledgment."""
    if not ack.strip():
        raise ValueError("kill-switch disengage requires a non-empty acknowledgment")
    Path(path).unlink(missing_ok=True)
