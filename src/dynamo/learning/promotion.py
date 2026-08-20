"""Pure promotion logic for the model/strategy registry (docs 08 §7, 10 §7).

Status changes flow only through ``PROMOTION_TRANSITIONS``; shadow → production
additionally requires a valid signed human-approval artifact, an evidence
bundle, and a clean git tree. The artifact is parsed and HMAC-verified by the
caller (same scheme as the live gate, ``approvals/promotion-<id>.yaml``) —
this module accepts the parsed dict plus a validity flag so it stays pure.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from dynamo.domain.models.learning import (
    PROMOTION_TRANSITIONS,
    PromotionRecord,
    RegistryStatus,
)


class PromotionError(ValueError):
    """Refused promotion: illegal transition or missing prerequisites."""


def validate_transition(from_status: RegistryStatus, to_status: RegistryStatus) -> None:
    allowed = PROMOTION_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise PromotionError(
            f"illegal promotion {from_status.value} -> {to_status.value}; "
            f"allowed: {sorted(s.value for s in allowed)}"
        )


def _require_production_prerequisites(
    target_id: str,
    approval: Mapping[str, Any] | None,
    approval_valid: bool,
    evidence_bundle_ref: str | None,
    clean_tree: bool,
) -> None:
    if approval is None:
        raise PromotionError(
            f"shadow -> production for {target_id} requires a signed approval artifact "
            f"(approvals/promotion-<id>.yaml)"
        )
    if not approval_valid:
        raise PromotionError(f"approval artifact for {target_id} failed signature/expiry checks")
    approved_target = str(approval.get("target_id", ""))
    if approved_target and approved_target != target_id:
        raise PromotionError(f"approval artifact targets {approved_target!r}, not {target_id!r}")
    if not evidence_bundle_ref:
        raise PromotionError(f"shadow -> production for {target_id} requires an evidence bundle")
    if not clean_tree:
        raise PromotionError(
            f"shadow -> production for {target_id} requires a clean git tree "
            f"(uncommitted changes would make the promotion unreproducible)"
        )


def promote(
    target_id: str,
    from_status: RegistryStatus,
    to_status: RegistryStatus,
    actor: str,
    at: datetime,
    *,
    approval: Mapping[str, Any] | None = None,
    approval_valid: bool = False,
    evidence_bundle_ref: str | None = None,
    clean_tree: bool = True,
    trial_count: int = 0,
    git_commit: str = "",
    config_hash: str = "",
    record_id: str | None = None,
) -> PromotionRecord:
    """Validate and describe one promotion as a ``PromotionRecord``.

    Pure: nothing is persisted here. The registry repository applies the record
    (its ONLY path for status changes), keeping "learning proposes, humans
    promote" enforceable in code.
    """
    validate_transition(from_status, to_status)
    if from_status == RegistryStatus.SHADOW and to_status == RegistryStatus.PRODUCTION:
        _require_production_prerequisites(
            target_id, approval, approval_valid, evidence_bundle_ref, clean_tree
        )
    if not actor.strip():
        raise PromotionError("promotion requires a named actor")
    return PromotionRecord(
        id=record_id
        or f"promo:{target_id}:{from_status.value}->{to_status.value}:{at.date().isoformat()}",
        target_id=target_id,
        from_status=from_status,
        to_status=to_status,
        actor=actor,
        evidence_bundle_ref=evidence_bundle_ref,
        trial_count=trial_count,
        git_commit=git_commit,
        config_hash=config_hash,
        at=at,
    )
