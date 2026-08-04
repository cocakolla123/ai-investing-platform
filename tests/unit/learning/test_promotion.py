"""Promotion logic (docs 08 §7, 10): illegal transitions refused; shadow →
production demands approval artifact + evidence + clean tree."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dynamo.domain.models.learning import PROMOTION_TRANSITIONS, RegistryStatus
from dynamo.learning.promotion import PromotionError, promote, validate_transition

AT = datetime(2026, 8, 3, 23, 0, tzinfo=UTC)
ALL = list(RegistryStatus)

APPROVAL = {"target_id": "ranker-v2", "expiry": "2026-08-31", "scope": "promotion"}


def _promote_to_production(**overrides: object):
    kwargs: dict[str, object] = {
        "approval": APPROVAL,
        "approval_valid": True,
        "evidence_bundle_ref": "artifacts/evidence/ranker-v2.tar.gz",
        "clean_tree": True,
    }
    kwargs.update(overrides)
    return promote(
        "ranker-v2", RegistryStatus.SHADOW, RegistryStatus.PRODUCTION, "owner", AT,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("from_status", ALL)
@pytest.mark.parametrize("to_status", ALL)
def test_full_transition_matrix(from_status: RegistryStatus, to_status: RegistryStatus) -> None:
    if to_status in PROMOTION_TRANSITIONS[from_status]:
        validate_transition(from_status, to_status)
    else:
        with pytest.raises(PromotionError, match="illegal promotion"):
            validate_transition(from_status, to_status)


@pytest.mark.parametrize("terminal", [RegistryStatus.RETIRED, RegistryStatus.REJECTED])
def test_terminal_statuses_absorbing(terminal: RegistryStatus) -> None:
    for target in ALL:
        with pytest.raises(PromotionError):
            promote("m", terminal, target, "owner", AT)


def test_legal_non_production_promotion_needs_no_approval() -> None:
    record = promote(
        "ranker-v2", RegistryStatus.CANDIDATE, RegistryStatus.VALIDATING, "owner", AT
    )
    assert record.target_id == "ranker-v2"
    assert record.from_status == RegistryStatus.CANDIDATE
    assert record.to_status == RegistryStatus.VALIDATING


def test_shadow_to_production_happy_path() -> None:
    record = _promote_to_production()
    assert record.to_status == RegistryStatus.PRODUCTION
    assert record.actor == "owner"
    assert record.evidence_bundle_ref == "artifacts/evidence/ranker-v2.tar.gz"
    assert record.at == AT


def test_production_requires_approval_artifact() -> None:
    with pytest.raises(PromotionError, match="signed approval artifact"):
        _promote_to_production(approval=None)


def test_production_requires_valid_signature() -> None:
    with pytest.raises(PromotionError, match="signature"):
        _promote_to_production(approval_valid=False)


def test_production_approval_must_target_the_model() -> None:
    with pytest.raises(PromotionError, match="targets"):
        _promote_to_production(approval={"target_id": "some-other-model"})


def test_production_requires_evidence_bundle() -> None:
    with pytest.raises(PromotionError, match="evidence bundle"):
        _promote_to_production(evidence_bundle_ref=None)


def test_production_requires_clean_tree() -> None:
    with pytest.raises(PromotionError, match="clean git tree"):
        _promote_to_production(clean_tree=False)


def test_promotion_requires_named_actor() -> None:
    with pytest.raises(PromotionError, match="actor"):
        promote("m", RegistryStatus.CANDIDATE, RegistryStatus.VALIDATING, "  ", AT)


def test_skipping_stages_is_illegal() -> None:
    for target in (RegistryStatus.PAPER, RegistryStatus.SHADOW, RegistryStatus.PRODUCTION):
        with pytest.raises(PromotionError):
            promote("m", RegistryStatus.CANDIDATE, target, "owner", AT)
