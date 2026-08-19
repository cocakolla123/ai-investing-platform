"""Hypothesis model, pre-registration rule, row round-trip, migration chain."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest

from dynamo.persistence.repositories.discovery import (
    hypothesis_from_row,
    hypothesis_to_row,
)
from dynamo.research.discovery.registry import (
    Hypothesis,
    HypothesisStatus,
    PreRegistrationError,
    ensure_auditionable,
    ensure_preregistered,
)
from tests.unit.discovery.conftest import NOW, make_hypothesis

MIGRATIONS = Path(__file__).resolve().parents[3] / "infra" / "migrations" / "versions"


# --- model validation ------------------------------------------------------


def test_hypothesis_slug_validation() -> None:
    with pytest.raises(ValueError):
        make_hypothesis("Bad Slug!")
    with pytest.raises(ValueError):
        make_hypothesis("UPPER")
    with pytest.raises(ValueError):
        make_hypothesis("ab")  # too short
    assert make_hypothesis("vol_skew-2").id == "vol_skew-2"


def test_hypothesis_requires_non_empty_precommitment() -> None:
    with pytest.raises(ValueError):
        Hypothesis(
            id="ok-slug",
            title="t",
            rationale="   ",  # empty pre-registration is no pre-registration
            prediction="p",
            trial_family="f",
            registered_at=NOW,
        )


# --- the pre-registration rule --------------------------------------------


def test_refuses_registration_at_or_after_audition_start() -> None:
    h = make_hypothesis(registered_at=NOW)
    with pytest.raises(PreRegistrationError):
        ensure_preregistered(h, NOW)  # same instant == same invocation
    with pytest.raises(PreRegistrationError):
        ensure_preregistered(h, NOW - timedelta(seconds=1))
    ensure_preregistered(h, NOW + timedelta(seconds=1))  # prior registration ok


def test_settled_hypotheses_are_not_auditionable() -> None:
    later = NOW + timedelta(hours=1)
    for status in (HypothesisStatus.PROPOSED, HypothesisStatus.REJECTED):
        with pytest.raises(PreRegistrationError):
            ensure_auditionable(make_hypothesis(status=status), later)
    ensure_auditionable(make_hypothesis(status=HypothesisStatus.REGISTERED), later)
    ensure_auditionable(make_hypothesis(status=HypothesisStatus.AUDITIONED), later)


# --- row round-trip --------------------------------------------------------


def test_hypothesis_row_round_trip() -> None:
    h = make_hypothesis(
        "round-trip",
        status=HypothesisStatus.AUDITIONED,
        audit_results={"verdict": {"passed": True}},
    )
    assert hypothesis_from_row(hypothesis_to_row(h)) == h


def test_round_trip_none_audit_results_and_empty_requirements() -> None:
    h = make_hypothesis("no-audit")
    row = hypothesis_to_row(h)
    assert row["audit_results"] is None
    assert hypothesis_from_row(row) == h


# --- migration chain -------------------------------------------------------


def _revisions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for path in MIGRATIONS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        rev = re.search(r'^revision = "([^"]+)"', text, re.M)
        down = re.search(r'^down_revision = (?:"([^"]+)"|None)', text, re.M)
        assert rev is not None, path
        out[rev.group(1)] = down.group(1) if down and down.group(1) else None
    return out


def test_migration_0019_chains_onto_0018_with_a_single_head() -> None:
    revs = _revisions()
    assert revs["0018"] == "0017"
    assert revs["0019"] == "0018"
    downs = {d for d in revs.values() if d is not None}
    heads = set(revs) - downs
    assert heads == {"0019"}


def test_schema_has_hypotheses_table_with_required_columns() -> None:
    from dynamo.persistence.schema import hypotheses

    cols = set(hypotheses.c.keys())
    assert {
        "id",
        "title",
        "rationale",
        "prediction",
        "trial_family",
        "registered_at",
        "data_requirements",
        "status",
        "audit_results",
    } <= cols
    assert hypotheses.c.id.primary_key
