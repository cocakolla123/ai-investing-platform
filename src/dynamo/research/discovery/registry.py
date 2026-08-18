"""Hypothesis pre-registration — the anti-data-snooping discipline (ADR-0015).

A candidate feature may only be auditioned if its hypothesis — WHY it should
work and WHAT it should show — was written down in a *prior* CLI invocation.
:func:`ensure_preregistered` is the rule in code: ``registered_at`` must be
strictly before the audition's start instant, so registering and testing in
the same breath (peek at the data, then "predict" it) is a refusal, not a
warning.

Pure module: domain models + stdlib only. Persistence lives in
``dynamo.persistence.repositories.discovery``; wiring in ``dynamo.cli.alpha``.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import field_validator

from dynamo.domain.models.market import Frozen

__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "PreRegistrationError",
    "ensure_auditionable",
    "ensure_preregistered",
]

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class HypothesisStatus(StrEnum):
    REGISTERED = "registered"  # written down, not yet tested
    AUDITIONED = "auditioned"  # tested and passed the audition thresholds
    PROPOSED = "proposed"  # NEW_FEATURE ImprovementProposal emitted
    REJECTED = "rejected"  # tested and failed — stays on record (it was a trial)


class Hypothesis(Frozen):
    """One pre-registered candidate-signal hypothesis.

    ``rationale`` and ``prediction`` are the pre-commitment: why the signal
    should work and what sign/magnitude it should show, written BEFORE any
    test runs. ``trial_family`` keys the persistent trial counter — every
    audition of any hypothesis in the family counts against the family's
    multiple-testing budget, whatever its outcome.
    """

    id: str  # slug: lowercase [a-z0-9_-], 3..64 chars
    title: str
    rationale: str  # WHY it should work — written before testing
    prediction: str  # expected sign/magnitude (e.g. "positive IC, ~0.02")
    trial_family: str
    registered_at: datetime
    data_requirements: str = ""
    status: HypothesisStatus = HypothesisStatus.REGISTERED
    audit_results: dict[str, Any] | None = None

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError(
                f"hypothesis id {value!r} must be a slug: lowercase letters, "
                "digits, '-', '_'; 3-64 chars; starting with [a-z0-9]"
            )
        return value

    @field_validator("title", "rationale", "prediction", "trial_family")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty — an empty pre-registration is no pre-registration")
        return value


class PreRegistrationError(RuntimeError):
    """The hypothesis was not registered before the audition started."""


def ensure_preregistered(hypothesis: Hypothesis, audition_start: datetime) -> None:
    """Refuse unless ``registered_at`` is STRICTLY before ``audition_start``.

    ``audition_start`` is captured at CLI-invocation start, so a hypothesis
    registered in the same invocation (or later) can never be tested by it —
    pre-registration must be a separate, earlier act.
    """
    if hypothesis.registered_at >= audition_start:
        raise PreRegistrationError(
            f"hypothesis {hypothesis.id!r} was registered at "
            f"{hypothesis.registered_at.isoformat()}, not strictly before this "
            f"audition's start ({audition_start.isoformat()}) — register first, "
            "audition in a later invocation"
        )


def ensure_auditionable(hypothesis: Hypothesis, audition_start: datetime) -> None:
    """Pre-registration rule + lifecycle rule (no re-testing a settled one)."""
    if hypothesis.status not in (HypothesisStatus.REGISTERED, HypothesisStatus.AUDITIONED):
        raise PreRegistrationError(
            f"hypothesis {hypothesis.id!r} is {hypothesis.status.value} — "
            "settled hypotheses are not re-auditioned; register a new one"
        )
    ensure_preregistered(hypothesis, audition_start)
