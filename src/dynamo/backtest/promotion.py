"""The promotion pipeline: stage gates as data (doc 08 section 7).

Only this module (via the CLI in ``dynamo/cli/promotion.py``) writes registry
status transitions, and it does so exclusively through a caller-supplied
repository Protocol — the core stays pure. Transition legality and the
shadow -> production human-approval requirement are enforced by the pure
``dynamo.learning.promotion`` logic, reused here.

Hard refusals: dirty git tree (unreproducible), non-event-driven evidence
(the research engine may reject ideas, never approve them), illegal
transitions, missing approval artifacts. There is no ``--force``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from dynamo.domain.models.learning import PromotionRecord, RegistryStatus
from dynamo.learning.promotion import PromotionError, promote

Op = Literal[">", ">=", "<", "<=", "==", "is_true", "is_false"]


@dataclass(frozen=True)
class Criterion:
    """One pass criterion evaluated against the evidence mapping."""

    key: str
    op: Op
    threshold: float | None
    description: str

    def evaluate(self, evidence: Mapping[str, Any]) -> tuple[bool, str]:
        value = evidence.get(self.key)
        if value is None:
            return False, f"{self.key}: missing from evidence ({self.description})"
        try:
            if self.op == "is_true":
                ok = bool(value) is True
            elif self.op == "is_false":
                ok = bool(value) is False
            else:
                v = float(value)  # type: ignore[arg-type]
                t = float(self.threshold if self.threshold is not None else 0.0)
                ok = {
                    ">": v > t,
                    ">=": v >= t,
                    "<": v < t,
                    "<=": v <= t,
                    "==": v == t,
                }[self.op]
        except (TypeError, ValueError):
            return False, f"{self.key}: unusable value {value!r}"
        detail = f"{self.key}={value!r} {self.op} {self.threshold!r}"
        return ok, detail if ok else f"FAILED {detail} ({self.description})"


@dataclass(frozen=True)
class StageGate:
    stage: int
    name: str
    criteria: tuple[Criterion, ...]
    # Registry transition applied when the stage passes (None: evidence only).
    transition: tuple[RegistryStatus, RegistryStatus] | None
    requires_approval: bool = False
    counts_trial: bool = False


# Stage gates as data (doc 08 section 7 table; per-artifact overrides live in
# the registry, these are the defaults).
PROMOTION_STAGES: tuple[StageGate, ...] = (
    StageGate(
        stage=1,
        name="backtest",
        criteria=(
            Criterion("beats_spy_risk_adjusted", "is_true", None, "beats SPY risk-adjusted"),
            Criterion("beats_naive_baselines", "is_true", None, "beats naive baselines"),
            Criterion("deflated_sharpe", ">", 0.0, "deflated Sharpe > 0"),
            Criterion("max_drawdown", ">=", -0.15, "max DD within bucket ladder tolerance"),
            Criterion("look_ahead_audit_passed", "is_true", None, "look-ahead audit passed"),
            Criterion("constraint_violations", "==", 0.0, "zero constraint violations"),
        ),
        transition=(RegistryStatus.CANDIDATE, RegistryStatus.VALIDATING),
        counts_trial=True,
    ),
    StageGate(
        stage=2,
        name="out_of_sample",
        criteria=(
            Criterion("oos_sharpe_fraction_of_is", ">=", 0.5, "OOS Sharpe >= 50% of IS"),
            Criterion("oos_alpha_same_sign", "is_true", None, "same sign of alpha"),
        ),
        transition=None,
    ),
    StageGate(
        stage=3,
        name="walk_forward",
        criteria=(
            Criterion("n_folds", ">=", 6.0, ">= 6 folds"),
            Criterion("median_fold_sharpe", ">", 0.0, "positive median fold Sharpe"),
            Criterion(
                "max_single_fold_return_share", "<=", 0.5,
                "no fold contributes > 50% of total return",
            ),
        ),
        transition=None,
    ),
    StageGate(
        stage=4,
        name="stress",
        criteria=(
            Criterion("ladder_dd15_breaches", "==", 0.0, "no ladder -15% breach"),
            Criterion("behavior_explained", "is_true", None, "behavior explainable at review"),
        ),
        transition=(RegistryStatus.VALIDATING, RegistryStatus.PAPER),
    ),
    StageGate(
        stage=5,
        name="paper",
        criteria=(
            Criterion("paper_trading_days", ">=", 30.0, ">= 30 trading days paper"),
            Criterion("tracking_error_within_band", "is_true", None, "tracking error in band"),
            Criterion("slippage_within_model", "is_true", None, "slippage within cost model"),
            Criterion("risk_constraint_violations", "==", 0.0, "zero risk violations"),
        ),
        transition=(RegistryStatus.PAPER, RegistryStatus.SHADOW),
    ),
    StageGate(
        stage=6,
        name="shadow",
        criteria=(
            Criterion("shadow_trading_days", ">=", 20.0, ">= 20 trading days shadow"),
            Criterion("divergence_report_reviewed", "is_true", None, "divergence reviewed"),
        ),
        transition=None,
    ),
    StageGate(
        stage=7,
        name="human_approval",
        criteria=(),  # the approval artifact IS the criterion
        transition=(RegistryStatus.SHADOW, RegistryStatus.PRODUCTION),
        requires_approval=True,
    ),
)

_STAGES_BY_NUMBER = {gate.stage: gate for gate in PROMOTION_STAGES}


class RegistryRepository(Protocol):
    """The only write path for registry status; supplied by the caller/CLI."""

    def current_status(self, target_id: str) -> RegistryStatus: ...

    def apply(self, record: PromotionRecord) -> None: ...

    def trial_count(self, family: str) -> int: ...

    def increment_trial(self, family: str) -> int: ...


@dataclass(frozen=True)
class StageResult:
    target_id: str
    stage: int
    stage_name: str
    passed: bool
    details: tuple[str, ...]
    trial_count: int
    record: PromotionRecord | None  # written transition, if any


def run_stage(
    target_id: str,
    stage: int,
    evidence: Mapping[str, Any],
    *,
    repo: RegistryRepository,
    actor: str,
    at: datetime,
    clean_tree: bool,
    strategy_family: str | None = None,
    approval: Mapping[str, Any] | None = None,
    approval_valid: bool = False,
    evidence_bundle_ref: str | None = None,
    git_commit: str = "",
    config_hash: str = "",
) -> StageResult:
    """Evaluate one stage gate; on pass, write its registry transition (if any).

    Raises :class:`PromotionError` for refusals (dirty tree, wrong engine,
    illegal transition, missing approval) — refusal is not the same as a
    failed gate, which returns ``passed=False``.
    """
    gate = _STAGES_BY_NUMBER.get(stage)
    if gate is None:
        raise PromotionError(f"unknown promotion stage {stage}")
    if not clean_tree:
        raise PromotionError(
            "dirty git tree: promotion-grade runs are refused (doc 08 section 8)"
        )
    engine = str(evidence.get("engine", ""))
    if gate.stage <= 4 and engine != "event_driven":
        raise PromotionError(
            f"stage {gate.stage} evidence must come from the event-driven validator, "
            f"got engine={engine!r} (vectorbt may reject ideas, never approve them)"
        )

    family = strategy_family or target_id
    trials = (
        repo.increment_trial(family) if gate.counts_trial else repo.trial_count(family)
    )

    details: list[str] = []
    passed = True
    for criterion in gate.criteria:
        ok, detail = criterion.evaluate(evidence)
        details.append(detail)
        passed = passed and ok

    record: PromotionRecord | None = None
    if passed and gate.transition is not None:
        from_status, to_status = gate.transition
        current = repo.current_status(target_id)
        if current != from_status:
            raise PromotionError(
                f"stage {gate.stage} transitions {from_status.value} -> "
                f"{to_status.value}, but {target_id} is {current.value}"
            )
        record = promote(
            target_id,
            from_status,
            to_status,
            actor,
            at,
            approval=approval,
            approval_valid=approval_valid,
            evidence_bundle_ref=evidence_bundle_ref,
            clean_tree=clean_tree,
            trial_count=trials,
            git_commit=git_commit,
            config_hash=config_hash,
        )
        repo.apply(record)

    return StageResult(
        target_id=target_id,
        stage=gate.stage,
        stage_name=gate.name,
        passed=passed,
        details=tuple(details),
        trial_count=trials,
        record=record,
    )


def reject(
    target_id: str,
    *,
    repo: RegistryRepository,
    actor: str,
    at: datetime,
    reason: str,
) -> PromotionRecord:
    """Fail a target back to rejected from any pre-production stage."""
    current = repo.current_status(target_id)
    record = promote(
        target_id,
        current,
        RegistryStatus.REJECTED,
        actor,
        at,
        evidence_bundle_ref=None,
        record_id=f"reject:{target_id}:{at.date().isoformat()}:{reason[:40]}",
    )
    repo.apply(record)
    return record
