"""Learning-loop & registry domain entities. Design: docs/design/08, 10."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import Field

from dynamo.domain.models.market import Frozen
from dynamo.domain.models.research import Evidence


class FailureClass(StrEnum):
    DATA_QUALITY = "data_quality"
    MISSING_INFORMATION = "missing_information"
    CLAUDE_REASONING = "claude_reasoning"
    QUANT_MODEL = "quant_model"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    POSITION_SIZING = "position_sizing"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    RISK_MANAGEMENT = "risk_management"
    STRATEGY_FAILURE = "strategy_failure"
    REGIME_CHANGE = "regime_change"
    NORMAL_UNCERTAINTY = "normal_uncertainty"  # a good decision with a bad outcome


class Assessment(StrEnum):
    TOO_SMALL = "too_small"
    RIGHT = "right"
    TOO_BIG = "too_big"
    EARLY = "early"
    ON_TIME = "on_time"
    LATE = "late"


class CalibrationCheck(Frozen):
    conviction_was_justified: bool
    comment: str


class PostTradeReview(Frozen):
    """Opus review of a closed trade against its frozen DecisionContext."""

    trade_id: str
    decision_context_id: str
    symbol: str
    as_of: date
    prompt_version: str
    model_id: str
    outcome_pnl_pct: float
    holding_days: int
    what_worked: tuple[str, ...] = ()
    what_failed: tuple[str, ...] = ()
    confidence_calibration: CalibrationCheck
    sizing_quality: Assessment
    timing_quality: Assessment
    missed_information: tuple[Evidence, ...] = ()  # knowable-then only
    thesis_remained_valid: bool
    exit_honored_exit_conditions: bool
    should_have_exited: str  # earlier | as_done | later
    failure_primary: FailureClass | None = None  # required iff pnl < 0
    failure_secondary: FailureClass | None = None
    lessons: tuple[str, ...] = ()


class JournalKind(StrEnum):
    THESIS = "thesis"
    REVIEW = "review"
    LESSON = "lesson"
    INCIDENT = "incident"
    UNIVERSE_CHANGE = "universe_change"
    META = "meta"


class JournalEntry(Frozen):
    id: str
    kind: JournalKind
    symbols: tuple[str, ...] = ()
    strategy: str | None = None
    regime_at_write: str | None = None
    sectors: tuple[str, ...] = ()
    text: str
    refs: dict[str, str] = {}
    created_at: datetime


class ProposalKind(StrEnum):
    RETRAIN = "retrain"
    NEW_FEATURE = "new_feature"
    NEW_DATA_SOURCE = "new_data_source"
    THRESHOLD_CHANGE = "threshold_change"
    PROMPT_REVISION = "prompt_revision"
    STRATEGY_RETIREMENT = "strategy_retirement"


class ProposalStatus(StrEnum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ImprovementProposal(Frozen):
    """Never applied directly — accepted proposals enter the promotion pipeline."""

    id: str
    created_at: datetime
    kind: ProposalKind
    target: str
    evidence: tuple[Evidence, ...] = ()
    expected_effect: str = ""
    proposed_change: dict[str, object] = {}
    status: ProposalStatus = ProposalStatus.OPEN


class RegistryStatus(StrEnum):
    """Model/strategy lifecycle (doc 08 section 7). Only the promotion tool writes."""

    CANDIDATE = "candidate"
    VALIDATING = "validating"
    PAPER = "paper"
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"
    REJECTED = "rejected"


PROMOTION_TRANSITIONS: dict[RegistryStatus, frozenset[RegistryStatus]] = {
    RegistryStatus.CANDIDATE: frozenset({RegistryStatus.VALIDATING, RegistryStatus.REJECTED}),
    RegistryStatus.VALIDATING: frozenset({RegistryStatus.PAPER, RegistryStatus.REJECTED}),
    RegistryStatus.PAPER: frozenset({RegistryStatus.SHADOW, RegistryStatus.REJECTED}),
    RegistryStatus.SHADOW: frozenset({RegistryStatus.PRODUCTION, RegistryStatus.REJECTED}),
    RegistryStatus.PRODUCTION: frozenset({RegistryStatus.RETIRED}),
    RegistryStatus.RETIRED: frozenset(),
    RegistryStatus.REJECTED: frozenset(),
}


class ModelVersion(Frozen):
    id: str  # e.g. ranker-lgbm-2026-08-01-a
    model_type: str  # ranker | downside | vol | strategy | config
    algo: str
    features_hash: str = ""
    train_start: date | None = None
    train_end: date | None = None
    hyperparams: dict[str, object] = {}
    metrics: dict[str, object] = {}
    artifact_path: str = ""
    status: RegistryStatus = RegistryStatus.CANDIDATE
    created_at: datetime | None = None
    promoted_at: datetime | None = None
    promoted_by: str | None = None


class PromotionRecord(Frozen):
    id: str
    target_id: str  # ModelVersion.id or strategy id
    from_status: RegistryStatus
    to_status: RegistryStatus
    actor: str
    evidence_bundle_ref: str | None = None
    trial_count: int = 0
    git_commit: str = ""
    config_hash: str = ""
    at: datetime


class CalibrationPoint(Frozen):
    scope_type: str  # agent | committee | strategy | prediction_market
    scope_id: str
    as_of: date
    brier: float = Field(ge=0.0, le=1.0)
    n: int
    bucket_stats: dict[str, float] = {}
