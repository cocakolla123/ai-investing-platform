"""Research domain entities: committee, quant, strategies, watchlist.

Design: docs/design/03-claude-committee.md, 04-quant-models.md, 05-strategies-and-scoring.md.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dynamo.domain.models.market import Frozen


class SourceType(StrEnum):
    FILING = "filing"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    PRICE = "price"
    MACRO = "macro"
    PREDICTION_MARKET = "prediction_market"
    JOURNAL = "journal"


class Evidence(Frozen):
    claim: str
    source_type: SourceType
    source_ref: str  # accession no. / news id / metric key / series id — resolvable
    as_of: date


class Stance(StrEnum):
    STRONG_SELL = "strong_sell"
    SELL = "sell"
    HOLD = "hold"
    BUY = "buy"
    STRONG_BUY = "strong_buy"


class Action(StrEnum):
    BUY = "buy"
    ADD = "add"
    HOLD = "hold"
    TRIM = "trim"
    SELL = "sell"
    AVOID = "avoid"
    NO_ACTION = "no_action"


STANCE_SCORE: dict[Stance, float] = {
    Stance.STRONG_SELL: -2.0,
    Stance.SELL: -1.0,
    Stance.HOLD: 0.0,
    Stance.BUY: 1.0,
    Stance.STRONG_BUY: 2.0,
}

ACTION_SCORE: dict[Action, float] = {
    Action.SELL: -2.0,
    Action.TRIM: -1.0,
    Action.AVOID: -0.5,
    Action.NO_ACTION: 0.0,
    Action.HOLD: 0.0,
    Action.ADD: 1.0,
    Action.BUY: 2.0,
}


class FundamentalsRead(Frozen):
    growth: str
    margins: str
    cash_flow: str
    balance_sheet: str
    evidence: tuple[Evidence, ...] = ()


class TechnicalRead(Frozen):
    trend: str
    relative_strength: str
    volatility_state: str
    key_levels: str
    evidence: tuple[Evidence, ...] = ()


class DossierAnalysis(Frozen):
    """Call 1 — stance-free fact base (doc 03 section 2)."""

    symbol: str
    as_of: date
    prompt_version: str
    business_summary: str
    fundamentals: FundamentalsRead
    filing_deltas: tuple[Evidence, ...] = ()
    technical_state: TechnicalRead
    news_catalysts: tuple[Evidence, ...] = ()
    data_gaps: tuple[str, ...] = ()


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class KillCase(Frozen):
    description: str
    severity: Severity
    trigger: str  # what would make it real


class Side(StrEnum):
    BULL = "bull"
    BEAR = "bear"


class AdvocateBrief(Frozen):
    """Calls 2 & 3 — independent adversarial advocates."""

    symbol: str
    side: Side
    thesis: str
    key_points: tuple[Evidence, ...]
    kill_cases: tuple[KillCase, ...] = ()
    expected_move_pct: float
    horizon_days: int
    conviction: float = Field(ge=0.0, le=1.0)


class Rebuttal(Frozen):
    kill_case: str  # description of the bear KillCase being rebutted
    rebuttal: str
    evidence: tuple[Evidence, ...] = ()


class ExitCondition(Frozen):
    description: str
    checkable: bool = False
    predicate: str | None = None  # machine-checkable form where possible


class Dissent(Frozen):
    agent: str
    stance: Stance
    reason: str


class QuantAlignment(StrEnum):
    AGREE = "agree"
    NEUTRAL = "neutral"
    CONFLICT = "conflict"


class CommitteeVerdict(Frozen):
    """Call 4 (Opus arbiter) output. Guardrails enforced in code, not prompts."""

    id: str
    symbol: str
    as_of: date
    action: Action
    conviction: float = Field(ge=0.0, le=1.0)
    thesis: str
    bull_case: str
    bear_case: str
    bear_rebuttals: tuple[Rebuttal, ...] = ()  # required non-empty for buy/add
    risks: tuple[str, ...] = ()
    exit_conditions: tuple[ExitCondition, ...] = ()
    expected_upside_pct: float = 0.0
    expected_downside_pct: float = 0.0
    horizon_days: int = 20
    position_size_hint_pct: float = 0.0  # advisory only — risk engine decides
    disagreement_score: float = Field(ge=0.0, le=1.0, default=0.0)
    dissents: tuple[Dissent, ...] = ()  # never deleted, always rendered
    self_critique: str = ""
    quant_alignment: QuantAlignment = QuantAlignment.NEUTRAL
    journal_context_used: tuple[str, ...] = ()
    prompt_version: str = ""
    model_id: str = ""


class QuantScore(Frozen):
    """Nightly quant model output (doc 04 section 8)."""

    symbol: str
    as_of: date
    model_version: str
    expected_excess_return: float | None = None  # 20d, vs sector ETF
    downside_prob: float | None = None  # calibrated P(< -10% within 20d)
    vol_forecast: float | None = None  # annualized 20d
    rank: int | None = None  # 1 = best, cross-sectional
    confidence: float = 0.0
    feature_coverage: float = 1.0


class QualityScore(Frozen):
    """Long-term quality composite 0-100 (doc 05 section 2)."""

    symbol: str
    as_of: date
    composite: float | None = None  # None = ineligible (>2 null sub-scores)
    subscores: dict[str, float | None] = {}
    config_hash: str = ""


class SignalStatus(StrEnum):
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class Signal(Frozen):
    """Tactical strategy signal (doc 05 section 3.3). Input to candidates, never orders."""

    strategy_id: str
    symbol: str
    as_of: date
    direction: str = "long"  # long-only in scope
    entry_ref_price: float | None = None
    stop_hint: float | None = None
    horizon_days: int = 20
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    rule_snapshot: dict[str, object] = {}
    status: SignalStatus = SignalStatus.ACTIVE


class Sleeve(StrEnum):
    LONG_TERM = "long_term"
    TACTICAL = "tactical"
    # Crypto sleeve (ADR-0013): Alpaca crypto pairs ("BTC/USD"), capped by
    # settings.crypto_sleeve_cap / crypto_max_position in the sizing cascade.
    CRYPTO = "crypto"
    # Idle-cash ballast: residual weight above the cash floor parked in a
    # diversified index ETF (SPY by default) instead of dollars. A cash
    # substitute, not an active bet — the constructor computes it AFTER the
    # active sleeves, so it never consumes their budgets. A distinct sleeve
    # tag lets orders / trades / attribution label it as its own line.
    BALLAST = "ballast"


class WatchlistRow(Frozen):
    as_of: date
    sleeve: Sleeve
    symbol: str
    rank: int
    blend_score: float
    committee_component: float | None = None
    quant_component: float | None = None
    quality_component: float | None = None
    badges: tuple[str, ...] = ()  # conflict | no-quant | stale-verdict | promoted
    blend_config_hash: str = ""
    verdict_ref: str | None = None
    signal_ref: int | None = None


class PredictionMarketView(Frozen):
    """Research-only prediction market analysis (doc 03 section 6)."""

    venue: str
    market_id: str
    question: str
    as_of: date
    market_implied_prob: float
    estimated_prob: float
    edge: float  # estimated - implied; research artifact, no execution hooks
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: tuple[str, ...]  # required non-empty (validated at call site)
    evidence: tuple[Evidence, ...] = ()
    historical_comparisons: tuple[str, ...] = ()
    why_wrong: tuple[str, ...] = ()


class MacroDigest(Frozen):
    """Once-daily whole-market context injected into every dossier."""

    as_of: date
    narrative: str
    rates_outlook: str
    inflation_outlook: str
    regime_comment: str
    prediction_market_context: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    prompt_version: str = ""


class BlendConfig(BaseModel):
    """Deterministic watchlist blend weights (doc 05 section 4.1)."""

    model_config = ConfigDict(frozen=True)

    w_committee: float = 0.40
    w_quant: float = 0.35
    w_quality: float = 0.25
    verdict_decay_per_week: float = 0.9
    conflict_rank_cap_pctile: float = 0.40
    downside_penalty: float = 0.5
