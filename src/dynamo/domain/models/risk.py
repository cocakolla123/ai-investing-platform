"""Risk-engine domain entities. Design: docs/design/06-risk-engine.md.

The risk engine's output is a hard constraint set; everything else is advisory.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dynamo.domain.models.market import Frozen


class Regime(StrEnum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


class RegimeState(Frozen):
    as_of: date
    regime: Regime
    composite_score: float  # -2..+2 rule-based composite (doc 06 section 8)
    inputs: dict[str, float] = {}
    sessions_in_state: int = 0


class RiskBucket(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LadderRung(StrEnum):
    """Drawdown ladder states (doc 06 section 3). Sticky with hysteresis."""

    NORMAL = "normal"
    DD5 = "dd5"  # -5%: new-position risk halved
    DD8 = "dd8"  # -8%: no new tactical
    DD10 = "dd10"  # -10%: no new positions
    DD15 = "dd15"  # -15%: de-risk to defensive floor
    DD20 = "dd20"  # -20%: full stop, kill-switch engaged


class RiskConfig(BaseModel):
    """All thresholds; Medium preset defaults. Promotion-gated production logic."""

    model_config = ConfigDict(frozen=True)

    bucket: RiskBucket = RiskBucket.MEDIUM
    per_position_cap: float = 0.08
    per_position_hard: float = 0.15
    sector_cap: float = 0.25
    sector_hard: float = 0.35
    gross_cap: float = 1.00
    tactical_position_cap: float = 0.04
    cash_floor: float = 0.05
    cluster_cap: float = 0.30
    cluster_hard: float = 0.40
    var95_cap: float = 0.025
    var95_hard: float = 0.04
    target_vol_annual: float = 0.12
    vol_band: float = 0.25  # +/- fraction around target before scaling
    kelly_fraction_cap: float = 0.25
    min_position_weight: float = 0.005
    atr_stop_multiple: float = 2.5
    meanrev_stop_multiple: float = 1.5
    disaster_stop_pct: float = 0.25
    liquidity_adv_participation: float = 0.10
    liquidity_max_days: float = 3.0
    # ABSOLUTE liquidity floor, in consolidated-equivalent dollars per day.
    #
    # The participation cap above is purely RELATIVE — it asks "can I exit
    # this position in 3 days at 10% of the tape?", which a small account
    # can always answer yes to. At $1,000 of equity it permitted a full
    # position in a name trading a few hundred dollars a day, where the
    # quoted spread is a large fraction of any edge and "3 days to exit" is
    # fiction because there is no continuous book to participate in.
    # Relative caps scale with account size; market microstructure does
    # not, so this floor has to be absolute. $1M/day is roughly where US
    # equity spreads stop being measured in whole percent.
    #
    # NOT a substitute for a price floor or a spread limit; see
    # dynamo.risk.liquidity for what those would need.
    min_adv_dollar: float = 1_000_000.0
    correlation_cluster_threshold: float = 0.7
    conviction_tilt_min: float = 0.5
    conviction_tilt_max: float = 1.5
    ewma_vol_span_days: int = 60
    hysteresis_sessions: int = 5
    # Crypto sleeve caps (ADR-0013). Crypto realized vol runs 3-5x equities,
    # so vol-parity sizing already shrinks crypto positions naturally; these
    # caps are belt-and-suspenders hard bounds enforced in the sizing cascade
    # AND the constructor's verdict-compliance pass. Defaults mirror
    # settings.crypto_sleeve_cap / crypto_max_position; callers may override
    # from Settings (risk stays pure — caps are passed in, never read here).
    crypto_sleeve_cap: float = 0.10  # all crypto positions combined
    crypto_max_position: float = 0.05  # any single crypto pair


class StopKind(StrEnum):
    ATR = "atr"
    TRAILING = "trailing"
    THESIS = "thesis"
    DISASTER = "disaster"
    TIME = "time"


class StopOrderSpec(Frozen):
    symbol: str
    kind: StopKind
    stop_price: float
    reason: str = ""


class Veto(Frozen):
    symbol: str
    rule: str
    reason: str
    inputs: dict[str, float | str] = {}


class RiskMetrics(Frozen):
    as_of: datetime
    portfolio_vol_annual: float | None = None
    var95_1d: float | None = None
    var99_1d: float | None = None
    cvar95_1d: float | None = None
    net_beta: float | None = None
    gross_exposure: float = 0.0
    cash_fraction: float = 1.0
    drawdown_from_hwm: float = 0.0
    var_exceptions_250d: int = 0


class RiskVerdict(Frozen):
    """Hard constraint set. Produced independently of any proposal; fail-closed."""

    as_of: datetime
    regime: RegimeState
    ladder: LadderRung
    new_risk_allowed: bool
    gross_exposure_cap: float
    net_exposure_cap: float
    cash_floor: float
    per_position_cap: dict[str, float] = {}  # symbol -> max weight (after clamps)
    sector_caps: dict[str, float] = {}
    cluster_caps: dict[str, float] = {}  # cluster id -> max weight
    clusters: dict[str, str] = {}  # symbol -> cluster id
    stop_updates: tuple[StopOrderSpec, ...] = ()
    vetoes: tuple[Veto, ...] = ()
    metrics: RiskMetrics
    degraded: bool = False  # True when inputs were incomplete -> no new risk


class SizedPosition(Frozen):
    """Output of the sizing cascade (doc 06 section 4)."""

    symbol: str
    weight: float = Field(ge=0.0)
    base_weight: float = 0.0
    conviction_tilt: float = 1.0
    kelly_ceiling: float | None = None
    clamps_applied: tuple[str, ...] = ()


class StressResult(Frozen):
    scenario: str
    equity_impact_pct: float
    ladder_implication: LadderRung | None = None
    details: dict[str, float] = {}
