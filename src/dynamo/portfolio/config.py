"""Construction configuration: risk buckets as construction presets.

Design: docs/design/07-portfolio-construction.md section 3. Low / Medium /
High are parameter presets over one code path; bucket selection is owner
config, changed only via the promotion pipeline.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamo.domain.models.risk import LadderRung, Regime, RiskBucket
from dynamo.domain.symbols import is_crypto_symbol

# Bucket presets (doc 07 section 3 table).
_PRESETS: dict[RiskBucket, dict[str, float | int]] = {
    RiskBucket.LOW: {
        "long_term_fraction": 0.85,
        "tactical_fraction": 0.15,
        "tactical_max": 0.20,
        "max_positions": 15,
        "max_tactical_positions": 4,
        "turnover_budget_monthly": 0.08,
    },
    RiskBucket.MEDIUM: {
        "long_term_fraction": 0.70,
        "tactical_fraction": 0.30,
        "tactical_max": 0.35,
        "max_positions": 25,
        "max_tactical_positions": 8,
        "turnover_budget_monthly": 0.15,
    },
    RiskBucket.HIGH: {
        "long_term_fraction": 0.60,
        "tactical_fraction": 0.40,
        "tactical_max": 0.50,
        "max_positions": 30,
        "max_tactical_positions": 12,
        "turnover_budget_monthly": 0.25,
    },
}


class ConstructionConfig(BaseModel):
    """Frozen, versioned construction parameters (doc 07 sections 3, 5, 8)."""

    model_config = ConfigDict(frozen=True)

    bucket: RiskBucket = RiskBucket.MEDIUM
    # Sleeve split (long-term / tactical) and the tactical hard bound.
    long_term_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    tactical_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    tactical_max: float = Field(default=0.35, ge=0.0, le=1.0)
    # Position-count caps.
    max_positions: int = Field(default=25, ge=0)
    max_tactical_positions: int = Field(default=8, ge=0)
    # Turnover discipline (doc 07 section 5).
    turnover_budget_monthly: float = Field(default=0.15, ge=0.0)
    min_trade_weight: float = Field(default=0.0025, ge=0.0)  # 0.25% noise floor
    drift_band_rel: float = Field(default=0.25, gt=0.0)  # 25% relative drift band
    # Candidate-selection floors (doc 07 section 2, step 2).
    quality_floor: float = Field(default=60.0)  # long-term quality composite 0-100
    long_term_top_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    # Regimes in which NEW tactical entries are eligible (doc 05 / doc 06 section 8).
    tactical_entry_regimes: tuple[Regime, ...] = (Regime.RISK_ON, Regime.NEUTRAL)
    # Tactical-sleeve regime multipliers ("split honors whichever is tighter").
    risk_off_tactical_multiplier: float = Field(default=0.5, ge=0.0, le=1.0)
    # --- Idle-cash ballast (empty symbol = disabled; bit-identical to the
    # pre-ballast constructor when disabled). Residual weight above
    # ``max(verdict.cash_floor, ballast_cash_floor)`` is parked in this
    # symbol AFTER active sleeves are sized and compliance-clamped, scaled by
    # the drawdown ladder's risk posture. The DEFAULT ballast in production
    # is the diversified index (SPY, settings.ballast_symbol): ballast is a
    # cash substitute deliberately exempt from per-position/sector caps, and
    # a single name carrying uncapped residual weight is not ballast — it is
    # a concentrated bet. Single-stock ballast stays technically possible
    # (any non-crypto symbol validates) but is not the default.
    ballast_symbol: str = ""
    ballast_cash_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    # Bankroll risk sleeve (dynamo.portfolio.bankroll): the most that
    # AdvisoryInputs.risk_sleeve_symbols may hold together. None = off
    # (construction identical to before the field existed).
    risk_sleeve_cap: float | None = Field(default=None, ge=0.0, le=1.0)
    # Per-rung multipliers on the ballast residual, overriding
    # ``dynamo.risk.ladder.BALLAST_RUNG_MULTIPLIERS`` (empty = that
    # documented default schedule). Ballast deliberately does NOT follow the
    # ACTIVE drawdown ladder: it is a passive index floor, and cutting it on
    # a -5% portfolio dip is market timing, not risk management (campaign
    # findings item 20 — the control run did exactly that and gave back most
    # of its benchmark gap). A tuple of pairs rather than a dict so the
    # frozen config stays hashable and ``config_hash`` stays stable.
    ballast_ladder_multipliers: tuple[tuple[LadderRung, float], ...] = ()

    @field_validator("ballast_ladder_multipliers")
    @classmethod
    def _ballast_multipliers_in_unit_range(
        cls, v: tuple[tuple[LadderRung, float], ...]
    ) -> tuple[tuple[LadderRung, float], ...]:
        for rung, multiplier in v:
            if not 0.0 <= multiplier <= 1.0:
                raise ValueError(
                    f"ballast multiplier for {rung.value} must be in [0, 1], got {multiplier}"
                )
        return v

    @field_validator("ballast_symbol")
    @classmethod
    def _ballast_not_crypto(cls, v: str) -> str:
        if v and is_crypto_symbol(v):
            raise ValueError(
                f"ballast_symbol {v!r} is a crypto pair; ballast must be an "
                "equity/ETF symbol (crypto vol makes it cash-substitute-unfit)"
            )
        return v

    @classmethod
    def for_bucket(cls, bucket: RiskBucket) -> ConstructionConfig:
        """Preset for a risk bucket (doc 07 section 3 table)."""
        return cls(bucket=bucket, **_PRESETS[bucket])  # type: ignore[arg-type]

    @property
    def config_hash(self) -> str:
        """Stable sha256 over the full frozen configuration."""
        payload = self.model_dump_json()  # pydantic emits keys in field order: stable
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
