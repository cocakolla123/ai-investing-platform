"""Construction configuration: risk buckets as construction presets.

Design: docs/design/07-portfolio-construction.md section 3. Low / Medium /
High are parameter presets over one code path; bucket selection is owner
config, changed only via the promotion pipeline.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from dynamo.domain.models.risk import Regime, RiskBucket

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

    @classmethod
    def for_bucket(cls, bucket: RiskBucket) -> ConstructionConfig:
        """Preset for a risk bucket (doc 07 section 3 table)."""
        return cls(bucket=bucket, **_PRESETS[bucket])  # type: ignore[arg-type]

    @property
    def config_hash(self) -> str:
        """Stable sha256 over the full frozen configuration."""
        payload = self.model_dump_json()  # pydantic emits keys in field order: stable
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
