"""Portfolio volatility targeting — the primary "how much risk" dial.

Design: docs/design/06-risk-engine.md section 9. Outside the +/-band around
the bucket target, exposure scales toward target; freed exposure goes to
cash. When vol spikes, sizing shrinks mechanically — nobody decides anything.
"""

from __future__ import annotations

import math

from dynamo.domain.models.risk import RiskConfig

MAX_SCALE = 1.5  # never lever more than 1.5x toward target (gross cap still binds)


def exposure_scale(realized_vol_annual: float, config: RiskConfig) -> float:
    """Exposure multiplier given realized portfolio vol vs the bucket target.

    - inside the band (target * (1 -/+ vol_band)): 1.0 (no churn)
    - outside: target / realized, clamped to [0, MAX_SCALE]
    - unusable realized vol: 1.0 (the engine flags degraded data elsewhere)
    """
    target = config.target_vol_annual
    if not (math.isfinite(realized_vol_annual) and realized_vol_annual > 0.0):
        return 1.0
    if not (math.isfinite(target) and target > 0.0):
        return 1.0
    low = target * (1.0 - config.vol_band)
    high = target * (1.0 + config.vol_band)
    if low <= realized_vol_annual <= high:
        return 1.0
    scale = target / realized_vol_annual
    return float(min(max(scale, 0.0), MAX_SCALE))


def scaled_gross_cap(realized_vol_annual: float, config: RiskConfig) -> float:
    """Gross-exposure cap after vol targeting (never above config.gross_cap)."""
    scale = exposure_scale(realized_vol_annual, config)
    return float(min(config.gross_cap, config.gross_cap * scale))
