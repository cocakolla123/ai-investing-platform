"""Cost and slippage model (doc 08 section 3). Deliberately conservative.

cost_per_side = half_spread(liquidity_decile) + impact
impact        = k * sigma_daily * sqrt(order_value / ADV_dollar)   # k = 1.0

Fills execute at the NEXT session's open after a signal date — no same-bar
fills, ever. That rule lives in the engine; this module prices one side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Half-spread by 20d-median-dollar-volume band (doc 08 section 3 table).
_MEGA_ADV = 100_000_000.0  # deciles 1-2
_MID_ADV = 20_000_000.0  # deciles 3-5
_SMALL_ADV = 5_000_000.0  # deciles 6-8; below -> 9-10

HALF_SPREAD_MEGA_BPS = 2.0
HALF_SPREAD_MID_BPS = 5.0
HALF_SPREAD_SMALL_BPS = 12.0
HALF_SPREAD_MICRO_BPS = 25.0


@dataclass(frozen=True)
class CostParams:
    """Versioned cost-model parameters; changes invalidate cached reports."""

    impact_k: float = 1.0
    flat_cost_bps: float | None = None  # override for parity runs (matched flat costs)


def half_spread_bps(adv_dollar_20d: float) -> float:
    """Half-spread assumption from the liquidity band. Unknown ADV fails expensive."""
    if not math.isfinite(adv_dollar_20d) or adv_dollar_20d <= 0.0:
        return HALF_SPREAD_MICRO_BPS
    if adv_dollar_20d > _MEGA_ADV:
        return HALF_SPREAD_MEGA_BPS
    if adv_dollar_20d > _MID_ADV:
        return HALF_SPREAD_MID_BPS
    if adv_dollar_20d > _SMALL_ADV:
        return HALF_SPREAD_SMALL_BPS
    return HALF_SPREAD_MICRO_BPS


def impact_fraction(
    order_value: float, sigma_daily: float, adv_dollar_20d: float, k: float = 1.0
) -> float:
    """Square-root market impact as a fraction of order value.

    Monotone non-decreasing in order value: bigger orders are never cheaper.
    Degenerate inputs price expensively (sigma missing -> treated as 2% daily).
    """
    if not math.isfinite(order_value) or order_value <= 0.0:
        return 0.0
    if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
        sigma_daily = 0.02
    if not math.isfinite(adv_dollar_20d) or adv_dollar_20d <= 0.0:
        adv_dollar_20d = 1_000_000.0  # unknown liquidity: assume thin
    return k * sigma_daily * math.sqrt(order_value / adv_dollar_20d)


def cost_per_side_fraction(
    order_value: float,
    sigma_daily: float,
    adv_dollar_20d: float,
    params: CostParams | None = None,
) -> float:
    """Total one-side cost as a fraction of order value (commission $0, Alpaca)."""
    p = params if params is not None else CostParams()
    if p.flat_cost_bps is not None:
        return p.flat_cost_bps / 10_000.0
    spread = half_spread_bps(adv_dollar_20d) / 10_000.0
    return spread + impact_fraction(order_value, sigma_daily, adv_dollar_20d, p.impact_k)
