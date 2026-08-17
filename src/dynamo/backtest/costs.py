"""Cost and slippage model (doc 08 section 3). Deliberately conservative.

cost_per_side = half_spread(liquidity_decile) + impact
impact        = k * sigma_daily * sqrt(order_value / ADV_dollar)   # k = 1.0

Fills execute at the NEXT session's open after a signal date — no same-bar
fills, ever. That rule lives in the engine; this module prices one side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dynamo.domain.symbols import is_crypto_symbol

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
    """Versioned cost-model parameters; changes invalidate cached reports.

    Bankroll-realism knobs (backtest-campaign Phase 3), defaults preserve
    bit-for-bit prior behavior:

    - ``commission_per_order``: flat dollar commission added to the cost of
      every EXECUTED order (Alpaca equities are $0 — nonzero models other
      brokers / small-account realism);
    - ``min_order_notional``: orders with ``|value| < min_order_notional``
      are skipped entirely (never partially filled) — brokers reject dust
      orders, and a $3 rebalance trade is fiction at any bankroll.
    """

    impact_k: float = 1.0
    flat_cost_bps: float | None = None  # override for parity runs (matched flat costs)
    commission_per_order: float = 0.0
    min_order_notional: float = 0.0


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


# --- bankroll-realism helpers (backtest-campaign Phase 3) -------------------
#
# Shared by BOTH fill paths (replay runner._execute and backtest
# engine._execute) so the two engines can never drift on what an order costs
# or when it is too small to exist. Defaults (0 / 0 / False) reproduce prior
# behavior bit-for-bit.


def order_skipped(order_value: float, params: CostParams | None = None) -> bool:
    """True when the order is below the minimum notional and must be skipped
    entirely (no partial fills). ``min_order_notional=0`` never skips."""
    p = params if params is not None else CostParams()
    return abs(order_value) < p.min_order_notional


def order_cost(
    order_value: float,
    sigma_daily: float,
    adv_dollar_20d: float,
    params: CostParams | None = None,
) -> float:
    """Total one-side dollar cost for one EXECUTED order: the fractional
    spread+impact model times value, plus the flat per-order commission.
    Callers round to cents exactly as they rounded the pre-commission cost,
    so ``commission_per_order=0`` is bit-identical to the old math."""
    p = params if params is not None else CostParams()
    return (
        order_value * cost_per_side_fraction(order_value, sigma_daily, adv_dollar_20d, p)
        + p.commission_per_order
    )


def whole_share_qty(symbol: str, qty: float, whole_shares: bool) -> float:
    """Desired share quantity under the whole-shares regime.

    Rule (documented, deliberately simple): in whole-shares mode ALL desired
    quantities are floored toward zero BEFORE the buy/sell delta is computed.
    Because every buy is then a whole number, positions accumulated under the
    regime are whole, so every delta (including partial trims) is whole and a
    zero target sells the exact held quantity. Crypto pairs are EXEMPT —
    fractional is native everywhere (Alpaca crypto included)."""
    if not whole_shares or is_crypto_symbol(symbol):
        return qty
    return float(math.floor(qty))
