"""Liquidity constraints: entry caps, days-to-liquidate, concentration flag.

Design: docs/design/06-risk-engine.md section 10.

UNITS CONTRACT: every ``adv_dollar_20d`` argument in this module is
**consolidated-equivalent** dollars — the whole-market tape, which is what
"average daily volume" conventionally means and what these caps were sized
against. A caller holding volume from a single-venue feed (Alpaca's free data
is IEX-only, on the order of 1/15th of the tape) must convert first, with
:func:`consolidated_equivalent_adv` below. Passing raw IEX dollars here does
not fail loudly — it silently shrinks every entry cap by that same factor and
reports days-to-liquidate correspondingly too pessimistic. See
:mod:`dynamo.domain.volume_source` for the measurement and its caveats.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from dynamo.domain.models.risk import RiskConfig
from dynamo.domain.volume_source import consolidated_equivalent_dollar_volume

# Re-exported so risk-side callers converting a single-venue ADV do not have to
# reach past this module for the units contract it documents.
consolidated_equivalent_adv = consolidated_equivalent_dollar_volume

EXIT_PARTICIPATION = 0.20  # portfolio exit test at 20% of ADV
CONCENTRATION_DAYS = 5.0
CONCENTRATION_EQUITY_FRACTION = 0.10


@dataclass(frozen=True)
class LiquidityReport:
    days_to_liquidate: dict[str, float]  # symbol -> est. sessions at 20% ADV
    portfolio_days_to_liquidate: float  # slowest position (bottleneck)
    illiquid_equity_fraction: float  # equity share needing > 5 sessions
    concentration_flag: bool


def max_entry_weight(adv_dollar_20d: float, equity: float, config: RiskConfig) -> float:
    """Largest openable weight: <= max_days x participation x ADV / equity,
    and 0 below the absolute ``min_adv_dollar`` floor.

    Unusable inputs give 0 — an illiquid or unknown name cannot be bought.

    The relative cap alone is vacuous for a small account: it scales with
    equity, so at $1,000 of equity it cleared a full position in a name
    trading a few hundred dollars a day. ``config.min_adv_dollar`` is the
    absolute floor that does not scale, because microstructure does not
    scale either.

    STILL MISSING, stated rather than left to be discovered: there is no
    **price floor** (a $0.40 name with a 1-cent spread pays 2.5% per side)
    and no **spread limit** anywhere in ``dynamo.risk``. Both need inputs
    the risk engine is not given today — last price and a quote — so both
    are plumbing changes to ``RiskInputs``, not one-line additions here.
    """
    if not (math.isfinite(adv_dollar_20d) and adv_dollar_20d > 0.0):
        return 0.0
    if adv_dollar_20d < config.min_adv_dollar:
        return 0.0
    if not (math.isfinite(equity) and equity > 0.0):
        return 0.0
    cap = config.liquidity_max_days * config.liquidity_adv_participation * adv_dollar_20d / equity
    return max(cap, 0.0) if math.isfinite(cap) else 0.0


def days_to_liquidate(
    position_dollars: float,
    adv_dollar_20d: float,
    participation: float = EXIT_PARTICIPATION,
) -> float:
    """Estimated sessions to exit a position at the given ADV participation."""
    if not (math.isfinite(position_dollars) and position_dollars >= 0.0):
        return float("inf")
    if position_dollars == 0.0:
        return 0.0
    if not (math.isfinite(adv_dollar_20d) and adv_dollar_20d > 0.0):
        return float("inf")
    if not (math.isfinite(participation) and participation > 0.0):
        return float("inf")
    return position_dollars / (adv_dollar_20d * participation)


def portfolio_liquidity(
    weights: Mapping[str, float],
    equity: float,
    adv_dollar_20d: Mapping[str, float],
    participation: float = EXIT_PARTICIPATION,
) -> LiquidityReport:
    """Weekly portfolio exit test: days to liquidate 100% at 20% participation.

    Concentration alert when positions worth > 10% of equity need > 5 sessions.
    """
    days: dict[str, float] = {}
    illiquid_fraction = 0.0
    worst = 0.0
    equity_ok = math.isfinite(equity) and equity > 0.0
    for symbol, weight in weights.items():
        if not math.isfinite(weight) or weight <= 0.0 or not equity_ok:
            d = 0.0 if weight == 0.0 else float("inf")
        else:
            d = days_to_liquidate(weight * equity, adv_dollar_20d.get(symbol, 0.0), participation)
        days[symbol] = d
        worst = max(worst, d if math.isfinite(d) else float("inf"))
        if d > CONCENTRATION_DAYS and math.isfinite(weight) and weight > 0.0:
            illiquid_fraction += weight
    return LiquidityReport(
        days_to_liquidate=days,
        portfolio_days_to_liquidate=worst,
        illiquid_equity_fraction=illiquid_fraction,
        concentration_flag=illiquid_fraction > CONCENTRATION_EQUITY_FRACTION,
    )
