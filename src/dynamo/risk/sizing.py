"""Position sizing cascade (doc 06 section 4) — risk engine decides, advice tilts.

Order is contractual:
  1. vol-normalized base within the sleeve budget
  2. conviction tilt clamp(0.5 + conviction, [0.5, 1.5])
  3. fractional-Kelly CEILING (0.25x; negative edge -> 0)
  4. clamp cascade: per-position -> sector -> cluster -> liquidity -> gross/cash
  5. 0.5% floor (noise trades are not opened)

Invariants (property-tested): no output exceeds any cap; NaN/inf/negative
inputs produce weight 0, never an exception; higher sigma never yields a
larger base size. Every binding clamp is recorded in clamps_applied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import RiskConfig, SizedPosition

CLAMP_INVALID = "invalid_input"
CLAMP_VETO = "veto"
CLAMP_KELLY = "kelly_ceiling"
CLAMP_PER_POSITION = "per_position_cap"
CLAMP_TACTICAL = "tactical_position_cap"
CLAMP_SECTOR = "sector_cap"
CLAMP_CLUSTER = "cluster_cap"
CLAMP_LIQUIDITY = "liquidity_cap"
CLAMP_GROSS = "gross_cap"
CLAMP_FLOOR = "min_weight_floor"

_HORIZON_SESSIONS = 20.0
_SESSIONS_PER_YEAR = 252.0


@dataclass(frozen=True)
class SizingCandidate:
    """One proposed position entering the cascade."""

    symbol: str
    sleeve: Sleeve
    conviction: float  # committee conviction 0..1 (advisory tilt only)
    sigma_annual: float  # EWMA(config.ewma_vol_span_days) annualized
    adv_dollar_20d: float  # 20d median dollar volume
    expected_excess_return: float | None = None  # quant 20d mu (for Kelly)
    vol_forecast_annual: float | None = None  # quant vol forecast (for Kelly)
    sector: str | None = None
    cluster_id: str | None = None
    # Meta-model calibrated P(win) (plan Phase 3). DARK by default: None
    # everywhere until meta_sizing_enabled is flipped post-promotion; when
    # present (and finite) it replaces conviction in the tilt formula ONLY —
    # the formula and every cap stay unchanged.
    effective_probability: float | None = None


@dataclass(frozen=True)
class SizingContext:
    """Portfolio state and risk-engine directives the cascade runs under."""

    equity: float
    existing_weights: Mapping[str, float] = field(default_factory=dict[str, float])
    sector_by_symbol: Mapping[str, str] = field(default_factory=dict[str, str])
    cluster_by_symbol: Mapping[str, str] = field(default_factory=dict[str, str])
    vetoed: frozenset[str] = frozenset()
    sleeve_budget: float = 1.0  # gross budget available to the new positions
    risk_multiplier: float = 1.0  # drawdown-ladder new-risk multiplier
    tactical_multiplier: float = 1.0  # ladder tactical-sleeve cap multiplier
    tighten_factor: float = 1.0  # Kupiec-red auto-tighten (0.75)


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _zero(candidate: SizingCandidate, reason: str) -> SizedPosition:
    return SizedPosition(
        symbol=candidate.symbol,
        weight=0.0,
        base_weight=0.0,
        conviction_tilt=1.0,
        kelly_ceiling=None,
        clamps_applied=(reason,),
    )


def _candidate_sizable(candidate: SizingCandidate, ctx: SizingContext) -> bool:
    return (
        candidate.symbol not in ctx.vetoed
        and _finite(candidate.sigma_annual)
        and candidate.sigma_annual > 0.0
        and _finite(candidate.conviction)
        and _finite(candidate.adv_dollar_20d)
        and candidate.adv_dollar_20d >= 0.0
    )


def _kelly_ceiling(candidate: SizingCandidate, config: RiskConfig) -> float | None:
    """0.25x Kelly ceiling from quant edge; negative/degenerate edge -> 0."""
    mu = candidate.expected_excess_return
    if mu is None:
        return None
    if not _finite(mu) or mu <= 0.0:
        return 0.0
    sigma = candidate.vol_forecast_annual
    if sigma is None or not _finite(sigma) or sigma <= 0.0:
        sigma = candidate.sigma_annual
    var_horizon = (sigma * sigma) * (_HORIZON_SESSIONS / _SESSIONS_PER_YEAR)
    if not _finite(var_horizon) or var_horizon <= 0.0:
        return 0.0
    kelly = mu / var_horizon
    ceiling = config.kelly_fraction_cap * kelly
    if not _finite(ceiling) or ceiling < 0.0:
        return 0.0
    return ceiling


def size_positions(
    candidates: Sequence[SizingCandidate],
    config: RiskConfig,
    ctx: SizingContext,
) -> list[SizedPosition]:
    """Run the full sizing cascade over the candidate list (order-deterministic).

    Sector/cluster/gross budgets are consumed cumulatively in input order, so
    the totals respect the caps even when many candidates share a bucket.
    """
    context_valid = _finite(ctx.equity) and ctx.equity > 0.0

    budget = ctx.sleeve_budget * ctx.risk_multiplier
    if not _finite(budget) or budget < 0.0:
        budget = 0.0
    tighten = ctx.tighten_factor
    if not _finite(tighten) or not 0.0 < tighten <= 1.0:
        tighten = 0.75  # unusable tighten directive -> fail tighter, not looser

    # Vol-normalized base weights (risk parity within the sleeve budget).
    inv_sigmas: dict[int, float] = {}
    for i, candidate in enumerate(candidates):
        if context_valid and _candidate_sizable(candidate, ctx):
            inv_sigmas[i] = 1.0 / candidate.sigma_annual
    inv_total = sum(inv_sigmas.values())

    # Existing usage per sector / cluster / gross.
    sector_used: dict[str, float] = {}
    cluster_used: dict[str, float] = {}
    existing_gross = 0.0
    for symbol, weight in ctx.existing_weights.items():
        if not _finite(weight) or weight <= 0.0:
            continue
        existing_gross += weight
        sector = ctx.sector_by_symbol.get(symbol)
        if sector is not None:
            sector_used[sector] = sector_used.get(sector, 0.0) + weight
        cluster = ctx.cluster_by_symbol.get(symbol)
        if cluster is not None:
            cluster_used[cluster] = cluster_used.get(cluster, 0.0) + weight

    new_gross = 0.0
    results: list[SizedPosition] = []

    for i, candidate in enumerate(candidates):
        if candidate.symbol in ctx.vetoed:
            results.append(_zero(candidate, CLAMP_VETO))
            continue
        if not context_valid or i not in inv_sigmas:
            results.append(_zero(candidate, CLAMP_INVALID))
            continue

        clamps: list[str] = []
        base = budget * inv_sigmas[i] / inv_total if inv_total > 0.0 else 0.0
        # Tilt input: the meta-model's effective probability when present
        # (dark path, plan Phase 3), raw conviction otherwise — formula
        # unchanged either way. Non-finite advice falls back to conviction.
        p = (
            candidate.effective_probability
            if candidate.effective_probability is not None
            and _finite(candidate.effective_probability)
            else candidate.conviction
        )
        tilt = min(
            max(0.5 + p, config.conviction_tilt_min),
            config.conviction_tilt_max,
        )
        weight = base * tilt

        kelly_ceiling = _kelly_ceiling(candidate, config)
        if kelly_ceiling is not None and weight > kelly_ceiling:
            weight = kelly_ceiling
            clamps.append(CLAMP_KELLY)

        # -- clamp cascade --------------------------------------------------
        if candidate.sleeve is Sleeve.TACTICAL:
            position_cap = config.tactical_position_cap * ctx.tactical_multiplier
            position_clamp = CLAMP_TACTICAL
        else:
            position_cap = config.per_position_cap
            position_clamp = CLAMP_PER_POSITION
        position_cap = min(position_cap * tighten, config.per_position_hard)
        position_cap = max(position_cap, 0.0)
        if weight > position_cap:
            weight = position_cap
            clamps.append(position_clamp)

        if candidate.sector is not None:
            available = max(
                config.sector_cap * tighten - sector_used.get(candidate.sector, 0.0),
                0.0,
            )
            if weight > available:
                weight = available
                clamps.append(CLAMP_SECTOR)

        if candidate.cluster_id is not None:
            available = max(
                config.cluster_cap * tighten
                - cluster_used.get(candidate.cluster_id, 0.0),
                0.0,
            )
            if weight > available:
                weight = available
                clamps.append(CLAMP_CLUSTER)

        liquidity_cap = (
            config.liquidity_max_days
            * config.liquidity_adv_participation
            * candidate.adv_dollar_20d
            / ctx.equity
        )
        if not _finite(liquidity_cap) or liquidity_cap < 0.0:
            liquidity_cap = 0.0
        if weight > liquidity_cap:
            weight = liquidity_cap
            clamps.append(CLAMP_LIQUIDITY)

        gross_available = max(
            config.gross_cap - config.cash_floor - existing_gross - new_gross, 0.0
        )
        if weight > gross_available:
            weight = gross_available
            clamps.append(CLAMP_GROSS)

        if not _finite(weight) or weight < 0.0:
            weight = 0.0
            clamps.append(CLAMP_INVALID)
        elif 0.0 < weight < config.min_position_weight:
            weight = 0.0
            clamps.append(CLAMP_FLOOR)

        if weight > 0.0:
            new_gross += weight
            if candidate.sector is not None:
                sector_used[candidate.sector] = (
                    sector_used.get(candidate.sector, 0.0) + weight
                )
            if candidate.cluster_id is not None:
                cluster_used[candidate.cluster_id] = (
                    cluster_used.get(candidate.cluster_id, 0.0) + weight
                )

        results.append(
            SizedPosition(
                symbol=candidate.symbol,
                weight=weight,
                base_weight=base,
                conviction_tilt=tilt,
                kelly_ceiling=kelly_ceiling,
                clamps_applied=tuple(clamps),
            )
        )
    return results
