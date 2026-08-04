"""Rebalance trigger evaluation (doc 07 section 5).

Evaluated nightly after the research loop; any single trigger suffices to
request reconstruction. Pure and deterministic — all state is explicit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from dynamo.domain.models.portfolio import RebalanceTrigger
from dynamo.domain.models.research import Action
from dynamo.domain.models.risk import Regime, RiskVerdict

_EXITISH = frozenset({Action.TRIM, Action.SELL, Action.AVOID})
_ENTRYISH = frozenset({Action.BUY, Action.ADD})


@dataclass(frozen=True)
class TriggerInputs:
    """Everything trigger evaluation needs, gathered by the caller."""

    current_weights: Mapping[str, float]
    last_target_weights: Mapping[str, float]
    verdict: RiskVerdict
    drift_band_rel: float = 0.25
    previous_actions: Mapping[str, Action] = field(default_factory=dict)
    current_actions: Mapping[str, Action] = field(default_factory=dict)
    previous_regime: Regime | None = None
    scheduled_review: bool = False  # first trading day of the month
    vol_scale: float = 1.0  # doc 06 section 9 exposure scale; != 1 -> vol_target


@dataclass(frozen=True)
class RebalanceDecision:
    triggers: frozenset[RebalanceTrigger]
    reconstruct: bool
    details: tuple[str, ...]  # one line per fired trigger, for the dashboard


def evaluate_triggers(inputs: TriggerInputs) -> RebalanceDecision:
    triggers: set[RebalanceTrigger] = set()
    details: list[str] = []

    # Drift: > band RELATIVE deviation from the last target (e.g. 8% target
    # drifting past 10% or below 6% at the default 25% band).
    for symbol in sorted(inputs.last_target_weights):
        target = inputs.last_target_weights[symbol]
        if not math.isfinite(target) or target <= 0.0:
            continue
        current = inputs.current_weights.get(symbol, 0.0)
        rel = abs(current - target) / target
        if rel > inputs.drift_band_rel:
            triggers.add(RebalanceTrigger.DRIFT)
            details.append(
                f"drift: {symbol} at {current:.4f} vs target {target:.4f} "
                f"({rel:.0%} > {inputs.drift_band_rel:.0%} band)"
            )

    # Verdict change: action changed for a held name, or a new buy/add enters
    # the selectable set.
    held = {s for s, w in inputs.current_weights.items() if w > 0.0}
    for symbol in sorted(set(inputs.current_actions) | set(inputs.previous_actions)):
        prev = inputs.previous_actions.get(symbol)
        curr = inputs.current_actions.get(symbol)
        if symbol in held and prev is not None and curr is not None and prev != curr:
            triggers.add(RebalanceTrigger.VERDICT_CHANGE)
            details.append(f"verdict change: {symbol} {prev.value} -> {curr.value}")
        elif symbol in held and curr is not None and curr in _EXITISH and prev is None:
            triggers.add(RebalanceTrigger.VERDICT_CHANGE)
            details.append(f"verdict change: held {symbol} now {curr.value}")
        elif (
            symbol not in held
            and curr is not None
            and curr in _ENTRYISH
            and prev not in _ENTRYISH
        ):
            triggers.add(RebalanceTrigger.VERDICT_CHANGE)
            details.append(f"verdict change: new {curr.value} candidate {symbol}")

    # Risk-limit breach by CURRENT holdings (clamp to compliance is mandatory
    # and exempt from the turnover budget).
    breaches = _risk_breaches(inputs.current_weights, inputs.verdict)
    if breaches:
        triggers.add(RebalanceTrigger.RISK_BREACH)
        details.extend(breaches)

    # Regime change (hysteresis already applied inside the risk engine).
    regime_now = inputs.verdict.regime.regime
    if inputs.previous_regime is not None and inputs.previous_regime != regime_now:
        triggers.add(RebalanceTrigger.REGIME_CHANGE)
        details.append(
            f"regime change: {inputs.previous_regime.value} -> {regime_now.value}"
        )

    if inputs.scheduled_review:
        triggers.add(RebalanceTrigger.SCHEDULED)
        details.append("scheduled monthly reconstruction")

    if math.isfinite(inputs.vol_scale) and inputs.vol_scale != 1.0:
        triggers.add(RebalanceTrigger.VOL_TARGET)
        details.append(f"vol targeting: exposure scale {inputs.vol_scale:.3f}")

    return RebalanceDecision(
        triggers=frozenset(triggers),
        reconstruct=bool(triggers),
        details=tuple(details),
    )


def _risk_breaches(weights: Mapping[str, float], verdict: RiskVerdict) -> list[str]:
    out: list[str] = []
    gross = math.fsum(w for w in weights.values() if w > 0.0)
    if gross > verdict.gross_exposure_cap + 1e-12:
        out.append(
            f"risk breach: gross {gross:.4f} > cap {verdict.gross_exposure_cap:.4f}"
        )
    if 1.0 - gross < verdict.cash_floor - 1e-12:
        out.append(
            f"risk breach: cash {1.0 - gross:.4f} < floor {verdict.cash_floor:.4f}"
        )
    vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}
    for symbol in sorted(weights):
        w = weights[symbol]
        if w <= 0.0:
            continue
        cap = verdict.per_position_cap.get(symbol)
        if cap is not None and w > cap + 1e-12:
            out.append(f"risk breach: {symbol} weight {w:.4f} > cap {cap:.4f}")
        if symbol in vetoed:
            out.append(f"risk breach: vetoed symbol {symbol} still held")
    return out
