"""Stop computation: ATR initial, trailing ratchet, disaster, time, thesis.

Design: docs/design/06-risk-engine.md section 5. Stops only ever tighten:
strategy overrides are tighter-only and the trailing stop ratchets up, never
down. Recomputed daily; the execution layer maintains the broker orders.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import RiskConfig, StopKind, StopOrderSpec


@dataclass(frozen=True)
class StopInputs:
    """Per-position facts required to recompute stops."""

    symbol: str
    sleeve: Sleeve
    entry_price: float
    current_price: float
    atr14: float
    highest_close_since_entry: float | None = None
    prev_stop: float | None = None
    strategy_stop_multiple: float | None = None  # tighter-only override
    sessions_held: int = 0
    horizon_days: int | None = None  # tactical time-stop horizon
    thesis_breaches: tuple[str, ...] = ()  # triggered exit-condition predicates


def effective_atr_multiple(
    config: RiskConfig, strategy_multiple: float | None, override: float | None = None
) -> float:
    """Base ATR multiple with tighter-only strategy/crisis overrides."""
    multiple = config.atr_stop_multiple
    if (
        strategy_multiple is not None
        and math.isfinite(strategy_multiple)
        and strategy_multiple > 0.0
    ):
        multiple = min(multiple, strategy_multiple)
    if override is not None and math.isfinite(override) and override > 0.0:
        multiple = min(multiple, override)
    return multiple


def initial_atr_stop(entry_price: float, atr14: float, multiple: float) -> float:
    return entry_price - multiple * atr14


def trailing_stop(
    highest_close: float, atr14: float, multiple: float, entry_price: float
) -> float | None:
    """Trailing level once the position is more than 1 ATR in profit, else None."""
    if not (
        math.isfinite(highest_close)
        and math.isfinite(atr14)
        and math.isfinite(entry_price)
        and atr14 > 0.0
    ):
        return None
    if highest_close <= entry_price + atr14:
        return None
    return highest_close - multiple * atr14


def disaster_stop(entry_price: float, config: RiskConfig) -> float:
    return entry_price * (1.0 - config.disaster_stop_pct)


def compute_position_stops(
    inputs: StopInputs,
    config: RiskConfig,
    atr_multiple_override: float | None = None,
) -> list[StopOrderSpec]:
    """All stop specs for one position; invalid inputs degrade to the widest
    stop still computable (disaster from entry) rather than raising."""
    specs: list[StopOrderSpec] = []
    entry = inputs.entry_price
    if not (math.isfinite(entry) and entry > 0.0):
        return specs  # nothing computable without an entry price

    disaster = disaster_stop(entry, config)
    atr_ok = math.isfinite(inputs.atr14) and inputs.atr14 > 0.0

    if inputs.sleeve is Sleeve.TACTICAL and atr_ok:
        multiple = effective_atr_multiple(
            config, inputs.strategy_stop_multiple, atr_multiple_override
        )
        level = initial_atr_stop(entry, inputs.atr14, multiple)
        kind = StopKind.ATR
        reason = f"initial {multiple:g}x ATR"

        highest = inputs.highest_close_since_entry
        if highest is None and math.isfinite(inputs.current_price):
            highest = max(entry, inputs.current_price)
        if highest is not None:
            trail = trailing_stop(highest, inputs.atr14, multiple, entry)
            if trail is not None and trail > level:
                level = trail
                kind = StopKind.TRAILING
                reason = f"trailing {multiple:g}x ATR from {highest:g}"

        # Ratchet: never below a previously set price stop.
        prev = inputs.prev_stop
        if prev is not None and math.isfinite(prev) and prev > level:
            level = prev
            kind = StopKind.TRAILING
            reason = "ratchet holds prior stop (stops only tighten)"

        level = max(level, disaster)
        if level == disaster and kind is StopKind.ATR:
            kind = StopKind.DISASTER
            reason = f"disaster stop -{config.disaster_stop_pct:.0%} from entry"
        specs.append(
            StopOrderSpec(
                symbol=inputs.symbol, kind=kind, stop_price=level, reason=reason
            )
        )
    else:
        # Long-term sleeve (or no usable ATR): disaster stop from entry;
        # thesis invalidation is handled via predicates below.
        prev = inputs.prev_stop
        level = disaster
        kind = StopKind.DISASTER
        reason = f"disaster stop -{config.disaster_stop_pct:.0%} from entry"
        if prev is not None and math.isfinite(prev) and prev > level:
            level = prev
            kind = StopKind.TRAILING
            reason = "ratchet holds prior stop (stops only tighten)"
        specs.append(
            StopOrderSpec(
                symbol=inputs.symbol, kind=kind, stop_price=level, reason=reason
            )
        )

    if (
        inputs.sleeve is Sleeve.TACTICAL
        and inputs.horizon_days is not None
        and inputs.horizon_days > 0
        and inputs.sessions_held >= inputs.horizon_days
        and math.isfinite(inputs.current_price)
        and inputs.current_price > 0.0
    ):
        specs.append(
            StopOrderSpec(
                symbol=inputs.symbol,
                kind=StopKind.TIME,
                stop_price=inputs.current_price,
                reason=(
                    f"time stop: held {inputs.sessions_held} sessions"
                    f" >= horizon {inputs.horizon_days} — exit review"
                ),
            )
        )

    if math.isfinite(inputs.current_price) and inputs.current_price > 0.0:
        for breach in inputs.thesis_breaches:
            specs.append(
                StopOrderSpec(
                    symbol=inputs.symbol,
                    kind=StopKind.THESIS,
                    stop_price=inputs.current_price,
                    reason=f"thesis invalidated: {breach}",
                )
            )
    return specs


def recompute_stops(
    positions: Sequence[StopInputs],
    config: RiskConfig,
    atr_multiple_override: float | None = None,
) -> list[StopOrderSpec]:
    """Recompute stop specs for every held position (daily / intraday check)."""
    specs: list[StopOrderSpec] = []
    for inputs in positions:
        specs.extend(compute_position_stops(inputs, config, atr_multiple_override))
    return specs
