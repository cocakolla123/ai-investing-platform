"""Stops: ATR initial, tighter-only overrides, trailing ratchet, disaster,
time flags, thesis predicates."""

from __future__ import annotations

import math

from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import RiskConfig, StopKind
from dynamo.risk.stops import StopInputs, compute_position_stops, recompute_stops

CFG = RiskConfig()  # atr_stop_multiple 2.5, disaster 25%


def _tactical(**overrides: object) -> StopInputs:
    base: dict[str, object] = {
        "symbol": "AAA",
        "sleeve": Sleeve.TACTICAL,
        "entry_price": 100.0,
        "current_price": 100.0,
        "atr14": 2.0,
    }
    base.update(overrides)
    return StopInputs(**base)  # type: ignore[arg-type]


def test_initial_atr_stop_default_multiple() -> None:
    (spec,) = compute_position_stops(_tactical(), CFG)
    assert spec.kind is StopKind.ATR
    assert math.isclose(spec.stop_price, 100.0 - 2.5 * 2.0)


def test_strategy_override_is_tighter_only() -> None:
    (tighter,) = compute_position_stops(_tactical(strategy_stop_multiple=1.5), CFG)
    assert math.isclose(tighter.stop_price, 100.0 - 1.5 * 2.0)
    (wider,) = compute_position_stops(_tactical(strategy_stop_multiple=4.0), CFG)
    assert math.isclose(wider.stop_price, 100.0 - 2.5 * 2.0)  # cannot widen


def test_trailing_engages_only_beyond_one_atr_profit() -> None:
    (flat,) = compute_position_stops(
        _tactical(current_price=101.0, highest_close_since_entry=101.0), CFG
    )
    assert flat.kind is StopKind.ATR  # 1 ATR not yet exceeded
    (trailing,) = compute_position_stops(
        _tactical(current_price=110.0, highest_close_since_entry=110.0), CFG
    )
    assert trailing.kind is StopKind.TRAILING
    assert math.isclose(trailing.stop_price, 110.0 - 2.5 * 2.0)


def test_trailing_ratchets_up_never_down() -> None:
    (spec,) = compute_position_stops(
        _tactical(current_price=104.0, highest_close_since_entry=110.0, prev_stop=106.0),
        CFG,
    )
    assert spec.stop_price == 106.0  # prior stop above recomputed trail: hold it
    (spec2,) = compute_position_stops(
        _tactical(current_price=112.0, highest_close_since_entry=112.0, prev_stop=95.5),
        CFG,
    )
    assert spec2.stop_price > 95.5  # higher trail wins


def test_disaster_stop_floors_everything() -> None:
    # ATR stop way below entry: the -25% disaster stop is the binding level.
    (spec,) = compute_position_stops(_tactical(atr14=20.0), CFG)
    assert spec.kind is StopKind.DISASTER
    assert math.isclose(spec.stop_price, 75.0)


def test_long_term_uses_disaster_not_atr() -> None:
    (spec,) = compute_position_stops(_tactical(sleeve=Sleeve.LONG_TERM), CFG)
    assert spec.kind is StopKind.DISASTER
    assert math.isclose(spec.stop_price, 75.0)


def test_time_stop_flags_at_horizon() -> None:
    specs = compute_position_stops(_tactical(sessions_held=21, horizon_days=20), CFG)
    kinds = {s.kind for s in specs}
    assert StopKind.TIME in kinds
    time_spec = next(s for s in specs if s.kind is StopKind.TIME)
    assert time_spec.stop_price == 100.0
    # Below horizon: no time stop.
    specs = compute_position_stops(_tactical(sessions_held=10, horizon_days=20), CFG)
    assert StopKind.TIME not in {s.kind for s in specs}


def test_thesis_predicate_hooks_emit_thesis_stops() -> None:
    specs = compute_position_stops(
        _tactical(
            sleeve=Sleeve.LONG_TERM,
            thesis_breaches=("gross margin < 30% next report",),
        ),
        CFG,
    )
    thesis = [s for s in specs if s.kind is StopKind.THESIS]
    assert len(thesis) == 1
    assert "gross margin" in thesis[0].reason


def test_crisis_override_tightens_atr_multiple() -> None:
    (normal,) = compute_position_stops(_tactical(), CFG)
    (crisis,) = compute_position_stops(_tactical(), CFG, atr_multiple_override=2.0)
    assert crisis.stop_price > normal.stop_price


def test_invalid_inputs_never_raise() -> None:
    assert compute_position_stops(_tactical(entry_price=float("nan")), CFG) == []
    (spec,) = compute_position_stops(_tactical(atr14=float("nan")), CFG)
    assert spec.kind is StopKind.DISASTER  # degrade to widest computable stop


def test_recompute_stops_covers_all_positions() -> None:
    specs = recompute_stops([_tactical(), _tactical(symbol="BBB", sleeve=Sleeve.LONG_TERM)], CFG)
    assert {s.symbol for s in specs} == {"AAA", "BBB"}
