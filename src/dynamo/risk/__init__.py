"""Dynamo risk engine — pure, deterministic, property-tested.

The highest-priority component: everything else proposes, the risk engine
disposes. Design: docs/design/06-risk-engine.md.
"""

from dynamo.risk.engine import RiskEngine, RiskEngineInputs
from dynamo.risk.ladder import LadderState, RungEffects, effects_for
from dynamo.risk.regime import RegimeHistory, backfill_regimes, latest_regime
from dynamo.risk.sizing import SizingCandidate, SizingContext, size_positions
from dynamo.risk.stops import StopInputs, recompute_stops
from dynamo.risk.stress import run_stress_suite
from dynamo.risk.types import ReturnsPanel, SymbolBars
from dynamo.risk.var import historical_var, kupiec_traffic_light

__all__ = [
    "LadderState",
    "RegimeHistory",
    "ReturnsPanel",
    "RiskEngine",
    "RiskEngineInputs",
    "RungEffects",
    "SizingCandidate",
    "SizingContext",
    "StopInputs",
    "SymbolBars",
    "backfill_regimes",
    "effects_for",
    "historical_var",
    "kupiec_traffic_light",
    "latest_regime",
    "recompute_stops",
    "run_stress_suite",
    "size_positions",
]
