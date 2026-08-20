"""Historical VaR/CVaR (full revaluation) and Kupiec traffic light.

Design: docs/design/06-risk-engine.md section 6. 95/99 one-day from 2y of
portfolio-weighted history; VaR model quality monitored via exception counts,
with automatic cap tightening (0.75x) in the red zone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dynamo.risk.types import FloatArray, ReturnsPanel, clean_finite
from dynamo.risk.vol import portfolio_return_series

DEFAULT_LOOKBACK_SESSIONS = 504  # ~2 years
RED_TIGHTEN_FACTOR = 0.75


@dataclass(frozen=True)
class VarResult:
    var95: float
    var99: float
    cvar95: float
    cvar99: float
    n_obs: int


class VarTrafficLight(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


def _var_cvar(losses: FloatArray, confidence: float) -> tuple[float, float]:
    """(VaR, CVaR) at the confidence level from a loss distribution (>0 = loss)."""
    q = float(np.quantile(losses, confidence))
    var = max(q, 0.0)
    tail = losses[losses >= q]
    cvar = float(np.mean(tail)) if tail.size else var
    return var, max(cvar, var)


def historical_var(
    panel: ReturnsPanel,
    weights: Mapping[str, float],
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
) -> VarResult | None:
    """Full-revaluation 1-day historical VaR/CVaR at 95% and 99%.

    Returns None when the portfolio series cannot be computed (missing symbol,
    non-finite weight) or has fewer than 60 usable observations — fail closed.
    """
    series = portfolio_return_series(panel, weights)
    if series is None:
        return None
    series = clean_finite(series[-max(lookback_sessions, 1) :])
    if series.size < 60:
        return None
    losses = -series
    var95, cvar95 = _var_cvar(losses, 0.95)
    var99, cvar99 = _var_cvar(losses, 0.99)
    var99 = max(var99, var95)
    cvar99 = max(cvar99, var99)
    return VarResult(var95=var95, var99=var99, cvar95=cvar95, cvar99=cvar99, n_obs=int(series.size))


def _binom_cdf(k: int, n: int, p: float) -> float:
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(total, 1.0)


def kupiec_traffic_light(
    exceptions: int, observations: int = 250, coverage: float = 0.95
) -> VarTrafficLight:
    """Basel-style traffic light on VaR exception counts.

    Green while the exception count is unsurprising under the model
    (binomial CDF <= 0.95), yellow to 0.9999, red beyond. Invalid inputs fail
    closed to red.
    """
    if observations <= 0 or exceptions < 0 or not 0.0 < coverage < 1.0:
        return VarTrafficLight.RED
    if exceptions > observations:
        return VarTrafficLight.RED
    cdf = _binom_cdf(exceptions, observations, 1.0 - coverage)
    if cdf <= 0.95:
        return VarTrafficLight.GREEN
    if cdf <= 0.9999:
        return VarTrafficLight.YELLOW
    return VarTrafficLight.RED


def kupiec_tighten_factor(light: VarTrafficLight) -> float:
    """Cap multiplier when the VaR model is degraded: red -> 0.75, else 1.0."""
    return RED_TIGHTEN_FACTOR if light is VarTrafficLight.RED else 1.0


# --- backtesting the VaR model (doc 06 §6) --------------------------------

EXCEPTION_WINDOW_SESSIONS = 250
MIN_BACKTEST_OBSERVATIONS = 60


@dataclass(frozen=True)
class VarBacktest:
    """Realized outcome of the VaR model over a trailing window."""

    exceptions: int
    observations: int
    light: VarTrafficLight
    tighten_factor: float
    worst_excess: float = 0.0  # largest (loss - predicted var95); 0.0 when none

    @property
    def caps_tightened(self) -> bool:
        return self.tighten_factor < 1.0


def count_var_exceptions(observations: Sequence[tuple[float, float]]) -> tuple[int, float]:
    """Days where the realized loss exceeded the VaR predicted *for that day*.

    ``observations`` is ``(predicted_var95, realized_return)`` in
    chronological order; ``realized_return`` is a simple return (-0.03 = down
    3%) and ``predicted_var95`` is a positive loss magnitude, matching
    :class:`VarResult`. Pairs with a non-finite or non-positive VaR are
    skipped rather than counted — a day on which the model produced no
    forecast is evidence neither for nor against it.

    Returns ``(exceptions, worst_excess)``.
    """
    exceptions = 0
    worst = 0.0
    for predicted, realized in observations:
        if not (math.isfinite(predicted) and predicted > 0.0):
            continue
        if not math.isfinite(realized):
            continue
        loss = -realized
        if loss > predicted:
            exceptions += 1
            worst = max(worst, loss - predicted)
    return exceptions, worst


def backtest_var(
    observations: Sequence[tuple[float, float]],
    *,
    window: int = EXCEPTION_WINDOW_SESSIONS,
    coverage: float = 0.95,
) -> VarBacktest:
    """Trailing-window VaR backtest and the cap multiplier it implies.

    doc 06 §6: "too many VaR exceptions -> caps tighten 25%". The count fed to
    :func:`kupiec_traffic_light` has to come from *realized* returns compared
    against the VaR actually stored on each day. A hard-coded 0 — the state
    before this function existed — makes the traffic light permanently green
    and the tightening rule unreachable.

    Fewer than 60 usable observations is not enough to judge the model: the
    light stays GREEN and caps are untouched. Tightening on noise would be
    its own risk.
    """
    recent = list(observations)[-max(window, 1) :]
    usable = [(p, r) for p, r in recent if math.isfinite(p) and p > 0.0 and math.isfinite(r)]
    exceptions, worst = count_var_exceptions(usable)
    if len(usable) < MIN_BACKTEST_OBSERVATIONS:
        return VarBacktest(exceptions, len(usable), VarTrafficLight.GREEN, 1.0, worst)
    light = kupiec_traffic_light(exceptions, len(usable), coverage)
    return VarBacktest(exceptions, len(usable), light, kupiec_tighten_factor(light), worst)
