"""Rule-based market regime composite with hysteresis.

Design: docs/design/06-risk-engine.md section 8. Five transparent components
(SPY 50/200 trend, realized-vol percentile, HY OAS level + change, curve
slope, breadth) each scored -2..+2; the composite mean maps to a Regime.
Regime informs caps; it never generates buy signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from dynamo.domain.models.risk import Regime, RegimeState, RiskConfig
from dynamo.risk.types import ANNUALIZATION, FloatArray

DEFAULT_WARMUP_SESSIONS = 200
_VOL_WINDOW = 20
_VOL_PERCENTILE_LOOKBACK = 504
_OAS_CHANGE_SESSIONS = 20

RISK_ON_THRESHOLD = 0.75
RISK_OFF_THRESHOLD = -0.5
CRISIS_THRESHOLD = -1.25


@dataclass(frozen=True)
class RegimeHistory:
    """Aligned daily inputs for regime backfill (15+ years for validation)."""

    dates: tuple[date, ...]
    spy_close: FloatArray
    hy_oas: FloatArray  # percent, e.g. 4.5
    curve_slope: FloatArray  # 10y minus 3m, percentage points
    breadth: FloatArray  # fraction of Tier-1 above 200d MA, 0..1

    def __post_init__(self) -> None:
        n = len(self.dates)
        for field in ("spy_close", "hy_oas", "curve_slope", "breadth"):
            arr = np.asarray(getattr(self, field), dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"RegimeHistory.{field} must align with dates")
            object.__setattr__(self, field, arr)

    @property
    def n_sessions(self) -> int:
        return len(self.dates)


def trend_score(price: float, ma50: float, ma200: float) -> float:
    if not (math.isfinite(price) and math.isfinite(ma50) and math.isfinite(ma200)):
        return 0.0
    if price > ma200 and ma50 > ma200:
        return 2.0
    if price > ma200:
        return 1.0
    if price < ma200 and ma50 < ma200:
        return -2.0
    if price < ma200:
        return -1.0
    return 0.0


def vol_percentile_score(percentile: float) -> float:
    if not math.isfinite(percentile):
        return 0.0
    if percentile <= 0.2:
        return 2.0
    if percentile <= 0.4:
        return 1.0
    if percentile <= 0.6:
        return 0.0
    if percentile <= 0.8:
        return -1.0
    return -2.0


def credit_score(oas: float, oas_change_20d: float) -> float:
    if not math.isfinite(oas):
        return 0.0
    if oas <= 3.5:
        level = 2.0
    elif oas <= 4.5:
        level = 1.0
    elif oas <= 5.5:
        level = 0.0
    elif oas <= 7.0:
        level = -1.0
    else:
        level = -2.0
    change = 0.0
    if math.isfinite(oas_change_20d):
        if oas_change_20d >= 1.0:
            change = -1.0
        elif oas_change_20d >= 0.5:
            change = -0.5
        elif oas_change_20d <= -0.5:
            change = 0.5
    return float(min(max(level + change, -2.0), 2.0))


def curve_score(slope: float) -> float:
    if not math.isfinite(slope):
        return 0.0
    if slope >= 1.5:
        return 2.0
    if slope >= 0.5:
        return 1.0
    if slope >= 0.0:
        return 0.0
    if slope >= -0.5:
        return -1.0
    return -2.0


def breadth_score(breadth: float) -> float:
    if not math.isfinite(breadth):
        return 0.0
    if breadth >= 0.75:
        return 2.0
    if breadth >= 0.60:
        return 1.0
    if breadth >= 0.45:
        return 0.0
    if breadth >= 0.30:
        return -1.0
    return -2.0


def classify_composite(score: float) -> Regime:
    if not math.isfinite(score):
        return Regime.RISK_OFF  # fail closed
    if score <= CRISIS_THRESHOLD:
        return Regime.CRISIS
    if score <= RISK_OFF_THRESHOLD:
        return Regime.RISK_OFF
    if score >= RISK_ON_THRESHOLD:
        return Regime.RISK_ON
    return Regime.NEUTRAL


def _rolling_mean(x: FloatArray, window: int) -> FloatArray:
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    if window < 1 or x.shape[0] < window:
        return out
    kernel = np.ones(window, dtype=np.float64) / float(window)
    valid = np.convolve(x, kernel, mode="valid")
    out[window - 1 :] = valid
    return out


def _rolling_std(x: FloatArray, window: int) -> FloatArray:
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    if window < 2 or x.shape[0] < window:
        return out
    for i in range(window - 1, x.shape[0]):
        seg = x[i - window + 1 : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2:
            out[i] = float(np.std(seg, ddof=1))
    return out


def _trailing_percentile(x: FloatArray, lookback: int) -> FloatArray:
    """Midrank percentile of the current value within the trailing window.

    Midrank (ties count half) so a flat vol history scores 0.5, not 1.0.
    """
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    for i in range(x.shape[0]):
        if not np.isfinite(x[i]):
            continue
        start = max(0, i - lookback + 1)
        seg = x[start : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 20:
            current = float(x[i])
            below = float(np.mean(seg < current))
            ties = float(np.mean(np.equal(seg, current)))
            out[i] = below + 0.5 * ties
    return out


def backfill_regimes(
    history: RegimeHistory,
    config: RiskConfig,
    warmup_sessions: int = DEFAULT_WARMUP_SESSIONS,
) -> tuple[RegimeState, ...]:
    """Compute the regime series over the whole history with hysteresis.

    A candidate regime must persist ``config.hysteresis_sessions`` consecutive
    sessions before it replaces the current one (no flapping on noise).
    Returns () when history is too short — callers fail closed.
    """
    t = history.n_sessions
    if t <= warmup_sessions:
        return ()

    close = history.spy_close
    ma50 = _rolling_mean(close, 50)
    ma200 = _rolling_mean(close, 200)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.full(t, np.nan, dtype=np.float64)
        returns[1:] = close[1:] / close[:-1] - 1.0
    realized_vol = _rolling_std(returns, _VOL_WINDOW) * ANNUALIZATION
    vol_pct = _trailing_percentile(realized_vol, _VOL_PERCENTILE_LOOKBACK)

    states: list[RegimeState] = []
    current = Regime.NEUTRAL
    sessions_in_state = 0
    pending: Regime | None = None
    streak = 0
    hysteresis = max(config.hysteresis_sessions, 1)

    for i in range(warmup_sessions, t):
        oas = float(history.hy_oas[i])
        oas_change = (
            float(history.hy_oas[i] - history.hy_oas[i - _OAS_CHANGE_SESSIONS])
            if i >= _OAS_CHANGE_SESSIONS
            else float("nan")
        )
        scores: dict[str, float] = {
            "trend": trend_score(float(close[i]), float(ma50[i]), float(ma200[i])),
            "vol": vol_percentile_score(float(vol_pct[i])),
            "credit": credit_score(oas, oas_change),
            "curve": curve_score(float(history.curve_slope[i])),
            "breadth": breadth_score(float(history.breadth[i])),
        }
        composite = float(np.mean(list(scores.values())))
        candidate = classify_composite(composite)

        if candidate == current:
            pending = None
            streak = 0
            sessions_in_state += 1
        else:
            if candidate == pending:
                streak += 1
            else:
                pending = candidate
                streak = 1
            if streak >= hysteresis:
                current = candidate
                sessions_in_state = 1
                pending = None
                streak = 0
            else:
                sessions_in_state += 1

        states.append(
            RegimeState(
                as_of=history.dates[i],
                regime=current,
                composite_score=composite,
                inputs=scores,
                sessions_in_state=sessions_in_state,
            )
        )
    return tuple(states)


def latest_regime(
    history: RegimeHistory,
    config: RiskConfig,
    warmup_sessions: int = DEFAULT_WARMUP_SESSIONS,
) -> RegimeState | None:
    states = backfill_regimes(history, config, warmup_sessions)
    return states[-1] if states else None
