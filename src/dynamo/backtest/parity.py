"""Parity harness (doc 08 section 9): event-driven vs vectorized MA-cross.

A simple moving-average-cross strategy implemented twice — once as a
vectorized reference, once through the event-driven validator via an
advisory provider — with matched flat costs and no risk constraints.
Terminal equity must agree within 0.5%.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dynamo.backtest.engine import (
    BacktestConfig,
    BacktestData,
    BacktestEngine,
    BacktestResult,
    SessionAdvice,
)
from dynamo.domain.models.research import Action, CommitteeVerdict, Signal, SignalStatus
from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.config import ConstructionConfig

PARITY_TOLERANCE = 0.005

__all__ = [
    "PARITY_TOLERANCE",
    "MACrossProvider",
    "ParityResult",
    "ma_cross_positions",
    "parity_backtest_config",
    "parity_construction_config",
    "parity_risk_config",
    "run_parity_check",
    "vectorized_ma_cross_equity",
]


# ------------------------------------------------------ vectorized reference
def ma_cross_positions(closes: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """1 when fast MA > slow MA at the close of t (decided at t), else 0."""
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()
    above = np.asarray(fast_ma > slow_ma, dtype=bool)
    return pd.Series(np.where(above, 1.0, 0.0), index=closes.index)


def vectorized_ma_cross_equity(
    opens: pd.Series,
    closes: pd.Series,
    fast: int = 5,
    slow: int = 20,
    cost_bps: float = 5.0,
    initial_equity: float = 100_000.0,
) -> pd.Series:
    """Vectorized equity curve; trades execute at the NEXT session's open.

    Session-t factor on equity (marked at closes):
      held through:            close_t / close_{t-1}
      bought at today's open:  close_t / (open_t * (1 + c))
      sold at today's open:    (open_t / close_{t-1}) * (1 - c)
      flat:                    1
    """
    c = cost_bps / 10_000.0
    pos = ma_cross_positions(closes, fast, slow).to_numpy()
    close = closes.to_numpy(dtype=np.float64)
    open_ = opens.to_numpy(dtype=np.float64)
    n = close.shape[0]
    factors = np.ones(n, dtype=np.float64)
    for t in range(1, n):
        held_now = pos[t - 1] > 0.0  # decided at close t-1, effective day t
        held_before = pos[t - 2] > 0.0 if t >= 2 else False
        if held_now and held_before:
            factors[t] = close[t] / close[t - 1]
        elif held_now and not held_before:
            factors[t] = close[t] / (open_[t] * (1.0 + c))
        elif not held_now and held_before:
            factors[t] = (open_[t] / close[t - 1]) * (1.0 - c)
    equity = initial_equity * np.cumprod(factors)
    return pd.Series(equity, index=closes.index, name="equity")


# ------------------------------------------------- event-driven counterpart
class MACrossProvider:
    """Advisory provider emitting a tactical signal while fast MA > slow MA."""

    def __init__(self, symbol: str, fast: int = 5, slow: int = 20) -> None:
        self.symbol = symbol
        self.fast = fast
        self.slow = slow

    def advise(
        self,
        as_of: date,
        visible_closes: pd.DataFrame,
        holdings: Mapping[str, float],
    ) -> SessionAdvice:
        series = visible_closes[self.symbol].dropna()
        if len(series) < self.slow:
            return SessionAdvice()
        fast_ma = float(series.iloc[-self.fast :].mean())
        slow_ma = float(series.iloc[-self.slow :].mean())
        if fast_ma > slow_ma:
            return SessionAdvice(
                signals=(
                    Signal(
                        strategy_id="ma_cross",
                        symbol=self.symbol,
                        as_of=as_of,
                        strength=0.5,  # conviction tilt 1.0: weight == budget
                        status=SignalStatus.ACTIVE,
                    ),
                )
            )
        if holdings.get(self.symbol, 0.0) > 0.0:
            # Cross turned off while held: exit via a recorded sell verdict.
            return SessionAdvice(
                verdicts=(
                    CommitteeVerdict(
                        id=f"ma-exit-{as_of.isoformat()}",
                        symbol=self.symbol,
                        as_of=as_of,
                        action=Action.SELL,
                        conviction=0.9,
                        thesis="ma cross off",
                        bull_case="",
                        bear_case="",
                    ),
                )
            )
        return SessionAdvice()


def parity_risk_config() -> RiskConfig:
    """No constraints: every cap wide open, no floors, no vol targeting."""
    return RiskConfig(
        per_position_cap=1.0,
        per_position_hard=1.0,
        sector_cap=1.0,
        sector_hard=1.0,
        gross_cap=1.0,
        tactical_position_cap=1.0,
        cash_floor=0.0,
        cluster_cap=1.0,
        cluster_hard=1.0,
        var95_cap=1.0,
        var95_hard=1.0,
        min_position_weight=0.0,
        liquidity_adv_participation=1.0,
        liquidity_max_days=1e9,
    )


def parity_construction_config() -> ConstructionConfig:
    """Tactical-only sleeve, no turnover budget, no noise floor."""
    return ConstructionConfig(
        long_term_fraction=0.0,
        tactical_fraction=1.0,
        tactical_max=1.0,
        max_positions=10,
        max_tactical_positions=10,
        turnover_budget_monthly=1e9,
        min_trade_weight=0.0,
    )


def parity_backtest_config(cost_bps: float, initial_equity: float) -> BacktestConfig:
    return BacktestConfig(
        initial_equity=initial_equity,
        flat_cost_bps=cost_bps,
        apply_ladder=False,  # parity is constraint-free by definition
    )


@dataclass(frozen=True)
class ParityResult:
    terminal_event_driven: float
    terminal_vectorized: float
    relative_difference: float
    tolerance: float
    passed: bool
    event_result: BacktestResult


def run_parity_check(
    opens: pd.Series,
    closes: pd.Series,
    *,
    symbol: str = "SPY",
    fast: int = 5,
    slow: int = 20,
    cost_bps: float = 5.0,
    initial_equity: float = 100_000.0,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityResult:
    """Run both engines on one symbol and compare terminal equity."""
    vec = vectorized_ma_cross_equity(
        opens, closes, fast, slow, cost_bps, initial_equity
    )
    first = _first_session(opens)
    data = BacktestData(
        opens=opens.to_frame(symbol),
        closes=closes.to_frame(symbol),
        universe={symbol: ((first, None),)},
        adv_dollar_20d={symbol: 1e12},
        sigma_annual_override={symbol: 0.20},
    )
    engine = BacktestEngine(
        parity_backtest_config(cost_bps, initial_equity),
        construction_config=parity_construction_config(),
        risk_config=parity_risk_config(),
    )
    result = engine.run(data, MACrossProvider(symbol, fast, slow))
    terminal_vec = float(vec.iloc[-1])
    terminal_ev = result.terminal_equity
    rel = abs(terminal_ev - terminal_vec) / terminal_vec if terminal_vec else float("inf")
    return ParityResult(
        terminal_event_driven=terminal_ev,
        terminal_vectorized=terminal_vec,
        relative_difference=rel,
        tolerance=tolerance,
        passed=rel <= tolerance,
        event_result=result,
    )


def _first_session(series: pd.Series) -> date:
    ts = series.index[0]
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, date):
        return ts
    parsed = pd.Timestamp(str(ts)).to_pydatetime()
    assert isinstance(parsed, date)
    return parsed.date()
