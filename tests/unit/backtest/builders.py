"""Shared builders for backtest tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import pandas as pd

from dynamo.backtest.engine import BacktestConfig, BacktestData, SessionAdvice
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    Signal,
    SignalStatus,
)
from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.config import ConstructionConfig

START = date(2026, 1, 5)  # a Monday


def sessions(n: int) -> list[date]:
    """n weekday sessions starting at START."""
    out: list[date] = []
    d = START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def frame(rows: Mapping[str, Sequence[float]], dates: Sequence[date]) -> pd.DataFrame:
    return pd.DataFrame(dict(rows), index=pd.Index(list(dates)))


def full_universe(symbols: Sequence[str]) -> dict[str, tuple[tuple[date, None], ...]]:
    return {s: ((date(2000, 1, 1), None),) for s in symbols}


def toy_risk_config() -> RiskConfig:
    """Clean caps for hand-computed cases: 30% tactical, 10% cash floor."""
    return RiskConfig(
        per_position_cap=0.35,
        per_position_hard=0.50,
        sector_cap=1.0,
        sector_hard=1.0,
        gross_cap=1.0,
        tactical_position_cap=0.30,
        cash_floor=0.10,
        cluster_cap=1.0,
        cluster_hard=1.0,
        liquidity_adv_participation=1.0,
        liquidity_max_days=1e9,
    )


def toy_construction_config() -> ConstructionConfig:
    return ConstructionConfig(
        long_term_fraction=0.0,
        tactical_fraction=1.0,
        tactical_max=1.0,
        max_positions=10,
        max_tactical_positions=10,
        turnover_budget_monthly=1e9,
    )


def toy_backtest_config(cost_bps: float = 10.0) -> BacktestConfig:
    return BacktestConfig(initial_equity=100_000.0, flat_cost_bps=cost_bps)


def signal(symbol: str, as_of: date, strength: float = 0.5) -> Signal:
    return Signal(
        strategy_id="toy",
        symbol=symbol,
        as_of=as_of,
        strength=strength,
        status=SignalStatus.ACTIVE,
    )


def sell_verdict(symbol: str, as_of: date) -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"sell-{symbol}-{as_of.isoformat()}",
        symbol=symbol,
        as_of=as_of,
        action=Action.SELL,
        conviction=0.9,
        thesis="exit",
        bull_case="",
        bear_case="",
    )


class ScriptedProvider:
    """Advisory provider replaying a fixed per-session script."""

    def __init__(self, script: Mapping[date, SessionAdvice]) -> None:
        self.script = dict(script)

    def advise(self, as_of, visible_closes, holdings) -> SessionAdvice:
        del visible_closes, holdings
        return self.script.get(as_of, SessionAdvice())


def toy_data(
    opens: Mapping[str, Sequence[float]],
    closes: Mapping[str, Sequence[float]],
    dates: Sequence[date],
    universe: Mapping[str, tuple[tuple[date, date | None], ...]] | None = None,
) -> BacktestData:
    symbols = sorted(opens)
    return BacktestData(
        opens=frame(opens, dates),
        closes=frame(closes, dates),
        universe=dict(universe) if universe is not None else full_universe(symbols),
        adv_dollar_20d=dict.fromkeys(symbols, 1e9),
        sigma_annual_override=dict.fromkeys(symbols, 0.20),
    )
