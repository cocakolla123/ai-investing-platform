"""Portfolio read endpoints: positions, performance, attribution."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

from fastapi import APIRouter

from dynamo.api.deps import DepsDep
from dynamo.api.envelopes import AsOf, Page

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

TRADING_DAYS = 252


def _sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return mean / std * math.sqrt(TRADING_DAYS)


def _max_drawdown(equity: list[float]) -> float:
    peak, max_dd = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd


@router.get("/positions")
def get_positions(deps: DepsDep) -> Page[dict[str, Any]]:
    """Open positions with stops and decision-context refs (doc 11 §1.1)."""
    return Page.of(deps.portfolio.positions())


@router.get("/performance")
def get_performance(deps: DepsDep) -> AsOf[dict[str, Any]]:
    series = deps.portfolio.equity_series()
    equity = [row["equity"] for row in series]
    returns = [b / a - 1.0 for a, b in pairwise(equity) if a > 0]
    data = {
        "series": series,
        "sharpe": _sharpe(returns),
        "max_drawdown": _max_drawdown(equity),
    }
    return AsOf(data=data, as_of=series[-1]["as_of"] if series else None)


@router.get("/attribution")
def get_attribution(deps: DepsDep) -> Page[dict[str, Any]]:
    """Per-sleeve / per-strategy cash-flow attribution from trades + lots."""
    return Page.of(deps.portfolio.attribution())
