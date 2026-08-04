"""Shared builders for risk-engine tests. Everything seeded — no wall clock."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pytest

from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.regime import RegimeHistory
from dynamo.risk.types import FloatArray, ReturnsPanel, SymbolBars


def make_dates(n: int, end: date = date(2024, 8, 1)) -> tuple[date, ...]:
    return tuple(end - timedelta(days=n - 1 - i) for i in range(n))


def make_panel(
    seed: int,
    symbols: Sequence[str],
    sessions: int,
    end: date = date(2024, 8, 1),
    mu: float = 0.0004,
    sigma: float = 0.012,
) -> ReturnsPanel:
    rng = np.random.default_rng(seed)
    values = rng.normal(mu, sigma, size=(sessions, len(symbols)))
    return ReturnsPanel(make_dates(sessions, end), tuple(symbols), values)


def make_bars(seed: int, sessions: int = 60, price: float = 50.0) -> SymbolBars:
    rng = np.random.default_rng(seed)
    close: FloatArray = price + np.cumsum(rng.normal(0.0, 0.5, sessions))
    high = close + np.abs(rng.normal(0.5, 0.2, sessions))
    low = close - np.abs(rng.normal(0.5, 0.2, sessions))
    return SymbolBars(make_dates(sessions), high, low, close)


def make_regime_history(
    sessions: int = 300,
    end: date = date(2024, 8, 1),
    seed: int = 3,
    drift: float = 0.0005,
    vol: float = 0.008,
    oas: float = 4.0,
    slope: float = 1.0,
    breadth: float = 0.65,
) -> RegimeHistory:
    rng = np.random.default_rng(seed)
    close = 400.0 * np.cumprod(1.0 + rng.normal(drift, vol, sessions))
    return RegimeHistory(
        dates=make_dates(sessions, end),
        spy_close=close,
        hy_oas=np.full(sessions, oas),
        curve_slope=np.full(sessions, slope),
        breadth=np.full(sessions, breadth),
    )


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig()
