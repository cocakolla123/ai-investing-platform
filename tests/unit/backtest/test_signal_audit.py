"""Signal audit: guarded alphalens import + smoke stats on a planted factor."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import dynamo.backtest.signal_audit as audit_mod
from dynamo.backtest.signal_audit import audit_factor

HAS_ALPHALENS = importlib.util.find_spec("alphalens") is not None


def _frames(n_days: int = 80, n_assets: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices where next-day return is (noisily) proportional to the factor."""
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    assets = [f"S{i}" for i in range(n_assets)]
    factor_rows = []
    prices = pd.DataFrame(100.0, index=days, columns=pd.Index(assets))
    factor_panel = rng.normal(size=(n_days, n_assets))
    for t in range(1, n_days):
        drift = 0.01 * factor_panel[t - 1]  # yesterday's factor drives today's return
        prices.iloc[t] = prices.iloc[t - 1].to_numpy() * (
            1.0 + drift + rng.normal(0.0, 0.002, n_assets)
        )
    for t, day in enumerate(days):
        for j, asset in enumerate(assets):
            factor_rows.append({"ts": day, "symbol": asset, "factor": factor_panel[t, j]})
    return pd.DataFrame(factor_rows), prices


class _BrokenImportlib:
    @staticmethod
    def import_module(name: str, package: str | None = None) -> object:
        raise ImportError("alphalens unavailable")


def test_audit_factor_unavailable_without_alphalens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_mod, "importlib", _BrokenImportlib)
    factor, prices = _frames(10, 3)
    result = audit_factor(factor, prices)
    assert result["available"] is False


@pytest.mark.skipif(not HAS_ALPHALENS, reason="alphalens not installed (analysis extra)")
def test_audit_factor_smoke_on_planted_signal() -> None:
    factor, prices = _frames()
    result = audit_factor(factor, prices, quantiles=3, periods=(1, 5))
    assert result["available"] is True
    assert result["n_obs"] > 0
    assert set(result["ic_mean"]) == set(result["ic_std"])
    # the planted 1-day signal shows up as positive IC and positive Q-spread
    one_day = next(k for k in result["ic_mean"] if k.startswith("1"))
    assert result["ic_mean"][one_day] > 0.1
    assert result["quantile_spread"][one_day] > 0.0
    assert np.isfinite(result["turnover_proxy"])
