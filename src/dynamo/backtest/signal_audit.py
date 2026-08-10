"""Alphalens factor audit — the standard audition for any NEW signal idea.

Before a candidate signal/factor earns a strategy definition (doc 05) or a
walk-forward run (doc 08), it goes through :func:`audit_factor` once, from a
test or notebook::

    from dynamo.backtest.signal_audit import audit_factor

    stats = audit_factor(factor_frame, prices_frame)
    # {'available': True, 'n_obs': ..., 'ic_mean': {'1D': ...}, 'ic_std': {...},
    #  'quantile_spread': {...}, 'rank_autocorrelation': ..., 'turnover_proxy': ...}

Inputs:

* ``factor_frame`` — long frame with columns ``ts``, ``symbol``, ``factor``
  (or a (ts, symbol)-MultiIndexed Series/single-column frame).
* ``prices_frame`` — wide close prices: DatetimeIndex rows, one column per
  symbol (canonical closes from ``MarketDataRepository.bars_frame`` pivoted).

Pass gate rule of thumb (doc 04 §6 mirrors it): mean IC positive with
consistent sign, monotone quantile spread, turnover you can afford. This
module only reports numbers — the ship gates stay in training/promotion.

``alphalens-reloaded`` ships in the optional ``analysis`` extra; without it
the function returns ``{"available": False, ...}`` instead of raising, so
auditions degrade gracefully on a slim install.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

__all__ = ["audit_factor"]

DEFAULT_QUANTILES = 5
DEFAULT_PERIODS = (1, 5, 20)


def _as_factor_series(factor_frame: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize the accepted factor shapes to a (ts, symbol)-indexed Series."""
    if isinstance(factor_frame, pd.Series):
        series = factor_frame.copy()
    elif isinstance(factor_frame.index, pd.MultiIndex):
        series = factor_frame.iloc[:, 0].copy()
    else:
        frame = factor_frame.rename(columns=str)
        series = frame.set_index(["ts", "symbol"])["factor"]
    series = pd.Series(np.asarray(series, dtype=float), index=series.index)
    ts = pd.to_datetime(series.index.get_level_values(0))
    symbols = series.index.get_level_values(1)
    series.index = pd.MultiIndex.from_arrays([ts, symbols], names=["date", "asset"])
    return series.sort_index()


def audit_factor(
    factor_frame: pd.DataFrame | pd.Series,
    prices_frame: pd.DataFrame,
    *,
    quantiles: int = DEFAULT_QUANTILES,
    periods: tuple[int, ...] = DEFAULT_PERIODS,
) -> dict[str, Any]:
    """Alphalens summary stats for one factor (see module docstring for shapes).

    Returns a JSON-friendly dict: per-period IC mean/std, top-minus-bottom
    quantile spreads, and a turnover proxy (1 - mean rank autocorrelation).
    ``{"available": False}`` when alphalens is not installed or preparation
    fails (e.g. every row dropped by the forward-returns join).
    """
    try:
        utils = importlib.import_module("alphalens.utils")
        performance = importlib.import_module("alphalens.performance")
    except Exception:
        log.info(
            "signal_audit_skipped",
            reason="alphalens-reloaded not installed (uv sync --extra analysis)",
        )
        return {"available": False, "reason": "alphalens not installed"}

    factor = _as_factor_series(factor_frame)
    prices = prices_frame.copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    try:
        clean = utils.get_clean_factor_and_forward_returns(
            factor,
            prices,
            quantiles=quantiles,
            periods=periods,
            max_loss=1.0,
        )
        ic = performance.factor_information_coefficient(clean)
        mean_q, _ = performance.mean_return_by_quantile(clean)
        autocorr = performance.factor_rank_autocorrelation(clean)
    except Exception as exc:
        log.warning("signal_audit_failed", error=str(exc))
        return {"available": False, "reason": str(exc)}

    spread = {
        str(col): float(mean_q[col].loc[mean_q.index.max()] - mean_q[col].loc[mean_q.index.min()])
        for col in mean_q.columns
    }
    rank_ac = float(np.nanmean(autocorr.to_numpy(dtype=float))) if len(autocorr) else float("nan")
    result: dict[str, Any] = {
        "available": True,
        "n_obs": len(clean),
        "quantiles": quantiles,
        "ic_mean": {str(c): float(ic[c].mean()) for c in ic.columns},
        "ic_std": {str(c): float(ic[c].std(ddof=1)) for c in ic.columns},
        "quantile_spread": spread,
        "rank_autocorrelation": rank_ac,
        # High autocorrelation = stable ranks = low turnover.
        "turnover_proxy": float(1.0 - rank_ac) if np.isfinite(rank_ac) else float("nan"),
    }
    return result
