"""Monte Carlo risk simulation: Student-t marginals + Gaussian copula.

Design: docs/design/06-risk-engine.md section 6. 10k paths over a 20-day
horizon; reports P(loss > x), expected shortfall, and drawdown-ladder trigger
probabilities. Deterministic: the seed is a required parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.special import ndtr, stdtrit  # type: ignore

from dynamo.domain.models.risk import LadderRung
from dynamo.risk.correlation import shrunk_correlation
from dynamo.risk.ladder import RUNG_TRIGGERS
from dynamo.risk.types import FloatArray, ReturnsPanel, clean_finite
from dynamo.risk.vol import ewma_sigma_daily

DEFAULT_N_PATHS = 10_000
DEFAULT_HORIZON_DAYS = 20
DEFAULT_LOSS_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
DF_MIN = 3.0
DF_MAX = 30.0
_DF_FALLBACK = 5.0  # fat-tailed default when a fit is not possible (conservative)


@dataclass(frozen=True)
class MonteCarloResult:
    n_paths: int
    horizon_days: int
    seed: int
    loss_probabilities: dict[float, float]  # threshold -> P(loss > threshold)
    expected_shortfall_95: float  # mean loss of the worst 5% terminal outcomes
    rung_hit_probabilities: dict[LadderRung, float]
    df_by_symbol: dict[str, float]


def fit_student_df(returns: FloatArray, low: float = DF_MIN, high: float = DF_MAX) -> float:
    """Method-of-moments Student-t degrees of freedom, clamped to [low, high].

    Excess kurtosis g2 = 6 / (df - 4) => df = 4 + 6 / g2. Thin-tailed or
    unusable samples map to the upper clamp; tiny samples fail conservative
    (fat-tailed fallback).
    """
    r = clean_finite(returns)
    if r.size < 30:
        return float(min(max(_DF_FALLBACK, low), high))
    mu = float(np.mean(r))
    m2 = float(np.mean((r - mu) ** 2))
    if m2 <= 0.0:
        return high
    m4 = float(np.mean((r - mu) ** 4))
    g2 = m4 / (m2 * m2) - 3.0
    if g2 <= 1e-9:
        return high
    df = 4.0 + 6.0 / g2
    return float(min(max(df, low), high))


def _safe_cholesky(corr: FloatArray) -> FloatArray:
    """Cholesky factor with diagonal jitter fallback for near-singular matrices."""
    n = corr.shape[0]
    jitter = 0.0
    for _ in range(6):
        try:
            return np.asarray(
                np.linalg.cholesky(corr + jitter * np.eye(n, dtype=np.float64)),
                dtype=np.float64,
            )
        except np.linalg.LinAlgError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 100.0
    return np.eye(n, dtype=np.float64)


def run_monte_carlo(
    panel: ReturnsPanel,
    weights: Mapping[str, float],
    *,
    equity: float,
    high_water_mark: float,
    seed: int,
    n_paths: int = DEFAULT_N_PATHS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    ewma_span: int = 60,
    loss_grid: Sequence[float] = DEFAULT_LOSS_GRID,
) -> MonteCarloResult | None:
    """Simulate 20d portfolio paths; None when inputs are unusable (fail closed).

    Marginals are standardized Student-t (per-symbol df) scaled by EWMA daily
    vol; dependence comes from a Gaussian copula over the shrunk EWMA
    correlation matrix.
    """
    if not (np.isfinite(equity) and equity > 0.0) or n_paths < 1 or horizon_days < 1:
        return None

    symbols: list[str] = []
    w_list: list[float] = []
    for symbol, weight in weights.items():
        if weight == 0.0:
            continue
        if not np.isfinite(weight) or panel.index_of(symbol) is None:
            return None
        symbols.append(symbol)
        w_list.append(float(weight))
    if not symbols:
        return None

    sub = panel.subset(symbols)
    sigmas: list[float] = []
    dfs: list[float] = []
    for i in range(sub.n_symbols):
        col = np.asarray(sub.values[:, i], dtype=np.float64)
        sigma = ewma_sigma_daily(col, ewma_span)
        if not (np.isfinite(sigma) and sigma >= 0.0):
            return None
        sigmas.append(sigma)
        dfs.append(fit_student_df(col))

    corr = shrunk_correlation(sub)
    chol = _safe_cholesky(corr)
    sigma_arr = np.asarray(sigmas, dtype=np.float64)
    df_arr = np.asarray(dfs, dtype=np.float64)
    w_arr = np.asarray(w_list, dtype=np.float64)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon_days, len(symbols)))
    z_corr = z @ chol.T
    u = np.clip(cast(FloatArray, ndtr(z_corr)), 1e-9, 1.0 - 1e-9)
    t_draws = cast(FloatArray, stdtrit(df_arr, u))
    # Standardize t to unit variance (df > 2 guaranteed by the clamp).
    t_std = t_draws / np.sqrt(df_arr / (df_arr - 2.0))
    asset_returns = t_std * sigma_arr
    port_returns = np.clip(asset_returns @ w_arr, -0.99, None)

    growth = np.cumprod(1.0 + port_returns, axis=1)
    equity_paths = equity * growth
    terminal_loss = 1.0 - growth[:, -1]

    loss_probabilities = {
        float(x): float(np.mean(terminal_loss > float(x))) for x in loss_grid
    }
    tail_n = max(1, round(0.05 * n_paths))
    worst = np.sort(terminal_loss)[-tail_n:]
    expected_shortfall_95 = float(np.mean(worst))

    hwm = high_water_mark if np.isfinite(high_water_mark) and high_water_mark > 0 else equity
    hwm = max(hwm, equity)
    min_equity = np.min(equity_paths, axis=1)
    path_dd = min_equity / hwm - 1.0
    rung_hit = {
        rung: float(np.mean(path_dd <= trigger))
        for rung, trigger in RUNG_TRIGGERS.items()
    }

    return MonteCarloResult(
        n_paths=n_paths,
        horizon_days=horizon_days,
        seed=seed,
        loss_probabilities=loss_probabilities,
        expected_shortfall_95=expected_shortfall_95,
        rung_hit_probabilities=rung_hit,
        df_by_symbol=dict(zip(symbols, dfs, strict=True)),
    )
