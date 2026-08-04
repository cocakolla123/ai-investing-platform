"""Volatility estimators: EWMA sigma, portfolio realized vol, ATR(14).

Design: docs/design/06-risk-engine.md sections 4 and 5.
Pure functions; NaN/inf inputs degrade to NaN results, never exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dynamo.risk.types import (
    ANNUALIZATION,
    FloatArray,
    ReturnsPanel,
    SymbolBars,
    clean_finite,
)


def ewma_sigma_daily(returns: FloatArray, span: int) -> float:
    """EWMA standard deviation (daily) of a return series with the given span.

    alpha = 2 / (span + 1); weights decay backwards from the most recent
    observation. Returns NaN when the series is unusable.
    """
    r = clean_finite(returns)
    if span <= 1 or r.size < 2:
        return float("nan")
    alpha = 2.0 / (float(span) + 1.0)
    ages = np.arange(r.size - 1, -1, -1, dtype=np.float64)
    weights = (1.0 - alpha) ** ages
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return float("nan")
    weights = weights / total
    mu = float(np.dot(weights, r))
    var = float(np.dot(weights, (r - mu) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def ewma_sigma_annual(returns: FloatArray, span: int) -> float:
    return ewma_sigma_daily(returns, span) * ANNUALIZATION


def ewma_vol_by_symbol(panel: ReturnsPanel, span: int) -> dict[str, float]:
    """Annualized EWMA vol per symbol. Unusable series map to NaN."""
    out: dict[str, float] = {}
    for i, symbol in enumerate(panel.symbols):
        out[symbol] = ewma_sigma_annual(
            np.asarray(panel.values[:, i], dtype=np.float64), span
        )
    return out


def portfolio_return_series(
    panel: ReturnsPanel, weights: Mapping[str, float]
) -> FloatArray | None:
    """Full-revaluation portfolio daily returns R @ w.

    Returns None if any weighted symbol is missing from the panel or a weight
    is non-finite (fail-closed: caller treats as not computable).
    """
    symbols: list[str] = []
    w_list: list[float] = []
    for symbol, weight in weights.items():
        if weight == 0.0:
            continue
        if not np.isfinite(weight):
            return None
        if panel.index_of(symbol) is None:
            return None
        symbols.append(symbol)
        w_list.append(float(weight))
    if not symbols:
        return np.zeros(panel.n_sessions, dtype=np.float64)
    sub = panel.subset(symbols)
    values = np.where(np.isfinite(sub.values), sub.values, 0.0)
    w = np.asarray(w_list, dtype=np.float64)
    return np.asarray(values @ w, dtype=np.float64)


def portfolio_realized_vol_annual(
    panel: ReturnsPanel, weights: Mapping[str, float], window: int = 20
) -> float:
    """Annualized realized vol of the weighted portfolio over the trailing window."""
    series = portfolio_return_series(panel, weights)
    if series is None or window < 2:
        return float("nan")
    tail = clean_finite(series[-window:])
    if tail.size < 2:
        return float("nan")
    return float(np.std(tail, ddof=1)) * ANNUALIZATION


def atr(bars: SymbolBars, period: int = 14) -> float:
    """Wilder's Average True Range over the bar history (last value).

    Returns NaN when fewer than two usable bars exist.
    """
    if period < 1 or bars.n_sessions < 2:
        return float("nan")
    high = bars.high[1:]
    low = bars.low[1:]
    prev_close = bars.close[:-1]
    tr = np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    tr = tr[np.isfinite(tr)]
    if tr.size == 0:
        return float("nan")
    if tr.size <= period:
        return float(np.mean(tr))
    value = float(np.mean(tr[:period]))
    alpha = 1.0 / float(period)
    for x in tr[period:]:
        value = value + alpha * (float(x) - value)
    return value
