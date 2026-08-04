"""EWMA correlation with Ledoit-Wolf shrinkage and hierarchical clusters.

Design: docs/design/06-risk-engine.md section 7. Ten "different" AI names are
one bet: clusters at the configured rho threshold become first-class limits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore
from sklearn.covariance import LedoitWolf  # type: ignore

from dynamo.risk.types import FloatArray, ReturnsPanel

DEFAULT_HALFLIFE_DAYS = 120.0
_FALLBACK_SHRINKAGE = 0.10


def ewma_correlation(
    panel: ReturnsPanel, halflife: float = DEFAULT_HALFLIFE_DAYS
) -> FloatArray:
    """EWMA correlation matrix (halflife-weighted, most recent heaviest).

    Non-finite returns are treated as zero; degenerate (zero-variance) symbols
    get zero off-diagonal correlation. Always returns an (N, N) matrix with
    unit diagonal, clipped to [-1, 1].
    """
    n = panel.n_symbols
    identity = np.eye(n, dtype=np.float64)
    if n == 0:
        return identity
    t = panel.n_sessions
    if t < 2 or halflife <= 0.0:
        return identity
    x = np.where(np.isfinite(panel.values), panel.values, 0.0)
    lam = 0.5 ** (1.0 / halflife)
    ages = np.arange(t - 1, -1, -1, dtype=np.float64)
    w = lam**ages
    w = w / float(w.sum())
    mu = w @ x
    centered = x - mu
    cov = (centered * w[:, None]).T @ centered
    diag = np.clip(np.diag(cov), 0.0, None)
    scale = np.sqrt(diag)
    denom = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, cov / denom, 0.0)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=np.float64)


def ledoit_wolf_shrinkage_intensity(panel: ReturnsPanel) -> float:
    """Ledoit-Wolf shrinkage intensity estimated from the (finite-row) history."""
    if panel.n_symbols < 2 or panel.n_sessions < 4:
        return _FALLBACK_SHRINKAGE
    x = panel.values
    finite_rows = np.all(np.isfinite(x), axis=1)
    x = x[finite_rows, :]
    if x.shape[0] < 4:
        return _FALLBACK_SHRINKAGE
    try:
        model = LedoitWolf().fit(x)  # pyright: ignore[reportUnknownMemberType]
        delta = float(
            cast(float, model.shrinkage_)  # pyright: ignore[reportUnknownMemberType]
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return _FALLBACK_SHRINKAGE
    if not np.isfinite(delta):
        return _FALLBACK_SHRINKAGE
    return float(min(max(delta, 0.0), 1.0))


def shrunk_correlation(
    panel: ReturnsPanel, halflife: float = DEFAULT_HALFLIFE_DAYS
) -> FloatArray:
    """EWMA correlation blended toward identity with LW intensity (stability)."""
    corr = ewma_correlation(panel, halflife)
    n = corr.shape[0]
    if n == 0:
        return corr
    delta = ledoit_wolf_shrinkage_intensity(panel)
    out = (1.0 - delta) * corr + delta * np.eye(n, dtype=np.float64)
    out = np.clip(out, -1.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return np.asarray(out, dtype=np.float64)


def correlation_clusters(
    corr: FloatArray, symbols: Sequence[str], threshold: float
) -> dict[str, str]:
    """Hierarchical (average-linkage) clusters at the rho threshold.

    Distance is 1 - rho; symbols whose average-linkage distance is below
    (1 - threshold) share a cluster. Returns {symbol: cluster_id}.
    """
    n = len(symbols)
    if n == 0:
        return {}
    if n == 1:
        return {symbols[0]: "c1"}
    c = np.where(np.isfinite(corr), corr, 0.0)
    c = np.clip(c, -1.0, 1.0)
    dist = 1.0 - c
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = cast(FloatArray, squareform(dist, checks=False))
    link = cast(FloatArray, linkage(condensed, method="average"))
    cut = max(1.0 - float(threshold), 1e-9)
    labels = cast(
        NDArray[np.int_], fcluster(link, t=cut, criterion="distance")
    )
    return {
        symbol: f"c{int(label)}"
        for symbol, label in zip(symbols, labels.tolist(), strict=True)
    }
