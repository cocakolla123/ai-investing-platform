"""EWMA correlation, LW shrinkage, and hierarchical clusters."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from risk_helpers import make_dates, make_panel

from dynamo.risk.correlation import (
    correlation_clusters,
    ewma_correlation,
    ledoit_wolf_shrinkage_intensity,
    shrunk_correlation,
)
from dynamo.risk.types import ReturnsPanel


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=9999),
    sessions=st.integers(min_value=2, max_value=300),
    n=st.integers(min_value=1, max_value=6),
)
def test_correlation_matrix_is_well_formed(seed: int, sessions: int, n: int) -> None:
    symbols = tuple(f"S{i}" for i in range(n))
    panel = make_panel(seed, symbols, sessions)
    for corr in (ewma_correlation(panel), shrunk_correlation(panel)):
        assert corr.shape == (n, n)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T, atol=1e-12)
        assert np.all(corr >= -1.0 - 1e-12) and np.all(corr <= 1.0 + 1e-12)


def test_correlated_pair_clusters_together_independent_apart() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.01, 500)
    twin = base + rng.normal(0.0, 0.001, 500)  # rho ~ 0.995
    other = rng.normal(0.0, 0.01, 500)
    values = np.column_stack([base, twin, other])
    panel = ReturnsPanel(make_dates(500), ("AI1", "AI2", "UTIL"), values)
    corr = shrunk_correlation(panel)
    clusters = correlation_clusters(corr, panel.symbols, 0.7)
    assert clusters["AI1"] == clusters["AI2"]
    assert clusters["UTIL"] != clusters["AI1"]


def test_cluster_edge_cases() -> None:
    assert correlation_clusters(np.eye(0), (), 0.7) == {}
    assert correlation_clusters(np.eye(1), ("ONLY",), 0.7) == {"ONLY": "c1"}


def test_nan_columns_do_not_break_correlation() -> None:
    values = np.full((100, 2), np.nan)
    panel = ReturnsPanel(make_dates(100), ("A", "B"), values)
    corr = shrunk_correlation(panel)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(np.isfinite(corr))


def test_shrinkage_intensity_bounded() -> None:
    panel = make_panel(7, ("A", "B", "C"), 300)
    delta = ledoit_wolf_shrinkage_intensity(panel)
    assert 0.0 <= delta <= 1.0
    tiny = make_panel(7, ("A", "B"), 3)
    assert 0.0 <= ledoit_wolf_shrinkage_intensity(tiny) <= 1.0


def test_shrinkage_pulls_offdiagonal_toward_zero() -> None:
    rng = np.random.default_rng(13)
    base = rng.normal(0.0, 0.01, 400)
    values = np.column_stack([base, base + rng.normal(0, 0.004, 400)])
    panel = ReturnsPanel(make_dates(400), ("A", "B"), values)
    raw = ewma_correlation(panel)
    shrunk = shrunk_correlation(panel)
    assert abs(shrunk[0, 1]) <= abs(raw[0, 1]) + 1e-12
