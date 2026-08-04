"""Normalization properties, label construction, determinism (doc 04 section 10)."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.research.quant.catalog import FeatureInputs
from dynamo.research.quant.features import (
    build_features,
    build_labels,
    build_raw_features,
    normalize_features,
)
from tests.unit.quant.conftest import last_session, make_bars, session_at


@settings(max_examples=30, deadline=None)
@given(
    scale=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False),
    shift=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    seed=st.integers(0, 1000),
)
def test_normalization_scale_invariance(scale: float, shift: float, seed: int) -> None:
    """Winsorized z-scores are invariant under positive affine rescaling."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {"ret_20d": rng.normal(size=40), "rsi_14": rng.uniform(0, 100, size=40)},
        index=pd.Index([f"S{i}" for i in range(40)]),
    )
    a = normalize_features(raw)
    b = normalize_features(raw * scale + shift)
    pd.testing.assert_frame_equal(a, b, atol=1e-8, rtol=1e-6)


def test_normalization_leaves_raw_groups_untouched() -> None:
    raw = pd.DataFrame(
        {"month_of_year": [3.0, 3.0, 3.0], "hy_oas_level": [4.2, 4.2, 4.2],
         "ret_20d": [0.1, 0.0, -0.1]},
        index=pd.Index(["A", "B", "C"]),
    )
    norm = normalize_features(raw)
    assert norm["month_of_year"].tolist() == [3.0, 3.0, 3.0]
    assert norm["hy_oas_level"].tolist() == [4.2, 4.2, 4.2]
    assert abs(norm["ret_20d"].mean()) < 1e-12  # z-scored


def test_normalization_preserves_nan_and_handles_constant_columns() -> None:
    raw = pd.DataFrame(
        {"ret_20d": [0.1, np.nan, -0.1, 0.05], "vol_cc_20d": [0.2, 0.2, 0.2, 0.2]},
        index=pd.Index(list("ABCD")),
    )
    norm = normalize_features(raw)
    assert np.isnan(norm.loc["B", "ret_20d"])
    assert (norm["vol_cc_20d"] == 0.0).all()  # zero-variance -> 0, not inf


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.round(12).to_csv().encode("utf-8")).hexdigest()


def test_build_features_deterministic(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    as_of = last_session(bars)
    universe = ["AAA", "BBB", "CCC"]
    h1 = _frame_hash(build_features(inputs, as_of, universe))
    h2 = _frame_hash(build_features(inputs, as_of, universe))
    assert h1 == h2


def test_features_use_only_data_up_to_as_of(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    """Truncating bars after as_of changes nothing: no future read exists."""
    as_of = session_at(bars, -30)
    truncated = cast(pd.DataFrame, bars[bars["ts"] <= as_of])
    full = build_raw_features(inputs, as_of, ["AAA", "BBB", "CCC"])
    trunc = build_raw_features(
        FeatureInputs(bars=truncated, sector_map=inputs.sector_map), as_of, ["AAA", "BBB", "CCC"]
    )
    pd.testing.assert_frame_equal(full, trunc)


def test_build_labels_excess_downside_and_vol() -> None:
    ts = pd.bdate_range("2025-01-02", periods=30)
    rows = []
    for sym, daily in (("AAA", 0.01), ("BBB", -0.02), ("ETF", 0.0)):
        px = 100.0
        for t in ts:
            rows.append({"symbol": sym, "ts": t, "close": px, "adj_close": px, "volume": 1000})
            px *= 1.0 + daily
    bars = pd.DataFrame(rows)
    sector_map = {"AAA": "ETF", "BBB": "ETF"}
    labels = build_labels(bars, sector_map, cast(pd.Timestamp, ts[0]))
    assert set(labels.index) == {"AAA", "BBB"}
    # AAA: +1%/day for 20 sessions vs flat ETF
    expected_aaa = 1.01**20 - 1.0
    assert abs(labels.loc["AAA", "excess_return_20d"] - expected_aaa) < 1e-9
    assert labels.loc["AAA", "downside_20d"] == 0.0
    # BBB: -2%/day breaches -10% within 20 sessions
    assert labels.loc["BBB", "downside_20d"] == 1.0
    assert labels.loc["BBB", "excess_return_20d"] < -0.3
    # constant daily log-return => (near) zero realized vol
    assert labels.loc["AAA", "realized_vol_20d"] < 1e-8


def test_build_labels_insufficient_forward_history_is_nan() -> None:
    bars = make_bars(["AAA", "ETF"], n_sessions=25)
    as_of = last_session(bars)  # no forward window at all
    labels = build_labels(bars, {"AAA": "ETF"}, as_of)
    assert bool(labels["excess_return_20d"].isna().all())


def test_build_labels_missing_sector_etf_gives_nan_excess() -> None:
    bars = make_bars(["AAA"], n_sessions=60)
    labels = build_labels(bars, {"AAA": "MISSING_ETF"}, session_at(bars, 10))
    assert np.isnan(labels.loc["AAA", "excess_return_20d"])
    # downside/vol only need the symbol's own path
    assert not np.isnan(labels.loc["AAA", "realized_vol_20d"])
