"""Crypto-like symbols through the feature/scoring path (ADR-0013).

Crypto pairs have bars but no fundamentals / insider / earnings data — the
feature builder must NaN those groups out (never raise) and the scoring path
must reflect the holes in ``feature_coverage`` while the bars-driven
momentum/vol/volume features still work.
"""

from __future__ import annotations

import numpy as np

from dynamo.research.quant.catalog import FeatureInputs
from dynamo.research.quant.features import build_features, build_raw_features
from dynamo.research.quant.scoring import ModelBundle, score_universe
from tests.unit.quant.conftest import last_session, make_bars

UNIVERSE = ["AAA", "BBB", "BTC/USD"]


def _inputs() -> FeatureInputs:
    # Crypto-like fixture: bars for everyone (the pair included), zero
    # fundamentals/insider/earnings anywhere — exactly what a crypto symbol
    # contributes to the nightly panel.
    return FeatureInputs(
        bars=make_bars(UNIVERSE, n_sessions=300),
        sector_map={"AAA": "XLK", "BBB": "XLK"},  # crypto has no sector ETF
    )


def test_build_features_tolerates_crypto_symbols() -> None:
    inputs = _inputs()
    as_of = last_session(inputs.bars)  # type: ignore[arg-type]
    features = build_features(inputs, as_of, UNIVERSE)
    assert list(features.index) == UNIVERSE  # crypto row present, no raise
    raw = build_raw_features(inputs, as_of, UNIVERSE)
    crypto_row = raw.loc["BTC/USD"]
    # Bars-driven momentum/vol/volume features exist for the pair...
    for feature in ("ret_20d", "vol_cc_20d", "dollar_vol_20d"):
        assert np.isfinite(crypto_row[feature]), feature
    # ...while fundamentals-driven features NaN out by design.
    for feature in ("earnings_yield", "rev_growth_yoy", "gross_margin_ttm"):
        assert np.isnan(crypto_row[feature]), feature


def test_score_universe_reports_partial_coverage_for_crypto() -> None:
    inputs = _inputs()
    as_of = last_session(inputs.bars)  # type: ignore[arg-type]
    features = build_features(inputs, as_of, UNIVERSE)
    scores = {
        s.symbol: s
        for s in score_universe(ModelBundle(model_version="baseline-v0"), features, as_of)
    }
    crypto = scores["BTC/USD"]
    assert 0.0 < crypto.feature_coverage < 1.0  # holes reflected, not fatal
