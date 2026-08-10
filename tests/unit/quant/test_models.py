"""Model determinism, calibration fallback, baseline behavior (doc 04 sections 1, 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.models import (
    LightGBMModel,
    MapieCalibratedClassifier,
    RandomForestModel,
    StatsforecastBaseline,
)


def _data(n: int = 400, seed: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(x["f1"] * 2 + rng.normal(scale=0.1, size=n))
    return x, y


def test_models_satisfy_forecast_model_port() -> None:
    # ForecastModel is not @runtime_checkable; assert the structural contract.
    for model in (LightGBMModel(), RandomForestModel(), StatsforecastBaseline()):
        for method in ("fit", "predict", "feature_importances"):
            assert callable(getattr(model, method))


def test_lightgbm_regression_deterministic() -> None:
    x, y = _data()
    p1 = LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5}).fit(
        x, y, {}
    ).predict(x)
    p2 = LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5}).fit(
        x, y, {}
    ).predict(x)
    np.testing.assert_array_equal(p1, p2)
    imp = LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5}).fit(
        x, y, {}
    ).feature_importances()
    assert set(imp) == {"f1", "f2"}
    assert imp["f1"] > imp["f2"]  # planted signal dominates


def test_lightgbm_binary_platt_fallback_below_500_positives() -> None:
    x, y = _data(400)
    yb = (y > y.quantile(0.7)).astype(float)  # ~120 positives < 500
    model = LightGBMModel("binary", n_estimators=20, params={"min_child_samples": 5})
    model.fit(x, yb, {})
    assert model.calibration_method == "platt"
    probs = model.predict(x)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_lightgbm_binary_isotonic_when_enough_positives() -> None:
    x, y = _data(600)
    yb = (y > y.median()).astype(float)
    model = LightGBMModel("binary", n_estimators=20, params={"min_child_samples": 5})
    model.fit(x, yb, {"min_isotonic_positives": 50})
    assert model.calibration_method == "isotonic"


def test_lightgbm_binary_mapie_calibration_smoke() -> None:
    """cfg calibration='mapie' -> conformal (Venn-ABERS) probabilities in [0, 1]."""
    if not MapieCalibratedClassifier.available():
        pytest.skip("mapie not installed")
    x, y = _data(240)
    yb = (y > y.quantile(0.6)).astype(float)
    model = LightGBMModel("binary", n_estimators=20, params={"min_child_samples": 5})
    model.fit(x, yb, {"calibration": "mapie"})
    assert model.calibration_method == "mapie"
    probs = model.predict(x)
    assert probs.shape == (len(x),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # deterministic: same data + seed -> identical calibrated scores
    again = LightGBMModel("binary", n_estimators=20, params={"min_child_samples": 5})
    again.fit(x, yb, {"calibration": "mapie"})
    np.testing.assert_allclose(probs, again.predict(x))


def test_lightgbm_isotonic_stays_the_default() -> None:
    x, y = _data(600)
    yb = (y > y.median()).astype(float)
    model = LightGBMModel("binary", n_estimators=20, params={"min_child_samples": 5})
    model.fit(x, yb, {"min_isotonic_positives": 50})  # no calibration key
    assert model.calibration_method == "isotonic"


def test_random_forest_handles_nan_via_train_fold_medians() -> None:
    x, y = _data(200)
    x_nan = x.copy()
    x_nan.iloc[::7, 0] = np.nan
    model = RandomForestModel("regression", n_estimators=10)
    model.fit(x_nan, y, {})
    preds = model.predict(x_nan)
    assert np.isfinite(preds).all()
    p2 = RandomForestModel("regression", n_estimators=10).fit(x_nan, y, {}).predict(x_nan)
    np.testing.assert_array_equal(preds, p2)  # deterministic


def test_statsforecast_historic_average_is_per_symbol_train_mean() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=10)
    idx = pd.MultiIndex.from_product([sessions, ["AAA", "BBB"]], names=["ts", "symbol"])
    y = pd.Series([1.0, -1.0] * 10, index=idx)  # AAA always 1, BBB always -1
    x = pd.DataFrame({"f": 0.0}, index=idx)
    model = StatsforecastBaseline("historic_average")
    model.fit(x, y, {})
    preds = pd.Series(model.predict(x), index=idx)
    assert np.allclose(preds.xs("AAA", level=1), 1.0)
    assert np.allclose(preds.xs("BBB", level=1), -1.0)
    # unseen symbol falls back to global mean
    idx_new = pd.MultiIndex.from_product([sessions[:1], ["ZZZ"]], names=["ts", "symbol"])
    assert np.allclose(model.predict(pd.DataFrame({"f": 0.0}, index=idx_new)), 0.0)


def test_statsforecast_auto_ets_never_raises_on_short_series() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    idx = pd.MultiIndex.from_product([sessions, ["AAA"]], names=["ts", "symbol"])
    y = pd.Series([0.1, 0.2, 0.3], index=idx)
    x = pd.DataFrame({"f": 0.0}, index=idx)
    model = StatsforecastBaseline("auto_ets")
    model.fit(x, y, {})
    assert np.isfinite(model.predict(x)).all()
