"""score_universe: determinism, ranks, confidence heuristic, coverage clamp."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from dynamo.research.quant.models import LightGBMModel
from dynamo.research.quant.scoring import ModelBundle, score_universe


def _bundle(train_n: int = 300, seed: int = 9) -> tuple[ModelBundle, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"f1": rng.normal(size=train_n), "f2": rng.normal(size=train_n)})
    y = pd.Series(x["f1"] + 0.1 * rng.normal(size=train_n))
    yb = (y < y.quantile(0.2)).astype(float)
    ranker = LightGBMModel("regression", n_estimators=15, params={"min_child_samples": 5}).fit(
        x, y, {}
    )
    downside = LightGBMModel("binary", n_estimators=15, params={"min_child_samples": 5}).fit(
        x, yb, {}
    )
    quantiles = pd.DataFrame(
        {c: [x[c].quantile(0.01), x[c].quantile(0.99)] for c in x.columns},
        index=pd.Index(["q01", "q99"]),
    )
    bundle = ModelBundle(
        model_version="ranker-lgbm-test-a",
        ranker=ranker,
        downside=downside,
        vol=None,
        train_feature_quantiles=quantiles,
        fold_ic_dispersion=0.1,
    )
    features = pd.DataFrame(
        {"f1": [1.0, -1.0, 0.2, np.nan], "f2": [0.0, 0.5, -0.3, np.nan]},
        index=pd.Index(["AAA", "BBB", "CCC", "DDD"]),
    )
    return bundle, features


def test_score_universe_shape_and_ranks() -> None:
    bundle, features = _bundle()
    scores = score_universe(bundle, features, date(2026, 8, 3))
    assert [s.symbol for s in scores] == ["AAA", "BBB", "CCC", "DDD"]
    ranks = [s.rank for s in scores if s.rank is not None]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))  # 1 = best, dense, unique
    by_symbol = {s.symbol: s for s in scores}
    # f1 drives the label, so AAA (f1=1.0) must outrank BBB (f1=-1.0).
    assert by_symbol["AAA"].rank is not None and by_symbol["BBB"].rank is not None
    assert by_symbol["AAA"].rank < by_symbol["BBB"].rank
    assert by_symbol["AAA"].model_version == "ranker-lgbm-test-a"
    for s in scores:
        if s.downside_prob is not None:
            assert 0.0 <= s.downside_prob <= 1.0
        assert 0.0 <= s.confidence <= 1.0
        assert 0.0 <= s.feature_coverage <= 1.0
    assert by_symbol["DDD"].feature_coverage == 0.0


def test_missing_over_30pct_clamps_confidence() -> None:
    bundle, features = _bundle()
    scores = {s.symbol: s for s in score_universe(bundle, features, date(2026, 8, 3))}
    # DDD is missing 100% of features -> confidence clamped low.
    assert scores["DDD"].confidence <= 0.2
    # AAA is fully populated and in-distribution -> higher confidence.
    assert scores["AAA"].confidence > scores["DDD"].confidence


def test_out_of_distribution_rows_lose_confidence() -> None:
    bundle, _ = _bundle()
    inlier = pd.DataFrame({"f1": [0.0], "f2": [0.0]}, index=pd.Index(["IN"]))
    outlier = pd.DataFrame({"f1": [50.0], "f2": [-50.0]}, index=pd.Index(["OUT"]))
    s_in = score_universe(bundle, inlier, date(2026, 8, 3))[0]
    s_out = score_universe(bundle, outlier, date(2026, 8, 3))[0]
    assert s_out.confidence < s_in.confidence


def test_score_universe_deterministic() -> None:
    bundle, features = _bundle()

    def run_hash() -> str:
        rows = [s.model_dump_json() for s in score_universe(bundle, features, date(2026, 8, 3))]
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    assert run_hash() == run_hash()


def test_empty_universe_and_missing_models() -> None:
    bundle, _ = _bundle()
    assert score_universe(bundle, pd.DataFrame(), date(2026, 8, 3)) == []
    no_models = ModelBundle(model_version="x")
    features = pd.DataFrame({"f1": [1.0]}, index=pd.Index(["AAA"]))
    scores = score_universe(no_models, features, date(2026, 8, 3))
    assert scores[0].expected_excess_return is None
    assert scores[0].rank is None


# -- AveragingEnsemble (seed ensembling, --ensemble-seeds) --------------------


class _ConstModel:
    """ForecastModel-shaped fake predicting a constant; records fit calls."""

    def __init__(self, value: float, importances: dict[str, float] | None = None) -> None:
        self.value = value
        self.importances = dict(importances or {})
        self.fit_calls: list[tuple[int, dict]] = []

    def fit(self, x, y, cfg):  # type: ignore[no-untyped-def]
        self.fit_calls.append((len(x), dict(cfg or {})))
        return self

    def predict(self, x):  # type: ignore[no-untyped-def]
        return np.full(len(x), self.value)

    def feature_importances(self) -> dict[str, float]:
        return dict(self.importances)


def test_averaging_ensemble_predicts_member_mean() -> None:
    from dynamo.research.quant.scoring import AveragingEnsemble

    ensemble = AveragingEnsemble([_ConstModel(1.0), _ConstModel(3.0), _ConstModel(5.0)])
    x = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    np.testing.assert_allclose(ensemble.predict(x), np.full(4, 3.0))


def test_averaging_ensemble_fit_delegates_to_every_member() -> None:
    from dynamo.research.quant.scoring import AveragingEnsemble

    members = [_ConstModel(1.0), _ConstModel(2.0)]
    ensemble = AveragingEnsemble(members)
    x = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0])
    assert ensemble.fit(x, y, {"k": "v"}) is ensemble
    assert all(m.fit_calls == [(3, {"k": "v"})] for m in members)


def test_averaging_ensemble_importances_average_across_members() -> None:
    from dynamo.research.quant.scoring import AveragingEnsemble

    ensemble = AveragingEnsemble(
        [_ConstModel(0.0, {"a": 1.0, "b": 0.0}), _ConstModel(0.0, {"a": 0.0, "b": 0.5})]
    )
    assert ensemble.feature_importances() == {"a": 0.5, "b": 0.25}


def test_averaging_ensemble_refuses_empty_members() -> None:
    import pytest

    from dynamo.research.quant.scoring import AveragingEnsemble

    with pytest.raises(ValueError, match="at least one member"):
        AveragingEnsemble([])


def test_single_member_ensemble_matches_wrapped_model() -> None:
    from dynamo.research.quant.scoring import AveragingEnsemble

    bundle, features = _bundle()
    assert bundle.ranker is not None
    wrapped = AveragingEnsemble([bundle.ranker])
    np.testing.assert_array_equal(
        wrapped.predict(features.fillna(0.0)),
        np.asarray(bundle.ranker.predict(features.fillna(0.0)), dtype=float),
    )
