"""TrendRule: momentum order, return-scale values, abstain on missing."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from dynamo.research.quant.bundle import bundle_from_version, save_bundle
from dynamo.research.quant.models import AbstainModel, TrendRule
from dynamo.research.quant.scoring import score_universe


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {"mom_12_1": [0.40, -0.10, np.nan, 0.05, 0.90]},
        index=pd.Index(["AAA", "BBB", "CCC", "DDD", "EEE"]),
    )


def test_order_is_momentum_and_scale_is_a_20d_return() -> None:
    pred = TrendRule().predict(_features())
    assert np.isnan(pred[2])  # missing momentum abstains
    ok = pred[~np.isnan(pred)]
    assert list(np.argsort(-ok)) == [3, 0, 2, 1]  # EEE > AAA > DDD > BBB
    assert np.all(np.abs(ok) < 0.01)  # 0.04 x 0.08 x |z| with n=4
    assert abs(float(np.mean(ok))) < 1e-12  # symmetric around zero


def test_bundle_round_trip_scores_like_the_rule(tmp_path) -> None:
    save_bundle(
        tmp_path,
        "trend-test",
        ranker=TrendRule(),
        downside=AbstainModel(),
        vol=AbstainModel(),
        feature_names=["mom_12_1"],
        train_feature_quantiles=None,
        train_start=date(2021, 3, 1),
        train_end=date(2026, 9, 23),
        n_samples=0,
        config_hash="test",
        metrics={},
        fold_ic_dispersion=0.0,
        vol_is_log=False,
    )
    bundle = bundle_from_version("trend-test", tmp_path)
    scores = {s.symbol: s for s in score_universe(bundle, _features(), date(2026, 8, 3))}
    assert scores["EEE"].rank == 1 and scores["BBB"].rank == 4
    assert scores["CCC"].expected_excess_return is None and scores["CCC"].rank is None
    assert scores["CCC"].feature_coverage == 0.0
    assert all(s.downside_prob is None and s.vol_forecast is None for s in scores.values())
