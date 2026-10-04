"""Shadow scoring: rows for each bundle, failures reported not raised."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from dynamo.research.quant.models import AbstainModel, TrendRule
from dynamo.research.quant.scoring import ModelBundle
from dynamo.research.quant.shadow import score_shadow_bundles, shadow_bundle_names

AS_OF = date(2026, 10, 5)


def _trend_bundle(name: str) -> ModelBundle:
    if name != "trend-rule-v1":
        raise FileNotFoundError(name)
    return ModelBundle(
        model_version=name,
        ranker=TrendRule(),
        downside=AbstainModel(),
        vol=AbstainModel(),
        feature_names=("mom_12_1",),
        vol_is_log=False,
    )


def test_names_parse_and_empty_disables() -> None:
    assert shadow_bundle_names(" trend-rule-v1 , x ,") == ["trend-rule-v1", "x"]
    assert shadow_bundle_names("") == []


def test_scores_each_bundle_and_isolates_failures() -> None:
    features = pd.DataFrame(
        {"mom_12_1": [0.3, -0.1, np.nan]}, index=pd.Index(["AAA", "BBB", "CCC"])
    )
    rows, errors = score_shadow_bundles(
        features, AS_OF, ["trend-rule-v1", "missing-bundle"], _trend_bundle
    )
    assert {r["model_version"] for r in rows} == {"trend-rule-v1"}
    assert {r["symbol"] for r in rows} == {"AAA", "BBB", "CCC"}
    assert next(r for r in rows if r["symbol"] == "AAA")["rank"] == 1
    assert set(errors) == {"missing-bundle"}
