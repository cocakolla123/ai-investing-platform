"""Look-ahead audit test — doc 04 sections 2.1 and 10.

Planted-signal synthetic data: features at t predict the label at t by
construction. Shifting all features forward one day destroys the alignment,
so a leak-free pipeline's OOS rank IC must collapse. A feature set that still
"predicts" after the shift contains leakage.
"""

from __future__ import annotations

from dynamo.backtest.walkforward import WalkForwardConfig
from dynamo.research.quant.models import LightGBMModel
from dynamo.research.quant.training import train_and_evaluate
from tests.unit.quant.conftest import planted_panel

CFG = WalkForwardConfig(
    train_sessions=200, test_sessions=60, purge_sessions=20, embargo_sessions=10, roll_sessions=60
)

COLLAPSE_RATIO = 0.25


def _pooled_ic(features, label) -> float:
    res = train_and_evaluate(
        features,
        label,
        task="rank",
        model_factories={
            "lgbm": lambda: LightGBMModel(
                "regression", n_estimators=20, seed=42, params={"min_child_samples": 5}
            )
        },
        baseline_factories={},
        wf_config=CFG,
    )
    return float(res["pooled"]["lgbm"]["rank_ic_pooled"])


def test_shifted_features_collapse_rank_ic() -> None:
    features, label, _ = planted_panel(seed=3)

    ic_unshifted = _pooled_ic(features, label)
    assert ic_unshifted > 0.8  # strong planted signal, fixed seed

    # Shift all features forward +1 session per symbol: feature date t now
    # carries t-1 information, against unshifted labels.
    shifted = features.groupby(level="symbol", group_keys=False).shift(1)

    ic_shifted = _pooled_ic(shifted, label)
    assert abs(ic_shifted) < COLLAPSE_RATIO * abs(ic_unshifted), (
        f"leakage: shifted IC {ic_shifted:.4f} did not collapse vs {ic_unshifted:.4f}"
    )
