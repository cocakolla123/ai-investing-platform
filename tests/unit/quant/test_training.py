"""train_and_evaluate: metrics shape, ship gates, freeze guard, determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dynamo.backtest.walkforward import (
    ConfigFrozenError,
    EvaluationRegistry,
    WalkForwardConfig,
    compute_config_hash,
)
from dynamo.research.quant.models import LightGBMModel, RandomForestModel
from dynamo.research.quant.training import train_and_evaluate
from tests.unit.quant.conftest import planted_panel

CFG = WalkForwardConfig(
    train_sessions=200, test_sessions=60, purge_sessions=20, embargo_sessions=10, roll_sessions=60
)


def _lgbm() -> LightGBMModel:
    return LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5})


def test_rank_task_metrics_and_ship_gate() -> None:
    features, label, _ = planted_panel()
    res = train_and_evaluate(
        features, label, task="rank", model_factories={"lgbm": _lgbm}, wf_config=CFG
    )
    assert res["task"] == "rank"
    assert res["n_folds"] >= 1
    pooled = res["pooled"]["lgbm"]
    # Planted signal: the model should crush the featureless baselines.
    assert pooled["rank_ic_pooled"] > 0.8
    assert pooled["rank_ic_t_stat"] >= 0.0
    assert pooled["win_share"] == 1.0
    assert res["ships_ok"]["lgbm"] in (True, False)
    assert json.dumps(res)  # JSON-serializable end to end
    for fold in res["folds"]:
        assert {"fold_id", "test_start", "test_end", "models"} <= set(fold)
        assert "rank_ic" in fold["models"]["lgbm"]


def test_downside_task_metrics() -> None:
    features, label, _ = planted_panel()
    yb = (label < label.quantile(0.2)).astype(float)
    res = train_and_evaluate(
        features,
        yb.rename("downside_20d"),
        task="downside",
        model_factories={
            "lgbm": lambda: LightGBMModel(
                "binary", n_estimators=20, params={"min_child_samples": 5}
            )
        },
        wf_config=CFG,
    )
    pooled = res["pooled"]["lgbm"]
    assert 0.0 <= pooled["brier_pooled"] <= 1.0
    assert pooled["brier_pooled"] < pooled["base_rate_brier_pooled"]
    assert json.dumps(res)


def test_vol_task_metrics() -> None:
    features, _label, _ = planted_panel()
    vol = np.exp(0.5 * features["signal"]) * 0.2
    res = train_and_evaluate(
        features,
        pd.Series(vol, index=features.index, name="realized_vol_20d"),
        task="vol",
        model_factories={"rf": lambda: RandomForestModel("regression", n_estimators=10)},
        wf_config=CFG,
    )
    pooled = res["pooled"]["rf"]
    assert pooled["rmse_log_vol_pooled"] >= 0.0
    assert pooled["qlike_pooled"] >= 0.0
    assert json.dumps(res)


def test_training_deterministic() -> None:
    features, label, _ = planted_panel()
    kwargs: dict = {"task": "rank", "model_factories": {"lgbm": _lgbm}, "wf_config": CFG}
    r1 = train_and_evaluate(features, label, **kwargs)
    r2 = train_and_evaluate(features, label, **kwargs)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_training_enforces_freeze_guard() -> None:
    features, label, _ = planted_panel()
    registry = EvaluationRegistry()
    h = compute_config_hash({"model": "lgbm", "v": 1})
    train_and_evaluate(
        features,
        label,
        task="rank",
        model_factories={"lgbm": _lgbm},
        wf_config=CFG,
        registry=registry,
        config_hash=h,
    )
    with pytest.raises(ConfigFrozenError):
        train_and_evaluate(
            features,
            label,
            task="rank",
            model_factories={"lgbm": _lgbm},
            wf_config=CFG,
            registry=registry,
            config_hash=h,
        )
    assert len(registry.records()) > 0
