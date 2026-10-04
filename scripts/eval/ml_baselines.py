"""Phase 5: quant baselines + feature-group ablations on campaign-v10's folds.

Implements docs/preregistration_ml_baselines.md. Rebuilds v10's panel through
the same code path quant_train uses (served from its panel cache), calls the
same ``train_and_evaluate`` with the same fold geometry, and logs every model
to MLflow (local SQLite store under artifacts/mlruns). Writes evals/ml/results_phase5.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TRAIN_START, TRAIN_END = date(2021, 3, 1), date(2026, 9, 23)
STRIDE = 2
SYMBOLS_FILE = "config/training_universe_wide.txt"
LGBM_PARAMS = {"max_depth": 3, "min_child_samples": 120, "reg_lambda": 10.0}  # v10's
TARGET = "excess_return_20d"
OUT = Path("evals/ml")


class MomentumBaseline:
    """Score = mom_12_1; no fitting. Missing values score neutral (0)."""

    def fit(self, x: Any, y: Any, cfg: Any = None) -> None:
        return None

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(x["mom_12_1"].to_numpy(float), nan=0.0)


class RidgeBaseline:
    """Ridge (alpha=1) on standardized features, missing -> 0 before scaling."""

    def __init__(self) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit(self, x: pd.DataFrame, y: pd.Series, cfg: Any = None) -> None:
        self._model.fit(np.nan_to_num(x.to_numpy(float)), y.to_numpy(float))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(np.nan_to_num(x.to_numpy(float))), dtype=float)


def build_v10_panel() -> Any:
    from dynamo.cli.backfill import read_symbols_file
    from dynamo.cli.quant_train import build_panel, label_sector_map
    from dynamo.config.settings import Settings
    from dynamo.data.universe import feature_input_symbols
    from dynamo.persistence.db import make_engine
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import _sessions_between
    from dynamo.research.quant.panel_cache import PanelFeatureCache

    engine = make_engine(Settings().database_url)
    gw_end = HistoricalDataGateway(engine, TRAIN_END)
    universe = read_symbols_file(Path(SYMBOLS_FILE))
    sector_map = label_sector_map(gw_end.sectors(universe), universe)
    sessions = _sessions_between(TRAIN_START, TRAIN_END)[::-1][::STRIDE][::-1]
    feature_symbols = feature_input_symbols(universe)
    label_bars = gw_end.bars(feature_symbols, (TRAIN_END - TRAIN_START).days + 60)
    cache = PanelFeatureCache(Path(Settings().artifact_dir) / "panel_cache", feature_symbols)
    print(f"panel cache: {cache.directory}", flush=True)
    return build_panel(
        lambda d: HistoricalDataGateway(engine, d).feature_inputs(feature_symbols),
        label_bars,
        sector_map,
        universe,
        sessions,
        progress=lambda m: print(m, flush=True),
        cache=cache,
    )


def dataset_hash(features: pd.DataFrame, labels: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes())  # pyright: ignore[reportAttributeAccessIssue]
    h.update(pd.util.hash_pandas_object(labels[[TARGET]], index=True).to_numpy().tobytes())  # pyright: ignore[reportAttributeAccessIssue]
    return h.hexdigest()


def fold_ics(report: dict[str, Any], name: str) -> list[float]:
    return [
        float(f["models"][name]["rank_ic"])
        if f["models"][name]["rank_ic"] is not None
        else math.nan
        for f in report["folds"]
    ]


def paired_t(a: list[float], b: list[float]) -> dict[str, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if d.size < 2 or d.std(ddof=1) == 0:
        return {
            "mean_diff": float(d.mean()) if d.size else math.nan,
            "t": math.nan,
            "n": int(d.size),
        }
    return {
        "mean_diff": float(d.mean()),
        "t": float(d.mean() / d.std(ddof=1) * math.sqrt(d.size)),
        "n": int(d.size),
    }


def main() -> None:
    import argparse

    import mlflow

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag",
        default="",
        help="re-run label: results_phase5_<tag>.json + its own MLflow experiment",
    )
    tag = ap.parse_args().tag
    out_name = f"results_phase5_{tag}.json" if tag else "results_phase5.json"
    experiment = f"quant-phase5-v10-folds-{tag}" if tag else "quant-phase5-v10-folds"

    from dynamo.cli.quant_train import walkforward_for
    from dynamo.research.quant.catalog import FEATURE_CATALOG, GROUP_TARGETS
    from dynamo.research.quant.models import LightGBMModel, StatsforecastBaseline
    from dynamo.research.quant.training import train_and_evaluate

    OUT.mkdir(parents=True, exist_ok=True)
    # Local, server-free tracking. MLflow 3 retired the plain-directory store
    # ("maintenance mode"); a local SQLite file is its recommended equivalent.
    root = Path("artifacts/mlruns").resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(experiment)

    panel = build_v10_panel()
    features, labels = panel.features, panel.labels
    n_sessions = len(sorted(set(features.index.get_level_values(0))))
    wf = walkforward_for(n_sessions, session_stride=STRIDE)
    ds_hash = dataset_hash(features, labels)
    print(f"panel {features.shape}, {n_sessions} sessions, dataset {ds_hash[:12]}", flush=True)

    def evaluate(name: str, feats: pd.DataFrame, factories: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        report = train_and_evaluate(
            feats,
            labels,
            task="rank",
            target=TARGET,
            model_factories=factories,
            baseline_factories={
                "baseline-historic-average": lambda: StatsforecastBaseline("historic_average")
            },
            wf_config=wf,
        )
        secs = time.perf_counter() - t0
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "dataset_hash": ds_hash,
                    "n_features": feats.shape[1],
                    "folds": len(report["folds"]),
                    "target": TARGET,
                    "stride": STRIDE,
                    **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()},
                }
            )
            mlflow.log_text("\n".join(map(str, feats.columns)), "feature_list.txt")
            for m in factories:
                for i, ic in enumerate(fold_ics(report, m)):
                    if not math.isnan(ic):
                        mlflow.log_metric(f"{m}.fold_rank_ic", ic, step=i)
                mlflow.log_metric(f"{m}.pooled_rank_ic", report["pooled"][m]["rank_ic_pooled"])
                mlflow.log_metric(f"{m}.t_stat", report["pooled"][m]["rank_ic_t_stat"])
            mlflow.log_metric("seconds", secs)
        print(f"{name}: {secs:.0f}s", flush=True)
        return report

    lgbm = lambda: LightGBMModel("regression", params=LGBM_PARAMS)  # noqa: E731
    main_report = evaluate(
        "models",
        features,
        {"lightgbm": lgbm, "baseline-momentum": MomentumBaseline, "baseline-linear": RidgeBaseline},
    )
    ics = {
        m: fold_ics(main_report, m)
        for m in ("lightgbm", "baseline-momentum", "baseline-linear", "baseline-historic-average")
    }
    models_summary = {
        m: {
            "pooled_rank_ic": main_report["pooled"][m]["rank_ic_pooled"],
            "fold_t_stat": main_report["pooled"][m]["rank_ic_t_stat"],
            "fold_ics": v,
        }
        for m, v in ics.items()
    }
    lgbm_vs = {b: paired_t(ics["lightgbm"], ics[b]) for b in ics if b != "lightgbm"}

    ablations: dict[str, Any] = {}
    for group in sorted(GROUP_TARGETS):
        keep = [
            c
            for c in features.columns
            if FEATURE_CATALOG.get(c) and FEATURE_CATALOG[c].group != group
        ]
        if len(keep) == features.shape[1]:
            continue
        rep = evaluate(f"drop-{group}", features[keep], {"lightgbm": lgbm})
        dropped = fold_ics(rep, "lightgbm")
        ablations[group] = {
            "features_removed": features.shape[1] - len(keep),
            "pooled_rank_ic": rep["pooled"]["lightgbm"]["rank_ic_pooled"],
            "dropped_minus_full": paired_t(dropped, ics["lightgbm"]),
        }

    result = {
        "preregistration": "docs/preregistration_ml_baselines.md",
        "dataset_hash": ds_hash,
        "panel_shape": list(features.shape),
        "folds": len(main_report["folds"]),
        "models": models_summary,
        "lightgbm_minus_baseline": lgbm_vs,
        "ablations": ablations,
        "thresholds": {"t_call": 2.0, "t_bonferroni_17": 3.4},
    }
    (OUT / out_name).write_text(json.dumps(result, indent=2, default=float))
    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(str(OUT / out_name))
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2, default=float))
    for m, s in models_summary.items():
        print(f"{m:28s} pooled IC {s['pooled_rank_ic']:+.4f}  fold t {s['fold_t_stat']:+.2f}")


if __name__ == "__main__":
    main()
