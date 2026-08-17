"""Walk-forward training and evaluation — doc 04 sections 5-6.

``train_and_evaluate`` runs candidate models and statsforecast baselines over
identical purged/embargoed folds (backtest/walkforward.py, the canonical
splitter) and reports per-fold + pooled metrics with the doc-04 section 6
ship gates:

* rank task: Spearman rank IC (mean + t-stat over folds), fold win share vs
  the best baseline; ships when pooled IC beats the best baseline, is positive
  with t >= 2, and the model wins >= 60% of folds.
* downside task: Brier score, compared against the base-rate Brier and the
  best baseline.
* vol task: RMSE on log-vol + QLIKE, compared against baselines.

The returned metrics dict is JSON-serializable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from dynamo.backtest.walkforward import (
    EvaluationRegistry,
    Fold,
    WalkForwardConfig,
    generate_folds,
)
from dynamo.domain.ports import ForecastModel

# Re-exported for the replay preflight hook (Phase-2 contract: the loader is
# discoverable as ``dynamo.research.quant.training.bundle_from_version``).
from dynamo.research.quant.bundle import QuantBundleError, bundle_from_version
from dynamo.research.quant.models import StatsforecastBaseline

Task = Literal["rank", "downside", "vol"]

MIN_CROSS_SECTION = 3
SHIP_T_STAT = 2.0
SHIP_WIN_SHARE = 0.60

__all__ = [
    "QuantBundleError",
    "bundle_from_version",
    "default_baseline_factories",
    "feature_quantiles",
    "fit_final_model",
    "train_and_evaluate",
]


def default_baseline_factories(task: Task) -> dict[str, Callable[[], ForecastModel]]:
    if task == "downside":
        return {"baseline-historic-average": lambda: StatsforecastBaseline("historic_average")}
    return {
        "baseline-historic-average": lambda: StatsforecastBaseline("historic_average"),
        "baseline-seasonal-naive": lambda: StatsforecastBaseline("seasonal_naive"),
        "baseline-auto-ets": lambda: StatsforecastBaseline("auto_ets"),
    }


def _ts_level(index: pd.MultiIndex) -> np.ndarray:
    return np.asarray(pd.to_datetime(index.get_level_values(0)))


def _spearman_ic(pred: pd.Series, actual: pd.Series) -> float:
    if len(pred) < MIN_CROSS_SECTION or pred.nunique() < 2 or actual.nunique() < 2:
        return float("nan")
    return float(pred.corr(actual, method="spearman"))


def _per_date_ics(pred: pd.Series, actual: pd.Series) -> list[float]:
    ts = pred.index.get_level_values(0)
    ics = []
    for _, idx in pd.Series(range(len(pred)), index=pred.index).groupby(ts).groups.items():
        ic = _spearman_ic(pred.loc[idx], actual.loc[idx])
        if not math.isnan(ic):
            ics.append(ic)
    return ics


def _t_stat(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd * math.sqrt(len(arr)))


def _brier(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((np.clip(pred, 0.0, 1.0) - actual) ** 2))


def _auc(pred: np.ndarray, actual: np.ndarray) -> float:
    """ROC AUC of a probability score; NaN when the slice is single-class."""
    y = (np.asarray(actual, dtype=float) > 0.5).astype(int)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, np.asarray(pred, dtype=float)))


def _qlike(vol_pred: np.ndarray, vol_true: np.ndarray) -> float:
    var_p = np.maximum(vol_pred, 1e-8) ** 2
    var_t = np.maximum(vol_true, 1e-8) ** 2
    ratio = var_t / var_p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _fit_predict(
    factory: Callable[[], ForecastModel],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    fit_cfg: dict[str, Any],
) -> np.ndarray:
    model = factory()
    model.fit(x_train, y_train, dict(fit_cfg))
    return np.asarray(model.predict(x_test), dtype=float).ravel()


def _fold_slices(
    features: pd.DataFrame, labels: pd.Series, fold: Fold
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ts = pd.DatetimeIndex(features.index.get_level_values(0))
    train_mask = ts.isin(fold.train_sessions)
    test_mask = ts.isin(fold.test_sessions)
    label_ok = labels.notna().to_numpy()
    x_train = features.loc[train_mask & label_ok]
    y_train = labels.loc[train_mask & label_ok]
    x_test = features.loc[test_mask & label_ok]
    y_test = labels.loc[test_mask & label_ok]
    return x_train, y_train, x_test, y_test


def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
    *,
    task: Task = "rank",
    target: str | None = None,
    model_factories: Mapping[str, Callable[[], ForecastModel]],
    baseline_factories: Mapping[str, Callable[[], ForecastModel]] | None = None,
    wf_config: WalkForwardConfig | None = None,
    sessions: pd.DatetimeIndex | None = None,
    fit_cfg: Mapping[str, Any] | None = None,
    config_hash: str | None = None,
    registry: EvaluationRegistry | None = None,
) -> dict[str, Any]:
    """Evaluate models vs baselines over the canonical walk-forward folds.

    ``features``/``labels`` share a (ts, symbol) MultiIndex. ``labels`` may be
    a frame (``target`` selects the column) or a Series. When ``registry`` and
    ``config_hash`` are given, the config-freeze guard claims every OOS fold
    window before any evaluation and refuses repeats.
    """
    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features must be indexed by a (ts, symbol) MultiIndex")
    if isinstance(labels, pd.DataFrame):
        default_target = {
            "rank": "excess_return_20d",
            "downside": "downside_20d",
            "vol": "realized_vol_20d",
        }
        col = target or default_target[task]
        label = labels[col].astype(float)
    else:
        label = labels.astype(float)
        col = target or str(label.name or task)
    label = label.reindex(features.index)

    if sessions is None:
        sessions = pd.DatetimeIndex(sorted(set(_ts_level(features.index))))
    folds = generate_folds(sessions, wf_config)
    if not folds:
        raise ValueError("not enough sessions for a single walk-forward fold")
    if registry is not None and config_hash is not None:
        registry.claim_folds(config_hash, folds)

    if task == "vol":
        # Regression on log-vol (doc 04 section 4); zero/negative vol rows drop.
        label = label.where(label > 0)
        fit_label = np.log(label)
    else:
        fit_label = label
    base_fit_cfg = dict(fit_cfg or {})

    if baseline_factories is not None:
        baselines = dict(baseline_factories)
    else:
        baselines = default_baseline_factories(task)
    all_names = list(model_factories) + list(baselines)
    all_factories: dict[str, Callable[[], ForecastModel]] = {**dict(model_factories), **baselines}

    fold_rows: list[dict[str, Any]] = []
    per_model_fold_metric: dict[str, list[float]] = {n: [] for n in all_names}
    per_model_date_ics: dict[str, list[float]] = {n: [] for n in all_names}
    pooled_pred: dict[str, list[np.ndarray]] = {n: [] for n in all_names}
    pooled_true: list[np.ndarray] = []

    for fold in folds:
        x_train, y_train, x_test, y_test = _fold_slices(features, cast(pd.Series, fit_label), fold)
        if len(x_train) == 0 or len(x_test) == 0:
            continue
        y_test_nat = label.loc[y_test.index]
        pooled_true.append(np.asarray(y_test_nat, dtype=float))
        fold_metrics: dict[str, dict[str, float]] = {}
        for name in all_names:
            pred = _fit_predict(all_factories[name], x_train, y_train, x_test, base_fit_cfg)
            pooled_pred[name].append(pred)
            pred_s = pd.Series(pred, index=x_test.index)
            if task == "rank":
                ics = _per_date_ics(pred_s, y_test_nat)
                metric = float(np.mean(ics)) if ics else float("nan")
                per_model_date_ics[name].extend(ics)
                fold_metrics[name] = {"rank_ic": metric}
            elif task == "downside":
                y_arr = np.asarray(y_test_nat, dtype=float)
                metric = _brier(pred, y_arr)
                base_rate = float(np.mean(y_train)) if len(y_train) else 0.5
                fold_metrics[name] = {
                    "brier": metric,
                    "auc": _auc(pred, y_arr),
                    "base_rate_brier": _brier(np.full(len(y_test_nat), base_rate), y_arr),
                }
            else:  # vol: models emit log-vol
                vol_pred = np.exp(pred)
                vol_true = np.asarray(y_test_nat, dtype=float)
                metric = float(np.sqrt(np.mean((pred - np.log(vol_true)) ** 2)))
                fold_metrics[name] = {
                    "rmse_log_vol": metric,
                    "qlike": _qlike(vol_pred, vol_true),
                }
            per_model_fold_metric[name].append(metric)
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train": len(x_train),
                "n_test": len(x_test),
                "models": fold_metrics,
            }
        )

    if not fold_rows:
        raise ValueError("no fold produced evaluable train/test rows")

    higher_is_better = task == "rank"
    n_folds = len(fold_rows)
    baseline_names = list(baselines)

    def _best_baseline_fold(fold_idx: int) -> float:
        vals = [per_model_fold_metric[b][fold_idx] for b in baseline_names]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return float("nan")
        return max(vals) if higher_is_better else min(vals)

    pooled: dict[str, dict[str, Any]] = {}
    ships_ok: dict[str, bool] = {}
    y_pooled = np.concatenate(pooled_true)

    for name in all_names:
        fold_vals = per_model_fold_metric[name]
        wins = 0
        comparable = 0
        for i, v in enumerate(fold_vals):
            bb = _best_baseline_fold(i)
            if math.isnan(v) or math.isnan(bb):
                continue
            comparable += 1
            if (v > bb) if higher_is_better else (v < bb):
                wins += 1
        win_share = wins / comparable if comparable else 0.0
        entry: dict[str, Any] = {
            "fold_metric_mean": float(np.nanmean(fold_vals)),
            "win_share": float(win_share),
            "n_folds": n_folds,
        }
        preds = np.concatenate(pooled_pred[name])
        if task == "rank":
            ics = per_model_date_ics[name]
            entry["rank_ic_pooled"] = float(np.mean(ics)) if ics else float("nan")
            entry["rank_ic_t_stat"] = _t_stat([v for v in fold_vals if not math.isnan(v)])
        elif task == "downside":
            entry["brier_pooled"] = _brier(preds, y_pooled)
            entry["auc_pooled"] = _auc(preds, y_pooled)
            entry["base_rate_brier_pooled"] = _brier(
                np.full(len(y_pooled), float(np.mean(y_pooled))), y_pooled
            )
        else:
            with np.errstate(invalid="ignore"):
                entry["rmse_log_vol_pooled"] = float(
                    np.sqrt(np.mean((preds - np.log(y_pooled)) ** 2))
                )
                entry["qlike_pooled"] = _qlike(np.exp(preds), y_pooled)
        pooled[name] = entry

    def _pooled_metric(name: str) -> float:
        e = pooled[name]
        if task == "rank":
            return float(e["rank_ic_pooled"])
        if task == "downside":
            return float(e["brier_pooled"])
        return float(e["rmse_log_vol_pooled"])

    baseline_pooled = [
        _pooled_metric(b) for b in baseline_names if not math.isnan(_pooled_metric(b))
    ]
    best_baseline = (
        (max(baseline_pooled) if higher_is_better else min(baseline_pooled))
        if baseline_pooled
        else float("nan")
    )

    for name in model_factories:
        e = pooled[name]
        m = _pooled_metric(name)
        beats_baseline = (
            not math.isnan(m)
            and not math.isnan(best_baseline)
            and ((m > best_baseline) if higher_is_better else (m < best_baseline))
        )
        wins_enough = float(e["win_share"]) >= SHIP_WIN_SHARE
        if task == "rank":
            ships = (
                beats_baseline
                and m > 0
                and float(e["rank_ic_t_stat"]) >= SHIP_T_STAT
                and wins_enough
            )
        elif task == "downside":
            ships = (
                beats_baseline
                and m < float(e["base_rate_brier_pooled"])
                and wins_enough
            )
        else:
            ships = beats_baseline and wins_enough
        ships_ok[name] = bool(ships)

    return _jsonify(
        {
            "task": task,
            "target": col,
            "n_folds": n_folds,
            "config_hash": config_hash,
            "folds": fold_rows,
            "pooled": pooled,
            "baselines": baseline_names,
            "best_baseline_pooled": best_baseline,
            "ships_ok": ships_ok,
        }
    )


# --------------------------------------------------------------------------
# final-model fitting for artifact save (backtest-campaign Phase 2)
# --------------------------------------------------------------------------


def _select_label(
    labels: pd.DataFrame | pd.Series, task: Task, target: str | None
) -> pd.Series:
    if isinstance(labels, pd.DataFrame):
        default_target = {
            "rank": "excess_return_20d",
            "downside": "downside_20d",
            "vol": "realized_vol_20d",
        }
        return cast(pd.Series, labels[target or default_target[task]].astype(float))
    return labels.astype(float)


def fit_final_model(
    features: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
    *,
    task: Task,
    model_factory: Callable[[], ForecastModel],
    target: str | None = None,
    fit_cfg: Mapping[str, Any] | None = None,
) -> ForecastModel:
    """Fit ONE production model for ``task`` on the FULL training window.

    Sibling of :func:`train_and_evaluate` (deliberately not folded in — the
    walk-forward report card keeps its exact signature and semantics): the
    fold loop measures, this fits the artifact that ships. Label handling
    mirrors the fold loop bit for bit — rows with NaN labels drop, the vol
    task trains on log-vol (positive rows only), and rows are sorted
    chronologically so the binary model's internal chronological calibration
    split (``LightGBMModel.fit``, isotonic/Platt on the tail slice) is honest.
    """
    if not isinstance(features.index, pd.MultiIndex):
        raise ValueError("features must be indexed by a (ts, symbol) MultiIndex")
    label = _select_label(labels, task, target).reindex(features.index)
    if task == "vol":
        label = label.where(label > 0)
        label = cast(pd.Series, np.log(label))
    keep = label.notna().to_numpy()
    x = features.loc[keep]
    y = cast(pd.Series, label.loc[keep])
    if len(x) == 0:
        raise ValueError(f"no rows with a usable {task!r} label to fit on")
    x = x.sort_index(level=[0, 1])
    y = y.sort_index(level=[0, 1])
    model = model_factory()
    model.fit(x, y, dict(fit_cfg or {}))
    return model


def feature_quantiles(features: pd.DataFrame) -> pd.DataFrame:
    """Training-window q01/q99 per feature — the coverage band the scoring
    confidence heuristic checks live rows against (part of the artifact)."""
    valid = features.astype(float)
    return pd.DataFrame(
        {
            col: {
                "q01": float(valid[col].quantile(0.01)),
                "q99": float(valid[col].quantile(0.99)),
            }
            for col in valid.columns
        }
    ).reindex(["q01", "q99"])


def _jsonify(obj: Any) -> Any:
    """Coerce numpy scalars to native Python so json.dumps always works."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list | tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating | np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj
