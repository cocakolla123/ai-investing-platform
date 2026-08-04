"""ForecastModel implementations — doc 04 sections 1 and 6.

* :class:`LightGBMModel` — primary (regression + binary). Binary scores are
  calibrated with isotonic regression on a chronological validation slice;
  Platt scaling is the fallback when the slice has < 500 positives (isotonic
  overfits small samples).
* :class:`RandomForestModel` — challenger; median-impute fitted on the
  training fold only (doc 04 section 2.2 missing-data policy).
* :class:`StatsforecastBaseline` — HistoricAverage / SeasonalNaive / AutoETS
  wrapped to the same interface. The bar to beat: per-symbol univariate
  forecasts of the label, no features.

All models are deterministic: fixed seeds, single-threaded tree building.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
MIN_ISOTONIC_POSITIVES = 500
DEFAULT_VALID_FRACTION = 0.25

Objective = Literal["regression", "binary"]

__all__ = [
    "LightGBMModel",
    "RandomForestModel",
    "StatsforecastBaseline",
]


def _as_frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


def _as_target(y: Any) -> np.ndarray:
    if isinstance(y, (pd.Series | pd.DataFrame)):
        return np.asarray(y).ravel().astype(float)
    return np.asarray(y, dtype=float).ravel()


class _Calibrator:
    """Isotonic on validation scores; Platt fallback below the positive floor."""

    def __init__(self, min_isotonic_positives: int = MIN_ISOTONIC_POSITIVES) -> None:
        self.min_isotonic_positives = min_isotonic_positives
        self.method: str = "identity"
        self._iso: Any = None
        self._platt: Any = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> _Calibrator:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        positives = int(np.sum(y_true > 0.5))
        negatives = int(np.sum(y_true <= 0.5))
        if positives == 0 or negatives == 0 or len(y_true) < 4:
            self.method = "identity"
            return self
        if positives >= self.min_isotonic_positives:
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._iso.fit(raw_scores, y_true)
            self.method = "isotonic"
        else:
            self._platt = LogisticRegression(random_state=DEFAULT_SEED)
            self._platt.fit(raw_scores.reshape(-1, 1), (y_true > 0.5).astype(int))
            self.method = "platt"
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(self._iso.predict(raw_scores), dtype=float)
        if self.method == "platt":
            return np.asarray(self._platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1])
        return np.clip(raw_scores, 0.0, 1.0)


def _chronological_split(
    x: pd.DataFrame, y: np.ndarray, valid_fraction: float
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    n = len(x)
    n_valid = max(1, round(n * valid_fraction))
    n_fit = max(1, n - n_valid)
    return x.iloc[:n_fit], y[:n_fit], x.iloc[n_fit:], y[n_fit:]


class LightGBMModel:
    """Primary tabular model. ``objective`` is ``regression`` or ``binary``."""

    def __init__(
        self,
        objective: Objective = "regression",
        *,
        seed: int = DEFAULT_SEED,
        n_estimators: int = 200,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.objective: Objective = objective
        self.seed = seed
        self.n_estimators = n_estimators
        self.extra_params = dict(params or {})
        self._model: Any = None
        self._calibrator: _Calibrator | None = None
        self._feature_names: list[str] = []

    def _params(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "random_state": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": 1,
            "verbosity": -1,
        }
        base.update(self.extra_params)
        return base

    def fit(self, x: Any, y: Any, cfg: dict[str, Any] | None = None) -> LightGBMModel:
        import lightgbm as lgb

        cfg = cfg or {}
        xf = _as_frame(x)
        yv = _as_target(y)
        self._feature_names = [str(c) for c in xf.columns]
        if self.objective == "regression":
            self._model = lgb.LGBMRegressor(objective="regression", **self._params())
            self._model.fit(xf, yv)
            self._calibrator = None
            return self

        # Binary: hold out a chronological validation slice for calibration.
        valid_fraction = float(cfg.get("valid_fraction", DEFAULT_VALID_FRACTION))
        x_valid = cfg.get("x_valid")
        y_valid = cfg.get("y_valid")
        if x_valid is not None and y_valid is not None:
            x_fit, y_fit = xf, yv
            xv_frame, yv_arr = _as_frame(x_valid), _as_target(y_valid)
        else:
            x_fit, y_fit, xv_frame, yv_arr = _chronological_split(xf, yv, valid_fraction)
        if len(np.unique(y_fit)) < 2:  # degenerate fold — fall back to base rate
            self._model = None
            self._base_rate = float(np.mean(yv))
            self._calibrator = None
            return self
        self._model = lgb.LGBMClassifier(objective="binary", **self._params())
        self._model.fit(x_fit, (y_fit > 0.5).astype(int))
        raw_valid = np.asarray(self._model.predict_proba(xv_frame)[:, 1])
        self._calibrator = _Calibrator(
            int(cfg.get("min_isotonic_positives", MIN_ISOTONIC_POSITIVES))
        ).fit(raw_valid, yv_arr)
        return self

    def predict(self, x: Any) -> np.ndarray:
        xf = _as_frame(x)
        if self.objective == "binary":
            if self._model is None:
                return np.full(len(xf), getattr(self, "_base_rate", 0.5))
            raw = np.asarray(self._model.predict_proba(xf)[:, 1])
            if self._calibrator is not None:
                return self._calibrator.transform(raw)
            return raw
        return np.asarray(self._model.predict(xf), dtype=float)

    def feature_importances(self) -> dict[str, float]:
        if self._model is None:
            return {}
        imp = np.asarray(self._model.feature_importances_, dtype=float)
        total = imp.sum()
        if total <= 0:
            return dict.fromkeys(self._feature_names, 0.0)
        return dict(zip(self._feature_names, (imp / total).tolist(), strict=True))

    @property
    def calibration_method(self) -> str | None:
        return self._calibrator.method if self._calibrator is not None else None


class RandomForestModel:
    """Challenger. Median impute is fitted on the training fold only."""

    def __init__(
        self,
        objective: Objective = "regression",
        *,
        seed: int = DEFAULT_SEED,
        n_estimators: int = 100,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.objective: Objective = objective
        self.seed = seed
        self.n_estimators = n_estimators
        self.extra_params = dict(params or {})
        self._model: Any = None
        self._medians: pd.Series | None = None
        self._calibrator: _Calibrator | None = None
        self._feature_names: list[str] = []
        self._constant: float | None = None

    def _impute(self, xf: pd.DataFrame) -> pd.DataFrame:
        assert self._medians is not None
        return xf.fillna(self._medians).fillna(0.0)

    def fit(self, x: Any, y: Any, cfg: dict[str, Any] | None = None) -> RandomForestModel:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cfg = cfg or {}
        xf = _as_frame(x)
        yv = _as_target(y)
        self._feature_names = [str(c) for c in xf.columns]
        self._medians = cast(pd.Series, xf.median(numeric_only=True))
        kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "random_state": self.seed,
            "n_jobs": 1,
        }
        kwargs.update(self.extra_params)
        if self.objective == "regression":
            self._model = RandomForestRegressor(**kwargs)
            self._model.fit(self._impute(xf), yv)
            return self
        valid_fraction = float(cfg.get("valid_fraction", DEFAULT_VALID_FRACTION))
        x_fit, y_fit, x_val, y_val = _chronological_split(xf, yv, valid_fraction)
        if len(np.unique(y_fit)) < 2:
            self._model = None
            self._constant = float(np.mean(yv))
            return self
        self._model = RandomForestClassifier(**kwargs)
        self._model.fit(self._impute(x_fit), (y_fit > 0.5).astype(int))
        raw_valid = np.asarray(self._model.predict_proba(self._impute(x_val))[:, 1])
        self._calibrator = _Calibrator(
            int(cfg.get("min_isotonic_positives", MIN_ISOTONIC_POSITIVES))
        ).fit(raw_valid, y_val)
        return self

    def predict(self, x: Any) -> np.ndarray:
        xf = _as_frame(x)
        if self.objective == "binary":
            if self._model is None:
                return np.full(len(xf), self._constant if self._constant is not None else 0.5)
            raw = np.asarray(self._model.predict_proba(self._impute(xf))[:, 1])
            return self._calibrator.transform(raw) if self._calibrator else raw
        return np.asarray(self._model.predict(self._impute(xf)), dtype=float)

    def feature_importances(self) -> dict[str, float]:
        if self._model is None:
            return {}
        imp = np.asarray(self._model.feature_importances_, dtype=float)
        total = imp.sum()
        if total <= 0:
            return dict.fromkeys(self._feature_names, 0.0)
        return dict(zip(self._feature_names, (imp / total).tolist(), strict=True))


BaselineName = Literal["historic_average", "seasonal_naive", "auto_ets"]


class StatsforecastBaseline:
    """Classical per-symbol baselines behind the ForecastModel interface.

    ``fit`` expects ``x``/``y`` indexed by a (ts, symbol) MultiIndex (or an
    explicit ``symbols`` array in ``cfg``); it fits one univariate model per
    symbol on that symbol's label history and forecasts a constant one-step
    value used for every test row of the symbol. Symbols unseen in training
    fall back to the global train mean. Failures fall back to the symbol mean
    — the baseline never raises.
    """

    def __init__(self, model_name: BaselineName = "historic_average", *, season_length: int = 5):
        self.model_name: BaselineName = model_name
        self.season_length = season_length
        self._per_symbol: dict[str, float] = {}
        self._global_mean: float = 0.0

    def _make_model(self) -> Any:
        from statsforecast.models import AutoETS, HistoricAverage, SeasonalNaive

        if self.model_name == "historic_average":
            return HistoricAverage()
        if self.model_name == "seasonal_naive":
            return SeasonalNaive(season_length=self.season_length)
        return AutoETS(season_length=1)

    @staticmethod
    def _symbols_of(x: Any, cfg: dict[str, Any] | None) -> np.ndarray:
        if cfg and "symbols" in cfg:
            return np.asarray(cfg["symbols"])
        xf = x if isinstance(x, (pd.DataFrame | pd.Series)) else pd.DataFrame(x)
        if isinstance(xf.index, pd.MultiIndex):
            return np.asarray(xf.index.get_level_values(-1))
        return np.asarray(xf.index)

    def fit(self, x: Any, y: Any, cfg: dict[str, Any] | None = None) -> StatsforecastBaseline:
        yv = _as_target(y)
        symbols = self._symbols_of(x, cfg)
        frame = pd.DataFrame({"symbol": symbols, "y": yv}).dropna(subset=["y"])
        self._global_mean = float(frame["y"].mean()) if len(frame) else 0.0
        self._per_symbol = {}
        for sym, grp in frame.groupby("symbol", sort=True):
            series = np.asarray(grp["y"], dtype=float)
            value = float(series.mean())
            if self.model_name != "historic_average" and len(series) >= 4:
                try:
                    model = self._make_model()
                    model.fit(y=series)
                    value = float(np.asarray(model.predict(h=1)["mean"]).ravel()[0])
                except Exception:
                    pass  # keep the mean fallback — baselines never raise
            elif self.model_name == "historic_average":
                value = float(series.mean())
            self._per_symbol[str(sym)] = value
        return self

    def predict(self, x: Any) -> np.ndarray:
        symbols = self._symbols_of(x, None)
        return np.asarray(
            [self._per_symbol.get(str(s), self._global_mean) for s in symbols], dtype=float
        )

    def feature_importances(self) -> dict[str, float]:
        return {}
