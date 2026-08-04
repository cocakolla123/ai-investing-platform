"""Nightly universe scoring — doc 04 section 8.

``score_universe`` turns a fitted :class:`ModelBundle` plus the night's
feature matrix into ``QuantScore`` rows: predictions, cross-sectional rank
(1 = best), a documented confidence heuristic, and feature coverage.

Confidence (doc 04 section 6) is a 0-1 heuristic, not a posterior:
``(1 - missing_fraction) x training-distribution coverage x 1/(1 + fold
dispersion)``. Symbols missing more than 30% of features are clamped low
(doc 04 section 2.2) and flagged via the low confidence itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from dynamo.domain.models.research import QuantScore
from dynamo.domain.ports import ForecastModel

MISSING_CLAMP_THRESHOLD = 0.30
MISSING_CLAMP_CEILING = 0.20

__all__ = ["ModelBundle", "score_universe"]


@dataclass(frozen=True)
class ModelBundle:
    """Fitted production models for one nightly scoring pass.

    ``train_feature_quantiles`` holds rows ``q01``/``q99`` per feature from
    the training window (part of the model artifact); ``fold_ic_dispersion``
    is the fold-to-fold std of the primary metric from the walk-forward
    report card. ``vol_is_log`` marks a vol model trained on log-vol.
    """

    model_version: str
    ranker: ForecastModel | None = None
    downside: ForecastModel | None = None
    vol: ForecastModel | None = None
    train_feature_quantiles: pd.DataFrame | None = None
    fold_ic_dispersion: float = 0.0
    vol_is_log: bool = True


def _predict(model: ForecastModel | None, features: pd.DataFrame) -> np.ndarray | None:
    if model is None:
        return None
    return np.asarray(model.predict(features), dtype=float).ravel()


def _coverage_fraction(row: pd.Series, quantiles: pd.DataFrame | None) -> float:
    """Fraction of non-missing features inside the training [q01, q99] band."""
    present = row.dropna()
    if quantiles is None or len(present) == 0:
        return 1.0
    common = [c for c in present.index if c in quantiles.columns]
    if not common:
        return 1.0
    lo = quantiles.loc["q01", common]
    hi = quantiles.loc["q99", common]
    inside = (present[common] >= lo) & (present[common] <= hi)
    return float(inside.mean())


def _confidence(missing_fraction: float, coverage: float, dispersion: float) -> float:
    conf = (1.0 - missing_fraction) * coverage / (1.0 + max(0.0, dispersion))
    if missing_fraction > MISSING_CLAMP_THRESHOLD:
        conf = min(conf, MISSING_CLAMP_CEILING)
    return float(min(1.0, max(0.0, conf)))


def _opt(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return float(value)


def score_universe(
    bundle: ModelBundle, features: pd.DataFrame, as_of: date
) -> list[QuantScore]:
    """Score every symbol in ``features`` (indexed by symbol) at ``as_of``.

    Deterministic: rank ties break by symbol; identical inputs produce
    identical rows.
    """
    if features.empty:
        return []
    symbols = [str(s) for s in features.index]
    expected = _predict(bundle.ranker, features)
    downside = _predict(bundle.downside, features)
    vol_raw = _predict(bundle.vol, features)
    vol = None
    if vol_raw is not None:
        vol = np.exp(vol_raw) if bundle.vol_is_log else vol_raw

    missing_fraction = np.asarray(features.isna().mean(axis=1), dtype=float)
    coverage = np.asarray(
        [
            _coverage_fraction(features.iloc[i], bundle.train_feature_quantiles)
            for i in range(len(features))
        ]
    )

    # Cross-sectional rank on expected excess return: 1 = best; ties break by
    # symbol so reruns are bit-identical.
    ranks: dict[str, int | None] = dict.fromkeys(symbols)
    if expected is not None:
        order = sorted(
            (i for i in range(len(symbols)) if not math.isnan(expected[i])),
            key=lambda i: (-expected[i], symbols[i]),
        )
        for pos, i in enumerate(order, start=1):
            ranks[symbols[i]] = pos

    scores: list[QuantScore] = []
    for i, sym in enumerate(symbols):
        scores.append(
            QuantScore(
                symbol=sym,
                as_of=as_of,
                model_version=bundle.model_version,
                expected_excess_return=_opt(None if expected is None else expected[i]),
                downside_prob=_opt(
                    None if downside is None else float(np.clip(downside[i], 0.0, 1.0))
                ),
                vol_forecast=_opt(None if vol is None else vol[i]),
                rank=ranks[sym],
                confidence=_confidence(
                    float(missing_fraction[i]), float(coverage[i]), bundle.fold_ic_dispersion
                ),
                feature_coverage=float(1.0 - missing_fraction[i]),
            )
        )
    return scores
