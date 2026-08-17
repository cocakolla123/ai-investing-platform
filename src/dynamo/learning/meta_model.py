"""Advisory meta-model: features + trainer (plan Phase 3).

Pure over inputs PASSED IN (domain models + plain dataclasses); import-linter
keeps this module off persistence. The model predicts P(win) for a committee
decision from everything knowable at decision time — verdict shape, quant
view, market context, hierarchical calibration, and nearest prior
experiences. Its output (``effective_probability``) is ADVISORY-FIRST:
logged to ``meta_decision_scores`` and surfaced in reports; the sizing path
consumes it ONLY behind ``meta_sizing_enabled`` (default False) after the
full promotion pipeline + human approval.

Two feature entry points share one builder so training and decision-time
scoring can never drift apart:

- :func:`build_feature_row` — from a resolved :class:`DecisionExperience`
  (training path);
- :func:`snapshot_from_decision` + :func:`build_features` — from tonight's
  verdict/quant/regime directly (the experience row does not exist yet at
  decision time).

NOTE (Phase-2 finding): ``decision_experience`` has NO strategy / event_type
columns — the platform does not record them on experiences yet — so those
scope dimensions are deliberately absent from the feature vector. They can
be added (with a model retrain) once the columns exist.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dynamo.domain.models.research import Action, QuantAlignment
from dynamo.learning.hierarchy import CalibratedProbability

if TYPE_CHECKING:
    from dynamo.domain.models.learning import DecisionExperience
    from dynamo.domain.models.research import CommitteeVerdict, QuantScore
    from dynamo.domain.models.risk import RegimeState

SEED = 42
MIN_TRAIN_EXPERIENCES = 150  # hard gate: the trainer refuses below this
VALID_FRACTION = 0.25  # chronological tail held out for calibration + OOS
MIN_ISOTONIC_POSITIVES = 500  # mirror research/quant/models.py:_Calibrator
_ENTRY_ACTIONS = frozenset({Action.BUY.value, Action.ADD.value})
_REGIMES = ("risk_on", "neutral", "risk_off", "crisis")
_ALIGNMENTS = ("agree", "neutral", "conflict")
_SLEEVES = ("long_term", "tactical")
_SECTOR_BUCKETS = 997  # prime bucket count for the deterministic sector hash

# Deterministic feature ordering — the single source of truth for both the
# trainer and every scorer. Append-only in spirit: reordering or renaming
# invalidates saved artifacts (the manifest records these names).
FEATURE_NAMES: tuple[str, ...] = (
    # verdict
    "conviction",
    "disagreement_score",
    "dissent_count",
    "has_self_critique",
    "qa_agree",
    "qa_neutral",
    "qa_conflict",
    "expected_upside_pct",
    "expected_downside_pct",
    "upside_downside_ratio",
    "horizon_days",
    "journal_context_used",
    # quant
    "quant_score",
    "quant_prob",
    "quant_veto",
    # context
    "regime_risk_on",
    "regime_neutral",
    "regime_risk_off",
    "regime_crisis",
    "regime_confidence",
    "sector_code",
    "sleeve_long_term",
    "sleeve_tactical",
    "vix",
    "spy_ret_20d",
    # calibration (plan Phase 2 output)
    "calibrated_probability",
    "calibration_n",
    # retrieval (k nearest prior experiences)
    "neighbor_count",
    "neighbor_mean_pnl_pct",
    "neighbor_hit_rate",
    # bankroll scale (backtest-campaign Phase 3; migration 0015) — appended
    # so the meta-model can learn scale-dependent trust. NaN-safe: legacy
    # rows with a NULL equity_at_decision collapse to 0.0.
    "log10_equity_at_decision",
)


@dataclass(frozen=True)
class DecisionSnapshot:
    """Everything the feature builder needs about ONE decision, decoupled
    from where it came from (resolved experience vs tonight's verdict).

    ``dissent_count`` / ``has_self_critique`` / ``journal_context_used`` /
    ``regime_confidence`` / ``spy_ret_20d`` are not stored on
    ``decision_experience`` rows, so the experience-derived snapshot leaves
    them ``None`` (NaN-safe defaults below) while the decision-time snapshot
    populates them from the live verdict/regime. The model simply learns
    nothing from near-constant training columns until the platform
    denormalizes them onto experiences.
    """

    conviction: float | None = None
    disagreement_score: float | None = None
    dissent_count: int | None = None
    has_self_critique: bool | None = None
    quant_alignment: str | None = None
    expected_upside_pct: float | None = None
    expected_downside_pct: float | None = None
    horizon_days: int | None = None
    journal_context_used: int | None = None
    action: str | None = None
    quant_score: float | None = None  # quant expected excess return
    quant_prob: float | None = None  # quant downside probability
    regime: str | None = None
    regime_confidence: float | None = None
    sector: str | None = None
    sleeve: str | None = None
    vix: float | None = None
    spy_ret_20d: float | None = None
    equity_at_decision: float | None = None


def sector_code(sector: str | None) -> float:
    """Deterministic sector hash in [0, 1): sha256 mod a prime bucket count.

    Stable across processes and Python versions (never ``hash()``); unknown
    or missing sectors land on 0.0.
    """
    if not sector:
        return 0.0
    digest = hashlib.sha256(sector.strip().lower().encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") % _SECTOR_BUCKETS) / _SECTOR_BUCKETS


def quant_veto_flag(action: str | None, quant_alignment: str | None) -> float:
    """1.0 when an entry (buy/add) was taken AGAINST a quant conflict — the
    same deterministic veto definition as ``learning/experience.py``."""
    return float(action in _ENTRY_ACTIONS and quant_alignment == QuantAlignment.CONFLICT.value)


def _num(value: float | int | None, default: float = 0.0) -> float:
    """NaN-safe scalar: ``None``/NaN/inf all collapse to ``default``."""
    if value is None:
        return default
    out = float(value)
    return out if math.isfinite(out) else default


def _one_hot(value: str | None, categories: Sequence[str]) -> list[float]:
    return [1.0 if value == cat else 0.0 for cat in categories]


def build_features(
    snap: DecisionSnapshot,
    calib: CalibratedProbability | None,
    neighbors: Sequence[DecisionExperience],
) -> dict[str, float]:
    """One deterministic, NaN-safe feature row keyed exactly by FEATURE_NAMES.

    Missing inputs degrade to neutral defaults (0.0; ``calibrated_probability``
    falls back to the raw conviction — the identity calibration) so every
    decision is scoreable, never raised on.
    """
    conviction = _num(snap.conviction)
    upside = _num(snap.expected_upside_pct)
    downside = _num(snap.expected_downside_pct)
    ratio = upside / abs(downside) if downside != 0.0 else 0.0
    qa = _one_hot(snap.quant_alignment, _ALIGNMENTS)
    regime = _one_hot(snap.regime, _REGIMES)
    sleeve = _one_hot(snap.sleeve, _SLEEVES)

    pnls = [float(n.pnl_pct) for n in neighbors if n.pnl_pct is not None]
    wins = [1.0 if float(p) >= 0.0 else 0.0 for p in pnls]

    row = {
        "conviction": conviction,
        "disagreement_score": _num(snap.disagreement_score),
        "dissent_count": _num(snap.dissent_count),
        "has_self_critique": (
            0.0 if snap.has_self_critique is None else float(snap.has_self_critique)
        ),
        "qa_agree": qa[0],
        "qa_neutral": qa[1],
        "qa_conflict": qa[2],
        "expected_upside_pct": upside,
        "expected_downside_pct": downside,
        "upside_downside_ratio": _num(ratio),
        "horizon_days": _num(snap.horizon_days),
        "journal_context_used": _num(snap.journal_context_used),
        "quant_score": _num(snap.quant_score),
        "quant_prob": _num(snap.quant_prob),
        "quant_veto": quant_veto_flag(snap.action, snap.quant_alignment),
        "regime_risk_on": regime[0],
        "regime_neutral": regime[1],
        "regime_risk_off": regime[2],
        "regime_crisis": regime[3],
        "regime_confidence": _num(snap.regime_confidence),
        "sector_code": sector_code(snap.sector),
        "sleeve_long_term": sleeve[0],
        "sleeve_tactical": sleeve[1],
        "vix": _num(snap.vix),
        "spy_ret_20d": _num(snap.spy_ret_20d),
        "calibrated_probability": (
            _num(calib.value, conviction) if calib is not None else conviction
        ),
        "calibration_n": float(calib.n_effective) if calib is not None else 0.0,
        "neighbor_count": float(len(pnls)),
        "neighbor_mean_pnl_pct": _num(sum(pnls) / len(pnls) if pnls else None),
        "neighbor_hit_rate": _num(sum(wins) / len(wins) if wins else None),
        # NaN-safe: None/NaN/non-positive equity (legacy rows) -> 0.0.
        "log10_equity_at_decision": (
            math.log10(_num(snap.equity_at_decision))
            if _num(snap.equity_at_decision) > 0.0
            else 0.0
        ),
    }
    assert tuple(row) == FEATURE_NAMES  # ordering contract, cheap and loud
    return row


def snapshot_from_experience(exp: DecisionExperience) -> DecisionSnapshot:
    """Training-path snapshot from a resolved experience row. See the class
    docstring for the fields experiences cannot populate (left ``None``)."""
    return DecisionSnapshot(
        conviction=exp.conviction,
        disagreement_score=exp.disagreement_score,
        quant_alignment=exp.quant_alignment,
        expected_upside_pct=exp.expected_upside_pct,
        expected_downside_pct=exp.expected_downside_pct,
        horizon_days=exp.horizon_days,
        action=exp.action,
        quant_score=exp.quant_expected_excess_return,
        quant_prob=exp.quant_downside_prob,
        regime=exp.regime,
        sector=exp.sector,
        sleeve=exp.sleeve,
        vix=exp.vix,
        equity_at_decision=exp.equity_at_decision,
    )


def snapshot_from_decision(
    verdict: CommitteeVerdict,
    *,
    quant: QuantScore | None = None,
    regime: RegimeState | None = None,
    sector: str | None = None,
    sleeve: str | None = None,
    vix: float | None = None,
    spy_ret_20d: float | None = None,
    equity_at_decision: float | None = None,
) -> DecisionSnapshot:
    """Decision-time snapshot for tonight's verdicts (no experience row yet)."""
    return DecisionSnapshot(
        conviction=verdict.conviction,
        disagreement_score=verdict.disagreement_score,
        dissent_count=len(verdict.dissents),
        has_self_critique=bool(verdict.self_critique.strip()),
        quant_alignment=verdict.quant_alignment.value,
        expected_upside_pct=verdict.expected_upside_pct,
        expected_downside_pct=verdict.expected_downside_pct,
        horizon_days=verdict.horizon_days,
        journal_context_used=len(verdict.journal_context_used),
        action=verdict.action.value,
        quant_score=quant.expected_excess_return if quant is not None else None,
        quant_prob=quant.downside_prob if quant is not None else None,
        regime=regime.regime.value if regime is not None else None,
        regime_confidence=regime.composite_score if regime is not None else None,
        sector=sector,
        sleeve=sleeve,
        vix=vix,
        spy_ret_20d=spy_ret_20d,
        equity_at_decision=equity_at_decision,
    )


def build_feature_row(
    exp: DecisionExperience,
    calib: CalibratedProbability | None,
    neighbors: Sequence[DecisionExperience],
) -> dict[str, float]:
    """Feature row for a resolved experience (training path)."""
    return build_features(snapshot_from_experience(exp), calib, neighbors)


def features_hash(names: Sequence[str] = FEATURE_NAMES) -> str:
    """Stable hash of the feature-name contract (registry ``features_hash``)."""
    return hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()[:16]


# --- calibrator (mirrors research/quant/models.py:_Calibrator) --------------
#
# Deliberately COPIED rather than imported: research and learning are both
# core modules, so the import would be legal under the import-linter
# contracts, but the repo keeps core modules siblings (domain-only imports in
# spirit). The wiring is ~30 lines; drift is caught by the trainer tests.


class _MetaCalibrator:
    """Isotonic on validation scores; Platt fallback below the positive floor;
    identity on degenerate slices. Same shape as the quant ``_Calibrator``."""

    def __init__(self, min_isotonic_positives: int = MIN_ISOTONIC_POSITIVES) -> None:
        self.min_isotonic_positives = min_isotonic_positives
        self.method = "identity"
        self._iso: Any = None
        self._platt: Any = None

    def fit(self, raw: np.ndarray, y_true: np.ndarray) -> _MetaCalibrator:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        positives = int(np.sum(y_true > 0.5))
        negatives = int(np.sum(y_true <= 0.5))
        if positives == 0 or negatives == 0 or len(y_true) < 4:
            self.method = "identity"
            return self
        if positives >= self.min_isotonic_positives:
            self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._iso.fit(raw, y_true)
            self.method = "isotonic"
        else:
            self._platt = LogisticRegression(random_state=SEED)
            self._platt.fit(raw.reshape(-1, 1), (y_true > 0.5).astype(int))
            self.method = "platt"
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(self._iso.predict(raw), dtype=float)
        if self.method == "platt":
            return np.asarray(self._platt.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


def brier(probs: Sequence[float], outcomes: Sequence[bool]) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray([1.0 if o else 0.0 for o in outcomes], dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else 0.0


def precision_by_conviction_bucket(
    frame: pd.DataFrame, probs: Sequence[float], outcomes: Sequence[bool]
) -> dict[str, float]:
    """OOS precision of predicted wins (p >= 0.5) per raw-conviction tercile."""
    edges = ((0.0, 1.0 / 3), (1.0 / 3, 2.0 / 3), (2.0 / 3, 1.0 + 1e-9))
    conviction = np.asarray(frame["conviction"], dtype=float)
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray([1.0 if o else 0.0 for o in outcomes], dtype=float)
    out: dict[str, float] = {}
    for lo, hi in edges:
        mask = (conviction >= lo) & (conviction < hi) & (p >= 0.5)
        key = f"precision_conviction_{lo:.2f}_{min(hi, 1.0):.2f}"
        out[key] = float(y[mask].mean()) if int(mask.sum()) else float("nan")
    return {k: v for k, v in out.items() if math.isfinite(v)}


class MetaModel:
    """Win/loss classifier + calibration; ``effective_probability`` IS the
    calibrated ``predict_proba`` output. joblib-picklable end to end."""

    def __init__(self, algo: str = "gradient_boosting", *, seed: int = SEED) -> None:
        self.algo = algo
        self.seed = seed
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        self._model: Any = None
        self._calibrator: _MetaCalibrator | None = None
        self._base_rate: float = 0.5

    def _make_estimator(self) -> Any:
        if self.algo == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(random_state=self.seed, max_iter=1000)
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=self.seed)

    def train(
        self,
        features_df: pd.DataFrame,
        labels: Sequence[bool],
        time_order: Sequence[Any],
    ) -> dict[str, float]:
        """Time-ordered fit + tail calibration/evaluation.

        Refuses (``ValueError``) below MIN_TRAIN_EXPERIENCES — a meta-model
        trained on a handful of trades is noise wearing a lab coat. Returns
        OOS metrics: ``oos_brier``, ``base_rate_brier`` (constant train-base-
        rate predictor on the same tail), ``oos_n``, ``train_n``, plus
        precision-by-conviction-bucket keys.
        """
        n = len(features_df)
        if n != len(labels) or n != len(time_order):
            raise ValueError("features, labels and time_order must align")
        if n < MIN_TRAIN_EXPERIENCES:
            raise ValueError(
                f"refusing to train on {n} experiences "
                f"(hard gate: {MIN_TRAIN_EXPERIENCES} resolved live experiences)"
            )
        frame = features_df.loc[:, list(FEATURE_NAMES)].astype(float).fillna(0.0)
        order = np.argsort(np.asarray(time_order, dtype="object"), kind="stable")
        frame = frame.iloc[order].reset_index(drop=True)
        y = np.asarray([1.0 if b else 0.0 for b in labels], dtype=float)[order]

        n_valid = max(1, round(n * VALID_FRACTION))
        n_fit = n - n_valid
        x_fit, y_fit = frame.iloc[:n_fit], y[:n_fit]
        x_val, y_val = frame.iloc[n_fit:], y[n_fit:]
        self._base_rate = float(np.mean(y_fit)) if n_fit else 0.5

        if len(np.unique(y_fit)) < 2:  # degenerate history — constant model
            self._model = None
            self._calibrator = None
            oos = np.full(len(x_val), self._base_rate)
        else:
            self._model = self._make_estimator()
            self._model.fit(x_fit, (y_fit > 0.5).astype(int))
            raw_val = np.asarray(self._model.predict_proba(x_val)[:, 1], dtype=float)
            self._calibrator = _MetaCalibrator().fit(raw_val, y_val)
            oos = self._calibrator.transform(raw_val)

        outcomes = [bool(v > 0.5) for v in y_val]
        metrics: dict[str, float] = {
            "train_n": float(n_fit),
            "oos_n": float(len(x_val)),
            "oos_brier": brier(oos.tolist(), outcomes),
            "base_rate_brier": brier([self._base_rate] * len(outcomes), outcomes),
            "base_rate": self._base_rate,
        }
        metrics.update(precision_by_conviction_bucket(x_val, oos.tolist(), outcomes))
        return metrics

    def predict_proba(self, rows: Sequence[Mapping[str, float]] | pd.DataFrame) -> list[float]:
        """Calibrated P(win) per row — the ``effective_probability``."""
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        frame = frame.loc[:, list(self.feature_names)].astype(float).fillna(0.0)
        if self._model is None:
            return [self._base_rate] * len(frame)
        raw = np.asarray(self._model.predict_proba(frame)[:, 1], dtype=float)
        if self._calibrator is not None:
            raw = self._calibrator.transform(raw)
        return [float(min(max(v, 0.0), 1.0)) for v in raw]
