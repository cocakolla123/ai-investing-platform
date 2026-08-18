"""Candidate-feature audition harness (ADR-0015) — pure logic + seams.

A "candidate" is a :class:`~dynamo.research.quant.catalog.FeatureSpec`-shaped
object loaded from a user-supplied python file. The audition runs the SAME
honest evaluation machinery production features went through:

1. **Alphalens signal audit** — ``dynamo.backtest.signal_audit.audit_factor``
   on the candidate series alone (optional extra; degrades to
   ``{"available": False}`` exactly like existing call sites).
2. **Incremental walk-forward value test** — does adding the candidate to the
   existing feature panel improve pooled OOS rank IC? Both arms share the
   identical purged/embargoed folds (same sessions, same
   ``WalkForwardConfig`` → ``generate_folds`` is deterministic).
3. **Permutation null + trial-counted verdict** — see below.

Candidate-file contract (documented for authors; enforced by
:func:`load_candidate`):

* the file must expose a module-level ``CANDIDATE`` that is an instance of
  ``dynamo.research.quant.catalog.FeatureSpec``;
* ``CANDIDATE.name`` must be a new name (not already in ``FEATURE_CATALOG``)
  and a valid python identifier;
* ``CANDIDATE.fn`` is a ``FeatureFn``: ``FeatureContext -> pd.Series`` indexed
  by the universe, NaN for missing inputs, never raising (raises are caught
  and become all-NaN, mirroring ``build_raw_features``);
* ``CANDIDATE.group`` chooses normalization: groups in ``RAW_GROUPS``
  (``macro``, ``calendar``) stay raw; anything else is cross-sectionally
  winsorized + z-scored per session, exactly like catalog groups 1-8.

Null-distribution design (the multiple-testing verdict), documented honestly:

The audition statistic is the mean per-session cross-sectional Spearman rank
IC between the candidate and the 20d excess-return label. The null is built
by **cross-sectional permutation**: within each session, the candidate's
values are shuffled across symbols (N=200 draws), which destroys any
candidate↔label association while preserving each session's value
distribution and the label structure exactly. We report where the real IC
falls in that null (``null_percentile``).

For the family-trials adjustment we reuse
``dynamo.backtest.metrics.expected_max_sharpe`` — the Bailey & López de Prado
E[max over N iid trials] formula. The mapping is structural, not a Sharpe
claim: the formula gives the expected maximum of N draws from a zero-mean
distribution with a given variance, and we feed it the *empirically
estimated* variance of the permutation-null ICs. The result is "the best
mean IC a zero-skill candidate family of this size would be expected to
show" — the bar the real IC must clear once the family's full trial count is
on the table. This is cleaner to defend than translating IC into
Sharpe-space (no return-distribution assumptions are smuggled in; the null
variance comes straight from the data via the permutations).

This module is pure of the database: callers inject ``feature_inputs_for``
(the point-in-time gateway seam), an optional panel cache
(``PanelFeatureCache``-shaped: ``get(day)`` / ``put(day, frame)``), and the
label bars frame. File I/O here is limited to loading the candidate file and
appending the research log (same stance as ``panel_cache``).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from dynamo.backtest.metrics import expected_max_sharpe
from dynamo.backtest.signal_audit import audit_factor
from dynamo.backtest.walkforward import WalkForwardConfig
from dynamo.domain.ports import ForecastModel
from dynamo.research.quant.catalog import (
    FEATURE_CATALOG,
    RAW_GROUPS,
    FeatureInputs,
    FeatureSpec,
    build_context,
)
from dynamo.research.quant.features import build_features, build_labels, normalize_features
from dynamo.research.quant.training import train_and_evaluate

__all__ = [
    "DEFAULT_N_PERMUTATIONS",
    "IC_DELTA_MIN",
    "LABEL_COLUMN",
    "NULL_PERCENTILE_MIN",
    "AuditionPanel",
    "AuditionVerdict",
    "CandidateLoadError",
    "append_research_log",
    "build_audition_panel",
    "decide_verdict",
    "format_log_row",
    "incremental_value_test",
    "load_candidate",
    "permutation_null_test",
]

LABEL_COLUMN = "excess_return_20d"
MIN_CROSS_SECTION = 3  # same floor as training._spearman_ic
DEFAULT_N_PERMUTATIONS = 200

# --- Verdict thresholds (constants, with rationale) -----------------------
#
# IC_DELTA_MIN = 0.0: the candidate must ADD pooled OOS rank IC on top of the
# existing feature panel. Zero is deliberate — the walk-forward delta is
# already out-of-sample and fold-purged, so any positive increment is real
# information the panel lacked; demanding a large delta here would double-
# count the significance requirement that the permutation null owns.
IC_DELTA_MIN = 0.0
# NULL_PERCENTILE_MIN = 95.0: the standalone IC must beat >= 95% of the
# cross-sectional permutation null — a one-sided 5% test against "this
# candidate is cross-sectional noise". With N=200 permutations the estimate's
# resolution is 0.5 percentile points, comfortably fine for a 95th cut.
NULL_PERCENTILE_MIN = 95.0
# The third bar has no constant: the real IC must exceed
# expected_max_sharpe(family_trials, var(null ICs)) — it rises automatically
# with every audition the family burns, which is the whole point of trial
# counting (small universe => wide null => high bar; many trials => higher).


class CandidateLoadError(RuntimeError):
    """The candidate file does not satisfy the CANDIDATE contract."""


def load_candidate(path: str | Path) -> FeatureSpec:
    """Load ``CANDIDATE: FeatureSpec`` from a python file (contract above)."""
    file = Path(path)
    if not file.is_file():
        raise CandidateLoadError(f"candidate file not found: {file}")
    spec = importlib.util.spec_from_file_location(f"_alpha_candidate_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise CandidateLoadError(f"cannot import candidate file: {file}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # syntax errors, import errors, top-level raises
        raise CandidateLoadError(f"candidate file failed to execute: {exc}") from exc
    candidate = getattr(module, "CANDIDATE", None)
    if candidate is None:
        raise CandidateLoadError(
            f"{file} exposes no module-level CANDIDATE (contract: CANDIDATE: FeatureSpec)"
        )
    if not isinstance(candidate, FeatureSpec):
        raise CandidateLoadError(
            f"CANDIDATE must be a dynamo.research.quant.catalog.FeatureSpec, "
            f"got {type(candidate).__name__}"
        )
    if not candidate.name.isidentifier():
        raise CandidateLoadError(f"candidate name {candidate.name!r} must be a valid identifier")
    if candidate.name in FEATURE_CATALOG:
        raise CandidateLoadError(
            f"candidate name {candidate.name!r} collides with the existing "
            "feature catalog — candidates must be new features"
        )
    if not callable(candidate.fn):
        raise CandidateLoadError("CANDIDATE.fn must be callable (FeatureContext -> Series)")
    return candidate


# --------------------------------------------------------------------------
# panel assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditionPanel:
    """Everything the audition evaluates, pooled over the window.

    ``base_features``/``labels`` are (ts, symbol)-MultiIndexed frames;
    ``candidate`` is the normalized candidate series on the same index;
    ``prices`` is the wide adjusted-close frame the alphalens audit needs.
    """

    base_features: pd.DataFrame
    candidate: pd.Series
    labels: pd.DataFrame
    prices: pd.DataFrame
    sessions: tuple[date, ...]


def _normalize_candidate(raw: pd.Series, group: str) -> pd.Series:
    """Catalog-parity normalization: raw groups stay raw, others winsor+z."""
    if group in RAW_GROUPS:
        return raw.astype(float)
    frame = normalize_features(pd.DataFrame({"candidate": raw.astype(float)}))
    return cast(pd.Series, frame["candidate"])


def _wide_prices(label_bars: pd.DataFrame | None) -> pd.DataFrame:
    if label_bars is None or label_bars.empty or "ts" not in label_bars.columns:
        return pd.DataFrame()
    df = label_bars.copy()
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    panel = df.pivot_table(index="ts", columns="symbol", values=price_col, aggfunc="last")
    return panel.sort_index()


def build_audition_panel(
    candidate: FeatureSpec,
    feature_inputs_for: Callable[[date], FeatureInputs],
    label_bars: pd.DataFrame | None,
    sector_map: Mapping[str, str],
    universe: Sequence[str],
    sessions: Sequence[date],
    *,
    horizon_sessions: int = 20,
    cache: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> AuditionPanel:
    """Pooled audition panel over the window.

    Base features reuse ``cache`` (a ``PanelFeatureCache``-shaped object)
    when provided — a cache hit skips rebuilding the 51-feature frame. The
    CANDIDATE series is always computed fresh through ``build_context`` on
    the point-in-time ``feature_inputs_for(day)`` frames: candidates are not
    in the cache key, so caching them would poison reuse.

    Mirrors ``quant_train.build_panel`` semantics: the trailing
    ``horizon_sessions`` sessions are excluded (their forward labels cannot
    be complete), and candidate raises become NaN, never a crash.
    """
    universe = [str(s) for s in universe]
    feature_sessions = list(sessions)[: -horizon_sessions or None]
    if horizon_sessions >= len(sessions):
        feature_sessions = []
    base_frames: list[pd.DataFrame] = []
    cand_frames: list[pd.Series] = []
    label_frames: list[pd.DataFrame] = []
    keys: list[pd.Timestamp] = []
    for i, day in enumerate(feature_sessions):
        inputs: FeatureInputs | None = None
        base = cache.get(day) if cache is not None else None
        if base is None:
            inputs = feature_inputs_for(day)
            base = build_features(inputs, day, universe)
            if cache is not None:
                cache.put(day, base)
        else:
            base = base.reindex(universe)
        if inputs is None:
            inputs = feature_inputs_for(day)
        ctx = build_context(inputs, day, universe)
        try:
            raw = candidate.fn(ctx)
        except Exception:  # candidate parity with catalog: NaN, never raise
            raw = pd.Series(np.nan, index=pd.Index(universe), dtype=float)
        raw = raw.reindex(universe).astype(float)
        cand_frames.append(_normalize_candidate(raw, candidate.group))
        base_frames.append(base)
        label_frames.append(
            build_labels(label_bars, sector_map, day, horizon_sessions=horizon_sessions).reindex(
                universe
            )
        )
        keys.append(cast(pd.Timestamp, pd.Timestamp(day)))
        if progress is not None and (i + 1) % 25 == 0:
            progress(f"audition panel: {i + 1}/{len(feature_sessions)} sessions built")
    if not base_frames:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["ts", "symbol"])
        return AuditionPanel(
            base_features=pd.DataFrame(index=empty_idx),
            candidate=pd.Series(dtype=float, index=empty_idx, name=candidate.name),
            labels=pd.DataFrame(index=empty_idx),
            prices=_wide_prices(label_bars),
            sessions=(),
        )
    base_panel = pd.concat(base_frames, keys=keys, names=["ts", "symbol"])
    cand_panel = pd.concat(cand_frames, keys=keys, names=["ts", "symbol"])
    cand_panel.name = candidate.name
    labels_panel = pd.concat(label_frames, keys=keys, names=["ts", "symbol"])
    return AuditionPanel(
        base_features=base_panel,
        candidate=cast(pd.Series, cand_panel),
        labels=labels_panel,
        prices=_wide_prices(label_bars),
        sessions=tuple(feature_sessions),
    )


# --------------------------------------------------------------------------
# evaluation (a) — alphalens signal audit (guarded degradation inside)
# --------------------------------------------------------------------------


def signal_audit(panel: AuditionPanel) -> dict[str, Any]:
    """``audit_factor`` on the candidate alone; ``available: False`` without
    the analysis extra or on preparation failure (existing call-site parity)."""
    factor = panel.candidate.dropna()
    if factor.empty or panel.prices.empty:
        return {"available": False, "reason": "no candidate values or prices"}
    return audit_factor(factor, panel.prices)


# --------------------------------------------------------------------------
# evaluation (b) — incremental walk-forward value test
# --------------------------------------------------------------------------


def incremental_value_test(
    panel: AuditionPanel,
    *,
    wf_config: WalkForwardConfig,
    model_factory: Callable[[], ForecastModel],
) -> dict[str, Any]:
    """Pooled OOS rank IC with vs without the candidate, identical folds.

    Both arms run ``train_and_evaluate`` on the SAME sessions with the SAME
    ``wf_config`` — ``generate_folds`` is deterministic, so the folds are
    bit-identical; the only difference is the one extra column.
    """
    features_without = panel.base_features
    features_with = panel.base_features.join(panel.candidate.to_frame())
    results: dict[str, dict[str, Any]] = {}
    for arm, features in (("without", features_without), ("with", features_with)):
        report = train_and_evaluate(
            features,
            panel.labels,
            task="rank",
            target=LABEL_COLUMN,
            model_factories={"model": model_factory},
            baseline_factories={},  # arms compare against each other, not ETS
            wf_config=wf_config,
        )
        pooled = report["pooled"]["model"]
        results[arm] = {
            "rank_ic_pooled": pooled["rank_ic_pooled"],
            "rank_ic_t_stat": pooled["rank_ic_t_stat"],
            "fold_ics": [f["models"]["model"]["rank_ic"] for f in report["folds"]],
            "n_folds": report["n_folds"],
        }
    ic_with = results["with"]["rank_ic_pooled"]
    ic_without = results["without"]["rank_ic_pooled"]
    delta = (
        float(ic_with) - float(ic_without)
        if ic_with is not None and ic_without is not None
        else float("nan")
    )
    return {
        "ic_with": ic_with,
        "ic_without": ic_without,
        "ic_delta": delta,
        "n_folds": results["with"]["n_folds"],
        "fold_ics_with": results["with"]["fold_ics"],
        "fold_ics_without": results["without"]["fold_ics"],
    }


# --------------------------------------------------------------------------
# evaluation (c) — permutation null
# --------------------------------------------------------------------------


def _date_rank_groups(
    candidate: pd.Series, label: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-session (candidate ranks, label ranks) for sessions with a usable
    cross-section (>= MIN_CROSS_SECTION pairs, both sides non-constant)."""
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    frame = pd.DataFrame({"c": candidate.astype(float), "y": label.astype(float)}).dropna()
    for _, sub in frame.groupby(level=0, sort=True):
        if len(sub) < MIN_CROSS_SECTION:
            continue
        if sub["c"].nunique() < 2 or sub["y"].nunique() < 2:
            continue
        cr = sub["c"].rank(method="average").to_numpy(dtype=float)
        yr = sub["y"].rank(method="average").to_numpy(dtype=float)
        groups.append((cr, yr))
    return groups


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom <= 0:
        return float("nan")
    return float(a @ b) / denom


def permutation_null_test(
    candidate: pd.Series,
    label: pd.Series,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 7,
) -> dict[str, Any]:
    """Cross-sectional permutation null for the mean per-session rank IC.

    Real statistic: mean over sessions of Spearman(candidate, label) within
    the session (Pearson on average ranks — exact Spearman with ties).
    Null draws: within EVERY session the candidate ranks are permuted across
    symbols, association destroyed, marginals preserved; the statistic is
    recomputed. ``null_percentile`` is the real statistic's midrank position
    in the null (ties count half).
    """
    groups = _date_rank_groups(candidate, label)
    if not groups:
        return {
            "real_ic": None,
            "null_percentile": None,
            "null_ic_mean": None,
            "null_ic_std": None,
            "n_sessions_used": 0,
            "n_permutations": int(n_permutations),
        }
    real_ics = [_pearson(cr, yr) for cr, yr in groups]
    real_ic = float(np.nanmean(real_ics))
    rng = np.random.default_rng(seed)
    null_ics = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        draws = [_pearson(rng.permutation(cr), yr) for cr, yr in groups]
        null_ics[k] = float(np.nanmean(draws))
    below = float(np.sum(null_ics < real_ic))
    ties = float(np.sum(null_ics == real_ic))
    percentile = 100.0 * (below + 0.5 * ties) / float(n_permutations)
    return {
        "real_ic": real_ic,
        "null_percentile": percentile,
        "null_ic_mean": float(null_ics.mean()),
        "null_ic_std": float(null_ics.std(ddof=1)) if n_permutations >= 2 else 0.0,
        "n_sessions_used": len(groups),
        "n_permutations": int(n_permutations),
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditionVerdict:
    passed: bool
    reasons: tuple[str, ...]  # empty when passed; failed checks otherwise
    ic_delta: float
    real_ic: float
    null_percentile: float
    expected_max_null_ic: float  # the family-trials-adjusted bar
    family_trials: int

    def as_dict(self) -> dict[str, Any]:
        def _num(v: float) -> float | None:
            return None if math.isnan(v) else float(v)

        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "ic_delta": _num(self.ic_delta),
            "real_ic": _num(self.real_ic),
            "null_percentile": _num(self.null_percentile),
            "expected_max_null_ic": _num(self.expected_max_null_ic),
            "family_trials": self.family_trials,
        }


def decide_verdict(
    *,
    incremental: Mapping[str, Any],
    null: Mapping[str, Any],
    family_trials: int,
) -> AuditionVerdict:
    """Three bars, all must clear (thresholds + rationale at module top):

    1. ``ic_delta > IC_DELTA_MIN`` — adds OOS value over the existing panel;
    2. ``null_percentile >= NULL_PERCENTILE_MIN`` — beats cross-sectional noise;
    3. ``real_ic > expected_max_sharpe(family_trials, var(null ICs))`` — the
       trials-adjusted bar: the expected BEST IC a zero-skill family of this
       size would show (see module docstring for why this mapping is honest).
    """

    def _f(value: Any) -> float:
        return float(value) if value is not None else float("nan")

    ic_delta = _f(incremental.get("ic_delta"))
    real_ic = _f(null.get("real_ic"))
    percentile = _f(null.get("null_percentile"))
    null_std = float(null.get("null_ic_std") or 0.0)
    trials = max(int(family_trials), 1)
    expected_max = expected_max_sharpe(trials, null_std * null_std)
    reasons: list[str] = []
    if not ic_delta > IC_DELTA_MIN:
        reasons.append(
            f"incremental OOS rank IC delta {ic_delta:+.4f} does not exceed "
            f"{IC_DELTA_MIN:+.4f} — the candidate adds nothing over the panel"
        )
    if not percentile >= NULL_PERCENTILE_MIN:
        reasons.append(
            f"null percentile {percentile:.1f} < {NULL_PERCENTILE_MIN:.0f} — "
            "indistinguishable from cross-sectional noise"
        )
    if not real_ic > expected_max:
        reasons.append(
            f"real IC {real_ic:+.4f} does not clear the {trials}-trial "
            f"expected-max-under-null bar {expected_max:+.4f}"
        )
    return AuditionVerdict(
        passed=not reasons,
        reasons=tuple(reasons),
        ic_delta=ic_delta,
        real_ic=real_ic,
        null_percentile=percentile,
        expected_max_null_ic=expected_max,
        family_trials=trials,
    )


# --------------------------------------------------------------------------
# research log (docs/alpha_research_log.md) — append-only trial record
# --------------------------------------------------------------------------

LOG_HEADER = """# Alpha research log

Append-only, trial-by-trial audition record (ADR-0015). EVERY audition lands
here, pass or fail — hiding failed trials is how false discoveries happen.

| date | hypothesis | family | family trials | IC delta | null pctile | verdict |
|---|---|---|---|---|---|---|
"""


def format_log_row(
    *,
    day: date,
    hypothesis_id: str,
    family: str,
    family_trials: int,
    verdict: AuditionVerdict,
) -> str:
    def _fmt(v: float, spec: str) -> str:
        return "n/a" if math.isnan(v) else format(v, spec)

    outcome = "PASS" if verdict.passed else "FAIL"
    return (
        f"| {day.isoformat()} | {hypothesis_id} | {family} | {family_trials} "
        f"| {_fmt(verdict.ic_delta, '+.4f')} | {_fmt(verdict.null_percentile, '.1f')} "
        f"| {outcome} |\n"
    )


def append_research_log(path: str | Path, row: str) -> None:
    """Append one row, creating the file with its header when missing."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        file.write_text(LOG_HEADER, encoding="utf-8")
    with file.open("a", encoding="utf-8") as fh:
        fh.write(row)
