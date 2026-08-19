"""Train the quant bundle (backtest-campaign Phase 2) — the campaign unblocker.

Usage:
    python -m dynamo.cli.quant_train --train-start 2024-01-01 \
        --train-end 2025-12-31 --label campaign-v1 [--universe] \
        [--symbols AAPL MSFT ...] [--database-url URL]

Manual only — NEVER scheduled. ZERO LLM calls. The flow:

1. **Panel** — for every NYSE session in the window (minus the trailing label
   horizon), features come through ``HistoricalDataGateway.feature_inputs``
   at that date — the SAME loading path replay uses (both knowledge-time
   bounds on every frame), so training can never see data replay would not.
   This is deliberately the per-date gateway rather than a hand-rolled bulk
   query: slower against the real database (~minutes for a 2-year window)
   but provably point-in-time by construction. Labels come from one bars
   frame bounded at ``train_end`` via ``build_labels`` per date; because the
   labels look 20 sessions forward, the last ``horizon_sessions`` sessions of
   the window are feature-excluded (they cannot have complete labels).
2. **Walk-forward report card** — ``train_and_evaluate`` per task (ranker =
   excess return, downside classifier, vol) over purged/embargoed folds.
   The doc-08 default fold geometry needs 902 sessions; shorter windows
   (the campaign window is ~500) get an honestly scaled-down geometry
   (documented in :func:`walkforward_for`) — purge stays >= the label
   horizon so no fold ever trains on rows whose labels overlap its test set.
3. **Final models** — ``fit_final_model`` per task on the FULL window (the
   binary model calibrates on its internal chronological tail slice), saved
   via ``learning/artifacts.py`` (sha256 manifests) + one bundle manifest
   (``research/quant/bundle.py``).
4. **Registry** — ONE CANDIDATE ``ModelVersion`` (model_type='quant_bundle',
   id = --label, metrics = pooled out-of-fold metrics, artifact_path = the
   bundle manifest). Replay may use the candidate explicitly via
   ``--quant-model-version``; production scoring requires promotion.
5. **Go/no-go table** — per-fold rank IC, pooled IC t-stat vs the ship gate,
   dispersion. GO/NO-GO #1 of the campaign runbook: never proceed to the
   paid committee run on a degenerate model.

Gate: refuses (exit 2) when the panel has < 60 feature sessions or < 20
symbols carrying both a usable label and at least one feature value.

Crypto note (ADR-0013): crypto pairs ride along with NaN fundamentals /
insider / estimates features (the builders never raise on them) — the models
train on whatever features exist and ``feature_coverage`` reflects the holes
at scoring time. Symbols without a sector ETF (crypto included) get SPY as
the excess-return benchmark in the label map.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast

MIN_PANEL_SESSIONS = 60
MIN_PANEL_SYMBOLS = 20
HORIZON_SESSIONS = 20
BENCHMARK_FALLBACK = "SPY"
MODEL_NAME = "lgbm"
ALGO = "lightgbm"

# Rank-task label selection (--label-kind). 'sector_excess' is the historical
# default and keeps config hashes bit-identical to pre-residual runs;
# 'residual' ranks on factor-residualized idiosyncratic return
# (features.build_labels residual_return_{h}d — betas vs SPY + sector ETF).
LABEL_KINDS = ("sector_excess", "residual")
DEFAULT_LABEL_KIND = "sector_excess"


class PanelTooSmallError(RuntimeError):
    """The pooled panel is below the training gate — refuse (exit 2)."""


@dataclass(frozen=True)
class TrainingPanel:
    """Pooled (ts, symbol)-indexed features + labels over the window."""

    features: Any  # pd.DataFrame, MultiIndex (ts, symbol)
    labels: Any  # pd.DataFrame, same index, the three doc-04 targets
    sessions: tuple[date, ...]  # feature sessions (window minus horizon tail)


@dataclass(frozen=True)
class TrainReport:
    """Everything main() needs to print the table and register the candidate."""

    version: str
    bundle_path: Path
    feature_names: tuple[str, ...]
    n_samples: int
    n_sessions: int
    n_symbols: int
    config_hash: str
    fold_ic_dispersion: float
    task_reports: dict[str, dict[str, Any]]
    registry_metrics: dict[str, Any]
    lgbm_params: dict[str, Any] = field(default_factory=dict)
    feature_groups: tuple[str, ...] | None = None
    label_kind: str = DEFAULT_LABEL_KIND


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-quant-train",
        description="Train + register the quant model bundle (manual, never scheduled)",
    )
    parser.add_argument("--train-start", required=True, type=date.fromisoformat)
    parser.add_argument("--train-end", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--label",
        required=True,
        help="bundle version AND registry id (e.g. campaign-v1); must be unused",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--symbols", nargs="+", default=None, help="explicit training symbols"
    )
    group.add_argument(
        "--universe",
        action="store_true",
        help="train on the active tier-1/3 universe at --train-end (the default)",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--label-kind",
        choices=LABEL_KINDS,
        default=DEFAULT_LABEL_KIND,
        help="rank-task label: 'sector_excess' (historical default, "
        "reproducible config hashes) or 'residual' (factor-residualized "
        "return — beta_mkt x SPY + beta_sector x sector-ETF excess stripped "
        "out, betas from ~250 trailing sessions <= as_of)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the on-disk per-session feature cache (artifacts/panel_cache)",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="append one JSON line {session_i, n_sessions, elapsed_s} per built "
        "session — machine-readable liveness for piped/buffered runs",
    )
    tuning = parser.add_argument_group(
        "tuning",
        "campaign-v3 retune knobs: each flag is one LightGBM hyperparameter "
        "or a feature-subset selection, threaded into all three task models",
    )
    tuning.add_argument("--lgbm-max-depth", type=int, default=None)
    tuning.add_argument("--lgbm-min-data-in-leaf", type=int, default=None)
    tuning.add_argument("--lgbm-lambda-l2", type=float, default=None)
    tuning.add_argument(
        "--feature-groups",
        nargs="+",
        default=None,
        metavar="GROUP",
        help="train on ONLY these feature-catalog groups (e.g. momentum "
        "volatility). NOTE: the bundle records the subset feature names, so "
        "replay's default loader refuses it as feature drift — subset bundles "
        "are for walk-forward report cards, not replay/production scoring.",
    )
    return parser


# --------------------------------------------------------------------------
# panel assembly (pure given injected loaders — unit-testable without a DB)
# --------------------------------------------------------------------------


def _hms(seconds: float) -> str:
    """``H:MM:SS`` for progress lines (negative/NaN-safe: clamps to zero)."""
    total = 0 if not seconds > 0 else int(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def progress_file_writer(path: Path) -> Callable[[int, int, float], None]:
    """``on_session`` hook appending one JSON line per built session.

    Append-per-call (open/write/close) so every line is on disk the moment
    the session finishes — a hard kill loses nothing, and ``tail -f`` on the
    file gives live progress even when stdout is piped and block-buffered
    (the failure mode that hid a 2h training run)."""
    import json

    def write(session_i: int, n_sessions: int, elapsed_s: float) -> None:
        line = json.dumps(
            {
                "session_i": session_i,
                "n_sessions": n_sessions,
                "elapsed_s": round(float(elapsed_s), 3),
            }
        )
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return write


def label_sector_map(
    sectors: Mapping[str, str], universe: Sequence[str]
) -> dict[str, str]:
    """symbol -> excess-return benchmark: the sector SPDR where one exists,
    else SPY (crypto pairs and unclassified names — documented decision:
    market-relative excess is the honest fallback, NaN would silently drop
    them from the ranker's training rows)."""
    from dynamo.data.universe import default_sector_etf_map

    etf = default_sector_etf_map(dict(sectors))
    return {sym: etf.get(sym, BENCHMARK_FALLBACK) for sym in universe}


def build_panel(
    feature_inputs_for: Callable[[date], Any],
    label_bars: Any,
    sector_map: Mapping[str, str],
    universe: Sequence[str],
    sessions: Sequence[date],
    *,
    horizon_sessions: int = HORIZON_SESSIONS,
    progress: Callable[[str], None] | None = None,
    cache: Any | None = None,
    on_session: Callable[[int, int, float], None] | None = None,
) -> TrainingPanel:
    """Pooled panel: ``build_features`` per session (explicit as_of, the
    replay loading path) + ``build_labels`` per session from the shared
    label-bars frame. The trailing ``horizon_sessions`` sessions are excluded
    — their forward windows extend past ``train_end`` so their labels cannot
    be complete.

    ``cache`` (a ``PanelFeatureCache``-shaped object: ``get(day)`` /
    ``put(day, frame)``) short-circuits the per-session gateway build on a
    hit and persists each freshly built frame immediately (crash-safe
    incremental warm-up). Labels are always rebuilt — they are cheap slices
    of the preloaded ``label_bars`` frame.

    ``on_session(session_i, n_sessions, elapsed_s)`` fires after EVERY
    session (the ``--progress-file`` hook); ``progress`` lines every 25
    sessions carry elapsed + ETA from the observed per-session rate."""
    import time

    import pandas as pd

    from dynamo.research.quant.features import build_features, build_labels

    universe = [str(s) for s in universe]
    feature_sessions = list(sessions)[: -horizon_sessions or None]
    if horizon_sessions >= len(sessions):
        feature_sessions = []
    feature_frames: list[Any] = []
    label_frames: list[Any] = []
    keys: list[Any] = []
    cache_hits = 0
    started = time.monotonic()
    n_sessions = len(feature_sessions)
    for i, day in enumerate(feature_sessions):
        features = cache.get(day) if cache is not None else None
        if features is None:
            inputs = feature_inputs_for(day)
            features = build_features(inputs, day, universe)
            if cache is not None:
                cache.put(day, features)
        else:
            cache_hits += 1
            features = features.reindex(universe)
        labels = build_labels(
            label_bars, sector_map, day, horizon_sessions=horizon_sessions
        ).reindex(universe)
        feature_frames.append(features)
        label_frames.append(labels)
        keys.append(pd.Timestamp(day))
        elapsed = time.monotonic() - started
        if on_session is not None:
            on_session(i + 1, n_sessions, elapsed)
        if progress is not None and (i + 1) % 25 == 0:
            hits = f" ({cache_hits} cache hits)" if cache is not None else ""
            eta = elapsed / (i + 1) * (n_sessions - i - 1)
            progress(
                f"panel: {i + 1}/{n_sessions} sessions built{hits} — "
                f"elapsed {_hms(elapsed)}, ETA {_hms(eta)}"
            )
    if progress is not None and cache is not None and feature_sessions:
        progress(
            f"panel: feature cache served {cache_hits}/{len(feature_sessions)} sessions"
        )
    if not feature_frames:
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["ts", "symbol"])
        )
        return TrainingPanel(features=empty, labels=empty, sessions=())
    features_panel = pd.concat(feature_frames, keys=keys, names=["ts", "symbol"])
    labels_panel = pd.concat(label_frames, keys=keys, names=["ts", "symbol"])
    return TrainingPanel(
        features=features_panel,
        labels=labels_panel,
        sessions=tuple(feature_sessions),
    )


def lgbm_params_from_flags(
    max_depth: int | None, min_data_in_leaf: int | None, lambda_l2: float | None
) -> dict[str, Any]:
    """CLI tuning flags -> LightGBM sklearn-API params (canonical names, so
    lightgbm never warns about alias resolution). Unset flags are omitted —
    an empty dict means 'library defaults', identical to pre-tuning runs."""
    params: dict[str, Any] = {}
    if max_depth is not None:
        params["max_depth"] = int(max_depth)
    if min_data_in_leaf is not None:
        params["min_child_samples"] = int(min_data_in_leaf)  # = min_data_in_leaf
    if lambda_l2 is not None:
        params["reg_lambda"] = float(lambda_l2)  # = lambda_l2
    return params


def select_feature_groups(features: Any, groups: Sequence[str]) -> Any:
    """Column-subset ``features`` to the named FEATURE_CATALOG groups.

    Unknown group names and empty selections raise ``ValueError`` — a tuning
    typo must refuse loudly, never silently train on nothing."""
    from dynamo.research.quant.catalog import FEATURE_CATALOG, GROUP_TARGETS

    wanted = [str(g) for g in groups]
    unknown = sorted(set(wanted) - set(GROUP_TARGETS))
    if unknown:
        raise ValueError(
            f"unknown feature group(s) {unknown}; valid: {sorted(GROUP_TARGETS)}"
        )
    keep = [
        c
        for c in features.columns
        if c in FEATURE_CATALOG and FEATURE_CATALOG[c].group in set(wanted)
    ]
    if not keep:
        raise ValueError(
            f"feature groups {wanted} select no columns from the panel "
            f"({len(features.columns)} columns)"
        )
    return features[keep]


def check_panel(panel: TrainingPanel) -> tuple[int, int]:
    """(n_sessions, n_usable_symbols); raises :class:`PanelTooSmallError`
    below the gate. A usable symbol has >= 1 non-NaN label AND >= 1 non-NaN
    feature somewhere in the window."""
    import pandas as pd

    features = cast(pd.DataFrame, panel.features)
    labels = cast(pd.DataFrame, panel.labels)
    n_sessions = len(panel.sessions)
    if features.empty or labels.empty:
        raise PanelTooSmallError(
            f"panel is empty ({n_sessions} feature sessions) — check bars/universe backfills"
        )
    label_any = cast(pd.Series, labels.notna().any(axis=1))
    feature_any = cast(pd.Series, features.notna().any(axis=1))
    by_symbol_label = label_any.groupby(level="symbol").any()
    by_symbol_feature = feature_any.groupby(level="symbol").any()
    usable = (by_symbol_label & by_symbol_feature.reindex(by_symbol_label.index, fill_value=False))
    n_symbols = int(usable.sum())
    problems = []
    if n_sessions < MIN_PANEL_SESSIONS:
        problems.append(
            f"{n_sessions} feature sessions < {MIN_PANEL_SESSIONS} minimum"
        )
    if n_symbols < MIN_PANEL_SYMBOLS:
        problems.append(
            f"{n_symbols} usable symbols (label + features) < {MIN_PANEL_SYMBOLS} minimum"
        )
    if problems:
        raise PanelTooSmallError(
            "refusing to train on a degenerate panel: " + "; ".join(problems)
        )
    return n_sessions, n_symbols


def walkforward_for(n_sessions: int) -> Any:
    """Fold geometry for the window. Doc-08 defaults when the window is long
    enough (>= 902 sessions); otherwise an honestly scaled-down geometry —
    purge stays 20 sessions (== the label horizon, so a fold can never train
    on rows whose forward windows overlap its test set), embargo stays 10,
    test ~15% of the window, train fills the rest leaving room for >= 2 folds
    when possible."""
    from dynamo.backtest.walkforward import WalkForwardConfig

    default = WalkForwardConfig()
    needed = default.train_sessions + default.purge_sessions + default.test_sessions
    if n_sessions >= needed:
        return default
    purge, embargo = default.purge_sessions, default.embargo_sessions
    test = max(10, int(n_sessions * 0.15))
    train = n_sessions - purge - 2 * test
    if train < test:  # tiny window — settle for a single fold
        train = n_sessions - purge - test
    if train <= 0:
        raise PanelTooSmallError(
            f"{n_sessions} sessions cannot form a single purged walk-forward fold"
        )
    return WalkForwardConfig(
        train_sessions=train,
        test_sessions=test,
        purge_sessions=purge,
        embargo_sessions=embargo,
        roll_sessions=test,
    )


# --------------------------------------------------------------------------
# training + artifact save (injectable — the smoke test drives this directly)
# --------------------------------------------------------------------------


def train_quant_bundle(
    panel: TrainingPanel,
    *,
    version: str,
    artifact_dir: str | Path,
    train_start: date,
    train_end: date,
    wf_config: Any | None = None,
    lgbm_params: Mapping[str, Any] | None = None,
    feature_groups: Sequence[str] | None = None,
    label_kind: str = DEFAULT_LABEL_KIND,
) -> TrainReport:
    """Walk-forward evaluation for the three tasks, final-model fits on the
    full window, artifact + bundle-manifest save. Pure of the database —
    callers hand in the assembled panel.

    ``lgbm_params`` (retune knobs) are handed to every task's
    :class:`LightGBMModel`; ``feature_groups`` column-subsets the panel by
    catalog group BEFORE the gate check. Both flow into the config hash and
    the registry hyperparams so a tuned run is never confusable with a
    default one. ``label_kind`` selects the RANK task's target column
    (``sector_excess`` -> excess_return_20d, ``residual`` ->
    residual_return_20d); the default leaves the config hash bit-identical
    to pre-residual runs, and a non-default kind moves the hash AND travels
    in the registry hyperparams."""
    import math

    import numpy as np

    from dynamo.backtest.walkforward import compute_config_hash
    from dynamo.research.quant.bundle import save_bundle
    from dynamo.research.quant.models import LightGBMModel, StatsforecastBaseline
    from dynamo.research.quant.training import (
        feature_quantiles,
        fit_final_model,
        train_and_evaluate,
    )

    if feature_groups is not None:
        panel = TrainingPanel(
            features=select_feature_groups(panel.features, feature_groups),
            labels=panel.labels,
            sessions=panel.sessions,
        )
    if label_kind not in LABEL_KINDS:
        raise ValueError(f"unknown label kind {label_kind!r}; valid: {list(LABEL_KINDS)}")
    tuned = dict(lgbm_params or {})
    n_sessions, n_symbols = check_panel(panel)
    features, labels = panel.features, panel.labels
    feature_names = tuple(str(c) for c in features.columns)
    rank_target = "residual_return_20d" if label_kind == "residual" else None
    wf = wf_config if wf_config is not None else walkforward_for(n_sessions)
    config_payload: dict[str, Any] = {
        "version": version,
        "train_start": train_start,
        "train_end": train_end,
        "feature_names": list(feature_names),
        "model": MODEL_NAME,
        "wf": {
            "train": wf.train_sessions,
            "test": wf.test_sessions,
            "purge": wf.purge_sessions,
            "embargo": wf.embargo_sessions,
            "roll": wf.roll_sessions,
        },
    }
    if tuned:  # only when set: default-run hashes stay identical to pre-tuning
        config_payload["lgbm_params"] = {k: tuned[k] for k in sorted(tuned)}
    if label_kind != DEFAULT_LABEL_KIND:  # default runs keep pre-residual hashes
        config_payload["label_kind"] = label_kind
    config_hash = compute_config_hash(config_payload)
    # Historic average only as the CLI's bar-to-beat: the full baseline suite
    # (seasonal-naive, AutoETS) stays available through train_and_evaluate
    # directly, but per-symbol ETS fits across folds would dominate runtime
    # for zero effect on the doc-04 ship gates' primary comparisons.
    baselines = {
        "baseline-historic-average": lambda: StatsforecastBaseline("historic_average")
    }

    factories = {
        "rank": lambda: LightGBMModel("regression", params=tuned),
        "downside": lambda: LightGBMModel("binary", params=tuned),
        "vol": lambda: LightGBMModel("regression", params=tuned),
    }
    task_reports: dict[str, dict[str, Any]] = {}
    for task in ("rank", "downside", "vol"):
        task_reports[task] = train_and_evaluate(
            features,
            labels,
            task=task,  # type: ignore[arg-type]
            target=rank_target if task == "rank" else None,
            model_factories={MODEL_NAME: factories[task]},
            baseline_factories=baselines,
            wf_config=wf,
            config_hash=config_hash,
        )

    rank_fold_ics = [
        f["models"][MODEL_NAME]["rank_ic"]
        for f in task_reports["rank"]["folds"]
        if f["models"][MODEL_NAME]["rank_ic"] is not None
    ]
    dispersion = (
        float(np.std(np.asarray(rank_fold_ics, dtype=float), ddof=1))
        if len(rank_fold_ics) >= 2
        else 0.0
    )

    ranker = fit_final_model(
        features, labels, task="rank", model_factory=factories["rank"], target=rank_target
    )
    downside = fit_final_model(
        features, labels, task="downside", model_factory=factories["downside"]
    )
    vol = fit_final_model(features, labels, task="vol", model_factory=factories["vol"])

    n_samples = len(features)
    rank_pooled = task_reports["rank"]["pooled"][MODEL_NAME]
    down_pooled = task_reports["downside"]["pooled"][MODEL_NAME]
    vol_pooled = task_reports["vol"]["pooled"][MODEL_NAME]

    def _num(value: Any) -> float | None:
        if value is None:
            return None
        v = float(value)
        return None if math.isnan(v) else v

    registry_metrics: dict[str, Any] = {
        "rank_ic_pooled": _num(rank_pooled.get("rank_ic_pooled")),
        "rank_ic_t_stat": _num(rank_pooled.get("rank_ic_t_stat")),
        "rank_win_share": _num(rank_pooled.get("win_share")),
        "rank_fold_ics": [_num(v) for v in rank_fold_ics],
        "rank_ships_ok": bool(task_reports["rank"]["ships_ok"][MODEL_NAME]),
        "downside_brier_pooled": _num(down_pooled.get("brier_pooled")),
        "downside_auc_pooled": _num(down_pooled.get("auc_pooled")),
        "downside_base_rate_brier_pooled": _num(
            down_pooled.get("base_rate_brier_pooled")
        ),
        "vol_rmse_log_pooled": _num(vol_pooled.get("rmse_log_vol_pooled")),
        "vol_qlike_pooled": _num(vol_pooled.get("qlike_pooled")),
        "n_folds": int(task_reports["rank"]["n_folds"]),
        "n_samples": n_samples,
        "n_sessions": n_sessions,
        "n_symbols": n_symbols,
        "fold_ic_dispersion": dispersion,
    }

    bundle_path = save_bundle(
        artifact_dir,
        version,
        ranker=ranker,
        downside=downside,
        vol=vol,
        feature_names=feature_names,
        train_feature_quantiles=feature_quantiles(features),
        train_start=train_start,
        train_end=train_end,
        n_samples=n_samples,
        config_hash=config_hash,
        metrics={"registry": registry_metrics, "tasks": task_reports},
        fold_ic_dispersion=dispersion,
        vol_is_log=True,
    )
    return TrainReport(
        version=version,
        bundle_path=bundle_path,
        feature_names=feature_names,
        n_samples=n_samples,
        n_sessions=n_sessions,
        n_symbols=n_symbols,
        config_hash=config_hash,
        fold_ic_dispersion=dispersion,
        task_reports=task_reports,
        registry_metrics=registry_metrics,
        lgbm_params=dict(tuned),
        feature_groups=tuple(feature_groups) if feature_groups is not None else None,
        label_kind=label_kind,
    )


def registry_model_version(report: TrainReport, train_start: date, train_end: date) -> Any:
    """The CANDIDATE row for the trained bundle (status is forced to
    candidate by ``insert_candidate``; production requires promotion)."""
    from datetime import UTC, datetime

    from dynamo.domain.models.learning import ModelVersion
    from dynamo.research.quant.bundle import feature_names_hash

    return ModelVersion(
        id=report.version,
        model_type="quant_bundle",
        algo=ALGO,
        features_hash=feature_names_hash(report.feature_names),
        train_start=train_start,
        train_end=train_end,
        hyperparams={
            "model": MODEL_NAME,
            "tasks": ["rank", "downside", "vol"],
            "config_hash": report.config_hash,
            # Retune knobs travel with the candidate row — a tuned bundle is
            # never confusable with a default one. Empty/None on default runs.
            **({"lgbm_params": dict(report.lgbm_params)} if report.lgbm_params else {}),
            **(
                {"feature_groups": list(report.feature_groups)}
                if report.feature_groups is not None
                else {}
            ),
            **(
                {"label_kind": report.label_kind}
                if report.label_kind != DEFAULT_LABEL_KIND
                else {}
            ),
        },
        metrics=dict(report.registry_metrics),
        artifact_path=str(report.bundle_path),
        created_at=datetime.now(UTC),
    )


def print_go_no_go(report: TrainReport, out: Callable[[str], None] = print) -> None:
    """The GO/NO-GO #1 table: per-fold rank IC, pooled t-stat vs the ship
    gate, dispersion, and the secondary-task pooled metrics."""
    from dynamo.research.quant.training import SHIP_T_STAT

    m = report.registry_metrics

    def _fmt(value: Any, spec: str = "+.4f") -> str:
        return "n/a" if value is None else format(float(value), spec)

    out(f"GO/NO-GO — quant bundle {report.version!r}")
    out(
        f"  panel: {report.n_samples} rows, {report.n_sessions} sessions, "
        f"{report.n_symbols} usable symbols, {len(report.feature_names)} features"
    )
    out("  fold  test window               rank IC")
    for fold in report.task_reports["rank"]["folds"]:
        ic = fold["models"][MODEL_NAME]["rank_ic"]
        out(
            f"  {fold['fold_id']:>4}  {fold['test_start']} .. {fold['test_end']}  "
            f"{_fmt(ic)}"
        )
    out(
        f"  pooled rank IC {_fmt(m['rank_ic_pooled'])}  "
        f"t-stat {_fmt(m['rank_ic_t_stat'], '.2f')} "
        f"(ship gate >= {SHIP_T_STAT:.2f})  "
        f"win share {_fmt(m['rank_win_share'], '.2f')}"
    )
    out(f"  fold IC dispersion {m['fold_ic_dispersion']:.4f}")
    out(
        f"  downside: Brier {_fmt(m['downside_brier_pooled'], '.4f')} "
        f"vs base-rate {_fmt(m['downside_base_rate_brier_pooled'], '.4f')}, "
        f"AUC {_fmt(m['downside_auc_pooled'], '.3f')}"
    )
    out(
        f"  vol: RMSE(log) {_fmt(m['vol_rmse_log_pooled'], '.4f')}, "
        f"QLIKE {_fmt(m['vol_qlike_pooled'], '.4f')}"
    )
    verdict = "GO" if m["rank_ships_ok"] else "NO-GO (rank ship gates not met)"
    out(f"  VERDICT: {verdict} — never proceed to the paid run on a NO-GO.")


# --------------------------------------------------------------------------
# composition (real database + gateway)
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Lazy imports: composition layer, import side-effect free module.
    from dynamo.config.settings import Settings
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.learning import ModelRegistryRepository
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import _sessions_between
    from dynamo.research.quant.bundle import default_bundle_dir

    args = _build_parser().parse_args(argv)
    if args.train_end <= args.train_start:
        print("--train-end must be after --train-start")
        return 1
    if args.feature_groups is not None:
        # Validate the group names NOW — a typo must fail in milliseconds,
        # not after an hours-long panel build.
        from dynamo.research.quant.catalog import GROUP_TARGETS

        unknown = sorted(set(args.feature_groups) - set(GROUP_TARGETS))
        if unknown:
            print(
                f"refused: unknown feature group(s) {unknown} — "
                f"valid: {sorted(GROUP_TARGETS)}"
            )
            return 1
    lgbm_params = lgbm_params_from_flags(
        args.lgbm_max_depth, args.lgbm_min_data_in_leaf, args.lgbm_lambda_l2
    )

    engine = make_engine(args.database_url or Settings().database_url)
    registry = ModelRegistryRepository(engine)
    if registry.status_of(args.label) is not None:
        print(
            f"refused: model version {args.label!r} already exists in the registry — "
            "pick a new --label (versions are immutable)."
        )
        return 1

    gw_end = HistoricalDataGateway(engine, args.train_end)
    universe = [str(s) for s in args.symbols] if args.symbols else gw_end.universe((1, 3))
    if not universe:
        print(
            f"refused: universe is empty at {args.train_end} — run "
            "`python -m dynamo.cli.backfill universe --as-of <date>` first."
        )
        return 1

    sector_map = label_sector_map(gw_end.sectors(universe), universe)
    sessions = _sessions_between(args.train_start, args.train_end)
    # flush=True on every progress print: piped/block-buffered stdout hid a
    # 2h panel build entirely — liveness must not depend on a tty.
    print(
        f"quant-train {args.label!r}: {len(universe)} symbols, "
        f"{len(sessions)} sessions {args.train_start} -> {args.train_end} "
        f"(last {HORIZON_SESSIONS} are label-horizon only)",
        flush=True,
    )
    label_bars = gw_end.bars(
        sorted({*universe, *sector_map.values(), BENCHMARK_FALLBACK}),
        (args.train_end - args.train_start).days + 60,
    )
    cache = None
    if not args.no_cache:
        from dynamo.research.quant.panel_cache import PanelFeatureCache

        cache = PanelFeatureCache(
            Path(Settings().artifact_dir) / "panel_cache", universe
        )
        print(f"panel cache: {cache.directory} (bypass with --no-cache)", flush=True)
    panel = build_panel(
        lambda d: HistoricalDataGateway(engine, d).feature_inputs(universe),
        label_bars,
        sector_map,
        universe,
        sessions,
        progress=lambda msg: print(msg, flush=True),
        cache=cache,
        on_session=(
            progress_file_writer(args.progress_file)
            if args.progress_file is not None
            else None
        ),
    )
    if lgbm_params or args.feature_groups or args.label_kind != DEFAULT_LABEL_KIND:
        print(
            f"tuning: lgbm_params={lgbm_params or '{}'} "
            f"feature_groups={args.feature_groups or 'all'} "
            f"label_kind={args.label_kind}",
            flush=True,
        )
    try:
        report = train_quant_bundle(
            panel,
            version=args.label,
            artifact_dir=default_bundle_dir(),
            train_start=args.train_start,
            train_end=args.train_end,
            lgbm_params=lgbm_params,
            feature_groups=args.feature_groups,
            label_kind=args.label_kind,
        )
    except PanelTooSmallError as exc:
        print(f"REFUSED: {exc}")
        return 2
    except ValueError as exc:  # bad tuning selection (e.g. groups match nothing)
        print(f"refused: {exc}")
        return 1

    print_go_no_go(report)
    registry.insert_candidate(
        registry_model_version(report, args.train_start, args.train_end)
    )
    print(f"  bundle: {report.bundle_path}")
    print(
        f"  registered as CANDIDATE (model_type='quant_bundle', id={report.version!r}) — "
        "replay uses it via --quant-model-version; production scoring requires "
        "the promotion pipeline."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
