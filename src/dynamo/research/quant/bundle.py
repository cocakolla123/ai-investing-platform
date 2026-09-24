"""Quant model-bundle artifacts (backtest-campaign Phase 2).

One trained "bundle" = the three nightly task models (ranker / downside /
vol) plus everything :func:`~dynamo.research.quant.scoring.score_universe`
needs to reproduce a scoring pass: the feature-name contract, the training
feature quantiles, and the fold-IC dispersion. On disk that is three
``learning/artifacts.py`` joblib+manifest pairs (sha256-verified) plus one
``<version>.bundle.json`` manifest tying them together.

Placement decision (import-linter contract 2): this module lives in core
research and therefore may NOT import ``dynamo.persistence``. It reads the
FILESYSTEM only — the registry lookup (candidate/production status, the
recorded ``artifact_path``) happens in composition layers (``cli/quant_train``,
``cli/replay``, ``replay/preflight``, ``orchestration/assets/research``), which
pass the artifact directory in or rely on the default directory convention
(``Settings().artifact_dir / models / quant`` — config import is contract-legal
for core logic).

Failures are LOUD by design:

* unknown version         -> :class:`QuantBundleError` (missing bundle manifest)
* artifact sha mismatch   -> :class:`~dynamo.learning.artifacts.ArtifactIntegrityError`
* feature-name drift      -> :class:`QuantBundleError` (manifest names vs the
  current FEATURE_CATALOG, i.e. exactly the columns ``build_features`` emits)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dynamo.domain.ports import ForecastModel
from dynamo.learning.artifacts import load_model, save_model
from dynamo.research.quant.catalog import FEATURE_CATALOG
from dynamo.research.quant.scoring import AveragingEnsemble, ModelBundle

BUNDLE_SUFFIX = ".bundle.json"
BUNDLE_TASKS = ("ranker", "downside", "vol")
QUANTILE_ROWS = ("q01", "q99")

__all__ = [
    "BUNDLE_TASKS",
    "QuantBundleError",
    "bundle_from_version",
    "default_bundle_dir",
    "feature_names_hash",
    "read_bundle_manifest",
    "save_bundle",
]


class QuantBundleError(RuntimeError):
    """A quant bundle is missing, malformed, or drifted from the catalog."""


def default_bundle_dir() -> Path:
    """``<artifact_dir>/models/quant`` — the trainer's save location and the
    loader's default, so ``bundle_from_version(version)`` with no directory
    resolves the same place ``dynamo.cli.quant_train`` wrote to."""
    from dynamo.config.settings import Settings

    return Path(Settings().artifact_dir) / "models" / "quant"


def feature_names_hash(names: Sequence[str]) -> str:
    """Stable sha256 of the ordered feature-name contract (registry column)."""
    payload = json.dumps(list(names))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bundle_path(directory: str | Path, version: str) -> Path:
    return Path(directory) / f"{version}{BUNDLE_SUFFIX}"


def _quantiles_to_json(quantiles: pd.DataFrame | None) -> dict[str, dict[str, float | None]]:
    if quantiles is None:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for row in QUANTILE_ROWS:
        if row not in quantiles.index:
            continue
        values: dict[str, float | None] = {}
        for col in quantiles.columns:
            v = float(quantiles.loc[row, col])
            values[str(col)] = None if math.isnan(v) else v
        out[row] = values
    return out


def _quantiles_from_json(payload: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame | None:
    if not payload:
        return None
    frame = pd.DataFrame({row: dict(values) for row, values in payload.items()}).T.astype(float)
    return frame.reindex([r for r in QUANTILE_ROWS if r in frame.index])


def save_bundle(
    directory: str | Path,
    version: str,
    *,
    ranker: ForecastModel,
    downside: ForecastModel,
    vol: ForecastModel,
    feature_names: Sequence[str],
    train_feature_quantiles: pd.DataFrame | None,
    train_start: date,
    train_end: date,
    n_samples: int,
    config_hash: str,
    metrics: Mapping[str, Any],
    fold_ic_dispersion: float,
    vol_is_log: bool = True,
) -> Path:
    """Persist the three task artifacts + the bundle manifest; returns the
    bundle-manifest path (this is what the registry records as
    ``artifact_path``). ``metrics`` should carry the per-fold + pooled
    out-of-fold report so the go/no-go evidence travels with the artifact."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, dict[str, Any]] = {}
    for task, model in (("ranker", ranker), ("downside", downside), ("vol", vol)):
        if isinstance(model, AveragingEnsemble):
            # Seed ensemble (--ensemble-seeds): one artifact PER MEMBER, tied
            # together by a 'members' list; the loader rebuilds the averaging
            # wrapper. Single-model bundles keep the original manifest shape.
            members: list[dict[str, str]] = []
            for i, member in enumerate(model.members):
                model_id = f"{version}-{task}-s{i:02d}"
                manifest = save_model(
                    member,
                    base,
                    model_id,
                    feature_names=feature_names,
                    train_start=train_start,
                    train_end=train_end,
                    n_samples=n_samples,
                    extra={"task": task, "bundle_version": version, "member": i},
                )
                members.append(
                    {
                        "model_id": model_id,
                        "sha256": manifest.sha256,
                        "artifact_file": manifest.artifact_file,
                    }
                )
            tasks[task] = {"members": members}
            continue
        model_id = f"{version}-{task}"
        manifest = save_model(
            model,
            base,
            model_id,
            feature_names=feature_names,
            train_start=train_start,
            train_end=train_end,
            n_samples=n_samples,
            extra={"task": task, "bundle_version": version},
        )
        tasks[task] = {
            "model_id": model_id,
            "sha256": manifest.sha256,
            "artifact_file": manifest.artifact_file,
        }
    payload: dict[str, Any] = {
        "bundle_version": version,
        "feature_names": [str(n) for n in feature_names],
        "features_hash": feature_names_hash(feature_names),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "n_samples": int(n_samples),
        "config_hash": config_hash,
        "tasks": tasks,
        "metrics": dict(metrics),
        "fold_ic_dispersion": float(fold_ic_dispersion),
        "vol_is_log": bool(vol_is_log),
        "train_feature_quantiles": _quantiles_to_json(train_feature_quantiles),
    }
    path = _bundle_path(base, version)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_bundle_manifest(directory: str | Path, version: str) -> dict[str, Any]:
    """The parsed ``<version>.bundle.json``; unknown versions are LOUD."""
    path = _bundle_path(directory, version)
    if not path.is_file():
        raise QuantBundleError(
            f"unknown quant model version {version!r}: no bundle manifest at {path} "
            "(train one with `python -m dynamo.cli.quant_train`)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuantBundleError(f"bundle manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "tasks" not in payload:
        raise QuantBundleError(f"bundle manifest {path} is malformed (no 'tasks' key)")
    return payload


def _check_feature_drift(
    version: str, manifest_names: Sequence[str], current_names: Sequence[str]
) -> None:
    """Refuse a bundle that needs a feature the catalog no longer computes.

    Only MISSING trained features are drift. Scoring selects the bundle's own
    columns by name in its training order (``ModelBundle.feature_names``), so
    catalog-only additions and reordering can no longer reach the model.
    This used to demand exact equality, which made every existing bundle
    unloadable the moment any feature was ADDED to the catalog — including
    campaign-v8, the model behind the committee campaign — and made a bundle
    trained with --feature-groups impossible to score at all.
    """
    current = set(current_names)
    missing = [n for n in manifest_names if n not in current]
    if not manifest_names:
        raise QuantBundleError(f"feature drift: bundle {version!r} records no feature names")
    if missing:
        raise QuantBundleError(
            f"feature drift: bundle {version!r} was trained on {len(manifest_names)} "
            f"features, {len(missing)} of which the current FEATURE_CATALOG no longer "
            f"computes: {missing[:8]}. Retrain the bundle against the current catalog."
        )


def bundle_from_version(
    model_version: str,
    artifact_dir: str | Path | None = None,
    *,
    current_feature_names: Sequence[str] | None = None,
) -> ModelBundle:
    """Load a trained :class:`ModelBundle` for ``model_version`` from disk.

    ``artifact_dir`` defaults to :func:`default_bundle_dir` (where the trainer
    saves); composition layers holding a registry row may pass
    ``Path(model_version_row.artifact_path).parent`` instead.
    ``current_feature_names`` defaults to the FEATURE_CATALOG order — exactly
    the columns ``build_features`` emits — so drift against today's catalog
    fails loudly before a single stale score is produced.
    """
    base = Path(artifact_dir) if artifact_dir is not None else default_bundle_dir()
    manifest = read_bundle_manifest(base, model_version)

    names = [str(n) for n in manifest.get("feature_names", ())]
    current = (
        list(current_feature_names) if current_feature_names is not None else list(FEATURE_CATALOG)
    )
    _check_feature_drift(model_version, names, current)

    models: dict[str, ForecastModel] = {}
    tasks = manifest["tasks"]
    for task in BUNDLE_TASKS:
        spec = tasks.get(task)
        if not spec:
            raise QuantBundleError(f"bundle {model_version!r} manifest has no {task!r} task entry")
        # load_model re-hashes the artifact and refuses on any mismatch with
        # the sidecar manifest AND the bundle-recorded sha (double-checked).
        members = spec.get("members")
        if members:
            # Seed-ensemble bundle: load every member artifact and rebuild
            # the averaging wrapper. Pre-ensemble manifests have no
            # 'members' key and take the single-model path unchanged.
            models[task] = AveragingEnsemble(
                [load_model(base, str(m["model_id"]), str(m["sha256"])) for m in members]
            )
        else:
            models[task] = load_model(base, str(spec["model_id"]), str(spec["sha256"]))

    return ModelBundle(
        model_version=model_version,
        ranker=models["ranker"],
        downside=models["downside"],
        vol=models["vol"],
        train_feature_quantiles=_quantiles_from_json(manifest.get("train_feature_quantiles") or {}),
        fold_ic_dispersion=float(manifest.get("fold_ic_dispersion") or 0.0),
        vol_is_log=bool(manifest.get("vol_is_log", True)),
        feature_names=tuple(names),
    )
