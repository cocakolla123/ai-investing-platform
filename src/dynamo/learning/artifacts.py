"""Model artifact save/load with integrity manifests (plan Phase 3).

First artifact persistence in the repo: joblib dump + a sidecar JSON manifest
carrying the file's sha256, the feature-name contract, the training window
and sample count. ``load_model`` refuses on any digest mismatch — a silently
swapped or truncated artifact must never score live decisions. Reused later
by replay (Phase 4+).

Pure-ish: filesystem only, no database, no network, no clock beyond the
caller-supplied metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

MANIFEST_SUFFIX = ".manifest.json"
ARTIFACT_SUFFIX = ".joblib"


class ArtifactIntegrityError(RuntimeError):
    """The artifact on disk does not match its recorded (or expected) sha256."""


@dataclass(frozen=True)
class ArtifactManifest:
    """Sidecar metadata for one saved model artifact."""

    model_id: str
    sha256: str
    artifact_file: str  # basename, next to the manifest
    feature_names: tuple[str, ...] = ()
    train_start: str | None = None  # ISO date
    train_end: str | None = None
    n_samples: int = 0
    extra: dict[str, Any] | None = None


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(directory: str | Path, model_id: str) -> tuple[Path, Path]:
    base = Path(directory)
    return base / f"{model_id}{ARTIFACT_SUFFIX}", base / f"{model_id}{MANIFEST_SUFFIX}"


def save_model(
    obj: Any,
    directory: str | Path,
    model_id: str,
    *,
    feature_names: Sequence[str] = (),
    train_start: date | None = None,
    train_end: date | None = None,
    n_samples: int = 0,
    extra: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """joblib-dump ``obj`` and write its manifest; returns the manifest.

    ``feature_names`` defaults from ``obj.feature_names`` when present so the
    trained :class:`~dynamo.learning.meta_model.MetaModel` self-describes.
    """
    import joblib

    artifact_path, manifest_path = _paths(directory, model_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, artifact_path)
    names = tuple(feature_names) or tuple(getattr(obj, "feature_names", ()) or ())
    manifest = ArtifactManifest(
        model_id=model_id,
        sha256=_sha256_of(artifact_path),
        artifact_file=artifact_path.name,
        feature_names=names,
        train_start=train_start.isoformat() if train_start else None,
        train_end=train_end.isoformat() if train_end else None,
        n_samples=int(n_samples),
        extra=extra,
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def read_manifest(directory: str | Path, model_id: str) -> ArtifactManifest:
    _, manifest_path = _paths(directory, model_id)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ArtifactManifest(
        model_id=str(raw["model_id"]),
        sha256=str(raw["sha256"]),
        artifact_file=str(raw["artifact_file"]),
        feature_names=tuple(raw.get("feature_names") or ()),
        train_start=raw.get("train_start"),
        train_end=raw.get("train_end"),
        n_samples=int(raw.get("n_samples") or 0),
        extra=raw.get("extra"),
    )


def load_model(directory: str | Path, model_id: str, expected_sha: str | None = None) -> Any:
    """Load a saved artifact, verifying its digest against the manifest (and
    against ``expected_sha`` when the caller carries one, e.g. from the
    registry row). Raises :class:`ArtifactIntegrityError` on any mismatch."""
    import joblib

    artifact_path, _ = _paths(directory, model_id)
    manifest = read_manifest(directory, model_id)
    actual = _sha256_of(artifact_path)
    if actual != manifest.sha256:
        raise ArtifactIntegrityError(
            f"{model_id}: artifact sha256 {actual[:12]}… does not match "
            f"manifest {manifest.sha256[:12]}…"
        )
    if expected_sha is not None and actual != expected_sha:
        raise ArtifactIntegrityError(
            f"{model_id}: artifact sha256 {actual[:12]}… does not match "
            f"expected {expected_sha[:12]}…"
        )
    return joblib.load(artifact_path)
