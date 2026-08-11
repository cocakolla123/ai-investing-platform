"""Artifact save/load round-trip + integrity refusal (plan Phase 3)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from dynamo.learning.artifacts import (
    ArtifactIntegrityError,
    load_model,
    read_manifest,
    save_model,
)


def test_save_load_round_trip(tmp_path: Path) -> None:
    obj = {"weights": [1.0, 2.0, 3.0], "algo": "toy"}
    manifest = save_model(
        obj,
        tmp_path,
        "meta-toy-1",
        feature_names=("a", "b"),
        train_start=date(2026, 1, 1),
        train_end=date(2026, 8, 1),
        n_samples=180,
    )
    assert manifest.model_id == "meta-toy-1"
    assert manifest.feature_names == ("a", "b")
    assert manifest.train_start == "2026-01-01"
    assert manifest.train_end == "2026-08-01"
    assert manifest.n_samples == 180
    assert len(manifest.sha256) == 64
    assert (tmp_path / "meta-toy-1.joblib").exists()
    assert (tmp_path / "meta-toy-1.manifest.json").exists()
    assert read_manifest(tmp_path, "meta-toy-1") == manifest
    loaded = load_model(tmp_path, "meta-toy-1")
    assert loaded == obj
    # An explicit expected sha (e.g. carried in the registry row) also passes.
    assert load_model(tmp_path, "meta-toy-1", expected_sha=manifest.sha256) == obj


class _WithNames:
    """Module-level (joblib/pickle needs an importable class)."""

    feature_names = ("x", "y", "z")


def test_feature_names_default_from_object(tmp_path: Path) -> None:
    manifest = save_model(_WithNames(), tmp_path, "meta-named")
    assert manifest.feature_names == ("x", "y", "z")


def test_tampered_artifact_raises(tmp_path: Path) -> None:
    save_model({"v": 1}, tmp_path, "meta-tamper")
    with (tmp_path / "meta-tamper.joblib").open("ab") as fh:
        fh.write(b"garbage")
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        load_model(tmp_path, "meta-tamper")


def test_expected_sha_mismatch_raises(tmp_path: Path) -> None:
    save_model({"v": 1}, tmp_path, "meta-sha")
    with pytest.raises(ArtifactIntegrityError, match="expected"):
        load_model(tmp_path, "meta-sha", expected_sha="0" * 64)
