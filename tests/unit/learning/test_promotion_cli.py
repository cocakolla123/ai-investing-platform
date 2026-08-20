"""Promotion CLI registry backends (plan Phase 3 — registry adoption).

JSON mode keeps working (backward compat, unit-level with a scratch git
repo); DB mode round-trips through the real ``ModelRegistryRepository``
(integration-marked, mirroring the other repository suites).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamo.cli.promotion import main
from dynamo.domain.models.learning import ModelVersion, RegistryStatus

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)

STAGE1_EVIDENCE = {
    "engine": "event_driven",
    "beats_spy_risk_adjusted": True,
    "beats_naive_baselines": True,
    "deflated_sharpe": 0.62,
    "max_drawdown": -0.09,
    "look_ahead_audit_passed": True,
    "constraint_violations": 0,
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def clean_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean scratch git repo so the CLI's dirty-tree gate passes.

    The evidence file is committed and the registry file gitignored — the
    tree must read clean when the gate checks it.
    """
    _git(tmp_path, "init")
    (tmp_path / "evidence.json").write_text(json.dumps(STAGE1_EVIDENCE))
    (tmp_path / ".gitignore").write_text("registry.json\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_json_mode_still_works(clean_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    evidence = clean_repo / "evidence.json"
    registry = clean_repo / "registry.json"
    code = main(
        [
            "run",
            "--target",
            "meta-x",
            "--stage",
            "1",
            "--evidence",
            str(evidence),
            "--registry",
            str(registry),
            "--actor",
            "owner",
        ]
    )
    assert code == 0
    data = json.loads(registry.read_text())
    assert data["status"]["meta-x"] == "validating"
    assert "PASS" in capsys.readouterr().out


def test_refuses_without_any_registry_backend(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    evidence = clean_repo / "evidence.json"
    code = main(
        [
            "run",
            "--target",
            "meta-x",
            "--stage",
            "1",
            "--evidence",
            str(evidence),
            "--actor",
            "owner",
        ]
    )
    assert code == 2
    assert "refused" in capsys.readouterr().out


# --- DB mode (integration; mirrors the other repository suites) -------------

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")


@pytest.fixture()
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine

    from dynamo.persistence.schema import (
        metadata,
        model_versions,
        promotion_records,
        trial_counter,
    )

    tables = [model_versions, promotion_records, trial_counter]
    eng = create_engine(INTEGRATION_URL, future=True)
    metadata.create_all(eng, tables=tables)
    yield eng
    metadata.drop_all(eng, tables=tables)
    eng.dispose()


@pytest.mark.integration
@pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set")
def test_db_registry_candidate_roundtrip_and_promotion(engine) -> None:  # type: ignore[no-untyped-def]
    """insert_candidate → status_of round-trip, then a stage transition
    through the CLI's DbRegistry adapter (the only legal status write)."""
    from dynamo.cli.promotion import DbRegistry
    from dynamo.domain.models.learning import PromotionRecord
    from dynamo.persistence.repositories.learning import ModelRegistryRepository

    repo = ModelRegistryRepository(engine)  # type: ignore[arg-type]
    repo.insert_candidate(
        ModelVersion(
            id="meta-gbm-1",
            model_type="meta",
            algo="gradient_boosting",
            features_hash="abc123",
            metrics={"oos_brier": 0.21, "artifact_sha256": "f" * 64},
            artifact_path="artifacts/models/meta-gbm-1.joblib",
            created_at=NOW,
        )
    )
    assert repo.status_of("meta-gbm-1") is RegistryStatus.CANDIDATE
    assert repo.latest_production("meta") is None  # the normal state

    db = DbRegistry(engine)
    assert db.current_status("meta-gbm-1") is RegistryStatus.CANDIDATE
    assert db.current_status("never-inserted") is RegistryStatus.CANDIDATE  # JSON parity
    assert db.trial_count("meta") == 0
    assert db.increment_trial("meta") == 1
    assert db.trial_count("meta") == 1

    db.apply(
        PromotionRecord(
            id="pr-1",
            target_id="meta-gbm-1",
            from_status=RegistryStatus.CANDIDATE,
            to_status=RegistryStatus.VALIDATING,
            actor="owner",
            at=NOW,
        )
    )
    assert repo.status_of("meta-gbm-1") is RegistryStatus.VALIDATING
