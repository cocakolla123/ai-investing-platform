"""meta_scoring asset core on in-memory fakes (plan Phase 3).

No production-stage meta-model ⇒ silent successful skip (the normal state);
a fake production model ⇒ deterministic ``meta_decision_scores`` rows.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from orchestration.assets.learning_ops import build_meta_score_rows, run_meta_scoring

from dynamo.learning.artifacts import ArtifactIntegrityError
from dynamo.learning.hierarchy import CalibratedProbability
from dynamo.learning.meta_model import DecisionSnapshot

NOW = datetime(2026, 8, 10, 23, 45, tzinfo=UTC)
DAY = date(2026, 8, 10)
LOG = logging.getLogger("test_meta_scoring")


@dataclass
class FakeModel:
    """predict_proba fake keyed off the conviction feature."""

    calls: list[list[Mapping[str, float]]] = field(default_factory=list)

    def predict_proba(self, rows: Sequence[Mapping[str, float]]) -> list[float]:
        self.calls.append(list(rows))
        return [min(max(0.1 + 0.5 * r["conviction"], 0.0), 1.0) for r in rows]


@dataclass(frozen=True)
class FakeProduction:
    id: str = "meta-gbm-1"
    artifact_path: str = "artifacts/models/meta-gbm-1.joblib"
    metrics: Mapping[str, Any] = field(default_factory=dict)


def _decisions() -> list[tuple[str, str | None, Any, Any, Sequence[Any]]]:
    calib = CalibratedProbability(value=0.55, scope_used="global", n_effective=12)
    return [
        ("AAPL", "cv-AAPL", DecisionSnapshot(conviction=0.8), calib, []),
        ("MSFT", "cv-MSFT", DecisionSnapshot(conviction=0.4), None, []),
    ]


def test_no_production_model_skips_silently() -> None:
    inserted: list[dict[str, Any]] = []
    result = run_meta_scoring(
        production=None,
        load=lambda mv: (_ for _ in ()).throw(AssertionError("must not load")),
        decisions=lambda: (_ for _ in ()).throw(AssertionError("must not query")),
        insert=lambda row: inserted.append(row) or True,
        now=NOW,
        as_of=DAY,
        log=LOG,
    )
    assert result == {"skipped": "no production meta-model", "scored": 0}
    assert inserted == []


def test_production_model_scores_and_writes_rows() -> None:
    model = FakeModel()
    inserted: list[dict[str, Any]] = []
    result = run_meta_scoring(
        production=FakeProduction(),
        load=lambda mv: model,
        decisions=_decisions,
        insert=lambda row: inserted.append(row) or True,
        now=NOW,
        as_of=DAY,
        log=LOG,
    )
    assert result["scored"] == 2 and result["written"] == 2
    assert result["model_version"] == "meta-gbm-1"
    by_symbol = {r["symbol"]: r for r in inserted}
    aapl = by_symbol["AAPL"]
    assert aapl["id"] == "mds:2026-08-10:AAPL:meta-gbm-1"  # deterministic, idempotent
    assert aapl["verdict_id"] == "cv-AAPL"
    assert aapl["as_of"] == DAY and aapl["created_at"] == NOW
    assert aapl["source"] == "live"
    assert aapl["meta_score"] == aapl["effective_probability"]
    assert aapl["effective_probability"] == 0.1 + 0.5 * 0.8
    assert len(aapl["features_hash"]) == 16
    # The calibration feature flowed through the shared builder.
    assert model.calls[0][0]["calibrated_probability"] == 0.55
    assert model.calls[0][1]["calibrated_probability"] == 0.4  # conviction fallback


def test_rows_are_deterministic() -> None:
    a = build_meta_score_rows(FakeModel(), "m1", _decisions(), NOW, DAY)
    b = build_meta_score_rows(FakeModel(), "m1", _decisions(), NOW, DAY)
    assert a == b
    assert build_meta_score_rows(FakeModel(), "m1", [], NOW, DAY) == []


def test_unusable_artifact_skips_with_reason() -> None:
    def _load(mv: Any) -> Any:
        raise ArtifactIntegrityError("sha mismatch")

    result = run_meta_scoring(
        production=FakeProduction(),
        load=_load,
        decisions=_decisions,
        insert=lambda row: True,
        now=NOW,
        as_of=DAY,
        log=LOG,
    )
    assert result["scored"] == 0
    assert "artifact unusable" in str(result["skipped"])
