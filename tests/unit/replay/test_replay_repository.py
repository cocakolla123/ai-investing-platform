"""ReplayRepository against a real Postgres (plan Phase 4)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date

import pytest

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]


@pytest.fixture()
def repo() -> Iterator[object]:
    from sqlalchemy import create_engine

    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.persistence.schema import metadata, replay_decisions, replay_runs

    eng = create_engine(INTEGRATION_URL, future=True)
    tables = [replay_runs, replay_decisions]
    metadata.create_all(eng, tables=tables)
    yield ReplayRepository(eng)
    metadata.drop_all(eng, tables=tables)
    eng.dispose()


def _create(repo, run_id: str = "replay-abc", status: str = "running") -> None:  # type: ignore[no-untyped-def]
    repo.create_run(
        run_id=run_id,
        label="test",
        mode="quant",
        start_as_of=date(2026, 6, 1),
        end_as_of=date(2026, 6, 10),
        config={"horizon_sessions": 20},
        code_version="deadbeef",
        budget_usd=None,
        status=status,
    )


def test_run_lifecycle(repo) -> None:  # type: ignore[no-untyped-def]
    _create(repo)
    run = repo.get_run("replay-abc")
    assert run is not None
    assert run["status"] == "running"
    assert run["mode"] == "quant"
    assert run["budget_usd"] is None
    assert run["spent_usd"] == 0.0
    assert run["config_json"] == {"horizon_sessions": 20}

    repo.update_status("replay-abc", "completed")
    assert repo.get_run("replay-abc")["status"] == "completed"

    repo.record_spend("replay-abc", 1.25)
    repo.record_spend("replay-abc", 0.75)
    assert repo.get_run("replay-abc")["spent_usd"] == 2.0

    assert [r["id"] for r in repo.list_runs()] == ["replay-abc"]
    assert repo.get_run("missing") is None


def test_create_run_is_idempotent(repo) -> None:  # type: ignore[no-untyped-def]
    _create(repo)
    repo.record_spend("replay-abc", 3.0)
    _create(repo, status="running")  # same deterministic id: upsert, no crash
    assert len(repo.list_runs()) == 1
    # spent_usd survives the re-create (run-local accounting is durable).
    assert repo.get_run("replay-abc")["spent_usd"] == 3.0


def test_decisions_upsert_idempotently(repo) -> None:  # type: ignore[no-untyped-def]
    _create(repo)
    rows = [
        {
            "id": "replay-abc:2026-06-01:AAPL",
            "run_id": "replay-abc",
            "as_of": date(2026, 6, 1),
            "symbol": "AAPL",
            "action": "buy",
            "conviction": 0.8,
            "target_weight": 0.05,
            "fill_price": None,
            "payload": {"verdict": None},
        }
    ]
    assert repo.insert_decisions(rows) == 1
    rows[0]["fill_price"] = 101.5
    repo.insert_decisions(rows)  # re-run: updates in place, no duplicate
    stored = repo.decisions_for("replay-abc")
    assert len(stored) == 1
    assert stored[0]["fill_price"] == 101.5
    assert repo.insert_decisions([]) == 0
