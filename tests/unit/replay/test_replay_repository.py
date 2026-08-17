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
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine

    from dynamo.persistence.schema import (
        evaluation_claims,
        generation_results,
        metadata,
        replay_decisions,
        replay_equity,
        replay_fills,
        replay_runs,
        replay_verdicts,
    )

    eng = create_engine(INTEGRATION_URL, future=True)
    tables = [
        replay_runs,
        replay_decisions,
        replay_verdicts,
        replay_equity,
        replay_fills,
        evaluation_claims,
        generation_results,
    ]
    metadata.create_all(eng, tables=tables)
    yield eng
    metadata.drop_all(eng, tables=tables)
    eng.dispose()


@pytest.fixture()
def repo(engine) -> Iterator[object]:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.replay import ReplayRepository

    yield ReplayRepository(engine)


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


def test_equity_upserts_idempotently(repo) -> None:  # type: ignore[no-untyped-def]
    """(run_id, as_of) PK: re-running the same config updates in place."""
    _create(repo)
    rows = [
        {
            "run_id": "replay-abc",
            "as_of": date(2026, 6, 1),
            "equity": 100_000.0,
            "cash": 40_000.0,
            "gross": 0.6,
            "regime": "risk_on",
            "ladder_rung": "normal",
            "vetoes": 2,
        },
        {
            "run_id": "replay-abc",
            "as_of": date(2026, 6, 2),
            "equity": 101_000.0,
            "cash": 41_000.0,
            "gross": 0.59,
            "regime": "risk_on",
            "ladder_rung": "normal",
            "vetoes": 0,
        },
    ]
    assert repo.insert_equity(rows) == 2
    rows[1]["equity"] = 101_500.0
    repo.insert_equity(rows)  # re-run: updates in place, no duplicate
    stored = repo.equity_for("replay-abc")
    assert [r["as_of"] for r in stored] == [date(2026, 6, 1), date(2026, 6, 2)]
    assert stored[1]["equity"] == 101_500.0
    assert stored[0]["vetoes"] == 2
    assert repo.insert_equity([]) == 0
    assert repo.equity_for("missing") == []


def test_fills_upsert_idempotently(repo) -> None:  # type: ignore[no-untyped-def]
    """(run_id, session, symbol) PK; realized_pnl NULL on buys survives."""
    _create(repo)
    rows = [
        {
            "run_id": "replay-abc",
            "session": date(2026, 6, 2),
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10.0,
            "price": 100.0,
            "trade_value": 1_000.0,
            "cost": 1.2,
            "realized_pnl": None,
        },
        {
            "run_id": "replay-abc",
            "session": date(2026, 6, 3),
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10.0,
            "price": 104.0,
            "trade_value": 1_040.0,
            "cost": 1.3,
            "realized_pnl": 37.5,
        },
    ]
    assert repo.insert_fills(rows) == 2
    rows[1]["realized_pnl"] = 37.6
    repo.insert_fills(rows)  # re-run: updates in place, no duplicate
    stored = repo.fills_for("replay-abc")
    assert len(stored) == 2
    assert stored[0]["realized_pnl"] is None
    assert stored[1]["realized_pnl"] == 37.6
    assert repo.insert_fills([]) == 0
    assert repo.fills_for("missing") == []


def test_verdicts_upsert_idempotently(repo) -> None:  # type: ignore[no-untyped-def]
    _create(repo, run_id="replay-cmt", status="running")
    rows = [
        {
            "id": "replay-cmt:AAPL-2026-06-01-deadbeef",
            "run_id": "replay-cmt",
            "symbol": "AAPL",
            "as_of": date(2026, 6, 1),
            "action": "buy",
            "conviction": 0.8,
            "disagreement_score": 0.2,
            "payload": {"thesis": "x"},
        }
    ]
    assert repo.insert_verdicts(rows) == 1
    rows[0]["conviction"] = 0.85
    repo.insert_verdicts(rows)  # re-run: updates in place, no duplicate
    stored = repo.verdicts_for("replay-cmt")
    assert len(stored) == 1
    assert stored[0]["conviction"] == 0.85
    assert repo.insert_verdicts([]) == 0
    assert repo.verdicts_for("missing") == []


def test_claim_period_first_wins_then_refuses(engine) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.replay import GenerationsRepository

    gens = GenerationsRepository(engine)
    start, end = date(2026, 7, 1), date(2026, 7, 31)
    assert gens.is_claimed(start, end, "final_holdout") is None
    assert gens.claim_period(
        period_start=start, period_end=end, purpose="final_holdout", config_hash="abc"
    )
    claim = gens.is_claimed(start, end, "final_holdout")
    assert claim is not None and claim["config_hash"] == "abc"
    # THE conflict refusal: the same period+purpose can never be claimed again,
    # not even by the identical config.
    assert not gens.claim_period(
        period_start=start, period_end=end, purpose="final_holdout", config_hash="abc"
    )
    assert not gens.claim_period(
        period_start=start, period_end=end, purpose="final_holdout", config_hash="other"
    )
    # A different purpose over the same window is a separate claim.
    assert gens.claim_period(
        period_start=start, period_end=end, purpose="something-else", config_hash="abc"
    )


def test_generation_results_upsert_and_list(engine) -> None:  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    from dynamo.persistence.repositories.replay import GenerationsRepository

    gens = GenerationsRepository(engine)
    row = {
        "id": "gen:abc123:001",
        "created_at": datetime(2026, 8, 11, tzinfo=UTC),
        "generation": 1,
        "train_start": date(2026, 1, 1),
        "train_end": date(2026, 1, 31),
        "test_start": date(2026, 2, 1),
        "test_end": date(2026, 2, 28),
        "model_version": "meta-gen001-logistic",
        "n_train": 160,
        "n_test": 40,
        "brier_meta": 0.18,
        "brier_control": 0.25,
        "precision_meta": {"precision_conviction_0.67_1.00": 0.7},
        "precision_control": {},
        "ev_meta": 0.5,
        "ev_control": 0.1,
        "value_of_learning": 0.07,
        "payload": {"skipped": False},
    }
    gens.insert_generation(row)
    row["brier_meta"] = 0.17
    row["value_of_learning"] = 0.08
    gens.insert_generation(row)  # idempotent upsert on id
    stored = gens.list_generations()
    assert len(stored) == 1
    assert stored[0]["brier_meta"] == 0.17
    assert stored[0]["value_of_learning"] == 0.08
