"""The committee must not hand in-process batch results to another process.

ADR-0014's ``ClaudeCodeResearchModel.submit_batch`` executes every request
immediately and keeps the results ONLY in that instance's memory. The batch
flow then records the batch as 'submitted' and leaves collection to the
sensor-triggered ``committee_advance_job`` — a separate Dagster run in a
separate process, whose fresh model instance reports the batch "ended" with
zero results. The advance op resolves it empty and every call the submitter
just paid for is silently discarded; the next night repeats the loop. With
the nightly ``deep_dive_cap_per_night`` (8) above ``SYNC_DEEP_DIVE_MAX`` (3),
the committee would burn ~32 CLI calls a night and never persist a verdict.

The fix under test: ``submit_committee_deep_dives`` routes models flagged
``executes_batches_in_process`` through the sync path for ANY set size, so
submission and persistence share one process.
"""

from __future__ import annotations

from typing import Any

import orchestration.assets.research as research
import pytest
from dagster import build_op_context
from orchestration.assets.research import submit_committee_deep_dives

from dynamo.llm.claude_code_client import ClaudeCodeResearchModel


class _InProcessBatchModel:
    """A research model whose batches never survive the submitting process."""

    executes_batches_in_process = True

    def submit_batch(self, requests: Any) -> str:  # pragma: no cover - must not run
        raise AssertionError(
            "submit_batch must not be used for an in-process batch model: "
            "its results would be lost to the advance job's process"
        )


class _CommitteeDynamo:
    """The two accessors the submit op touches, nothing else."""

    def __init__(self) -> None:
        self.model = _InProcessBatchModel()

    def research_model(self) -> Any:
        return self.model

    def committee_engine(self) -> Any:
        return object()  # non-None: research is configured


def test_claude_code_model_declares_in_process_batches() -> None:
    """The client's own flag — the orchestration routing keys off it."""
    assert ClaudeCodeResearchModel.executes_batches_in_process is True


def test_large_set_runs_sync_for_in_process_batch_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """8 selected symbols (> SYNC_DEEP_DIVE_MAX) must still run sync when the
    model executes batches in-process; submit_batch raising proves the batch
    path was never entered."""
    symbols = [f"SYM{i}" for i in range(8)]
    assert len(symbols) > research.SYNC_DEEP_DIVE_MAX
    monkeypatch.setattr(research, "select_deep_dive_symbols", lambda dynamo, as_of: symbols)

    ran: dict[str, Any] = {}

    def fake_sync(context: Any, dynamo: Any, syms: Any, as_of: Any) -> int:
        ran["symbols"] = list(syms)
        return len(syms)

    monkeypatch.setattr(research, "run_deep_dives_sync_impl", fake_sync)

    submit_committee_deep_dives(build_op_context(), _CommitteeDynamo())  # type: ignore[arg-type]

    assert ran["symbols"] == symbols


def test_api_backend_keeps_the_batch_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real Message Batches backend (no flag) still submits large sets as a
    batch — the routing is scoped to in-process models only."""
    symbols = [f"SYM{i}" for i in range(8)]
    monkeypatch.setattr(research, "select_deep_dive_symbols", lambda dynamo, as_of: symbols)
    monkeypatch.setattr(
        research,
        "build_deep_dive_contexts",
        lambda dynamo, syms, as_of: {s: "ctx" for s in syms},
    )
    recorded: dict[str, Any] = {}
    monkeypatch.setattr(
        research,
        "_record_batch",
        lambda dynamo, batch_id, stage, n: recorded.update(
            {"batch_id": batch_id, "stage": stage, "count": n}
        ),
    )

    class _BatchModel:  # no executes_batches_in_process attribute
        def submit_batch(self, requests: Any) -> str:
            recorded["submitted"] = list(requests)
            return "batch-123"

    class _Engine:
        def build_batch_requests(self, stage: str, items: Any, as_of: Any) -> list[dict[str, Any]]:
            return [{"custom_id": f"{as_of}-{s}-{stage}"} for s in sorted(items)]

    class _ApiDynamo:
        def research_model(self) -> Any:
            return _BatchModel()

        def committee_engine(self) -> Any:
            return _Engine()

    submit_committee_deep_dives(build_op_context(), _ApiDynamo())  # type: ignore[arg-type]

    assert recorded["batch_id"] == "batch-123"
    assert recorded["stage"] == "dossier"
    assert recorded["count"] == len(symbols)
