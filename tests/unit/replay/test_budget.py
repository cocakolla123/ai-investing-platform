"""ReplayBudget + estimator tests (plan Phase 5): refusal at the boundary,
spend accumulation and durable checkpoints, resume-after-restart accounting,
the ledger tee, and deterministic dry-run projection math. ZERO network,
ZERO LLM — pure objects and fakes only."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dynamo.llm.cost_meter import CostMeter, InMemoryLedgerWriter
from dynamo.llm.pricing import calculate_cost
from dynamo.replay.budget import (
    ASSUMED_CONTEXT_CHARS,
    CHARS_PER_TOKEN,
    BudgetLedgerTee,
    ReplayBudget,
    ReplayBudgetExhausted,
    estimate_batch_cost,
    estimate_deep_dive_cost,
    estimate_request_cost,
    estimate_run_cost,
    estimate_tokens,
)


class FakeRunRepo:
    def __init__(self) -> None:
        self.spends: list[tuple[str, float]] = []
        self.statuses: list[tuple[str, str]] = []

    def record_spend(self, run_id: str, delta_usd: float) -> None:
        self.spends.append((run_id, delta_usd))

    def update_status(self, run_id: str, status: str) -> None:
        self.statuses.append((run_id, status))


def _budget(cap: float = 10.0, prior: float = 0.0) -> tuple[ReplayBudget, FakeRunRepo]:
    repo = FakeRunRepo()
    return (
        ReplayBudget(cap_usd=cap, run_repo=repo, run_id="replay-x", prior_spent_usd=prior),
        repo,
    )


# --------------------------------------------------------------------------
# cap enforcement
# --------------------------------------------------------------------------


def test_refusal_at_the_boundary() -> None:
    budget, _ = _budget(cap=10.0, prior=8.0)
    budget.check_projection(2.0)  # exactly consumes the remainder — allowed
    with pytest.raises(ReplayBudgetExhausted):
        budget.check_projection(2.01)  # any overshoot refuses


def test_projection_exceeding_cap_refuses_before_any_spend() -> None:
    budget, repo = _budget(cap=1.0)
    with pytest.raises(ReplayBudgetExhausted):
        budget.check_projection(1.5)
    assert repo.spends == []  # nothing recorded — nothing was submitted


def test_rejects_non_positive_cap() -> None:
    repo = FakeRunRepo()
    with pytest.raises(ValueError):
        ReplayBudget(cap_usd=0.0, run_repo=repo, run_id="replay-x")


# --------------------------------------------------------------------------
# spend accumulation + durable checkpoints
# --------------------------------------------------------------------------


def test_spend_accumulates_across_checkpoints() -> None:
    budget, repo = _budget(cap=10.0)
    budget.record_actual(1.25)
    budget.record_actual(0.75)
    assert budget.spent_usd == pytest.approx(2.0)
    assert budget.remaining_usd == pytest.approx(8.0)
    # Every delta is checkpointed durably through the run repo.
    assert repo.spends == [("replay-x", 1.25), ("replay-x", 0.75)]
    with pytest.raises(ReplayBudgetExhausted):
        budget.check_projection(8.01)


def test_zero_and_negative_deltas_are_ignored() -> None:
    budget, repo = _budget()
    budget.record_actual(0.0)
    budget.record_actual(-1.0)
    assert budget.spent_usd == 0.0
    assert repo.spends == []


def test_resume_after_restart_uses_prior_spent() -> None:
    """A restarted run seeds prior spend from replay_runs.spent_usd — the
    cap covers the WHOLE run, not just the current process."""
    budget, _ = _budget(cap=10.0, prior=9.5)
    assert budget.spent_usd == pytest.approx(9.5)
    budget.check_projection(0.5)
    with pytest.raises(ReplayBudgetExhausted):
        budget.check_projection(0.6)


# --------------------------------------------------------------------------
# ledger tee: actuals flow from the client's own metering
# --------------------------------------------------------------------------


def test_ledger_tee_feeds_budget_and_monthly_meter() -> None:
    budget, repo = _budget(cap=10.0)
    inner = InMemoryLedgerWriter()
    meter = CostMeter(
        BudgetLedgerTee(inner, budget),
        monthly_budget_usd=90.0,
        now_fn=lambda: datetime(2026, 8, 11, tzinfo=UTC),
    )
    meter.record(
        model_id="claude-sonnet-5",
        call_kind="batch",
        input_tokens=1000,
        output_tokens=100,
        cost_usd=0.42,
        batch=True,
    )
    # Actual metered cost reached BOTH ledgers: the monthly meter and the run.
    assert [e.cost_usd for e in inner.entries] == [0.42]
    assert meter.month_to_date_usd() == pytest.approx(0.42)
    assert budget.spent_usd == pytest.approx(0.42)
    assert repo.spends == [("replay-x", 0.42)]


# --------------------------------------------------------------------------
# estimator: deterministic projection math, zero API calls
# --------------------------------------------------------------------------


def test_estimate_tokens_is_conservative_ceiling() -> None:
    assert estimate_tokens(0) == 0
    assert estimate_tokens(3) == 1
    assert estimate_tokens(4) == 2  # ceil(4/3)
    assert estimate_tokens(300) == 100


def test_estimate_request_cost_matches_pricing_table() -> None:
    cost = estimate_request_cost("claude-sonnet-5", prompt_chars=3000, max_tokens=1000)
    expected = calculate_cost("claude-sonnet-5", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(expected)
    # No batch discount applied — conservative by construction.
    assert cost == pytest.approx((1000 * 3.0 + 1000 * 15.0) / 1_000_000)


def test_estimate_batch_cost_sums_per_request_projections() -> None:
    tier_models = {"analyst": "claude-sonnet-5", "arbiter": "claude-opus-5"}
    requests = [
        {
            "custom_id": "a",
            "model_tier": "analyst",
            "system": "s" * 300,
            "context": "c" * 300,
            "instruction": "i" * 300,
            "schema": {},
            "max_tokens": 100,
        },
        {
            "custom_id": "b",
            "model_tier": "arbiter",
            "system": "s" * 600,
            "context": "c" * 300,
            "instruction": "",
            "schema": {},
            "max_tokens": 200,
        },
    ]
    expected = estimate_request_cost(
        "claude-sonnet-5", prompt_chars=900, max_tokens=100
    ) + estimate_request_cost("claude-opus-5", prompt_chars=900, max_tokens=200)
    assert estimate_batch_cost(requests, tier_models) == pytest.approx(expected)


def test_estimate_deep_dive_cost_is_deterministic_and_stage_aware() -> None:
    tier_models = {"analyst": "claude-sonnet-5", "arbiter": "claude-opus-5"}
    specs = {
        "dossier": ("analyst", 1200, 4096),
        "bull": ("analyst", 900, 3072),
        "bear": ("analyst", 900, 3072),
        "arbiter": ("arbiter", 1500, 4096),
    }
    once = estimate_deep_dive_cost(tier_models, specs)
    again = estimate_deep_dive_cost(tier_models, specs)
    assert once == again > 0.0
    # Manual reconstruction: later stages re-bill earlier stages' output.
    base = ASSUMED_CONTEXT_CHARS
    dossier_out = int(4096 * CHARS_PER_TOKEN)
    advocate_out = int(3072 * CHARS_PER_TOKEN)
    expected = (
        estimate_request_cost("claude-sonnet-5", prompt_chars=1200 + base, max_tokens=4096)
        + 2
        * estimate_request_cost(
            "claude-sonnet-5", prompt_chars=900 + base + dossier_out, max_tokens=3072
        )
        + estimate_request_cost(
            "claude-opus-5",
            prompt_chars=1500 + base + dossier_out + 2 * advocate_out,
            max_tokens=4096,
        )
    )
    assert once == pytest.approx(expected)


def test_estimate_run_cost_multiplies_sessions_and_cap() -> None:
    assert estimate_run_cost(n_sessions=10, deep_dives_per_day=4, per_dive_usd=0.25) == (
        pytest.approx(10.0)
    )
    assert estimate_run_cost(n_sessions=0, deep_dives_per_day=4, per_dive_usd=1.0) == 0.0
