"""Promotion pipeline tests (doc 08 sections 7, 9): illegal transitions
rejected, missing approval refused, dirty tree refused, trial counter
increments, and only event-driven evidence is admissible."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dynamo.backtest.promotion import (
    PROMOTION_STAGES,
    reject,
    run_stage,
)
from dynamo.domain.models.learning import PromotionRecord, RegistryStatus
from dynamo.learning.promotion import PromotionError

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class FakeRepo:
    def __init__(self, status: RegistryStatus = RegistryStatus.CANDIDATE) -> None:
        self.status: dict[str, RegistryStatus] = {}
        self.default = status
        self.trials: dict[str, int] = {}
        self.records: list[PromotionRecord] = []

    def current_status(self, target_id: str) -> RegistryStatus:
        return self.status.get(target_id, self.default)

    def apply(self, record: PromotionRecord) -> None:
        self.status[record.target_id] = record.to_status
        self.records.append(record)

    def trial_count(self, family: str) -> int:
        return self.trials.get(family, 0)

    def increment_trial(self, family: str) -> int:
        self.trials[family] = self.trials.get(family, 0) + 1
        return self.trials[family]


GOOD_STAGE1 = {
    "engine": "event_driven",
    "beats_spy_risk_adjusted": True,
    "beats_naive_baselines": True,
    "deflated_sharpe": 0.62,
    "max_drawdown": -0.09,
    "look_ahead_audit_passed": True,
    "constraint_violations": 0,
}


def test_stage_table_covers_all_seven_stages():
    assert [g.stage for g in PROMOTION_STAGES] == [1, 2, 3, 4, 5, 6, 7]
    assert PROMOTION_STAGES[-1].requires_approval


class TestStage1:
    def test_pass_transitions_candidate_to_validating_and_counts_trial(self):
        repo = FakeRepo()
        result = run_stage(
            "strat-a", 1, GOOD_STAGE1, repo=repo, actor="owner", at=NOW,
            clean_tree=True,
        )
        assert result.passed
        assert result.trial_count == 1
        assert repo.current_status("strat-a") is RegistryStatus.VALIDATING
        # A second run of the family increments the persistent counter.
        repo.status["strat-a"] = RegistryStatus.CANDIDATE
        again = run_stage(
            "strat-a", 1, GOOD_STAGE1, repo=repo, actor="owner", at=NOW,
            clean_tree=True,
        )
        assert again.trial_count == 2

    def test_failed_gate_writes_no_transition(self):
        repo = FakeRepo()
        bad = dict(GOOD_STAGE1, deflated_sharpe=-0.1)
        result = run_stage(
            "strat-a", 1, bad, repo=repo, actor="owner", at=NOW, clean_tree=True
        )
        assert not result.passed
        assert repo.records == []
        assert repo.current_status("strat-a") is RegistryStatus.CANDIDATE

    def test_dirty_tree_refused(self):
        with pytest.raises(PromotionError, match="dirty git tree"):
            run_stage(
                "strat-a", 1, GOOD_STAGE1, repo=FakeRepo(), actor="owner",
                at=NOW, clean_tree=False,
            )

    def test_vectorbt_evidence_refused(self):
        evidence = dict(GOOD_STAGE1, engine="vectorbt")
        with pytest.raises(PromotionError, match="event-driven"):
            run_stage(
                "strat-a", 1, evidence, repo=FakeRepo(), actor="owner",
                at=NOW, clean_tree=True,
            )


class TestStateMachine:
    def test_illegal_transition_rejected(self):
        # Stage 4 transitions validating -> paper; a candidate cannot take it.
        repo = FakeRepo(RegistryStatus.CANDIDATE)
        evidence = {
            "engine": "event_driven",
            "ladder_dd15_breaches": 0,
            "behavior_explained": True,
        }
        with pytest.raises(PromotionError, match="candidate"):
            run_stage(
                "strat-a", 4, evidence, repo=repo, actor="owner", at=NOW,
                clean_tree=True,
            )

    def test_missing_approval_refused_for_production(self):
        repo = FakeRepo(RegistryStatus.SHADOW)
        with pytest.raises(PromotionError, match="approval"):
            run_stage(
                "strat-a", 7, {"engine": "event_driven"}, repo=repo,
                actor="owner", at=NOW, clean_tree=True,
                evidence_bundle_ref="artifacts/backtests/run-1/",
            )
        assert repo.records == []

    def test_wrong_target_approval_refused(self):
        repo = FakeRepo(RegistryStatus.SHADOW)
        approval = {"target_id": "someone-else", "signature": "sig"}
        with pytest.raises(PromotionError, match="someone-else"):
            run_stage(
                "strat-a", 7, {"engine": "event_driven"}, repo=repo,
                actor="owner", at=NOW, clean_tree=True,
                approval=approval, approval_valid=True,
                evidence_bundle_ref="artifacts/backtests/run-1/",
            )

    def test_valid_approval_promotes_to_production(self):
        repo = FakeRepo(RegistryStatus.SHADOW)
        approval = {"target_id": "strat-a", "signature": "sig"}
        result = run_stage(
            "strat-a", 7, {"engine": "event_driven"}, repo=repo, actor="owner",
            at=NOW, clean_tree=True, approval=approval, approval_valid=True,
            evidence_bundle_ref="artifacts/backtests/run-1/",
            git_commit="abc123", config_hash="deadbeef",
        )
        assert result.passed
        assert repo.current_status("strat-a") is RegistryStatus.PRODUCTION
        record = repo.records[-1]
        assert record.git_commit == "abc123"
        assert record.evidence_bundle_ref == "artifacts/backtests/run-1/"

    def test_reject_from_any_pre_production_stage(self):
        repo = FakeRepo(RegistryStatus.VALIDATING)
        record = reject(
            "strat-a", repo=repo, actor="owner", at=NOW, reason="fold instability"
        )
        assert record.to_status is RegistryStatus.REJECTED
        assert repo.current_status("strat-a") is RegistryStatus.REJECTED

    def test_production_cannot_reject_only_retire(self):
        repo = FakeRepo(RegistryStatus.PRODUCTION)
        with pytest.raises(PromotionError, match="illegal"):
            reject("strat-a", repo=repo, actor="owner", at=NOW, reason="x")

    def test_unknown_stage(self):
        with pytest.raises(PromotionError, match="unknown"):
            run_stage(
                "strat-a", 9, {"engine": "event_driven"}, repo=FakeRepo(),
                actor="owner", at=NOW, clean_tree=True,
            )


class TestPaperAndShadowGates:
    def test_paper_gate_requires_thirty_days_and_zero_violations(self):
        repo = FakeRepo(RegistryStatus.PAPER)
        evidence = {
            "engine": "event_driven",
            "paper_trading_days": 31,
            "tracking_error_within_band": True,
            "slippage_within_model": True,
            "risk_constraint_violations": 0,
        }
        result = run_stage(
            "strat-a", 5, evidence, repo=repo, actor="owner", at=NOW,
            clean_tree=True,
        )
        assert result.passed
        assert repo.current_status("strat-a") is RegistryStatus.SHADOW

        short = dict(evidence, paper_trading_days=12)
        repo2 = FakeRepo(RegistryStatus.PAPER)
        result = run_stage(
            "strat-b", 5, short, repo=repo2, actor="owner", at=NOW,
            clean_tree=True,
        )
        assert not result.passed
        assert repo2.current_status("strat-b") is RegistryStatus.PAPER
