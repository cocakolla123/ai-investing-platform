"""CLI rails: arg plumbing, pre-registration enforcement, trial counting,
audit persistence, research-log rows, propose gating. All stores are fakes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from dynamo.backtest.walkforward import WalkForwardConfig
from dynamo.cli.alpha import (
    _build_parser,
    audition_command,
    list_command,
    propose_command,
    register_command,
)
from dynamo.research.discovery.registry import HypothesisStatus
from dynamo.research.quant.models import LightGBMModel
from tests.unit.discovery.conftest import (
    NOW,
    FakeHypothesisStore,
    FakeProposalStore,
    FakeTrialCounter,
    make_hypothesis,
    planted_audition_panel,
)

LATER = NOW + timedelta(hours=2)
SMALL_WF = WalkForwardConfig(
    train_sessions=60, test_sessions=20, purge_sessions=10, embargo_sessions=5, roll_sessions=20
)


def _fast_model() -> LightGBMModel:
    return LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5})


def _run_audition(
    store: FakeHypothesisStore,
    trials: FakeTrialCounter,
    candidate_file: Path,
    tmp_path: Path,
    *,
    hypothesis_id: str = "test-hypo",
    audition_start: datetime = LATER,
    signal_strength: float = 2.0,
    out: Any = None,
) -> tuple[int, Path]:
    log_path = tmp_path / "docs" / "alpha_research_log.md"
    code = audition_command(
        hypothesis_id=hypothesis_id,
        candidate_file=candidate_file,
        audition_start=audition_start,
        hypotheses=store,
        trials=trials,
        panel_builder=lambda candidate: planted_audition_panel(
            n_sessions=140, signal_strength=signal_strength
        ),
        log_path=log_path,
        wf_config=SMALL_WF,
        model_factory=_fast_model,
        n_permutations=100,
        out=out or (lambda _msg: None),
    )
    return code, log_path


# --- arg plumbing ----------------------------------------------------------


def test_parser_register_args() -> None:
    ns = _build_parser().parse_args(
        [
            "register",
            "--id",
            "vol-skew",
            "--title",
            "t",
            "--rationale",
            "r",
            "--prediction",
            "p",
            "--trial-family",
            "price-vol",
        ]
    )
    assert ns.command == "register"
    assert ns.id == "vol-skew" and ns.trial_family == "price-vol"
    assert ns.file is None


def test_parser_audition_args() -> None:
    ns = _build_parser().parse_args(
        [
            "audition",
            "--hypothesis-id",
            "vol-skew",
            "--candidate-file",
            "cand.py",
            "--train-start",
            "2024-01-01",
            "--train-end",
            "2025-06-30",
            "--symbols",
            "AAPL",
            "MSFT",
            "--n-permutations",
            "50",
            "--no-cache",
        ]
    )
    assert ns.command == "audition"
    assert ns.hypothesis_id == "vol-skew"
    assert ns.candidate_file == Path("cand.py")
    assert str(ns.train_start) == "2024-01-01" and str(ns.train_end) == "2025-06-30"
    assert ns.symbols == ["AAPL", "MSFT"] and ns.n_permutations == 50 and ns.no_cache


def test_parser_propose_and_list() -> None:
    assert _build_parser().parse_args(["propose", "--hypothesis-id", "x-y"]).command == "propose"
    assert _build_parser().parse_args(["list"]).command == "list"


def test_parser_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


# --- register --------------------------------------------------------------


def test_register_writes_hypothesis() -> None:
    store = FakeHypothesisStore()
    fields = {
        "id": "vol-skew",
        "title": "Vol skew",
        "rationale": "because reasons written beforehand",
        "prediction": "positive IC",
        "trial_family": "price-vol",
    }
    assert register_command(fields, store, now=NOW, out=lambda _m: None) == 0
    h = store.get("vol-skew")
    assert h is not None
    assert h.status == HypothesisStatus.REGISTERED
    assert h.registered_at == NOW


def test_register_refuses_duplicate_and_invalid() -> None:
    store = FakeHypothesisStore()
    store.insert(make_hypothesis("vol-skew"))
    fields = {
        "id": "vol-skew",
        "title": "t",
        "rationale": "r",
        "prediction": "p",
        "trial_family": "f",
    }
    assert register_command(fields, store, now=NOW, out=lambda _m: None) == 1
    assert register_command({"id": "Bad Slug"}, store, now=NOW, out=lambda _m: None) == 1
    assert len(store.rows) == 1


# --- audition: pre-registration enforcement --------------------------------


def test_audition_refuses_unregistered_hypothesis(candidate_file: Path, tmp_path: Path) -> None:
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    code, log_path = _run_audition(
        store, trials, candidate_file, tmp_path, hypothesis_id="never-registered"
    )
    assert code == 1
    assert trials.counts == {}  # a refusal burns no trial
    assert not log_path.exists()


def test_audition_refuses_same_invocation_registration(
    candidate_file: Path, tmp_path: Path
) -> None:
    """registered_at == audition_start (or later) — registered by THIS
    invocation — is a refusal; strictly-earlier registration passes."""
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    store.insert(make_hypothesis(registered_at=LATER))
    code, _ = _run_audition(store, trials, candidate_file, tmp_path, audition_start=LATER)
    assert code == 1
    assert trials.counts == {}


def test_audition_refuses_malformed_candidate(tmp_path: Path) -> None:
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    store.insert(make_hypothesis())
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    code, _ = _run_audition(store, trials, bad, tmp_path)
    assert code == 1
    assert trials.counts == {}


# --- audition: full flow ---------------------------------------------------


def test_audition_pass_counts_trial_persists_audit_and_logs(
    candidate_file: Path, tmp_path: Path
) -> None:
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    store.insert(make_hypothesis())
    lines: list[str] = []
    code, log_path = _run_audition(store, trials, candidate_file, tmp_path, out=lines.append)
    assert code == 0
    assert trials.counts == {"test-family": 1}
    ((hid, status, results),) = store.audits
    assert hid == "test-hypo" and status == HypothesisStatus.AUDITIONED
    assert results["verdict"]["passed"] is True
    assert results["family_trials"] == 1
    assert results["candidate"]["name"] == "ret_5d_test_candidate"
    assert results["permutation_null"]["n_permutations"] == 100
    import json

    json.dumps(results)  # audit_results must be storable as clean JSON
    text = log_path.read_text(encoding="utf-8")
    assert "| test-hypo | test-family | 1 |" in text and "| PASS |" in text
    assert any("VERDICT: PASSED" in line for line in lines)


def test_audition_fail_still_counts_trial_and_logs(candidate_file: Path, tmp_path: Path) -> None:
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    store.insert(make_hypothesis())
    code, log_path = _run_audition(store, trials, candidate_file, tmp_path, signal_strength=0.0)
    assert code == 2
    assert trials.counts == {"test-family": 1}  # failed look = spent trial
    ((_, status, results),) = store.audits
    assert status == HypothesisStatus.REJECTED
    assert results["verdict"]["passed"] is False
    assert "| FAIL |" in log_path.read_text(encoding="utf-8")


def test_each_audition_increments_the_family_counter(candidate_file: Path, tmp_path: Path) -> None:
    store, trials = FakeHypothesisStore(), FakeTrialCounter()
    store.insert(make_hypothesis())
    _run_audition(store, trials, candidate_file, tmp_path)
    # first audition passed -> status auditioned -> still auditionable
    _run_audition(store, trials, candidate_file, tmp_path)
    assert trials.counts == {"test-family": 2}
    assert len(store.audits) == 2


# --- propose ---------------------------------------------------------------


def _passed_results() -> dict[str, Any]:
    return {
        "candidate": {"name": "ret_5d_test_candidate", "group": "momentum", "file": "c.py"},
        "verdict": {
            "passed": True,
            "ic_delta": 0.03,
            "null_percentile": 99.5,
            "family_trials": 2,
        },
    }


def test_propose_emits_new_feature_proposal_with_evidence() -> None:
    from dynamo.domain.models.learning import ProposalKind, ProposalStatus

    store, proposals = FakeHypothesisStore(), FakeProposalStore()
    store.insert(
        make_hypothesis(status=HypothesisStatus.AUDITIONED, audit_results=_passed_results())
    )
    code = propose_command(
        hypothesis_id="test-hypo",
        hypotheses=store,
        proposals=proposals,
        now=LATER,
        out=lambda _m: None,
    )
    assert code == 0
    (proposal,) = proposals.inserted
    assert proposal.kind == ProposalKind.NEW_FEATURE
    assert proposal.status == ProposalStatus.OPEN
    assert proposal.target == "research.quant.catalog:ret_5d_test_candidate"
    assert proposal.proposed_change["hypothesis_id"] == "test-hypo"
    assert proposal.evidence and "audition passed" in proposal.evidence[0].claim
    h = store.get("test-hypo")
    assert h is not None and h.status == HypothesisStatus.PROPOSED


@pytest.mark.parametrize(
    ("status", "results"),
    [
        (HypothesisStatus.REGISTERED, None),  # never auditioned
        (HypothesisStatus.REJECTED, {"verdict": {"passed": False}}),  # failed audition
        (HypothesisStatus.PROPOSED, {"verdict": {"passed": True}}),  # already proposed
        (HypothesisStatus.AUDITIONED, {"verdict": {"passed": False}}),  # inconsistent row
        (HypothesisStatus.AUDITIONED, None),  # no stored evidence
    ],
)
def test_propose_refuses_without_a_passed_verdict(
    status: HypothesisStatus, results: dict[str, Any] | None
) -> None:
    store, proposals = FakeHypothesisStore(), FakeProposalStore()
    store.insert(make_hypothesis(status=status, audit_results=results))
    code = propose_command(
        hypothesis_id="test-hypo",
        hypotheses=store,
        proposals=proposals,
        now=LATER,
        out=lambda _m: None,
    )
    assert code == 1
    assert proposals.inserted == []


def test_propose_refuses_unknown_hypothesis() -> None:
    code = propose_command(
        hypothesis_id="ghost",
        hypotheses=FakeHypothesisStore(),
        proposals=FakeProposalStore(),
        now=datetime.now(UTC),
        out=lambda _m: None,
    )
    assert code == 1


# --- list ------------------------------------------------------------------


def test_list_prints_rows() -> None:
    store = FakeHypothesisStore()
    store.insert(make_hypothesis("one-hypo"))
    store.insert(make_hypothesis("two-hypo", registered_at=NOW + timedelta(minutes=1)))
    lines: list[str] = []
    assert list_command(store, out=lines.append) == 0
    assert any("one-hypo" in line for line in lines)
    assert any("two-hypo" in line for line in lines)
