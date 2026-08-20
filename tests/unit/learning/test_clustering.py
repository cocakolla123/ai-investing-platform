"""Failure-clustering tests (plan Phase 6): deterministic clusters on fixture
embeddings, the min-size proposal threshold, and the failure-class →
ProposalKind mapping. Pure — no database, no network, no LLM."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import FailureClass, ProposalKind, ProposalStatus
from dynamo.learning.clustering import (
    KIND_BY_FAILURE_CLASS,
    FailureCase,
    FailureCluster,
    cluster_failures,
    proposals_from_clusters,
)

AS_OF = date(2026, 8, 9)
NOW = datetime(2026, 8, 9, 8, 0, tzinfo=UTC)

# Two well-separated semantic directions (cosine distance ~1 across, ~0 within).
_DIR_A = (1.0, 0.0, 0.0)
_DIR_A2 = (0.995, 0.1, 0.0)
_DIR_B = (0.0, 1.0, 0.0)
_DIR_B2 = (0.0, 0.995, 0.1)


def _case(
    i: int,
    failure_class: FailureClass = FailureClass.CLAUDE_REASONING,
    embedding: tuple[float, ...] | None = _DIR_A,
) -> FailureCase:
    return FailureCase(
        case_id=f"exp:live:{i:03d}",
        symbol=f"SYM{i}",
        as_of=AS_OF,
        failure_class=failure_class,
        text=f"failure case {i}",
        embedding=embedding,
    )


# --- clustering ---------------------------------------------------------------


def test_semantic_subclusters_split_on_fixture_embeddings() -> None:
    cases = [
        _case(1, embedding=_DIR_A),
        _case(2, embedding=_DIR_A2),
        _case(3, embedding=_DIR_B),
        _case(4, embedding=_DIR_B2),
    ]
    clusters = cluster_failures(cases)
    assert [c.size for c in clusters] == [2, 2]
    refs = {c.label: set(c.example_refs) for c in clusters}
    assert refs["claude_reasoning/c1"] == {"exp:live:001", "exp:live:002"}
    assert refs["claude_reasoning/c2"] == {"exp:live:003", "exp:live:004"}
    assert all(c.failure_class == FailureClass.CLAUDE_REASONING for c in clusters)
    assert all(c.suggested_kind == ProposalKind.PROMPT_REVISION for c in clusters)


def test_clusters_deterministic_under_input_order() -> None:
    cases = [
        _case(1, embedding=_DIR_A),
        _case(2, embedding=_DIR_A2),
        _case(3, embedding=_DIR_B),
        _case(4, FailureClass.DATA_QUALITY, embedding=_DIR_A),
    ]
    forward = cluster_failures(cases)
    backward = cluster_failures(list(reversed(cases)))
    assert forward == backward


def test_failure_classes_group_first_then_semantics() -> None:
    cases = [
        _case(1, FailureClass.QUANT_MODEL, embedding=_DIR_A),
        _case(2, FailureClass.QUANT_MODEL, embedding=_DIR_A2),
        _case(3, FailureClass.DATA_QUALITY, embedding=_DIR_A),  # same direction,
        # different class -> its own cluster regardless of similarity
    ]
    clusters = cluster_failures(cases)
    assert {(c.failure_class, c.size) for c in clusters} == {
        (FailureClass.QUANT_MODEL, 2),
        (FailureClass.DATA_QUALITY, 1),
    }


def test_unembedded_cases_form_a_catch_all_never_dropped() -> None:
    cases = [
        _case(1, embedding=_DIR_A),
        _case(2, embedding=None),
        _case(3, embedding=None),
    ]
    clusters = cluster_failures(cases)
    assert sum(c.size for c in clusters) == 3
    # biggest first: the 2-case unembedded catch-all labels c1
    assert clusters[0].size == 2
    assert set(clusters[0].example_refs) == {"exp:live:002", "exp:live:003"}


def test_normal_uncertainty_is_not_a_failure_and_is_skipped() -> None:
    cases = [_case(1, FailureClass.NORMAL_UNCERTAINTY)]
    assert cluster_failures(cases) == []


def test_empty_input_is_total() -> None:
    assert cluster_failures([]) == []
    assert proposals_from_clusters([], AS_OF, NOW) == []


# --- proposal mapping + min-size threshold ------------------------------------


def _cluster(failure_class: FailureClass, size: int) -> FailureCluster:
    return FailureCluster(
        label=f"{failure_class.value}/c1",
        failure_class=failure_class,
        size=size,
        example_refs=tuple(f"exp:live:{i:03d}" for i in range(min(size, 5))),
        suggested_kind=KIND_BY_FAILURE_CLASS[failure_class],
    )


@pytest.mark.parametrize(
    ("failure_class", "kind"),
    [
        (FailureClass.CLAUDE_REASONING, ProposalKind.PROMPT_REVISION),
        (FailureClass.MISSING_INFORMATION, ProposalKind.PROMPT_REVISION),
        (FailureClass.CONFIDENCE_CALIBRATION, ProposalKind.PROMPT_REVISION),
        (FailureClass.QUANT_MODEL, ProposalKind.RETRAIN),
        (FailureClass.POSITION_SIZING, ProposalKind.THRESHOLD_CHANGE),
        (FailureClass.RISK_MANAGEMENT, ProposalKind.THRESHOLD_CHANGE),
        (FailureClass.PORTFOLIO_ALLOCATION, ProposalKind.THRESHOLD_CHANGE),
        (FailureClass.DATA_QUALITY, ProposalKind.NEW_DATA_SOURCE),
        (FailureClass.STRATEGY_FAILURE, ProposalKind.STRATEGY_RETIREMENT),
        (FailureClass.REGIME_CHANGE, ProposalKind.STRATEGY_RETIREMENT),
    ],
)
def test_proposal_kind_mapping_per_failure_class(
    failure_class: FailureClass, kind: ProposalKind
) -> None:
    [proposal] = proposals_from_clusters([_cluster(failure_class, 6)], AS_OF, NOW)
    assert proposal.kind == kind
    assert proposal.target == f"{failure_class.value}/c1"
    assert proposal.status == ProposalStatus.OPEN
    assert proposal.proposed_change["failure_class"] == failure_class.value
    assert proposal.evidence[0].source_ref.startswith("decision_experience:")


def test_no_proposals_below_min_cluster_size() -> None:
    clusters = [
        _cluster(FailureClass.QUANT_MODEL, 4),  # below the default 5
        _cluster(FailureClass.DATA_QUALITY, 5),  # at threshold
    ]
    proposals = proposals_from_clusters(clusters, AS_OF, NOW)
    assert [p.kind for p in proposals] == [ProposalKind.NEW_DATA_SOURCE]


def test_proposal_ids_deterministic_for_reruns() -> None:
    first = proposals_from_clusters([_cluster(FailureClass.QUANT_MODEL, 6)], AS_OF, NOW)
    second = proposals_from_clusters([_cluster(FailureClass.QUANT_MODEL, 6)], AS_OF, NOW)
    assert first == second
    assert first[0].id == f"retrain:quant_model/c1:{AS_OF.isoformat()}"


def test_mapping_covers_every_failure_class_except_normal_uncertainty() -> None:
    assert set(KIND_BY_FAILURE_CLASS) == set(FailureClass) - {FailureClass.NORMAL_UNCERTAINTY}
