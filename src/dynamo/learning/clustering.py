"""Failure clustering: failed experiences → clusters → ImprovementProposals.

PURE — cases (including any embeddings) are PASSED IN; the composition layer
(the ``failure_clustering`` asset) does the joins and the embedding, this
module only groups and maps. Deterministic by construction: inputs are sorted
by case id, scipy's average-linkage clustering is deterministic, and labels
are assigned by (size desc, first case id).

Approach (plan Phase 6, deliberately simple):

1. group failed cases by ``failure_class`` — the review already named the
   dominant axis, no algorithm needed;
2. within a class, sub-cluster semantically via embedding cosine distance +
   agglomerative average linkage at a fixed threshold (the exact style of
   ``risk/correlation.py::correlation_clusters``). Cases without an
   embedding fall into one per-class "unembedded" subcluster rather than
   being dropped;
3. each cluster suggests a ``ProposalKind`` via ``KIND_BY_FAILURE_CLASS``
   (mapping documented below); :func:`proposals_from_clusters` turns clusters
   of at least ``min_cluster_size`` cases into OPEN ``ImprovementProposal``
   rows — a human approves them in the dashboard, nothing is auto-applied.

``failure_class -> ProposalKind`` mapping (documented, load-bearing):

- claude_reasoning / missing_information / confidence_calibration →
  PROMPT_REVISION (the committee's reasoning or inputs need better prompts);
- quant_model → RETRAIN;
- position_sizing / risk_management / portfolio_allocation →
  THRESHOLD_CHANGE (sizing/risk knobs, doc 08 promotion pipeline);
- data_quality → NEW_DATA_SOURCE;
- strategy_failure / regime_change → STRATEGY_RETIREMENT (re-validate or
  retire the strategy under current conditions);
- normal_uncertainty is NOT a failure (doc 10 §4) — callers exclude those
  rows before clustering; :func:`cluster_failures` also skips them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore

from dynamo.domain.models.learning import FailureClass, ImprovementProposal, ProposalKind
from dynamo.domain.models.research import SourceType
from dynamo.learning.drift import _evidence, _proposal

DISTANCE_THRESHOLD = 0.4  # cosine distance cut inside a failure class
MIN_CLUSTER_SIZE = 5  # proposals only from clusters at least this big
MAX_EXAMPLE_REFS = 5

KIND_BY_FAILURE_CLASS: dict[FailureClass, ProposalKind] = {
    FailureClass.CLAUDE_REASONING: ProposalKind.PROMPT_REVISION,
    FailureClass.MISSING_INFORMATION: ProposalKind.PROMPT_REVISION,
    FailureClass.CONFIDENCE_CALIBRATION: ProposalKind.PROMPT_REVISION,
    FailureClass.QUANT_MODEL: ProposalKind.RETRAIN,
    FailureClass.POSITION_SIZING: ProposalKind.THRESHOLD_CHANGE,
    FailureClass.RISK_MANAGEMENT: ProposalKind.THRESHOLD_CHANGE,
    FailureClass.PORTFOLIO_ALLOCATION: ProposalKind.THRESHOLD_CHANGE,
    FailureClass.DATA_QUALITY: ProposalKind.NEW_DATA_SOURCE,
    FailureClass.STRATEGY_FAILURE: ProposalKind.STRATEGY_RETIREMENT,
    FailureClass.REGIME_CHANGE: ProposalKind.STRATEGY_RETIREMENT,
    # normal_uncertainty intentionally absent — not a failure.
}


@dataclass(frozen=True)
class FailureCase:
    """One failed decision: the experience row + its review text (+ optional
    embedding of that text, computed by the caller)."""

    case_id: str  # experience id — the example ref proposals carry
    symbol: str
    as_of: date
    failure_class: FailureClass
    text: str
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FailureCluster:
    label: str  # "<failure_class>/c<i>" — deterministic
    failure_class: FailureClass  # dominant by construction (grouped first)
    size: int
    example_refs: tuple[str, ...]  # up to MAX_EXAMPLE_REFS case ids
    suggested_kind: ProposalKind


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance in [0, 2]; zero vectors get distance 1 to
    everything (no similarity signal)."""
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    unit = vectors / safe[:, None]
    sim = unit @ unit.T
    sim = np.where(np.isfinite(sim), sim, 0.0)
    sim[norms == 0.0, :] = 0.0
    sim[:, norms == 0.0] = 0.0
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _semantic_groups(
    cases: Sequence[FailureCase], distance_threshold: float
) -> list[list[FailureCase]]:
    """Agglomerative (average-linkage) subclusters over embedding cosine
    distance — mirrors ``risk/correlation.py::correlation_clusters``."""
    if len(cases) == 1:
        return [list(cases)]
    vectors = np.asarray([c.embedding for c in cases], dtype=np.float64)
    condensed = cast(np.ndarray, squareform(_cosine_distances(vectors), checks=False))
    link = cast(np.ndarray, linkage(condensed, method="average"))
    labels = fcluster(link, t=max(distance_threshold, 1e-9), criterion="distance")
    grouped: dict[int, list[FailureCase]] = {}
    for case, label in zip(cases, labels.tolist(), strict=True):
        grouped.setdefault(int(label), []).append(case)
    return list(grouped.values())


def cluster_failures(
    cases: Sequence[FailureCase],
    *,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[FailureCluster]:
    """Group by failure class, then semantic subclusters within each class
    (module docstring). Total and deterministic: empty input → empty list;
    ``normal_uncertainty`` cases are skipped (not failures)."""
    by_class: dict[FailureClass, list[FailureCase]] = {}
    for case in sorted(cases, key=lambda c: c.case_id):
        if case.failure_class not in KIND_BY_FAILURE_CLASS:
            continue  # normal_uncertainty (or future non-failure classes)
        by_class.setdefault(case.failure_class, []).append(case)

    clusters: list[FailureCluster] = []
    for failure_class in sorted(by_class, key=lambda fc: fc.value):
        members = by_class[failure_class]
        embedded = [c for c in members if c.embedding is not None]
        unembedded = [c for c in members if c.embedding is None]
        groups = _semantic_groups(embedded, distance_threshold) if embedded else []
        if unembedded:
            groups.append(unembedded)  # one honest catch-all, never dropped
        # Deterministic labels: biggest first, ties by first case id.
        groups.sort(key=lambda g: (-len(g), g[0].case_id))
        for i, group in enumerate(groups, start=1):
            refs = tuple(c.case_id for c in group[:MAX_EXAMPLE_REFS])
            clusters.append(
                FailureCluster(
                    label=f"{failure_class.value}/c{i}",
                    failure_class=failure_class,
                    size=len(group),
                    example_refs=refs,
                    suggested_kind=KIND_BY_FAILURE_CLASS[failure_class],
                )
            )
    return clusters


def proposals_from_clusters(
    clusters: Sequence[FailureCluster],
    as_of: date,
    now: datetime,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> list[ImprovementProposal]:
    """OPEN proposals for clusters of at least ``min_cluster_size`` failures
    (drift.py's ``_proposal`` builder; ids ``<kind>:<label>:<as_of>`` are
    deterministic, so a same-day re-run re-derives identical rows). Smaller
    clusters produce NOTHING — thin evidence never opens a proposal."""
    proposals: list[ImprovementProposal] = []
    for cluster in clusters:
        if cluster.size < min_cluster_size:
            continue
        refs = ", ".join(cluster.example_refs)
        proposals.append(
            _proposal(
                cluster.suggested_kind,
                cluster.label,
                as_of,
                now,
                [
                    _evidence(
                        f"{cluster.size} failed decisions cluster on failure class "
                        f"{cluster.failure_class.value!r} (examples: {refs})",
                        f"decision_experience:{cluster.label}",
                        as_of,
                        SourceType.JOURNAL,
                    )
                ],
                f"address the shared {cluster.failure_class.value} failure mode "
                f"behind {cluster.size} losses",
                {
                    "cluster": cluster.label,
                    "failure_class": cluster.failure_class.value,
                    "size": cluster.size,
                    "example_refs": list(cluster.example_refs),
                },
            )
        )
    return proposals
