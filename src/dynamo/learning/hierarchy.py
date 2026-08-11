"""Hierarchical calibration over nested scopes (plan Phase 2).

Scope path, general → specific::

    global → sleeve → strategy → regime → sector → quant_alignment → event_type

A query is a :class:`ScopeKey` — every dimension optional. Each populated
key maps to a chain of path-string scope nodes, most-specific first, dropping
the most-specific populated dimension at each step until only ``global``
remains, e.g. ``sleeve=core|strategy=momentum-rs|regime=risk_off`` →
``sleeve=core|strategy=momentum-rs`` → ``sleeve=core`` → ``global``.
Dimensions without a value are simply omitted from every node.

Algorithm (:meth:`HierarchicalCalibrator.calibrate`), computed recursively
from the global root down the chain:

1. **Root**: when the global node has at least ``min_n`` observations, the
   raw probability is recalibrated through the existing
   :func:`~dynamo.learning.calibration.conformal_adjust` (Venn-ABERS) fit on
   the root's history. An EMPTY (or thin) global root is the identity — the
   raw probability flows through unchanged.
2. **Each child node down the chain**: when the node has ``n >= min_n``
   observations, its own conformal fit produces ``p_child``, and the running
   value is updated by empirical-Bayes shrinkage toward the parent::

       p = (n_child * p_child + k * p_parent) / (n_child + k),   k = 30

   Below ``min_n`` there is no per-node conformal fit — the node contributes
   only its (small) ``n`` to the shrinkage blend, and since the child value
   defaults to the parent's, the parent value flows through unchanged.

Total and deterministic by construction: no exceptions on empty data, no
randomness, no I/O — observations are PASSED IN (import-linter keeps this
module off persistence). ``conformal_adjust`` itself degrades to identity on
degenerate history, so every path yields a probability in [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from dynamo.learning.calibration import (
    MIN_CONFORMAL_N,
    CalibrationObservation,
    conformal_adjust,
)

if TYPE_CHECKING:
    from dynamo.domain.models.learning import DecisionExperience

# General → specific. A node's path string lists its populated dimensions in
# THIS order; scope_chain drops the most-specific populated one per step.
SCOPE_DIMENSIONS = ("sleeve", "strategy", "regime", "sector", "quant_alignment", "event_type")
GLOBAL_SCOPE = "global"
HIER_SCOPE_TYPE = "hier"  # calibration_points.scope_type for these nodes
SHRINKAGE_K = 30

# One observed (conviction, outcome) resolution at a scope node.
Observation = tuple[float, bool]
# A per-node fit: (p_raw, node history) -> calibrated probability. Injected
# for tests; the default wraps conformal_adjust (min_n enforced by caller).
NodeFit = Callable[[float, Sequence[Observation]], float]


@dataclass(frozen=True)
class ScopeKey:
    """One decision's coordinates on the scope path (all dims optional).

    ``strategy`` and ``event_type`` are part of the path for forward
    compatibility but are not yet recorded on live experiences — they stay
    empty (omitted) until the platform captures them.
    """

    sleeve: str | None = None
    strategy: str | None = None
    regime: str | None = None
    sector: str | None = None
    quant_alignment: str | None = None
    event_type: str | None = None


@dataclass(frozen=True)
class CalibratedProbability:
    """``value`` in [0, 1]; ``scope_used`` is the most specific node whose
    conformal fit informed it (``global`` when identity); ``n_effective`` is
    that node's observation count (0 for the identity fallback)."""

    value: float
    scope_used: str
    n_effective: int


def scope_path(dims: Sequence[tuple[str, str]]) -> str:
    """Deterministic path string for populated dims: ``a=x|b=y`` (``global``
    for the empty root). Dims must already be in SCOPE_DIMENSIONS order."""
    if not dims:
        return GLOBAL_SCOPE
    return "|".join(f"{name}={value}" for name, value in dims)


def _populated(key: ScopeKey) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in SCOPE_DIMENSIONS:
        value = getattr(key, name)
        if value is not None and str(value).strip():
            out.append((name, str(value).strip()))
    return out


def scope_chain(key: ScopeKey) -> list[str]:
    """Path-string scope ids, most-specific first, ending at ``global``.

    Each step drops the most-specific POPULATED dimension (event_type first,
    sleeve last); empty dimensions are omitted from every node. Deterministic:
    a fixed key always yields the identical chain.
    """
    dims = _populated(key)
    chain: list[str] = []
    while dims:
        chain.append(scope_path(dims))
        dims = dims[:-1]
    chain.append(GLOBAL_SCOPE)
    return chain


def shrink(p_child: float, n_child: int, p_parent: float, k: int = SHRINKAGE_K) -> float:
    """Empirical-Bayes blend of a node's value toward its parent's:
    ``(n*p_child + k*p_parent) / (n + k)``. ``n_child=0`` → parent exactly;
    ``n_child → ∞`` → child dominates; monotonic in between."""
    n = max(int(n_child), 0)
    if n == 0:
        return p_parent
    return (n * p_child + k * p_parent) / (n + k)


def _default_fit(p_raw: float, history: Sequence[Observation]) -> float:
    # min_n=1: the calibrator gates on min_n itself; conformal_adjust still
    # degrades to identity on degenerate/unfittable history.
    return conformal_adjust([p_raw], list(history), min_n=1)[0]


class HierarchicalCalibrator:
    """Per-node conformal fits + parent-shrinkage over a scope chain.

    Built from observations grouped per scope node (path-string scope id →
    list of ``(conviction, outcome)``). Total: unknown scopes, empty nodes,
    or an empty root all degrade gracefully (see module docstring).
    """

    def __init__(
        self,
        observations: Mapping[str, Sequence[Observation]],
        *,
        min_n: int = MIN_CONFORMAL_N,
        k: int = SHRINKAGE_K,
        node_fit: NodeFit = _default_fit,
    ) -> None:
        self._obs: dict[str, list[Observation]] = {
            scope: list(pairs) for scope, pairs in observations.items()
        }
        self._min_n = min_n
        self._k = k
        self._node_fit = node_fit

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[CalibrationObservation],
        *,
        min_n: int = MIN_CONFORMAL_N,
        k: int = SHRINKAGE_K,
        node_fit: NodeFit = _default_fit,
    ) -> HierarchicalCalibrator:
        """Group flat ``hier``-scope calibration observations per node."""
        grouped: dict[str, list[Observation]] = {}
        for o in observations:
            if o.scope_type != HIER_SCOPE_TYPE:
                continue
            grouped.setdefault(o.scope_id, []).append((o.conviction, o.outcome))
        return cls(grouped, min_n=min_n, k=k, node_fit=node_fit)

    def calibrate(self, p_raw: float, key: ScopeKey) -> CalibratedProbability:
        """Calibrated probability for ``p_raw`` under ``key``'s scope chain."""
        p = min(max(float(p_raw), 0.0), 1.0)
        scope_used = GLOBAL_SCOPE
        n_effective = 0
        # Walk general → specific (the chain is most-specific first).
        for scope in reversed(scope_chain(key)):
            history = self._obs.get(scope, [])
            n = len(history)
            if n >= self._min_n:
                p_node = min(max(float(self._node_fit(p_raw, history)), 0.0), 1.0)
                p = p_node if scope == GLOBAL_SCOPE else shrink(p_node, n, p, self._k)
                scope_used = scope
                n_effective = n
            # Below min_n: no conformal fit — the node's child value defaults
            # to the parent's, so the blend leaves p unchanged.
        return CalibratedProbability(value=p, scope_used=scope_used, n_effective=n_effective)


# --- DecisionExperience adapters (domain-only inputs; still pure) -----------


def experience_scope_key(exp: DecisionExperience) -> ScopeKey:
    """The experience's coordinates on the scope path. ``strategy`` and
    ``event_type`` are not recorded on experiences yet — always empty."""
    return ScopeKey(
        sleeve=exp.sleeve,
        regime=exp.regime,
        sector=exp.sector,
        quant_alignment=exp.quant_alignment,
    )


def experience_outcome(exp: DecisionExperience) -> bool | None:
    """Resolution label: path-based ``target_achieved`` first, the win/loss
    ``label`` as fallback; ``None`` when neither is known (row is skipped)."""
    if exp.target_achieved is not None:
        return bool(exp.target_achieved)
    if exp.label in ("win", "loss"):
        return exp.label == "win"
    return None


def hier_observations(
    experiences: Sequence[DecisionExperience],
) -> list[CalibrationObservation]:
    """Flat ``hier``-scope observations: one per chain prefix per resolvable
    experience (conviction and an outcome both required). Deterministic and
    total — unusable rows are skipped, never raised on."""
    out: list[CalibrationObservation] = []
    for exp in experiences:
        if exp.conviction is None:
            continue
        outcome = experience_outcome(exp)
        if outcome is None:
            continue
        conviction = min(max(float(exp.conviction), 0.0), 1.0)
        for scope_id in scope_chain(experience_scope_key(exp)):
            out.append(
                CalibrationObservation(
                    scope_type=HIER_SCOPE_TYPE,
                    scope_id=scope_id,
                    conviction=conviction,
                    outcome=outcome,
                )
            )
    return out


# Keep the dataclass-fields import honest (ScopeKey introspection for tests
# and future dimensions must track SCOPE_DIMENSIONS exactly).
assert tuple(f.name for f in fields(ScopeKey)) == SCOPE_DIMENSIONS
