"""PortfolioOptimizer port (doc 07 section 2, "Designed-in upgrade path").

Phase 1 registers :class:`RuleBasedConstructor`. Phase 2 — a ``CvxpyOptimizer``
maximizing risk-adjusted score under the identical ``RiskVerdict`` constraint
set with a turnover penalty — slots in behind this same Protocol once its
entry criteria fire (sustained > 25 positions, binding-constraint waste, or a
multi-objective need) AND it passes the identical property-test suite. Until
then it is documented, not built.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from dynamo.domain.models.portfolio import (
    PortfolioState,
    RebalanceTrigger,
    TargetPortfolio,
)
from dynamo.domain.models.risk import RiskConfig, RiskVerdict
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor


@runtime_checkable
class PortfolioOptimizer(Protocol):
    """The construction port: advisory state in, TargetPortfolio out. Pure."""

    def construct(
        self,
        candidates: AdvisoryInputs,
        verdict: RiskVerdict,
        current: PortfolioState,
        config: ConstructionConfig,
        *,
        risk_config: RiskConfig | None = None,
        current_weights: Mapping[str, float] | None = None,
        turnover_used_mtd: float = 0.0,
        triggers: frozenset[RebalanceTrigger] = frozenset(),
        risk_verdict_ref: str | None = None,
    ) -> TargetPortfolio: ...


# Phase-1 registry. "cvxpy" joins here only after passing the identical
# property-test suite (doc 07 section 9, optimizer-parity gate).
OPTIMIZERS: dict[str, type] = {
    "rule_based": RuleBasedConstructor,
}

DEFAULT_OPTIMIZER = "rule_based"


def get_optimizer(name: str = DEFAULT_OPTIMIZER) -> PortfolioOptimizer:
    try:
        cls = OPTIMIZERS[name]
    except KeyError:
        known = ", ".join(sorted(OPTIMIZERS))
        raise KeyError(f"unknown optimizer {name!r}; registered: {known}") from None
    instance = cls()
    assert isinstance(instance, PortfolioOptimizer)
    return instance
