"""Portfolio construction — pure, deterministic, verdict-constrained.

Design: docs/design/07-portfolio-construction.md. The optimizer proposes;
``RiskVerdict`` disposes. Imported unchanged by backtest, paper, and live.
"""

from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import (
    MIN_POSITION_WEIGHT,
    AdvisoryInputs,
    ConstructionOutput,
    Exclusion,
    RuleBasedConstructor,
)
from dynamo.portfolio.optimizer_port import (
    DEFAULT_OPTIMIZER,
    OPTIMIZERS,
    PortfolioOptimizer,
    get_optimizer,
)
from dynamo.portfolio.rebalance import (
    RebalanceDecision,
    TriggerInputs,
    evaluate_triggers,
)

__all__ = [
    "DEFAULT_OPTIMIZER",
    "MIN_POSITION_WEIGHT",
    "OPTIMIZERS",
    "AdvisoryInputs",
    "ConstructionConfig",
    "ConstructionOutput",
    "Exclusion",
    "PortfolioOptimizer",
    "RebalanceDecision",
    "RuleBasedConstructor",
    "TriggerInputs",
    "evaluate_triggers",
    "get_optimizer",
]
