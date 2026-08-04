"""Optimizer port registry (doc 07 section 2 upgrade path)."""

from __future__ import annotations

import pytest

from dynamo.portfolio.constructor import RuleBasedConstructor
from dynamo.portfolio.optimizer_port import (
    DEFAULT_OPTIMIZER,
    PortfolioOptimizer,
    get_optimizer,
)


def test_default_is_rule_based_and_satisfies_protocol():
    optimizer = get_optimizer()
    assert isinstance(optimizer, RuleBasedConstructor)
    assert isinstance(optimizer, PortfolioOptimizer)
    assert DEFAULT_OPTIMIZER == "rule_based"


def test_unknown_optimizer_rejected():
    with pytest.raises(KeyError, match="cvxpy"):
        get_optimizer("cvxpy")
