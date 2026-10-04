"""Bankroll risk profiles: tiers by account size, the risk-sleeve cap, and
off-by-default parity."""

from __future__ import annotations

import dataclasses

import pytest

from dynamo.domain.models.research import Action
from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.bankroll import apply_tier, tier_for
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_state,
    make_verdict,
)

CORE = ("AAA", "BBB", "CCC", "DDD")
RISKY = ("ZZA", "ZZB", "BTC/USD")


def _cfg(**overrides: object) -> ConstructionConfig:
    """Replay's construction settings: every candidate is eligible."""
    return ConstructionConfig(quality_floor=0.0, long_term_top_fraction=1.0, **overrides)  # type: ignore[arg-type]


def _inputs(risk_sleeve: frozenset[str] = frozenset()) -> AdvisoryInputs:
    names = (*CORE, *RISKY)
    base = advisory(
        watchlist=[make_row(s, i + 1) for i, s in enumerate(names)],
        verdicts=[make_cverdict(s, Action.BUY, conviction=0.9) for s in names],
        sectors={s: f"sector-{i}" for i, s in enumerate(names)},
    )
    return dataclasses.replace(base, risk_sleeve_symbols=risk_sleeve)


def test_tiers_by_account_size() -> None:
    assert tier_for("v1", 1_000).name == "small"
    assert tier_for("v1", 4_999.99).name == "small"
    assert tier_for("v1", 5_000).name == "medium"
    assert tier_for("v1", 50_000).name == "large"
    assert tier_for("v1", 10_000_000).name == "large"
    with pytest.raises(ValueError, match="unknown bankroll profile"):
        tier_for("v9", 1_000)


def test_apply_tier_changes_the_shared_caps() -> None:
    risk, construction = apply_tier(tier_for("v1", 1_000), RiskConfig(), ConstructionConfig())
    assert risk.per_position_cap == 0.15 and risk.cash_floor == 0.02
    assert risk.crypto_sleeve_cap == 0.0  # small accounts: no speculative sleeve
    assert construction.max_positions == 8 and construction.risk_sleeve_cap == 0.0
    risk, construction = apply_tier(tier_for("v1", 100_000), RiskConfig(), ConstructionConfig())
    assert risk.per_position_cap == 0.08 and construction.risk_sleeve_cap == 0.15
    assert risk.crypto_sleeve_cap == 0.10  # never above the ADR-0013 crypto cap


def test_off_by_default_is_identical() -> None:
    """No cap set -> sleeve symbols change nothing."""
    ctor, verdict, state = RuleBasedConstructor(), make_verdict(), make_state({})
    plain = ctor.construct(_inputs(), verdict, state, _cfg())
    tagged = ctor.construct(_inputs(frozenset(RISKY)), verdict, state, _cfg())
    assert [(w.symbol, w.weight) for w in plain.weights] == [
        (w.symbol, w.weight) for w in tagged.weights
    ]


def test_risk_sleeve_caps_risky_names_together() -> None:
    cap = 0.06
    target = RuleBasedConstructor().construct(
        _inputs(frozenset(RISKY)),
        make_verdict(),
        make_state({}),
        _cfg(risk_sleeve_cap=cap),
    )
    risky = sum(w.weight for w in target.weights if w.symbol in RISKY)
    core = sum(w.weight for w in target.weights if w.symbol in CORE)
    assert 0.0 < risky <= cap + 1e-9
    assert core > 0.0


def test_zero_sleeve_buys_only_core() -> None:
    target = RuleBasedConstructor().construct(
        _inputs(frozenset(RISKY)),
        make_verdict(),
        make_state({}),
        _cfg(risk_sleeve_cap=0.0),
    )
    assert {w.symbol for w in target.weights} <= set(CORE)
