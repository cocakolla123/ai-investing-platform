"""Dark-sizing-path parity (plan Phase 3 — THE critical guarantee).

Flag off / empty ``effective_probability`` mapping ⇒ ``construct()`` output
is bit-identical to the pre-meta constructor; a populated mapping is the ONLY
thing that changes the tilt, and only through the unchanged formula.
"""

from __future__ import annotations

import pytest

from dynamo.domain.models.research import Action, Sleeve
from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor
from dynamo.risk.sizing import SizingCandidate, SizingContext, size_positions
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_signal,
    make_state,
    make_verdict,
)


def _fixture_inputs(**overrides: object) -> AdvisoryInputs:
    """A portfolio with retained holdings, new LT entries and a tactical
    signal — every sizing path exercised."""
    base = advisory(
        watchlist=[make_row("AAPL", 1), make_row("MSFT", 2), make_row("NVDA", 3)],
        verdicts=[
            make_cverdict("AAPL", Action.BUY, 0.8),
            make_cverdict("MSFT", Action.BUY, 0.6),
            make_cverdict("TSM", Action.HOLD, 0.5),
        ],
        signals=[make_signal("SMCI", 0.7)],
        symbols=["TSM", "SMCI"],
        sectors={"AAPL": "tech", "MSFT": "tech", "SMCI": "tech", "TSM": "tech"},
    )
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)  # type: ignore[arg-type]


def _construct(inputs: AdvisoryInputs) -> str:
    state = make_state({"TSM": 0.05})
    # Roomy caps so the conviction tilt (not a binding clamp) determines the
    # sized weight — the tilt difference must be visible in the output.
    rcfg = RiskConfig(per_position_cap=0.5, per_position_hard=0.6, sector_cap=0.9)
    target = RuleBasedConstructor().construct(
        inputs, make_verdict(), state, ConstructionConfig(), risk_config=rcfg
    )
    return target.model_dump_json()


def test_default_and_empty_mapping_are_bit_identical() -> None:
    """The parity contract: constructing with the field left at its default
    and with an explicitly empty mapping produce byte-identical JSON."""
    assert _construct(_fixture_inputs()) == _construct(_fixture_inputs(effective_probability={}))


def test_mapping_for_absent_symbols_is_still_bit_identical() -> None:
    """Scores for symbols not being sized must change nothing."""
    assert _construct(_fixture_inputs()) == _construct(
        _fixture_inputs(effective_probability={"ZZZ": 0.9, "TSM": 0.9})
        # TSM is a retained holding: retained lots keep carried weights and
        # never re-enter the cascade, so its score is inert too.
    )


def test_effective_probability_changes_only_the_tilt() -> None:
    baseline = _construct(_fixture_inputs())
    tilted = _construct(_fixture_inputs(effective_probability={"AAPL": 0.2}))
    assert baseline != tilted


def test_sizing_tilt_prefers_effective_probability() -> None:
    config = RiskConfig()
    # Small budget so no cap binds and the tilt shows through in the weight.
    ctx = SizingContext(equity=1_000_000.0, sleeve_budget=0.05)

    def _cand(p: float | None) -> SizingCandidate:
        return SizingCandidate(
            symbol="AAPL",
            sleeve=Sleeve.LONG_TERM,
            conviction=0.8,
            sigma_annual=0.25,
            adv_dollar_20d=1e9,
            effective_probability=p,
        )

    none_tilt = size_positions([_cand(None)], config, ctx)[0]
    meta_tilt = size_positions([_cand(0.3)], config, ctx)[0]
    raw_equiv = size_positions([_cand(0.8)], config, ctx)[0]
    assert none_tilt.conviction_tilt == pytest.approx(1.3)  # 0.5 + conviction
    assert meta_tilt.conviction_tilt == pytest.approx(0.8)  # 0.5 + p, same formula
    assert raw_equiv.conviction_tilt == none_tilt.conviction_tilt
    assert meta_tilt.weight < none_tilt.weight


def test_non_finite_effective_probability_falls_back_to_conviction() -> None:
    config = RiskConfig()
    ctx = SizingContext(equity=1_000_000.0, sleeve_budget=0.5)
    nan_cand = SizingCandidate(
        symbol="AAPL",
        sleeve=Sleeve.LONG_TERM,
        conviction=0.8,
        sigma_annual=0.25,
        adv_dollar_20d=1e9,
        effective_probability=float("nan"),
    )
    clean = size_positions(
        [
            SizingCandidate(
                symbol="AAPL",
                sleeve=Sleeve.LONG_TERM,
                conviction=0.8,
                sigma_annual=0.25,
                adv_dollar_20d=1e9,
            )
        ],
        config,
        ctx,
    )[0]
    assert size_positions([nan_cand], config, ctx)[0] == clean
