"""Property tests (doc 07 section 9): the constructed portfolio NEVER
violates any RiskVerdict constraint, is deterministic under input shuffles,
and respects the turnover budget for discretionary trades."""

from __future__ import annotations

import math
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.research import Action, Sleeve
from dynamo.domain.models.risk import LadderRung, Regime, RiskConfig
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_signal,
    make_state,
    make_verdict,
)

POOL = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
SECTORS = {s: ("tech" if i % 2 == 0 else "energy") for i, s in enumerate(POOL)}
TOL = 1e-9


@st.composite
def scenario(draw):
    symbols = draw(st.lists(st.sampled_from(POOL), min_size=2, max_size=8, unique=True))
    held = draw(st.lists(st.sampled_from(symbols), max_size=4, unique=True))
    weights = {s: draw(st.floats(0.01, 0.12, allow_nan=False, allow_infinity=False)) for s in held}
    actions = {
        s: draw(st.sampled_from(list(Action)))
        for s in draw(st.lists(st.sampled_from(symbols), max_size=6, unique=True))
    }
    signal_symbols = draw(st.lists(st.sampled_from(symbols), max_size=4, unique=True))
    vetoed = draw(st.lists(st.sampled_from(symbols), max_size=2, unique=True))
    verdict = make_verdict(
        regime=draw(st.sampled_from(list(Regime))),
        ladder=draw(
            st.sampled_from([LadderRung.NORMAL, LadderRung.DD5, LadderRung.DD8, LadderRung.DD10])
        ),
        new_risk=draw(st.booleans()),
        gross_cap=draw(st.floats(0.3, 1.0)),
        cash_floor=draw(st.floats(0.0, 0.4)),
        per_position_cap={s: draw(st.floats(0.0, 0.15)) for s in symbols},
        sector_caps={"tech": draw(st.floats(0.05, 0.5)), "energy": 0.4},
        vetoed=vetoed,
    )
    quality = draw(st.floats(0.0, 100.0))
    inputs = advisory(
        watchlist=[make_row(s, rank=i + 1, quality=quality) for i, s in enumerate(sorted(symbols))],
        verdicts=[make_cverdict(s, a) for s, a in sorted(actions.items())],
        signals=[make_signal(s) for s in sorted(signal_symbols)],
        symbols=symbols,
        sectors=SECTORS,
    )
    turnover_used = draw(st.floats(0.0, 0.2))
    return inputs, verdict, weights, turnover_used


def _by_symbol(target):
    out: dict[str, float] = {}
    for w in target.weights:
        out[w.symbol] = out.get(w.symbol, 0.0) + w.weight
    return out


@settings(max_examples=100, deadline=None)
@given(scenario())
def test_never_violates_verdict_constraints(sc):
    inputs, verdict, weights, turnover_used = sc
    state = make_state(weights)
    target = RuleBasedConstructor().construct(
        inputs,
        verdict,
        state,
        ConstructionConfig(),
        risk_config=RiskConfig(),
        turnover_used_mtd=turnover_used,
    )

    by_symbol = _by_symbol(target)
    total = math.fsum(by_symbol.values())

    # weights + cash == 1 within 1e-9
    assert abs(total + target.cash - 1.0) <= TOL
    # cash >= floor
    assert target.cash >= verdict.cash_floor - TOL
    # gross <= cap
    assert total <= verdict.gross_exposure_cap + TOL
    # per-symbol combined weight <= verdict cap
    for symbol, w in by_symbol.items():
        cap = verdict.per_position_cap.get(symbol)
        if cap is not None:
            assert w <= cap + TOL
        assert w >= 0.0
    # sector caps on combined exposure
    sector_tot: dict[str, float] = {}
    for symbol, w in by_symbol.items():
        sec = SECTORS.get(symbol)
        if sec is not None:
            sector_tot[sec] = sector_tot.get(sec, 0.0) + w
    for sec, cap in verdict.sector_caps.items():
        assert sector_tot.get(sec, 0.0) <= cap + TOL
    # no vetoed symbol
    vetoed = {v.symbol for v in verdict.vetoes}
    assert not vetoed & set(by_symbol)
    # new_risk_allowed=False -> no new names
    if not verdict.new_risk_allowed:
        assert set(by_symbol) <= set(weights)


@settings(max_examples=100, deadline=None)
@given(scenario(), st.integers(0, 2**32 - 1))
def test_determinism_under_input_shuffles(sc, seed):
    inputs, verdict, weights, turnover_used = sc
    state = make_state(weights)
    constructor = RuleBasedConstructor()
    cfg = ConstructionConfig()

    baseline = constructor.construct(inputs, verdict, state, cfg, turnover_used_mtd=turnover_used)

    rng = random.Random(seed)
    watchlist = list(inputs.watchlist)
    signals = list(inputs.signals)
    rng.shuffle(watchlist)
    rng.shuffle(signals)
    verdict_items = list(inputs.verdicts.items())
    rng.shuffle(verdict_items)
    shuffled = AdvisoryInputs(
        as_of=inputs.as_of,
        watchlist=tuple(watchlist),
        verdicts=dict(verdict_items),
        quant_scores=inputs.quant_scores,
        quality_scores=inputs.quality_scores,
        signals=tuple(signals),
        sigma_annual=dict(reversed(list(inputs.sigma_annual.items()))),
        adv_dollar_20d=inputs.adv_dollar_20d,
        sector_by_symbol=inputs.sector_by_symbol,
    )
    again = constructor.construct(shuffled, verdict, state, cfg, turnover_used_mtd=turnover_used)
    assert baseline.model_dump_json() == again.model_dump_json()


@settings(max_examples=100, deadline=None)
@given(scenario())
def test_turnover_budget_respected(sc):
    """Discretionary one-way turnover never exceeds the remaining budget."""
    inputs, verdict, weights, turnover_used = sc
    state = make_state(weights)
    cfg = ConstructionConfig()
    target = RuleBasedConstructor().construct(
        inputs, verdict, state, cfg, turnover_used_mtd=turnover_used
    )
    mandatory_triggers = {"risk_breach", "verdict_change"}
    discretionary = math.fsum(
        abs(c.target_weight - c.current_weight) / 2.0
        for c in target.changes
        if c.trigger.value not in mandatory_triggers
    )
    remaining = max(cfg.turnover_budget_monthly - turnover_used, 0.0)
    assert discretionary <= remaining + TOL


@settings(max_examples=50, deadline=None)
@given(scenario())
def test_all_weights_above_floor_and_position_counts(sc):
    inputs, verdict, weights, _ = sc
    state = make_state(weights)
    cfg = ConstructionConfig()
    target = RuleBasedConstructor().construct(inputs, verdict, state, cfg)
    for w in target.weights:
        assert w.weight >= 0.005 - TOL  # no weight below the 0.5% floor
    tactical = [w for w in target.weights if w.sleeve is Sleeve.TACTICAL]
    assert len(tactical) <= cfg.max_tactical_positions
