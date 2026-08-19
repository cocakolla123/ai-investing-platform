"""Idle-cash index ballast: constructor overlay + config/settings validation.

Semantics under test (see ``RuleBasedConstructor._ballast_lot``):
- disabled (empty symbol) is bit-identical to the pre-ballast constructor;
- enabled, the residual above ``max(verdict.cash_floor, ballast_cash_floor)``
  rides the ballast symbol, scaled by ballast's OWN, much deeper de-risking
  schedule (campaign findings item 20) — NOT the active ladder's risk
  multiplier, which whipsawed the passive floor at DD5;
- active positions always take precedence (ballast is the residual on top,
  never a competitor for sleeve budgets or sector headroom);
- blocked rungs / no-new-risk / degraded verdicts hold ballast but never
  increase it; a risk veto on the ballast symbol exits it;
- ballast trades never charge the monthly turnover budget;
- crypto pairs are refused as ballast (config AND settings validation).
"""

from __future__ import annotations

import math

import pytest

from dynamo.config.settings import Settings
from dynamo.domain.models.research import Action, Sleeve
from dynamo.domain.models.risk import LadderRung
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import RuleBasedConstructor
from dynamo.risk.ladder import BALLAST_RUNG_MULTIPLIERS, RUNG_EFFECTS
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_state,
    make_verdict,
)

BALLAST = "SPY"


def cfg(**kwargs: object) -> ConstructionConfig:
    return ConstructionConfig(ballast_symbol=BALLAST, **kwargs)  # type: ignore[arg-type]


def construct(config: ConstructionConfig, *, verdict=None, state=None, inputs=None):
    verdict = verdict if verdict is not None else make_verdict()
    state = state if state is not None else make_state({})
    inputs = inputs if inputs is not None else advisory()
    return RuleBasedConstructor().construct(inputs, verdict, state, config)


def weight_of(target, symbol: str, sleeve: Sleeve | None = None) -> float:
    return math.fsum(
        w.weight
        for w in target.weights
        if w.symbol == symbol and (sleeve is None or w.sleeve is sleeve)
    )


def active_total(target) -> float:
    return math.fsum(w.weight for w in target.weights if w.sleeve is not Sleeve.BALLAST)


# --- disabled = parity ------------------------------------------------------


def test_disabled_is_bit_identical_no_ballast_weight() -> None:
    inputs = advisory(
        watchlist=[make_row("AAA", 1)], verdicts=[make_cverdict("AAA", Action.BUY)]
    )
    off = construct(ConstructionConfig(), inputs=inputs)
    off_again = construct(ConstructionConfig(), inputs=inputs)
    assert off.model_dump_json() == off_again.model_dump_json()  # determinism
    assert all(w.sleeve is not Sleeve.BALLAST for w in off.weights)
    assert weight_of(off, BALLAST) == 0.0


def test_enabled_active_weights_identical_to_disabled_run() -> None:
    """Ballast is purely additive: it never crowds out or resizes actives."""
    inputs = advisory(
        watchlist=[make_row("AAA", 1), make_row("BBB", 2)],
        verdicts=[
            make_cverdict("AAA", Action.BUY, 0.8),
            make_cverdict("BBB", Action.BUY, 0.6),
        ],
    )
    off = construct(ConstructionConfig(), inputs=inputs)
    on = construct(cfg(), inputs=inputs)
    actives_off = {(w.symbol, w.sleeve): w.weight for w in off.weights}
    actives_on = {
        (w.symbol, w.sleeve): w.weight
        for w in on.weights
        if w.sleeve is not Sleeve.BALLAST
    }
    assert actives_on == actives_off


# --- residual fill ----------------------------------------------------------


def test_ballast_fills_residual_to_one_minus_floor() -> None:
    inputs = advisory(
        watchlist=[make_row("AAA", 1)], verdicts=[make_cverdict("AAA", Action.BUY)]
    )
    verdict = make_verdict(gross_cap=0.95, cash_floor=0.05)
    target = construct(cfg(), verdict=verdict, inputs=inputs)
    ballast = weight_of(target, BALLAST, Sleeve.BALLAST)
    assert ballast > 0.0
    assert active_total(target) + ballast == pytest.approx(0.95, abs=1e-9)
    assert target.cash == pytest.approx(0.05, abs=1e-9)
    lot = next(w for w in target.weights if w.sleeve is Sleeve.BALLAST)
    assert lot.strategy_id == "ballast"
    assert "ballast" in lot.rationale


def test_ballast_alone_when_no_active_candidates() -> None:
    target = construct(cfg())
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95)
    assert target.cash == pytest.approx(0.05)


def test_ballast_cash_floor_binds_when_higher_than_verdict_floor() -> None:
    verdict = make_verdict(gross_cap=1.0, cash_floor=0.02)
    target = construct(cfg(ballast_cash_floor=0.10), verdict=verdict)
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.90)


# --- ladder / risk posture --------------------------------------------------


def test_shallow_rungs_leave_ballast_untouched() -> None:
    """Campaign findings item 20: ballast does NOT follow the active ladder.

    The control run halved the passive index floor at DD5 (invested exposure
    94.9% -> 47.7% on 2026-04-01) and rebought higher two weeks later. A -5%
    portfolio drawdown is noise for a passive floor, so the shallow rungs no
    longer trim it — while ACTIVE positions keep the old ladder exactly.
    """
    for rung in (LadderRung.NORMAL, LadderRung.DD5, LadderRung.DD8):
        target = construct(cfg(), verdict=make_verdict(ladder=rung))
        assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95), rung
        assert target.cash == pytest.approx(0.05), rung
    # DD10 still blocks INCREASES (new_positions_allowed False) but no longer
    # trims what is already held.
    held = construct(
        cfg(),
        verdict=make_verdict(ladder=LadderRung.DD10),
        state=make_state({BALLAST: 0.90}),
    )
    assert weight_of(held, BALLAST, Sleeve.BALLAST) == pytest.approx(0.90)


def test_ballast_ladder_schedule_is_a_configurable_table() -> None:
    """The schedule is a documented, overridable table — not magic numbers."""
    verdict = make_verdict(ladder=LadderRung.DD5)
    strict = cfg(ballast_ladder_multipliers=((LadderRung.DD5, 0.5),))
    assert weight_of(
        construct(strict, verdict=verdict), BALLAST, Sleeve.BALLAST
    ) == pytest.approx(0.95 * 0.5)
    assert ConstructionConfig(ballast_symbol=BALLAST).config_hash != strict.config_hash
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        cfg(ballast_ladder_multipliers=((LadderRung.DD5, 1.5),))


def test_blocked_rung_holds_existing_ballast_but_never_increases() -> None:
    verdict = make_verdict(ladder=LadderRung.DD10)  # new_positions_allowed False
    held = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.30}))
    assert weight_of(held, BALLAST, Sleeve.BALLAST) == pytest.approx(0.30)
    empty = construct(cfg(), verdict=verdict, state=make_state({}))
    assert weight_of(empty, BALLAST) == 0.0  # nothing held -> nothing opened


def test_new_risk_blocked_verdict_holds_not_increases() -> None:
    verdict = make_verdict(new_risk=False)
    held = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.20}))
    assert weight_of(held, BALLAST, Sleeve.BALLAST) == pytest.approx(0.20)
    empty = construct(cfg(), verdict=verdict, state=make_state({}))
    assert weight_of(empty, BALLAST) == 0.0


def test_dd15_trims_held_ballast_by_gross_override_and_ballast_table() -> None:
    """Deep rungs DO de-risk ballast: the 0.5 gross override still binds (the
    hard protection), and the ballast table's 0.75 trims modestly on top."""
    verdict = make_verdict(ladder=LadderRung.DD15, new_risk=False, gross_cap=0.5)
    target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.90}))
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.5 * 0.75)


def test_dd20_kill_switch_exits_ballast() -> None:
    verdict = make_verdict(ladder=LadderRung.DD20, new_risk=False, gross_cap=0.0)
    target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.90}))
    assert weight_of(target, BALLAST) == 0.0
    exit_change = next(c for c in target.changes if c.symbol == BALLAST)
    assert exit_change.target_weight == 0.0


def test_dd20_exits_ballast_even_without_the_gross_override() -> None:
    """The ballast table alone zeroes DD20 — the exit does not depend on the
    risk engine having already set gross_cap=0."""
    verdict = make_verdict(ladder=LadderRung.DD20, new_risk=False, gross_cap=0.95)
    target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.90}))
    assert weight_of(target, BALLAST) == 0.0


def test_veto_on_ballast_symbol_exits_it() -> None:
    verdict = make_verdict(vetoed=[BALLAST])
    target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.40}))
    assert weight_of(target, BALLAST) == 0.0
    exit_change = next(c for c in target.changes if c.symbol == BALLAST)
    assert exit_change.target_weight == 0.0


def test_veto_still_exits_ballast_at_a_shallow_rung() -> None:
    """The ladder change must not weaken the veto: DD5 no longer trims, but a
    risk veto on the ballast symbol still exits it outright."""
    verdict = make_verdict(ladder=LadderRung.DD5, vetoed=[BALLAST])
    target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.40}))
    assert weight_of(target, BALLAST) == 0.0


def test_gross_cap_and_cash_floor_still_bound_ballast_at_every_rung() -> None:
    """The hard protections are structural, not table-driven: whatever the
    ballast schedule says, the residual can never exceed the risk engine's
    gross cap nor eat the cash floor."""
    for rung in LadderRung:
        verdict = make_verdict(ladder=rung, gross_cap=0.60, cash_floor=0.25)
        target = construct(cfg(), verdict=verdict, state=make_state({BALLAST: 0.90}))
        ballast = weight_of(target, BALLAST, Sleeve.BALLAST)
        assert ballast <= 0.60 + 1e-12, rung
        assert target.cash >= 0.25 - 1e-12, rung


# --- active-position ladder parity ------------------------------------------


def test_active_ladder_behavior_is_bit_identical_with_and_without_ballast() -> None:
    """Parity: the ballast schedule is consumed ONLY by ``_ballast_lot``.

    The ballast-disabled path is literally the pre-change code path (it never
    reaches ``_ballast_lot``), so equality of every active weight, change and
    rationale at every rung proves the new table cannot leak into
    thesis-driven positions.
    """
    inputs = advisory(
        watchlist=[make_row("AAA", 1), make_row("BBB", 2)],
        verdicts=[
            make_cverdict("AAA", Action.BUY, 0.8),
            make_cverdict("BBB", Action.BUY, 0.6),
        ],
    )
    for rung in LadderRung:
        verdict = make_verdict(ladder=rung)
        off = construct(ConstructionConfig(), verdict=verdict, inputs=inputs)
        on = construct(cfg(), verdict=verdict, inputs=inputs)
        actives_off = [w for w in off.weights if w.sleeve is not Sleeve.BALLAST]
        actives_on = [w for w in on.weights if w.sleeve is not Sleeve.BALLAST]
        assert [w.model_dump() for w in actives_on] == [
            w.model_dump() for w in actives_off
        ], rung


def test_active_ladder_table_is_unchanged() -> None:
    """The counterpart of the parity test: the ACTIVE ladder table itself is
    untouched (doc 06 section 3), and ballast diverges from it exactly at the
    shallow rungs that were whipsawing the index floor."""
    assert {rung: RUNG_EFFECTS[rung].risk_multiplier for rung in LadderRung} == {
        LadderRung.NORMAL: 1.0,
        LadderRung.DD5: 0.5,
        LadderRung.DD8: 0.5,
        LadderRung.DD10: 0.0,
        LadderRung.DD15: 0.0,
        LadderRung.DD20: 0.0,
    }
    assert BALLAST_RUNG_MULTIPLIERS == {
        LadderRung.NORMAL: 1.0,
        LadderRung.DD5: 1.0,
        LadderRung.DD8: 1.0,
        LadderRung.DD10: 1.0,
        LadderRung.DD15: 0.75,
        LadderRung.DD20: 0.0,
    }


def test_ballast_holds_the_portfolio_invested_through_a_shallow_drawdown() -> None:
    """The measured regression: at DD5 the control run's invested exposure
    fell 94.9% -> 47.7%. With ballast on its own schedule, the shallow rung
    leaves total invested exposure unchanged — the ladder reallocates
    between active risk and the passive floor instead of going to cash."""
    inputs = advisory(
        watchlist=[make_row("AAA", 1)], verdicts=[make_cverdict("AAA", Action.BUY, 0.8)]
    )
    normal = construct(cfg(), verdict=make_verdict(), inputs=inputs)
    dd5 = construct(cfg(), verdict=make_verdict(ladder=LadderRung.DD5), inputs=inputs)
    invested = active_total(dd5) + weight_of(dd5, BALLAST, Sleeve.BALLAST)
    assert invested == pytest.approx(
        active_total(normal) + weight_of(normal, BALLAST, Sleeve.BALLAST)
    )
    assert dd5.cash == pytest.approx(normal.cash)


# --- active interplay -------------------------------------------------------


def test_active_verdict_on_ballast_symbol_active_wins_ballast_residual_on_top() -> None:
    inputs = advisory(
        watchlist=[make_row(BALLAST, 1)],
        verdicts=[make_cverdict(BALLAST, Action.BUY, 0.9)],
    )
    baseline = construct(ConstructionConfig(), inputs=inputs)
    active_w = weight_of(baseline, BALLAST)
    assert active_w > 0.0  # SPY is a real active pick in the baseline
    target = construct(cfg(), inputs=inputs)
    assert weight_of(target, BALLAST, Sleeve.LONG_TERM) == pytest.approx(active_w)
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(
        0.95 - active_w, abs=1e-9
    )
    assert weight_of(target, BALLAST) == pytest.approx(0.95, abs=1e-9)


def test_held_ballast_is_not_auto_retained_as_an_active_incumbent() -> None:
    """A held ballast position is overlay-governed: no LONG_TERM lot appears."""
    target = construct(cfg(), state=make_state({BALLAST: 0.50}))
    assert weight_of(target, BALLAST, Sleeve.LONG_TERM) == 0.0
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95)


# --- turnover / churn -------------------------------------------------------


def test_ballast_trades_never_charge_the_turnover_budget() -> None:
    target = construct(cfg())  # all-cash -> 0.95 ballast buy
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95)
    assert target.turnover_used_mtd == 0.0


def test_active_trades_still_charge_the_budget_with_ballast_on() -> None:
    inputs = advisory(
        watchlist=[make_row("AAA", 1)], verdicts=[make_cverdict("AAA", Action.BUY)]
    )
    target = construct(cfg(), inputs=inputs)
    active_w = weight_of(target, "AAA")
    assert active_w > 0.0
    assert target.turnover_used_mtd == pytest.approx(active_w / 2.0)


def test_churn_guard_pins_ballast_below_min_trade_weight() -> None:
    held = 0.949  # within min_trade_weight (0.25%) of the 0.95 residual
    target = construct(cfg(), state=make_state({BALLAST: held}))
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(held)
    assert not any(c.symbol == BALLAST for c in target.changes)


def test_ballast_never_deferred_by_an_exhausted_turnover_budget() -> None:
    target = construct(cfg(), verdict=make_verdict())
    # Budget fully consumed before this run: ballast still deploys.
    out = RuleBasedConstructor().construct_detailed(
        advisory(),
        make_verdict(),
        make_state({}),
        cfg(),
        turnover_used_mtd=10.0,
    )
    assert weight_of(out.target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95)
    assert not any(c.symbol == BALLAST for c in out.deferred)
    assert weight_of(target, BALLAST, Sleeve.BALLAST) == pytest.approx(0.95)


# --- validation -------------------------------------------------------------


def test_construction_config_refuses_crypto_ballast() -> None:
    with pytest.raises(ValueError, match="crypto"):
        ConstructionConfig(ballast_symbol="BTC/USD")


def test_settings_refuse_crypto_ballast_and_default_to_spy_on() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ballast_enabled is True
    assert s.ballast_symbol == "SPY"
    with pytest.raises(ValueError, match="crypto"):
        Settings(BALLAST_SYMBOL="ETH/USD", _env_file=None)  # type: ignore[call-arg]


def test_config_hash_changes_with_ballast() -> None:
    assert ConstructionConfig().config_hash != cfg().config_hash
