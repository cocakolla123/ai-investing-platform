"""Golden construction scenarios (doc 07 section 9): frozen inputs, asserted
behavior — normal night, risk_off, ladder dd8, candidate scarcity, turnover
exhaustion, infeasible set -> fewer names + more cash, never an exception."""

from __future__ import annotations

import math

from dynamo.domain.models.research import Action, Sleeve
from dynamo.domain.models.risk import LadderRung, Regime, RiskConfig
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import RuleBasedConstructor
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_signal,
    make_state,
    make_verdict,
)

CONSTRUCTOR = RuleBasedConstructor()
CFG = ConstructionConfig()
RCFG = RiskConfig()


def _by_symbol(target):
    out: dict[str, float] = {}
    for w in target.weights:
        out[w.symbol] = out.get(w.symbol, 0.0) + w.weight
    return out


def caps(symbols, cap=0.08):
    return dict.fromkeys(symbols, cap)


class TestNormalNight:
    def test_buys_enter_and_cash_floor_holds(self):
        symbols = ["AAA", "BBB", "CCC", "DDD", "TTT"]
        inputs = advisory(
            # 4 ranked rows; the default top-50% cutoff admits ranks 1-2.
            watchlist=[
                make_row("AAA", 1),
                make_row("BBB", 2),
                make_row("CCC", 3),
                make_row("DDD", 4),
            ],
            verdicts=[
                make_cverdict("AAA", Action.BUY),
                make_cverdict("BBB", Action.BUY),
            ],
            signals=[make_signal("TTT")],
            symbols=symbols,
        )
        verdict = make_verdict(per_position_cap=caps(symbols))
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({}), CFG, risk_config=RCFG
        )
        by = _by_symbol(target)
        assert "AAA" in by and "BBB" in by  # long-term entries admitted
        assert "TTT" in by  # tactical signal admitted in neutral regime
        assert target.cash >= verdict.cash_floor
        assert abs(sum(by.values()) + target.cash - 1.0) < 1e-9
        sleeves = {w.symbol: w.sleeve for w in target.weights}
        assert sleeves["TTT"] is Sleeve.TACTICAL
        assert target.changes  # proposed diffs vs empty book

    def test_ids_are_stable(self):
        inputs = advisory(
            watchlist=[make_row("AAA", 1)],
            verdicts=[make_cverdict("AAA", Action.BUY)],
        )
        verdict = make_verdict(per_position_cap={"AAA": 0.08})
        t1 = CONSTRUCTOR.construct(inputs, verdict, make_state({}), CFG)
        t2 = CONSTRUCTOR.construct(inputs, verdict, make_state({}), CFG)
        assert t1.id == t2.id
        assert t1.model_dump_json() == t2.model_dump_json()


class TestBothSleeves:
    def test_symbol_can_hold_separate_lots_with_combined_cap(self):
        # AAA qualifies for the long-term sleeve AND carries a tactical
        # signal: two lots, two positions against count caps, and the
        # combined exposure respects the per-symbol verdict cap (doc 05 s5).
        inputs = advisory(
            watchlist=[make_row("AAA", 1)],
            verdicts=[make_cverdict("AAA", Action.BUY)],
            signals=[make_signal("AAA")],
        )
        verdict = make_verdict(per_position_cap={"AAA": 0.09})
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({}), CFG, risk_config=RCFG
        )
        lots = [w for w in target.weights if w.symbol == "AAA"]
        assert {w.sleeve for w in lots} == {Sleeve.LONG_TERM, Sleeve.TACTICAL}
        assert math.fsum(w.weight for w in lots) <= 0.09 + 1e-9


class TestRiskOffRegime:
    def test_no_tactical_entries_and_gross_shrinks(self):
        symbols = ["AAA", "TTT"]
        inputs = advisory(
            watchlist=[make_row("AAA", 1)],
            verdicts=[make_cverdict("AAA", Action.BUY)],
            signals=[make_signal("TTT")],
            symbols=symbols,
        )
        verdict = make_verdict(
            regime=Regime.RISK_OFF, gross_cap=0.76, per_position_cap=caps(symbols)
        )
        out = CONSTRUCTOR.construct_detailed(
            inputs, verdict, make_state({}), CFG, risk_config=RCFG
        )
        by = _by_symbol(out.target)
        # New tactical entries blocked in risk_off (default entry regimes).
        assert "TTT" not in by
        assert any(
            e.symbol == "TTT" and "regime" in e.reason for e in out.exclusions
        )
        assert math.fsum(by.values()) <= 0.76 + 1e-9
        # Regime-driven cash carries attribution.
        assert out.target.cash_attribution["regime"] > 0.0


class TestLadderDd8:
    def test_no_tactical_sleeve_at_dd8(self):
        symbols = ["AAA", "TTT"]
        inputs = advisory(
            watchlist=[make_row("AAA", 1)],
            verdicts=[make_cverdict("AAA", Action.BUY)],
            signals=[make_signal("TTT")],
            symbols=symbols,
        )
        verdict = make_verdict(
            ladder=LadderRung.DD8, per_position_cap=caps(symbols)
        )
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({}), CFG, risk_config=RCFG
        )
        sleeves = {w.symbol: w.sleeve for w in target.weights}
        assert all(s is not Sleeve.TACTICAL for s in sleeves.values())
        # dd8 halves the new-risk multiplier: still a valid, high-cash book.
        assert target.cash >= verdict.cash_floor


class TestCandidateScarcity:
    def test_budget_stays_in_cash_with_scarcity_attribution(self):
        # One tiny candidate; capacity nowhere near investable budget.
        inputs = advisory(
            watchlist=[make_row("AAA", 1)],
            verdicts=[make_cverdict("AAA", Action.BUY)],
        )
        verdict = make_verdict(per_position_cap={"AAA": 0.05})
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({}), CFG, risk_config=RCFG
        )
        assert target.cash > 0.5  # never reaches down the list to "get invested"
        assert target.cash_attribution["scarcity"] > 0.0
        assert target.cash_attribution["floor"] == verdict.cash_floor
        assert (
            abs(math.fsum(target.cash_attribution.values()) - target.cash) < 1e-9
        )

    def test_empty_candidate_list_yields_all_cash_no_exception(self):
        inputs = advisory(symbols=["AAA"])
        verdict = make_verdict()
        target = CONSTRUCTOR.construct(inputs, verdict, make_state({}), CFG)
        assert target.weights == ()
        assert target.cash == 1.0


class TestTurnoverExhaustion:
    def test_discretionary_deferred_mandatory_executed(self):
        held = {"AAA": 0.06, "SSS": 0.05}
        inputs = advisory(
            # NEW is inside the top-50% rank cutoff (order 0-1 of 3 rows).
            watchlist=[
                make_row("AAA", 1),
                make_row("NEW", 2),
                make_row("SSS", 3),
            ],
            verdicts=[
                make_cverdict("AAA", Action.HOLD),
                make_cverdict("SSS", Action.SELL),
                make_cverdict("NEW", Action.BUY),
            ],
            symbols=[*held, "NEW"],
        )
        verdict = make_verdict(per_position_cap=caps(["AAA", "SSS", "NEW"]))
        out = CONSTRUCTOR.construct_detailed(
            inputs,
            verdict,
            make_state(held),
            CFG,
            risk_config=RCFG,
            turnover_used_mtd=CFG.turnover_budget_monthly,  # exhausted
        )
        by = _by_symbol(out.target)
        assert "SSS" not in by  # mandatory sell executed
        assert "NEW" not in by  # discretionary buy deferred
        assert any(c.symbol == "NEW" for c in out.deferred)
        assert by["AAA"] == held["AAA"]  # hold retained untouched
        sell = [c for c in out.target.changes if c.symbol == "SSS"]
        assert sell and sell[0].target_weight == 0.0


class TestInfeasibleSet:
    def test_fewer_names_more_cash_never_exception(self):
        # Three buys in one sector against a sector cap that fits ~1.25 names:
        # the greedy pass gives rank order what fits and DROPS the rest —
        # it never shaves everyone below the 0.5% floor to squeeze them in.
        symbols = ["AAA", "BBB", "CCC"]
        inputs = advisory(
            watchlist=[make_row(s, i + 1) for i, s in enumerate(symbols)],
            verdicts=[make_cverdict(s, Action.BUY) for s in symbols],
            symbols=symbols,
            sectors=dict.fromkeys(symbols, "tech"),
        )
        verdict = make_verdict(
            per_position_cap=caps(symbols), sector_caps={"tech": 0.10}
        )
        cfg = ConstructionConfig(long_term_top_fraction=1.0)
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({}), cfg, risk_config=RCFG
        )
        by = _by_symbol(target)
        assert math.fsum(by.values()) <= 0.10 + 1e-9  # sector cap held
        assert len(by) < 3  # lowest-ranked dropped, not shaved below floor
        assert all(w >= 0.005 for w in by.values())
        assert target.cash >= verdict.cash_floor

    def test_degraded_inputs_never_raise(self):
        # NaN sigma, missing quality, vetoed half the book.
        inputs = advisory(
            watchlist=[make_row("AAA", 1, quality=None), make_row("BBB", 2)],
            verdicts=[
                make_cverdict("AAA", Action.BUY),
                make_cverdict("BBB", Action.BUY),
            ],
            sigma=float("nan"),
        )
        verdict = make_verdict(vetoed=["BBB"], per_position_cap={"AAA": 0.08})
        target = CONSTRUCTOR.construct(
            inputs, verdict, make_state({"BBB": 0.05}), CFG
        )
        assert target.cash >= verdict.cash_floor
        assert "BBB" not in _by_symbol(target)  # vetoed holding exited
