"""Quality separation, excursion/benchmark arithmetic, and the experience
builder (plan Phase 1). All pure — fixtures in, DecisionExperience out."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import (
    Assessment,
    CalibrationCheck,
    DecisionQuality,
    FailureClass,
    ForecastQuality,
    OutcomeQuality,
    PostTradeReview,
)
from dynamo.domain.models.portfolio import DecisionContext
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    QuantAlignment,
    QuantScore,
    Rebuttal,
    Sleeve,
)
from dynamo.domain.models.risk import Regime, RegimeState
from dynamo.learning.experience import (
    PathInputs,
    TradeFacts,
    benchmark_excess_pct,
    build_experience,
    is_wrong_but_won,
    mae_mfe_pct,
    quant_vetoed,
    score_decision_quality,
    score_forecast_quality,
    score_outcome_quality,
)

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
OPENED = datetime(2026, 7, 1, 14, 0, tzinfo=UTC)
CLOSED = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)


def _verdict(**overrides: object) -> CommitteeVerdict:
    base: dict[str, object] = {
        "id": "v-1",
        "symbol": "AAPL",
        "as_of": date(2026, 6, 30),
        "action": Action.BUY,
        "conviction": 0.7,
        "thesis": "t",
        "bull_case": "b",
        "bear_case": "r",
        "bear_rebuttals": (Rebuttal(kill_case="k", rebuttal="answered"),),
        "risks": ("competition",),
        "expected_upside_pct": 10.0,
        "expected_downside_pct": 5.0,
        "horizon_days": 20,
        "quant_alignment": QuantAlignment.AGREE,
    }
    base.update(overrides)
    return CommitteeVerdict.model_validate(base)


def _dc(**overrides: object) -> DecisionContext:
    base: dict[str, object] = {
        "id": "dc-1",
        "created_at": datetime(2026, 6, 30, 23, 0, tzinfo=UTC),
        "env": "PAPER",
        "symbol": "AAPL",
        "sleeve": Sleeve.LONG_TERM,
        "verdict_id": "v-1",
        "target_portfolio_id": "tp-1",
        "payload": {},
        "expected_upside_pct": 10.0,
        "expected_downside_pct": 5.0,
        "horizon_days": 20,
    }
    base.update(overrides)
    return DecisionContext.model_validate(base)


def _trade(pnl_pct: float = 4.0, entry: float = 100.0) -> TradeFacts:
    exit_price = entry * (1.0 + pnl_pct / 100.0)
    return TradeFacts(
        trade_id="AAPL-2026-07-10",
        symbol="AAPL",
        qty=10.0,
        opened_at=OPENED,
        closed_at=CLOSED,
        entry_price=entry,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        holding_days=9,
    )


def _review(pnl_pct: float) -> PostTradeReview:
    return PostTradeReview(
        trade_id="AAPL-2026-07-10",
        decision_context_id="dc-1",
        symbol="AAPL",
        as_of=date(2026, 7, 11),
        prompt_version="v1",
        model_id="m",
        outcome_pnl_pct=pnl_pct,
        holding_days=9,
        confidence_calibration=CalibrationCheck(conviction_was_justified=True, comment=""),
        sizing_quality=Assessment.RIGHT,
        timing_quality=Assessment.ON_TIME,
        thesis_remained_valid=True,
        exit_honored_exit_conditions=True,
        should_have_exited="as_done",
        failure_primary=FailureClass.NORMAL_UNCERTAINTY if pnl_pct < 0 else None,
    )


# --- quality separation: three independent axes ------------------------------


def test_good_decision_bad_outcome_is_normal_uncertainty() -> None:
    # Sound decision, lost 3% with 5% expected downside → variance, not error.
    decision = score_decision_quality(_verdict())
    assert decision == DecisionQuality.SOUND
    outcome = score_outcome_quality(-3.0, decision, expected_downside_pct=5.0)
    assert outcome == OutcomeQuality.NORMAL_UNCERTAINTY


def test_sound_decision_losing_beyond_expected_downside_is_a_loss() -> None:
    decision = score_decision_quality(_verdict())
    assert score_outcome_quality(-9.0, decision, 5.0) == OutcomeQuality.LOSS


def test_wrong_but_won_is_flagged() -> None:
    # Entry against the quant conflict — unsound — that happened to win.
    verdict = _verdict(quant_alignment=QuantAlignment.CONFLICT)
    decision = score_decision_quality(verdict)
    assert decision == DecisionQuality.UNSOUND
    outcome = score_outcome_quality(6.0, decision, 5.0)
    assert outcome == OutcomeQuality.WIN
    assert is_wrong_but_won(decision, outcome) is True
    # The sound winner is NOT flagged.
    assert is_wrong_but_won(DecisionQuality.SOUND, OutcomeQuality.WIN) is False


def test_unsound_loss_never_gets_normal_uncertainty() -> None:
    verdict = _verdict(quant_alignment=QuantAlignment.CONFLICT)
    decision = score_decision_quality(verdict)
    assert score_outcome_quality(-3.0, decision, 5.0) == OutcomeQuality.LOSS


def test_unrebutted_risks_on_entry_are_unsound() -> None:
    verdict = _verdict(bear_rebuttals=(), risks=("thesis risk",))
    assert score_decision_quality(verdict) == DecisionQuality.UNSOUND


def test_abstention_and_non_entries_are_sound() -> None:
    for action in (Action.NO_ACTION, Action.AVOID, Action.HOLD, Action.TRIM, Action.SELL):
        # Even under a quant conflict: not entering IS consistent with it.
        verdict = _verdict(action=action, quant_alignment=QuantAlignment.CONFLICT)
        assert score_decision_quality(verdict) == DecisionQuality.SOUND


def test_missing_verdict_is_unknown_quality() -> None:
    assert score_decision_quality(None) == DecisionQuality.UNKNOWN
    assert quant_vetoed(None) is None
    assert score_outcome_quality(None, DecisionQuality.SOUND, 5.0) == OutcomeQuality.UNKNOWN


def test_forecast_quality_independent_of_decision_quality() -> None:
    # Confident forecast + target achieved → accurate, regardless of rubric.
    assert score_forecast_quality(0.8, True) == ForecastQuality.ACCURATE
    assert score_forecast_quality(0.8, False) == ForecastQuality.INACCURATE
    # Unconfident forecast that missed → correctly unconfident.
    assert score_forecast_quality(0.3, False) == ForecastQuality.ACCURATE
    assert score_forecast_quality(0.3, True) == ForecastQuality.INACCURATE
    assert score_forecast_quality(None, True) == ForecastQuality.UNKNOWN
    assert score_forecast_quality(0.8, None) == ForecastQuality.UNKNOWN


# --- MAE/MFE + benchmark excess arithmetic -----------------------------------


def test_mae_mfe_arithmetic() -> None:
    lows = (99.0, 94.0, 97.0, 80.0)  # post-exit collapse must NOT count
    highs = (101.0, 106.0, 103.0, 120.0)
    mae, mfe = mae_mfe_pct(100.0, lows, highs, exit_session_index=2)
    assert mae == pytest.approx(-6.0)  # 94/100 - 1
    assert mfe == pytest.approx(6.0)  # 106/100 - 1


def test_mae_mfe_insufficient_data() -> None:
    assert mae_mfe_pct(0.0, (99.0,), (101.0,), 0) == (None, None)
    assert mae_mfe_pct(100.0, (), (), 0) == (None, None)


def test_benchmark_excess_arithmetic() -> None:
    # Trade made +4% while SPY went 400 → 410 (+2.5%): excess = +1.5%.
    assert benchmark_excess_pct(4.0, (400.0, 405.0, 410.0)) == pytest.approx(1.5)
    # Underperformance is negative.
    assert benchmark_excess_pct(1.0, (400.0, 420.0)) == pytest.approx(-4.0)


def test_benchmark_excess_insufficient_data() -> None:
    assert benchmark_excess_pct(4.0, (400.0,)) is None
    assert benchmark_excess_pct(None, (400.0, 410.0)) is None
    assert benchmark_excess_pct(4.0, ()) is None


# --- the builder --------------------------------------------------------------


def _path() -> PathInputs:
    return PathInputs(
        opens=(100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0),
        highs=(101.0, 103.0, 105.0, 106.0, 107.0, 108.0, 106.0),
        lows=(99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0),
        closes=(100.5, 102.0, 104.0, 105.0, 106.0, 104.0, 105.0),
        exit_session_index=5,
        benchmark_closes=(400.0, 401.0, 402.0, 403.0, 404.0, 404.0),
    )


def test_builder_produces_fully_populated_row() -> None:
    trade = _trade(pnl_pct=4.0)
    exp = build_experience(
        trade=trade,
        now=NOW,
        decision_context=_dc(),
        verdict=_verdict(),
        quant=QuantScore(
            symbol="AAPL",
            as_of=date(2026, 6, 30),
            model_version="baseline-v0",
            expected_excess_return=0.02,
            downside_prob=0.2,
            vol_forecast=0.25,
            confidence=0.6,
        ),
        regime=RegimeState(as_of=date(2026, 6, 30), regime=Regime.NEUTRAL, composite_score=0.1),
        review=_review(4.0),
        path=_path(),
        sector="Technology",
        vix=17.5,
    )
    # identity
    assert exp.id == "exp:live:dc-1"
    assert exp.decision_context_id == "dc-1"
    assert exp.trade_id == "AAPL-2026-07-10"
    assert exp.source == "live" and exp.replay_run_id is None
    assert exp.symbol == "AAPL" and exp.as_of == date(2026, 6, 30)
    assert exp.sleeve == "long_term"
    # context
    assert exp.regime == "neutral" and exp.sector == "Technology" and exp.vix == 17.5
    assert exp.quant_expected_excess_return == 0.02
    assert exp.quant_downside_prob == 0.2
    assert exp.quant_vol_forecast == 0.25
    assert exp.quant_confidence == 0.6
    # committee
    assert exp.action == "buy" and exp.conviction == 0.7
    assert exp.quant_alignment == "agree"
    assert exp.expected_upside_pct == 10.0 and exp.expected_downside_pct == 5.0
    assert exp.horizon_days == 20
    # execution
    assert exp.entry_at == OPENED and exp.exit_at == CLOSED
    assert exp.entry_price == 100.0 and exp.qty == 10.0 and exp.holding_days == 9
    # outcome
    assert exp.resolved is True
    assert exp.pnl_pct == 4.0
    assert exp.benchmark_excess_pct == pytest.approx(3.0)  # 4.0 - (404/400 - 1)*100
    assert exp.mae_pct == pytest.approx(-1.0)  # low 99 vs entry 100
    assert exp.mfe_pct == pytest.approx(8.0)  # high 108 vs entry 100
    # cum returns hit +5% (half of 10% upside) at closes[3]=105 before -5%.
    assert exp.target_achieved is True
    assert exp.forecast_quality == ForecastQuality.ACCURATE  # confident + achieved
    assert exp.decision_quality == DecisionQuality.SOUND
    assert exp.outcome_quality == OutcomeQuality.WIN
    # counterfactuals
    assert exp.cf_hold_pnl_pct == 0.0
    assert exp.cf_half_size_pnl_pct == 2.0
    assert exp.cf_delayed_entry_pnl_pct == pytest.approx((104.0 / 101.0 - 1.0) * 100.0)
    assert exp.cf_hold_to_horizon_pnl_pct is None  # horizon 20 beyond the path
    assert exp.cf_obey_quant_veto_pnl_pct == 4.0  # no veto → realized
    assert exp.ev_per_decision == 4.0
    # review / label
    assert exp.failure_primary is None
    assert exp.label == "win"


def test_builder_handles_missing_verdict_and_review_gracefully() -> None:
    # Pre-repair context: hardcoded zero expectations, no verdict, no review.
    exp = build_experience(
        trade=_trade(pnl_pct=-2.0),
        now=NOW,
        decision_context=_dc(
            expected_upside_pct=0.0, expected_downside_pct=0.0, verdict_id=None
        ),
        verdict=None,
        quant=None,
        regime=None,
        review=None,
        path=None,
        sector=None,
        vix=None,
    )
    assert exp.action is None and exp.conviction is None
    assert exp.expected_upside_pct is None and exp.expected_downside_pct is None
    assert exp.horizon_days == 20  # context horizon still usable
    assert exp.target_achieved is None
    assert exp.mae_pct is None and exp.benchmark_excess_pct is None
    assert exp.forecast_quality == ForecastQuality.UNKNOWN
    assert exp.decision_quality == DecisionQuality.UNKNOWN
    assert exp.outcome_quality == OutcomeQuality.LOSS  # unknown decision ≠ sound
    assert exp.cf_obey_quant_veto_pnl_pct is None  # veto status unknowable
    assert exp.cf_half_size_pnl_pct == -1.0
    assert exp.failure_primary is None
    assert exp.label == "loss"


def test_builder_without_decision_context_uses_verdict_expectations() -> None:
    exp = build_experience(
        trade=_trade(pnl_pct=-6.0),
        now=NOW,
        decision_context=None,
        verdict=_verdict(),
        review=_review(-6.0),
        path=_path(),
    )
    assert exp.id == "exp:live:AAPL-2026-07-10"
    assert exp.decision_context_id is None
    assert exp.as_of == date(2026, 6, 30)  # verdict as_of
    assert exp.expected_upside_pct == 10.0  # verdict fallback
    assert exp.failure_primary == FailureClass.NORMAL_UNCERTAINTY
    assert exp.outcome_quality == OutcomeQuality.LOSS  # -6% beyond the 5% downside


def test_builder_flags_veto_override_in_counterfactual() -> None:
    exp = build_experience(
        trade=_trade(pnl_pct=7.0),
        now=NOW,
        decision_context=_dc(),
        verdict=_verdict(quant_alignment=QuantAlignment.CONFLICT),
        path=_path(),
    )
    assert exp.cf_obey_quant_veto_pnl_pct == 0.0  # obeying the veto = no trade
    assert exp.decision_quality == DecisionQuality.UNSOUND
    assert exp.outcome_quality == OutcomeQuality.WIN  # wrong but won
