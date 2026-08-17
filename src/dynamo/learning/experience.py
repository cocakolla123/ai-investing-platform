"""DecisionExperience builder + quality scorers (plan Phase 1).

Pure: everything is computed from inputs PASSED IN (domain models, plain
dataclass facts, bar paths). The composition layer (orchestration asset /
backfill CLI) does all the joins and hands the pieces to
:func:`build_experience`; import-linter keeps this module off persistence.

Quality separation — three independent axes, never conflated:

- **forecast quality** (:func:`score_forecast_quality`): was the stated
  conviction right about THIS path? Reuses the doc 10 §5 path definition via
  :func:`~dynamo.learning.calibration.trade_outcome_achieved`.
- **decision quality** (:func:`score_decision_quality`): was the decision
  defensible given only what was knowable then? Deterministic rubric,
  outcome-blind by construction.
- **outcome quality** (:func:`score_outcome_quality`): the realized label,
  preserving ``normal_uncertainty`` — a sound decision that lost within its
  expected downside is NOT a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from dynamo.domain.models.learning import (
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    OutcomeQuality,
    PostTradeReview,
)
from dynamo.domain.models.research import Action, QuantAlignment
from dynamo.learning import counterfactuals as cf
from dynamo.learning.calibration import trade_outcome_achieved

if TYPE_CHECKING:
    from dynamo.domain.models.portfolio import DecisionContext
    from dynamo.domain.models.research import CommitteeVerdict, QuantScore
    from dynamo.domain.models.risk import RegimeState

ENTRY_ACTIONS = frozenset({Action.BUY, Action.ADD})
CONFIDENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TradeFacts:
    """The realized round trip, decoupled from any persistence type."""

    trade_id: str
    symbol: str
    qty: float
    opened_at: datetime
    closed_at: datetime
    entry_price: float
    exit_price: float
    pnl_pct: float
    holding_days: int


@dataclass(frozen=True)
class PathInputs:
    """Session-ordered bars from the entry session onward (index 0 = entry).

    ``benchmark_closes`` are the benchmark's (SPY) closes over the holding
    window only: first element = entry session, last element = exit session.
    """

    opens: tuple[float, ...]
    highs: tuple[float, ...]
    lows: tuple[float, ...]
    closes: tuple[float, ...]
    exit_session_index: int
    benchmark_closes: tuple[float, ...] = ()


def mae_mfe_pct(
    entry_price: float,
    lows: tuple[float, ...] | list[float],
    highs: tuple[float, ...] | list[float],
    exit_session_index: int,
) -> tuple[float | None, float | None]:
    """Max adverse / favorable excursion vs entry over the holding window.

    MAE = worst low vs entry (<= 0 for longs unless gapped up), MFE = best
    high vs entry, both as percentages; ``(None, None)`` when the window is
    empty or the entry price is degenerate.
    """
    if entry_price <= 0 or exit_session_index < 0:
        return None, None
    window_lows = [x for x in lows[: exit_session_index + 1] if x > 0]
    window_highs = [x for x in highs[: exit_session_index + 1] if x > 0]
    if not window_lows or not window_highs:
        return None, None
    mae = (min(window_lows) / entry_price - 1.0) * 100.0
    mfe = (max(window_highs) / entry_price - 1.0) * 100.0
    return mae, mfe


def benchmark_excess_pct(
    pnl_pct: float | None,
    benchmark_closes: tuple[float, ...] | list[float],
) -> float | None:
    """Realized pnl minus the benchmark's return over the same window.

    ``benchmark_closes``: first = entry session, last = exit session. Needs at
    least two positive closes; otherwise ``None``.
    """
    if pnl_pct is None or len(benchmark_closes) < 2:
        return None
    first, last = benchmark_closes[0], benchmark_closes[-1]
    if first <= 0 or last <= 0:
        return None
    return pnl_pct - (last / first - 1.0) * 100.0


def cum_returns_pct(entry_price: float, closes: tuple[float, ...] | list[float]) -> list[float]:
    """Daily cumulative returns (%) vs the actual entry price, from the first
    session AFTER entry — the path :func:`trade_outcome_achieved` walks."""
    if entry_price <= 0:
        return []
    return [(c / entry_price - 1.0) * 100.0 for c in closes[1:]]


def quant_vetoed(verdict: CommitteeVerdict | None) -> bool | None:
    """Did the committee enter AGAINST the quant view? (the 'quant veto').

    Defined as an entry action carried out under ``quant_alignment=conflict``
    — the only deterministic veto signal the platform records today. ``None``
    when there is no verdict to judge.
    """
    if verdict is None:
        return None
    return verdict.action in ENTRY_ACTIONS and verdict.quant_alignment == QuantAlignment.CONFLICT


def score_forecast_quality(
    conviction: float | None, target_achieved: bool | None
) -> ForecastQuality:
    """Per-decision forecast correctness at the 0.5 conviction threshold.

    A confident forecast (conviction >= 0.5) is accurate iff the target was
    achieved; an unconfident one is accurate iff it was not. Population-level
    calibration stays in ``learning/calibration.py`` — this is the per-row cut.
    """
    if conviction is None or target_achieved is None:
        return ForecastQuality.UNKNOWN
    confident = conviction >= CONFIDENT_THRESHOLD
    return ForecastQuality.ACCURATE if confident == target_achieved else ForecastQuality.INACCURATE


def score_decision_quality(verdict: CommitteeVerdict | None) -> DecisionQuality:
    """Outcome-blind rubric: was the decision defensible given info then?

    Deterministic, deliberately simple (plan Phase 1):

    - UNSOUND: an entry (buy/add) taken against a quant conflict — acting
      against the quant veto needs extraordinary evidence the row can't show;
    - UNSOUND: an entry with named risks and zero bear rebuttals — an
      unrebutted bear case means the committee never answered its critics;
    - SOUND: everything else — including abstention (no_action/avoid), which
      is always defensible, and non-entry actions (hold/trim/sell);
    - UNKNOWN: no verdict to judge.
    """
    if verdict is None:
        return DecisionQuality.UNKNOWN
    if verdict.action in ENTRY_ACTIONS:
        if verdict.quant_alignment == QuantAlignment.CONFLICT:
            return DecisionQuality.UNSOUND
        if verdict.risks and not verdict.bear_rebuttals:
            return DecisionQuality.UNSOUND
    return DecisionQuality.SOUND


def score_outcome_quality(
    pnl_pct: float | None,
    decision_quality: DecisionQuality,
    expected_downside_pct: float | None,
) -> OutcomeQuality:
    """Realized label with ``normal_uncertainty`` preserved.

    - WIN: pnl >= 0;
    - NORMAL_UNCERTAINTY: a SOUND decision that lost no more than its stated
      expected downside — variance, not error (doc 10 §4 failure classes);
    - LOSS: any other loss;
    - UNKNOWN: unresolved.
    """
    if pnl_pct is None:
        return OutcomeQuality.UNKNOWN
    if pnl_pct >= 0:
        return OutcomeQuality.WIN
    if (
        decision_quality == DecisionQuality.SOUND
        and expected_downside_pct is not None
        and expected_downside_pct != 0.0
        and pnl_pct >= -abs(expected_downside_pct)
    ):
        return OutcomeQuality.NORMAL_UNCERTAINTY
    return OutcomeQuality.LOSS


def is_wrong_but_won(decision: DecisionQuality, outcome: OutcomeQuality) -> bool:
    """The dangerous quadrant: a bad decision rewarded by luck. Flagged so the
    learning loop never reinforces it (quality separation's whole point)."""
    return decision == DecisionQuality.UNSOUND and outcome == OutcomeQuality.WIN


def build_experience(
    *,
    trade: TradeFacts,
    now: datetime,
    decision_context: DecisionContext | None = None,
    verdict: CommitteeVerdict | None = None,
    quant: QuantScore | None = None,
    regime: RegimeState | None = None,
    review: PostTradeReview | None = None,
    path: PathInputs | None = None,
    sector: str | None = None,
    vix: float | None = None,
    source: str = "live",
    replay_run_id: str | None = None,
    equity_at_decision: float | None = None,
    position_notional: float | None = None,
) -> DecisionExperience:
    """Assemble one resolved ``DecisionExperience`` from the joined pieces.

    Every optional input may be ``None`` (broken joins degrade per-field, the
    row is still built); ``trade`` is the only hard requirement — no outcome,
    no experience. Expectations prefer the frozen decision context (Phase 0
    denormalized real verdict numbers into it) and fall back to the verdict.
    """
    expected_upside, expected_downside, horizon = _expectations(decision_context, verdict)

    target_achieved: bool | None = None
    mae: float | None = None
    mfe: float | None = None
    bench_excess: float | None = None
    cf_delayed: float | None = None
    cf_horizon: float | None = None
    if path is not None:
        mae, mfe = mae_mfe_pct(trade.entry_price, path.lows, path.highs, path.exit_session_index)
        bench_excess = benchmark_excess_pct(trade.pnl_pct, path.benchmark_closes)
        cf_delayed = cf.delayed_entry_pnl_pct(
            path.opens, path.exit_session_index, trade.exit_price
        )
        cf_horizon = cf.hold_to_horizon_pnl_pct(trade.entry_price, path.closes, horizon)
        if (
            expected_upside is not None
            and expected_upside > 0.0
            and expected_downside is not None
            and expected_downside != 0.0
            and horizon is not None
            and horizon > 0
        ):
            returns = cum_returns_pct(trade.entry_price, path.closes)
            if returns:
                target_achieved = trade_outcome_achieved(
                    returns, expected_upside, expected_downside, horizon
                )

    vetoed = quant_vetoed(verdict)
    decision_quality = score_decision_quality(verdict)
    conviction = verdict.conviction if verdict is not None else None
    as_of = _as_of(decision_context, verdict, trade)

    return DecisionExperience(
        # identity
        id=_experience_id(decision_context, trade, source, replay_run_id),
        decision_context_id=decision_context.id if decision_context is not None else None,
        trade_id=trade.trade_id,
        source=source,
        replay_run_id=replay_run_id,
        created_at=now,
        symbol=trade.symbol,
        as_of=as_of,
        sleeve=decision_context.sleeve.value if decision_context is not None else None,
        # context
        regime=regime.regime.value if regime is not None else None,
        sector=sector,
        vix=vix,
        quant_expected_excess_return=quant.expected_excess_return if quant else None,
        quant_downside_prob=quant.downside_prob if quant else None,
        quant_vol_forecast=quant.vol_forecast if quant else None,
        quant_confidence=quant.confidence if quant else None,
        # committee
        action=verdict.action.value if verdict is not None else None,
        conviction=conviction,
        disagreement_score=verdict.disagreement_score if verdict is not None else None,
        quant_alignment=verdict.quant_alignment.value if verdict is not None else None,
        expected_upside_pct=expected_upside,
        expected_downside_pct=expected_downside,
        horizon_days=horizon,
        # decision / execution
        entry_at=trade.opened_at,
        exit_at=trade.closed_at,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        qty=trade.qty,
        holding_days=trade.holding_days,
        # bankroll context (campaign Phase 3) — None on broken joins/legacy
        equity_at_decision=equity_at_decision,
        position_notional=position_notional,
        # outcome
        resolved=True,
        pnl_pct=trade.pnl_pct,
        benchmark_excess_pct=bench_excess,
        mae_pct=mae,
        mfe_pct=mfe,
        target_achieved=target_achieved,
        forecast_quality=score_forecast_quality(conviction, target_achieved),
        decision_quality=decision_quality,
        outcome_quality=score_outcome_quality(trade.pnl_pct, decision_quality, expected_downside),
        # counterfactuals
        cf_hold_pnl_pct=cf.hold_pnl_pct(),
        cf_half_size_pnl_pct=cf.half_size_pnl_pct(trade.pnl_pct),
        cf_delayed_entry_pnl_pct=cf_delayed,
        cf_hold_to_horizon_pnl_pct=cf_horizon,
        cf_obey_quant_veto_pnl_pct=cf.obey_quant_veto_pnl_pct(vetoed, trade.pnl_pct),
        ev_per_decision=cf.ev_per_decision(trade.pnl_pct),
        # review / label
        failure_primary=review.failure_primary if review is not None else None,
        label="win" if trade.pnl_pct >= 0 else "loss",
    )


def _expectations(
    decision_context: DecisionContext | None, verdict: CommitteeVerdict | None
) -> tuple[float | None, float | None, int | None]:
    """Committee expectations: the frozen context first (denormalized, real
    since Phase 0), the verdict as fallback; ``None`` when neither carries
    non-zero numbers (pre-repair rows hardcoded zeros)."""
    for source_obj in (decision_context, verdict):
        if source_obj is None:
            continue
        if source_obj.expected_upside_pct > 0.0 and source_obj.expected_downside_pct != 0.0:
            return (
                source_obj.expected_upside_pct,
                source_obj.expected_downside_pct,
                source_obj.horizon_days,
            )
    if verdict is not None:
        return None, None, verdict.horizon_days if verdict.horizon_days > 0 else None
    if decision_context is not None and decision_context.horizon_days > 0:
        return None, None, decision_context.horizon_days
    return None, None, None


def _as_of(
    decision_context: DecisionContext | None,
    verdict: CommitteeVerdict | None,
    trade: TradeFacts,
) -> date:
    if decision_context is not None:
        return decision_context.created_at.date()
    if verdict is not None:
        return verdict.as_of
    return trade.opened_at.date()


def _experience_id(
    decision_context: DecisionContext | None,
    trade: TradeFacts,
    source: str,
    replay_run_id: str | None,
) -> str:
    """Deterministic id: re-running a resolution asset re-derives the same row
    (idempotent with the live partial unique / plain PK for replay)."""
    if replay_run_id:
        return f"exp:{replay_run_id}:{trade.trade_id}"
    if decision_context is not None:
        return f"exp:{source}:{decision_context.id}"
    return f"exp:{source}:{trade.trade_id}"
