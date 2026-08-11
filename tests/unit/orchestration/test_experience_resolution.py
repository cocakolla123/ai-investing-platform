"""experience_resolution spine on in-memory fakes (plan Phase 1): a complete
fixture spine builds + inserts a row; broken joins skip; a bad trade logs and
continues instead of raising."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from orchestration.assets.learning_ops import build_experiences

from dynamo.domain.models.learning import (
    Assessment,
    CalibrationCheck,
    DecisionExperience,
    PostTradeReview,
)
from dynamo.domain.models.portfolio import DecisionContext
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    QuantAlignment,
    Rebuttal,
    Sleeve,
)
from dynamo.learning.experience import PathInputs

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
LOG = logging.getLogger("test_experience_resolution")


@dataclass(frozen=True)
class _Closed:
    symbol: str
    closed_at: datetime
    opened_at: datetime = NOW - timedelta(days=9)
    qty: float = 10.0
    entry_avg_price: float = 100.0
    exit_avg_price: float = 104.0
    pnl_pct: float = 4.0
    holding_days: int = 9


def _dc(dc_id: str = "dc-1", verdict_id: str | None = "v-1") -> DecisionContext:
    return DecisionContext(
        id=dc_id,
        created_at=NOW - timedelta(days=10),
        env="PAPER",
        symbol="AAPL",
        sleeve=Sleeve.LONG_TERM,
        verdict_id=verdict_id,
        target_portfolio_id="tp-1",
        payload={},
        expected_upside_pct=10.0,
        expected_downside_pct=5.0,
        horizon_days=20,
    )


def _verdict() -> CommitteeVerdict:
    return CommitteeVerdict(
        id="v-1",
        symbol="AAPL",
        as_of=date(2026, 7, 31),
        action=Action.BUY,
        conviction=0.7,
        thesis="t",
        bull_case="b",
        bear_case="r",
        bear_rebuttals=(Rebuttal(kill_case="k", rebuttal="a"),),
        risks=("risk",),
        expected_upside_pct=10.0,
        expected_downside_pct=5.0,
        horizon_days=20,
        quant_alignment=QuantAlignment.AGREE,
    )


def _review(trade_id: str) -> PostTradeReview:
    return PostTradeReview(
        trade_id=trade_id,
        decision_context_id="dc-1",
        symbol="AAPL",
        as_of=date(2026, 8, 10),
        prompt_version="v1",
        model_id="m",
        outcome_pnl_pct=4.0,
        holding_days=9,
        confidence_calibration=CalibrationCheck(conviction_was_justified=True, comment=""),
        sizing_quality=Assessment.RIGHT,
        timing_quality=Assessment.ON_TIME,
        thesis_remained_valid=True,
        exit_honored_exit_conditions=True,
        should_have_exited="as_done",
    )


def _path() -> PathInputs:
    return PathInputs(
        opens=(100.0, 101.0, 102.0),
        highs=(101.0, 103.0, 105.0),
        lows=(99.0, 100.0, 101.0),
        closes=(100.5, 102.0, 104.0),
        exit_session_index=2,
        benchmark_closes=(400.0, 402.0, 404.0),
    )


@dataclass
class _Sink:
    rows: list[DecisionExperience] = field(default_factory=list)

    def insert(self, exp: DecisionExperience) -> bool:
        self.rows.append(exp)
        return True


def _run(
    *,
    closed: list[_Closed],
    review_for: Any = None,
    dc_for: Any = None,
    verdict_for: Any = None,
    existing: frozenset[str] = frozenset(),
    sink: _Sink | None = None,
) -> tuple[dict[str, int], _Sink]:
    sink = sink or _Sink()
    counts = build_experiences(
        closed_trades=closed,
        review_for=review_for or (lambda tid: _review(tid)),
        decision_context_for=dc_for or (lambda c: _dc()),
        verdict_for=verdict_for or (lambda vid: _verdict()),
        quant_for=lambda symbol, as_of: None,
        regime_for=lambda as_of: None,
        vix_for=lambda as_of: 18.0,
        sector_for=lambda symbol: "Technology",
        path_for=lambda c, horizon: _path(),
        existing_dc_ids=existing,
        insert=sink.insert,
        now=NOW,
        log=LOG,
    )
    return counts, sink


def test_fixture_spine_inserts_experience_row() -> None:
    counts, sink = _run(closed=[_Closed("AAPL", NOW)])
    assert counts == {"built": 1, "skipped": 0, "existing": 0, "failed": 0}
    [exp] = sink.rows
    assert exp.source == "live"
    assert exp.decision_context_id == "dc-1"
    assert exp.trade_id == "AAPL-2026-08-10"
    assert exp.resolved is True
    assert exp.pnl_pct == 4.0
    assert exp.action == "buy" and exp.sector == "Technology" and exp.vix == 18.0


def test_unreviewed_trade_is_skipped() -> None:
    counts, sink = _run(closed=[_Closed("AAPL", NOW)], review_for=lambda tid: None)
    assert counts["skipped"] == 1 and counts["built"] == 0
    assert sink.rows == []


def test_broken_decision_context_join_skips_not_raises() -> None:
    counts, sink = _run(closed=[_Closed("AAPL", NOW)], dc_for=lambda c: None)
    assert counts == {"built": 0, "skipped": 1, "existing": 0, "failed": 0}
    assert sink.rows == []


def test_existing_experience_not_rebuilt() -> None:
    counts, sink = _run(closed=[_Closed("AAPL", NOW)], existing=frozenset({"dc-1"}))
    assert counts["existing"] == 1 and counts["built"] == 0
    assert sink.rows == []


def test_one_bad_trade_logs_and_continues() -> None:
    def exploding_verdict_for(vid: str | None) -> CommitteeVerdict:
        raise RuntimeError("boom")

    good = _Closed("MSFT", NOW - timedelta(days=1))
    bad = _Closed("AAPL", NOW)

    calls: list[str] = []

    def verdict_for(vid: str | None) -> CommitteeVerdict:
        # First trade explodes mid-join; the second must still be built.
        calls.append(vid or "")
        if len(calls) == 1:
            raise RuntimeError("boom")
        return _verdict()

    counts, sink = _run(closed=[bad, good], verdict_for=verdict_for)
    assert counts["failed"] == 1
    assert counts["built"] == 1
    assert [e.symbol for e in sink.rows] == ["MSFT"]


def test_missing_verdict_still_builds_row() -> None:
    counts, sink = _run(closed=[_Closed("AAPL", NOW)], verdict_for=lambda vid: None)
    assert counts["built"] == 1
    [exp] = sink.rows
    assert exp.action is None and exp.conviction is None
    assert exp.expected_upside_pct == 10.0  # frozen-context expectations survive
