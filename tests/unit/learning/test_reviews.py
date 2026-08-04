"""Post-trade review rules (doc 10 §3-4): no-hindsight allowlist, failure
classification requirement, completeness deadline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from dynamo.domain.models.learning import FailureClass
from dynamo.domain.models.portfolio import DecisionContext
from dynamo.domain.models.research import Sleeve
from dynamo.learning.reviews import (
    REVIEW_CONTEXT_ALLOWED_KEYS,
    HindsightLeak,
    ReviewValidationError,
    TradeOutcome,
    assert_no_hindsight,
    build_review_request,
    completeness_check,
    parse_review,
)

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

DC = DecisionContext(
    id="dc-1",
    created_at=NOW - timedelta(days=30),
    env="PAPER",
    symbol="AAPL",
    sleeve=Sleeve.LONG_TERM,
    verdict_id="v-1",
    target_portfolio_id="tp-1",
    payload={"prices": {"AAPL": 100.0}, "quant_scores": {"rank": 3}},
    expected_upside_pct=12.0,
    expected_downside_pct=6.0,
    horizon_days=60,
)


def _outcome(pnl: float = -4.2) -> TradeOutcome:
    return TradeOutcome(
        trade_id="t-1", symbol="AAPL", pnl_pct=pnl, holding_days=28,
        entry_price=100.0, exit_price=100.0 * (1 + pnl / 100), closed_at=NOW,
    )


def _review_payload(pnl: float, failure: FailureClass | None) -> dict[str, Any]:
    return {
        "trade_id": "t-1",
        "decision_context_id": "dc-1",
        "symbol": "AAPL",
        "as_of": date(2026, 8, 3),
        "prompt_version": "review-v1",
        "model_id": "claude-opus-4-6",
        "outcome_pnl_pct": pnl,
        "holding_days": 28,
        "confidence_calibration": {"conviction_was_justified": True, "comment": "ok"},
        "sizing_quality": "right",
        "timing_quality": "on_time",
        "thesis_remained_valid": True,
        "exit_honored_exit_conditions": True,
        "should_have_exited": "as_done",
        "failure_primary": failure.value if failure else None,
    }


def test_review_request_contains_only_snapshot_and_outcome() -> None:
    context = build_review_request(DC, _outcome())
    assert set(context) == REVIEW_CONTEXT_ALLOWED_KEYS == {"decision_context", "outcome"}
    assert context["decision_context"]["payload"]["prices"] == {"AAPL": 100.0}
    assert context["outcome"]["pnl_pct"] == -4.2


def test_hindsight_leak_extra_key_rejected() -> None:
    context = build_review_request(DC, _outcome())
    context["news_after_decision"] = ["surprise guidance cut"]  # hindsight!
    with pytest.raises(HindsightLeak, match="news_after_decision"):
        assert_no_hindsight(context)


def test_hindsight_leak_inside_outcome_rejected() -> None:
    context = build_review_request(DC, _outcome())
    outcome = dict(context["outcome"])
    outcome["price_path_after_exit"] = [1, 2, 3]
    context["outcome"] = outcome
    with pytest.raises(HindsightLeak, match="price_path_after_exit"):
        assert_no_hindsight(context)


def test_missing_required_key_rejected() -> None:
    with pytest.raises(HindsightLeak, match="missing"):
        assert_no_hindsight({"decision_context": {}})


def test_losing_trade_requires_failure_class() -> None:
    with pytest.raises(ReviewValidationError, match="failure classification"):
        parse_review(_review_payload(-4.2, None), _outcome(-4.2))


def test_losing_trade_with_failure_class_ok() -> None:
    review = parse_review(
        _review_payload(-4.2, FailureClass.NORMAL_UNCERTAINTY), _outcome(-4.2)
    )
    assert review.failure_primary == FailureClass.NORMAL_UNCERTAINTY


def test_winning_trade_must_not_have_failure_class() -> None:
    with pytest.raises(ReviewValidationError, match="losing trades only"):
        parse_review(_review_payload(3.0, FailureClass.QUANT_MODEL), _outcome(3.0))
    review = parse_review(_review_payload(3.0, None), _outcome(3.0))
    assert review.failure_primary is None


def test_secondary_without_primary_rejected() -> None:
    payload = _review_payload(3.0, None)
    payload["failure_secondary"] = FailureClass.POSITION_SIZING.value
    with pytest.raises(ReviewValidationError, match="failure_secondary"):
        parse_review(payload, _outcome(3.0))


def test_cross_wired_identity_rejected() -> None:
    payload = _review_payload(-1.0, FailureClass.NORMAL_UNCERTAINTY)
    payload["trade_id"] = "t-999"
    with pytest.raises(ReviewValidationError, match="trade_id"):
        parse_review(payload, _outcome(-1.0))
    payload = _review_payload(-1.0, FailureClass.NORMAL_UNCERTAINTY)
    payload["symbol"] = "TSLA"
    with pytest.raises(ReviewValidationError, match="symbol"):
        parse_review(payload, _outcome(-1.0))


def test_completeness_check_flags_only_overdue_unreviewed() -> None:
    closed = [
        ("t-old-unreviewed", NOW - timedelta(hours=30)),
        ("t-old-reviewed", NOW - timedelta(hours=30)),
        ("t-fresh", NOW - timedelta(hours=2)),
    ]
    overdue = completeness_check(closed, frozenset({"t-old-reviewed"}), NOW)
    assert overdue == ["t-old-unreviewed"]


def test_completeness_boundary_exactly_24h_not_overdue() -> None:
    closed = [("t-1", NOW - timedelta(hours=24))]
    assert completeness_check(closed, frozenset(), NOW) == []
    assert completeness_check(closed, frozenset(), NOW + timedelta(seconds=1)) == ["t-1"]
