"""Post-trade reviews: no-hindsight context assembly + validation (doc 10 §3-4).

The review prompt receives ONLY the frozen ``DecisionContext`` plus the
outcome — enforced here with an allowlist, and asserted by the no-hindsight
test. ``failure_primary`` is required iff the trade lost money, and must be
absent for winners (classes exist for losing trades only).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from dynamo.domain.models.learning import PostTradeReview
from dynamo.domain.models.portfolio import DecisionContext

REVIEW_OVERDUE = timedelta(hours=24)

# The ONLY top-level keys a review request may carry (no-hindsight allowlist).
REVIEW_CONTEXT_ALLOWED_KEYS = frozenset({"decision_context", "outcome"})

OUTCOME_ALLOWED_KEYS = frozenset(
    {
        "trade_id",
        "symbol",
        "pnl_pct",
        "holding_days",
        "entry_price",
        "exit_price",
        "slippage_bps",
        "closed_at",
    }
)


class HindsightLeak(ValueError):
    """The review context contains data beyond the frozen snapshot + outcome."""


class ReviewValidationError(ValueError):
    """The review payload violates a structural rule (e.g. failure class)."""


@dataclass(frozen=True)
class TradeOutcome:
    trade_id: str
    symbol: str
    pnl_pct: float
    holding_days: int
    entry_price: float | None = None
    exit_price: float | None = None
    slippage_bps: float | None = None
    closed_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "slippage_bps": self.slippage_bps,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }


def assert_no_hindsight(context: Mapping[str, Any]) -> None:
    """Raise ``HindsightLeak`` unless the context is exactly snapshot + outcome."""
    extra = set(context) - REVIEW_CONTEXT_ALLOWED_KEYS
    if extra:
        raise HindsightLeak(f"review context carries non-snapshot keys: {sorted(extra)}")
    missing = REVIEW_CONTEXT_ALLOWED_KEYS - set(context)
    if missing:
        raise HindsightLeak(f"review context missing required keys: {sorted(missing)}")
    outcome = context["outcome"]
    if isinstance(outcome, Mapping):
        bad = set(outcome) - OUTCOME_ALLOWED_KEYS
        if bad:
            raise HindsightLeak(f"outcome carries unexpected keys: {sorted(bad)}")


def build_review_request(
    decision_context: DecisionContext, outcome: TradeOutcome
) -> dict[str, Any]:
    """Assemble the review prompt context: the frozen snapshot + the outcome,
    and nothing else. 'What was knowable' is judged strictly against the
    snapshot (doc 10 §3)."""
    context: dict[str, Any] = {
        "decision_context": decision_context.model_dump(mode="json"),
        "outcome": outcome.as_dict(),
    }
    assert_no_hindsight(context)
    return context


def parse_review(payload: Mapping[str, Any], outcome: TradeOutcome) -> PostTradeReview:
    """Validate a structured review against the outcome it reviews.

    - ``failure_primary`` REQUIRED iff ``pnl < 0`` (doc 10 §4); winners must
      not carry a failure class.
    - trade/symbol identity must match the outcome (no cross-wiring).
    """
    review = PostTradeReview.model_validate(payload)
    if review.trade_id != outcome.trade_id:
        raise ReviewValidationError(
            f"review trade_id {review.trade_id!r} != outcome {outcome.trade_id!r}"
        )
    if review.symbol != outcome.symbol:
        raise ReviewValidationError(
            f"review symbol {review.symbol!r} != outcome {outcome.symbol!r}"
        )
    if review.outcome_pnl_pct < 0 and review.failure_primary is None:
        raise ReviewValidationError(
            "losing trade requires a primary failure classification (doc 10 §4)"
        )
    if review.outcome_pnl_pct >= 0 and review.failure_primary is not None:
        raise ReviewValidationError(
            "failure classes apply to losing trades only; remove failure_primary"
        )
    if review.failure_primary is None and review.failure_secondary is not None:
        raise ReviewValidationError("failure_secondary requires failure_primary")
    return review


def completeness_check(
    closed_trades: Iterable[tuple[str, datetime]],  # (trade_id, closed_at)
    reviewed_trade_ids: Sequence[str] | frozenset[str],
    now: datetime,
) -> list[str]:
    """Trade closed > 24h ago with no review → overdue (doc 00 §4 process
    metric; enforced in production by a sensor that alerts on breach)."""
    reviewed = set(reviewed_trade_ids)
    return [
        trade_id
        for trade_id, closed_at in closed_trades
        if trade_id not in reviewed and now - closed_at > REVIEW_OVERDUE
    ]
