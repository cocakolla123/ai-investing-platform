"""Portfolio & execution domain entities. Design: docs/design/07, 09."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import Field

from dynamo.domain.models.market import Frozen
from dynamo.domain.models.research import Sleeve


class Lot(Frozen):
    """Tax/attribution lot; tags immutable for life of the lot (doc 07 section 7)."""

    symbol: str
    sleeve: Sleeve
    strategy_id: str | None = None
    verdict_id: str | None = None
    qty: float
    entry_price: float
    entry_date: date


class Position(Frozen):
    symbol: str
    qty: float
    avg_entry_price: float
    lots: tuple[Lot, ...] = ()
    current_stop: float | None = None
    stop_kind: str | None = None


class PortfolioState(Frozen):
    as_of: datetime
    equity: float
    cash: float
    positions: tuple[Position, ...] = ()
    high_water_mark: float | None = None

    @property
    def weights(self) -> dict[str, float]:
        if self.equity <= 0:
            return {}
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.symbol] = out.get(p.symbol, 0.0) + p.qty * p.avg_entry_price / self.equity
        return out


class RebalanceTrigger(StrEnum):
    DRIFT = "drift"
    VERDICT_CHANGE = "verdict_change"
    RISK_BREACH = "risk_breach"
    REGIME_CHANGE = "regime_change"
    SCHEDULED = "scheduled"
    VOL_TARGET = "vol_target"


class TargetWeight(Frozen):
    symbol: str
    weight: float = Field(ge=0.0)
    sleeve: Sleeve
    strategy_id: str | None = None
    verdict_id: str | None = None
    rationale: str = ""


class ProposedChange(Frozen):
    symbol: str
    current_weight: float
    target_weight: float
    source: str
    trigger: RebalanceTrigger
    rationale: str = ""
    constraints_applied: tuple[str, ...] = ()


class TargetPortfolio(Frozen):
    """Deterministic output of portfolio construction (doc 07 section 6)."""

    id: str
    as_of: datetime
    risk_verdict_ref: str
    weights: tuple[TargetWeight, ...]
    cash: float
    cash_attribution: dict[str, float] = {}  # floor/vol_target/regime/scarcity
    changes: tuple[ProposedChange, ...] = ()
    turnover_used_mtd: float = 0.0


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET_ON_OPEN = "market_on_open"
    MARKETABLE_LIMIT = "marketable_limit"
    STOP = "stop"
    SYNTHETIC_STOP_EXIT = "synthetic_stop_exit"


class OrderState(StrEnum):
    PROPOSED = "proposed"
    RISK_CHECKED = "risk_checked"
    PENDING_APPROVAL = "pending_approval"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


TERMINAL_ORDER_STATES = frozenset(
    {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}
)

# Legal transitions of the order state machine (doc 09 section 2).
ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PROPOSED: frozenset({OrderState.RISK_CHECKED, OrderState.REJECTED}),
    OrderState.RISK_CHECKED: frozenset({OrderState.PENDING_APPROVAL}),
    OrderState.PENDING_APPROVAL: frozenset({OrderState.SUBMITTED, OrderState.CANCELLED}),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


class Order(Frozen):
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "day"
    state: OrderState = OrderState.PROPOSED
    sleeve: Sleeve | None = None
    decision_context_id: str | None = None  # REQUIRED before submission
    broker_order_id: str | None = None
    created_at: datetime | None = None


class Trade(Frozen):
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: float
    fill_price: float
    decision_price: float | None = None
    slippage_bps: float | None = None
    fees: float = 0.0
    filled_at: datetime


class DecisionContext(Frozen):
    """Immutable pre-order snapshot (doc 09 section 6, doc 10 section 2).

    No order without a snapshot — the order manager requires the id.
    """

    id: str
    created_at: datetime
    env: str
    symbol: str
    sleeve: Sleeve
    verdict_id: str | None
    target_portfolio_id: str
    payload: dict[str, object]  # full snapshot: scores, risk verdict, prices, versions
    expected_upside_pct: float
    expected_downside_pct: float
    horizon_days: int
