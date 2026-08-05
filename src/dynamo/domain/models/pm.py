"""Prediction-market trading domain entities (Kalshi sleeve). Design: ADR-0009.

The sleeve reuses the equity order lifecycle (``OrderState`` +
``ORDER_TRANSITIONS`` from :mod:`dynamo.domain.models.portfolio`) so there is
exactly one state machine to reason about. Prices are integer cents (Kalshi
native); contracts are non-negative integers; every contract settles at $1
(100 cents) when it resolves in the holder's favour, $0 otherwise.

Polymarket has NO trading entities anywhere — it stays research-only.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from dynamo.domain.models.market import Frozen
from dynamo.domain.models.portfolio import OrderState

PM_VENUE_KALSHI = "kalshi"
CONTRACT_PAYOUT_CENTS = 100.0


class PMSide(StrEnum):
    YES = "yes"
    NO = "no"


class PMOrder(Frozen):
    """A limit order for binary-outcome contracts (doc 09 discipline)."""

    client_order_id: str
    venue: str = PM_VENUE_KALSHI
    market_id: str
    side: PMSide
    contracts: int = Field(ge=1)
    limit_price_cents: int = Field(ge=1, le=99)
    state: OrderState = OrderState.PROPOSED
    decision_context_id: str | None = None  # REQUIRED before submission
    broker_order_id: str | None = None
    created_at: datetime | None = None


class PMPosition(Frozen):
    """Open contracts in one market, with the question snapshotted at entry."""

    venue: str = PM_VENUE_KALSHI
    market_id: str
    side: PMSide
    contracts: int = Field(ge=0)
    avg_price_cents: float = Field(ge=0.0, le=100.0)
    question: str = ""  # market question snapshot at entry

    @property
    def exposure_dollars(self) -> float:
        return self.contracts * self.avg_price_cents / 100.0


class PMTrade(Frozen):
    """A fill (``kind='fill'``) or a resolution settlement (``kind='settlement'``)."""

    client_order_id: str | None
    venue: str = PM_VENUE_KALSHI
    market_id: str
    side: PMSide
    contracts: int = Field(ge=0)
    price_cents: float
    fees_cents: float = 0.0
    kind: str = "fill"  # fill | settlement
    realized_pnl_cents: float | None = None  # settlements only
    filled_at: datetime


class PMTradeDecision(Frozen):
    """Sized output of :func:`dynamo.risk.pm_sizing.select_and_size_pm_trades`."""

    venue: str = PM_VENUE_KALSHI
    market_id: str
    side: PMSide
    edge: float  # estimated_prob - current implied_prob (yes terms)
    estimated_prob: float = Field(ge=0.0, le=1.0)
    implied_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    kelly_fraction: float = Field(ge=0.0)  # capped fractional-Kelly stake, as equity fraction
    contracts: int = Field(ge=1)
    price_cents: int = Field(ge=1, le=99)  # entry price per contract used for sizing
    rationale: str
    view_ref: str  # "{venue}:{market_id}:{as_of}"


def settlement_pnl_cents(
    side: PMSide, contracts: int, avg_price_cents: float, resolved_yes: bool
) -> float:
    """Realized P&L in cents when a market resolves (pure; ADR-0009).

    A winning contract pays 100 cents; a losing one pays 0. P&L is payout
    minus cost basis, per contract.
    """
    won = resolved_yes if side is PMSide.YES else not resolved_yes
    payout = CONTRACT_PAYOUT_CENTS if won else 0.0
    return contracts * (payout - avg_price_cents)
