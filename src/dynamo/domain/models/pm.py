"""Prediction-market trading domain entities. Design: ADR-0009, ADR-0011.

The sleeves reuse the equity order lifecycle (``OrderState`` +
``ORDER_TRANSITIONS`` from :mod:`dynamo.domain.models.portfolio`) so there is
exactly one state machine to reason about. Prices are integer cents (Kalshi
native); contracts are non-negative integers; every contract settles at $1
(100 cents) when it resolves in the holder's favour, $0 otherwise.

Two venues can appear in these entities:

- ``kalshi`` — the real sleeve (demo/prod behind the live gate, ADR-0009);
- ``polymarket`` — the shadow-trading sleeve (ADR-0011): decided by the same
  pipeline but ALWAYS SIMULATED. Polymarket has no demo exchange, so fills are
  simulated internally and no real order is ever sent. ``venue`` alone marks a
  row as simulated (:func:`is_simulated_venue`); there is deliberately no real
  Polymarket execution path anywhere in the codebase.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from dynamo.domain.models.market import Frozen
from dynamo.domain.models.portfolio import OrderState

PM_VENUE_KALSHI = "kalshi"
PM_VENUE_POLYMARKET = "polymarket"
# Venues whose executions are SIMULATED — internal fills only, no real orders.
PM_SIM_VENUES = frozenset({PM_VENUE_POLYMARKET})
CONTRACT_PAYOUT_CENTS = 100.0


def is_simulated_venue(venue: str) -> bool:
    """True when every execution on ``venue`` is simulated (ADR-0011).

    The venue string IS the simulation marker: Polymarket has no real
    execution path (no broker, no credentials, no CLOB endpoints), so any
    order/trade/position row carrying ``venue='polymarket'`` is by
    construction an internal simulation.
    """
    return venue in PM_SIM_VENUES


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


class PMSimLedgerEntry(Frozen):
    """One idempotent cash movement on a SIMULATED sleeve's bankroll (ADR-0011).

    ``(venue, kind, ref)`` is the idempotency key: the seed deposit uses
    ``ref='initial'``, fill debits the client order id, settlement credits the
    market id. Amounts are signed cents — debits negative, credits positive —
    and the bankroll is the signed sum of all entries.
    """

    venue: str = PM_VENUE_POLYMARKET
    kind: str  # deposit | fill_debit | settlement_credit
    ref: str
    market_id: str | None = None
    amount_cents: float
    note: str = ""
    ts: datetime


class CrossVenueSpread(Frozen):
    """One matched Kalshi/Polymarket pair from a cross-venue aggregator.

    Enrichment only (ADR-0009 addendum): the pair corroborates — or vetoes —
    a Kalshi sleeve candidate in :mod:`dynamo.risk.pm_sizing`; it never sizes
    a trade by itself. Prices are YES probabilities in [0, 1].
    """

    question: str
    kalshi_market_id: str
    kalshi_yes: float = Field(ge=0.0, le=1.0)
    polymarket_id: str
    polymarket_yes: float = Field(ge=0.0, le=1.0)
    yes_diff: float  # kalshi_yes - polymarket_yes as reported (sign preserved)
    as_of: datetime


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
