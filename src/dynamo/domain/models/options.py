"""Options trading domain entities — defined-risk only (ADR-0010).

The sleeve reuses the equity order lifecycle (``OrderState`` +
``ORDER_TRANSITIONS`` from :mod:`dynamo.domain.models.portfolio`) so there is
exactly one state machine to reason about. Prices are dollars per share (one
contract controls ``CONTRACT_MULTIPLIER`` shares); contracts are positive
integers.

DEFINED-RISK IS A TYPE INVARIANT, not a convention: the only representable
structures are long calls, long puts, covered calls, and cash-secured puts
(``OptionStructure``). Every order/position/decision model validates that its
``(side, right)`` leg matches its structure — a naked short call or naked
short put is UNREPRESENTABLE and raises at construction. Collateral for the
short structures is additionally validated at proposal (decision construction
carries the collateral proof) and re-checked at submission by the order
manager.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from dynamo.domain.models.market import Frozen
from dynamo.domain.models.portfolio import OrderSide, OrderState

CONTRACT_MULTIPLIER = 100
OPTIONS_VENUE_ALPACA = "alpaca"


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class OptionStructure(StrEnum):
    """The ONLY four representable structures — all have finite max loss."""

    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"  # short call fully covered by >= 100 shares/contract
    CASH_SECURED_PUT = "cash_secured_put"  # short put with cash reserved for assignment


# The single legal (side, right) leg per structure. Any other combination is
# an unbounded-loss (or undefined) structure and must never construct.
STRUCTURE_LEGS: dict[OptionStructure, tuple[OrderSide, OptionRight]] = {
    OptionStructure.LONG_CALL: (OrderSide.BUY, OptionRight.CALL),
    OptionStructure.LONG_PUT: (OrderSide.BUY, OptionRight.PUT),
    OptionStructure.COVERED_CALL: (OrderSide.SELL, OptionRight.CALL),
    OptionStructure.CASH_SECURED_PUT: (OrderSide.SELL, OptionRight.PUT),
}

SHORT_STRUCTURES = frozenset({OptionStructure.COVERED_CALL, OptionStructure.CASH_SECURED_PUT})


class CollateralShortfall(Exception):
    """Covered-call/CSP collateral is missing — the structure may not exist.

    Deliberately NOT a ``ValueError``: pydantic would swallow one inside a
    model validator, and this failure must surface as itself at proposal time.
    """


def occ_symbol(underlying: str, expiry: date, right: OptionRight, strike: float) -> str:
    """OCC option symbol, e.g. ``AAPL261218C00150000``."""
    letter = "C" if right is OptionRight.CALL else "P"
    return f"{underlying.upper()}{expiry.strftime('%y%m%d')}{letter}{round(strike * 1000):08d}"


def parse_occ_symbol(symbol: str) -> tuple[str, date, OptionRight, float]:
    """Inverse of :func:`occ_symbol`; raises ``ValueError`` on malformed input."""
    if len(symbol) < 16:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    strike = int(symbol[-8:]) / 1000.0
    letter = symbol[-9]
    if letter not in ("C", "P"):
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    right = OptionRight.CALL if letter == "C" else OptionRight.PUT
    expiry = datetime.strptime(symbol[-15:-9], "%y%m%d").date()
    underlying = symbol[:-15]
    if not underlying:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    return underlying, expiry, right, strike


def validate_covered_call_collateral(contracts: int, shares_held: float) -> None:
    """A covered call must be FULLY covered: >= 100 shares per contract."""
    required = contracts * CONTRACT_MULTIPLIER
    if not math.isfinite(shares_held) or shares_held < required:
        raise CollateralShortfall(f"covered call needs {required} shares, have {shares_held}")


def validate_csp_collateral(contracts: int, strike: float, cash_reserved: float) -> None:
    """A cash-secured put must reserve full assignment cash: strike x 100 x n."""
    required = contracts * strike * CONTRACT_MULTIPLIER
    if not math.isfinite(cash_reserved) or cash_reserved < required:
        raise CollateralShortfall(
            f"cash-secured put needs ${required:,.2f} reserved, have ${cash_reserved:,.2f}"
        )


def max_loss_dollars(
    structure: OptionStructure, strike: float, premium_per_share: float, contracts: int
) -> float:
    """Finite max loss for every representable structure (the defined-risk proof).

    Long premium structures lose at most the premium paid; a cash-secured put
    loses at most ``(strike - premium) x 100 x n`` (underlying to zero); the
    covered-call OPTION leg loses nothing beyond delivering shares already
    held — the share downside is owned by the equity risk engine, not this
    sleeve.
    """
    if structure in (OptionStructure.LONG_CALL, OptionStructure.LONG_PUT):
        return premium_per_share * CONTRACT_MULTIPLIER * contracts
    if structure is OptionStructure.CASH_SECURED_PUT:
        return max(strike - premium_per_share, 0.0) * CONTRACT_MULTIPLIER * contracts
    return 0.0  # covered call — fully share-covered by construction


class _StructureLegged(Frozen):
    """Shared invariant: the (side, right) leg must match the structure."""

    structure: OptionStructure
    side: OrderSide
    right: OptionRight

    @model_validator(mode="after")
    def _leg_matches_structure(self) -> _StructureLegged:
        expected = STRUCTURE_LEGS[self.structure]
        if (self.side, self.right) != expected:
            raise ValueError(
                f"unrepresentable structure: {self.structure.value} requires "
                f"{expected[0].value} {expected[1].value}, got "
                f"{self.side.value} {self.right.value} — a mismatched leg could "
                "have unbounded loss and must never exist"
            )
        return self


class OptionOrder(_StructureLegged):
    """A limit order for one defined-risk option leg (doc 09 discipline)."""

    client_order_id: str
    occ_symbol: str
    underlying: str
    expiry: date
    strike: float = Field(gt=0.0)
    contracts: int = Field(ge=1)
    limit_price: float = Field(gt=0.0)  # dollars per share; limit orders ONLY
    state: OrderState = OrderState.PROPOSED
    decision_context_id: str | None = None  # REQUIRED before submission
    broker_order_id: str | None = None
    created_at: datetime | None = None


class OptionPosition(_StructureLegged):
    """Open contracts in one OCC symbol."""

    occ_symbol: str
    underlying: str
    expiry: date
    strike: float = Field(gt=0.0)
    contracts: int = Field(ge=0)
    avg_price: float = Field(ge=0.0)  # dollars per share

    @property
    def premium_at_risk(self) -> float:
        """Dollars this position contributes to the sleeve risk cap.

        Long premium for long structures, the full assignment obligation for
        cash-secured puts, zero for covered calls (share-covered).
        """
        if self.structure in (OptionStructure.LONG_CALL, OptionStructure.LONG_PUT):
            return self.contracts * self.avg_price * CONTRACT_MULTIPLIER
        if self.structure is OptionStructure.CASH_SECURED_PUT:
            return self.contracts * self.strike * CONTRACT_MULTIPLIER
        return 0.0


class OptionTrade(Frozen):
    """A fill on an option order (``kind='fill'``) or an expiry close-out."""

    client_order_id: str | None
    occ_symbol: str
    underlying: str
    side: OrderSide
    right: OptionRight
    contracts: int = Field(ge=0)
    price: float = Field(ge=0.0)  # dollars per share
    fees: float = 0.0
    kind: str = "fill"  # fill | expiry
    filled_at: datetime


class OptionTradeDecision(_StructureLegged):
    """Sized output of :func:`dynamo.risk.options_sizing.select_and_size_option_trades`.

    Collateral is validated AT PROPOSAL: constructing a covered-call decision
    without full share coverage, or a CSP decision without full cash reserve,
    raises ``CollateralShortfall``.
    """

    underlying: str
    occ_symbol: str
    expiry: date
    strike: float = Field(gt=0.0)
    contracts: int = Field(ge=1)
    limit_price: float = Field(gt=0.0)  # dollars per share
    premium_at_risk: float = Field(ge=0.0)  # dollars counted against the sleeve cap
    collateral_shares: float = 0.0  # covered calls: shares backing the short call
    collateral_cash: float = 0.0  # CSPs: cash reserved for assignment
    conviction: float = Field(ge=0.0, le=1.0)
    iv_rank: float = Field(ge=0.0, le=1.0)
    dte: int = Field(ge=0)
    rationale: str
    verdict_ref: str

    @model_validator(mode="after")
    def _collateral_present(self) -> OptionTradeDecision:
        if self.structure is OptionStructure.COVERED_CALL:
            validate_covered_call_collateral(self.contracts, self.collateral_shares)
        elif self.structure is OptionStructure.CASH_SECURED_PUT:
            validate_csp_collateral(self.contracts, self.strike, self.collateral_cash)
        return self

    @property
    def max_loss(self) -> float:
        return max_loss_dollars(self.structure, self.strike, self.limit_price, self.contracts)


class OptionChainRow(Frozen):
    """One strike of a chain snapshot (persisted minimally near targets)."""

    underlying: str
    as_of: datetime
    expiry: date
    strike: float = Field(gt=0.0)
    right: OptionRight
    bid: float | None = None
    ask: float | None = None
    open_interest: int | None = None
    iv: float | None = None
    delta: float | None = None

    @property
    def occ(self) -> str:
        return occ_symbol(self.underlying, self.expiry, self.right, self.strike)
