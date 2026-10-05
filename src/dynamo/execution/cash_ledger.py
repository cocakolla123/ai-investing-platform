"""Internal cash ledger: the source of truth for cash (doc 09 §6 reconciliation).

Before this module there was no ledger. Internal cash was read from the
latest ``equity_curve`` row, and the nightly equity snapshot wrote that same
value back, so cash never moved. After the first paper fill (2026-09-24,
123.7092 SPY at 764.62) internal cash stayed at $100,000 while the broker
showed $5,409.29. Equity was then counted as cash plus the full position,
about 1.95x the real value, and every reconciliation since diverged.

Cash is now derived: the opening cash (the first equity snapshot, taken from
the broker at bootstrap) plus every recorded fill's cash flow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dynamo.domain.models.portfolio import Position


@dataclass(frozen=True)
class Fill:
    side: str  # "buy" | "sell"
    qty: float
    price: float
    fees: float = 0.0


def cash_flow(fill: Fill) -> float:
    """Signed cash effect of one fill: a buy pays notional + fees, a sell
    receives notional - fees."""
    notional = fill.qty * fill.price
    side = fill.side.lower()
    if side == "buy":
        return -(notional + fill.fees)
    if side == "sell":
        return notional - fill.fees
    raise ValueError(f"unknown fill side {fill.side!r}")


def ledger_cash(opening_cash: float, fills: Iterable[Fill]) -> float:
    return opening_cash + sum(cash_flow(f) for f in fills)


def mark_to_market(
    cash: float, positions: Iterable[Position], prices: Mapping[str, float]
) -> tuple[float, float]:
    """-> (equity, gross exposure). A position without a price is valued at
    its average entry price, never dropped."""
    market_value = sum(p.qty * prices.get(p.symbol, p.avg_entry_price) for p in positions)
    equity = cash + market_value
    return equity, (market_value / equity if equity > 0 else 0.0)
