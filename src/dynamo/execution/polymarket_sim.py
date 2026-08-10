"""Polymarket shadow-trading sleeve — the SIMULATED broker (ADR-0011).

Polymarket has no demo exchange, so this module simulates the broker
ourselves: :class:`PolymarketSimBroker` implements the ``PMBroker`` port over
an internal book and fills every order immediately at the live Gamma YES
price plus a slippage penalty applied AGAINST us. It is deliberately
incapable of real execution — it holds no HTTP client, imports no transport
library, and knows no endpoint or credential; the only inputs are quote
snapshots handed in by the caller (fetched through the read-only
:mod:`dynamo.data.polymarket` adapter) and the sim bankroll's available cash.

Fill model (documented assumption): the Gamma API quotes one price per
outcome, not a bid/ask book, so the current YES outcome price stands in for
the ask and ``slippage_cents`` is the crossing penalty — a YES buy fills at
``yes*100 + slippage`` cents, a NO buy at ``(1-yes)*100 + slippage``. Orders
are rejected (never silently mis-filled) when the quote is missing, stale,
degenerate, the market is closed, the penalized price reaches the 100c
payout, or the fill would overdraw the sim bankroll.

Money moves only through :class:`PMSimLedgerEntry` rows (seed deposit, fill
debits, settlement credits); the ledger's (venue, kind, ref) uniqueness makes
fills and settlements idempotent. :func:`settle_sim_positions` realizes P&L
when Gamma reports a market resolved: winners credit ``contracts * 100c``
back to the bankroll, losers credit nothing, and settling twice is a no-op.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from dynamo.domain.models.pm import (
    CONTRACT_PAYOUT_CENTS,
    PMOrder,
    PMPosition,
    PMSide,
    PMSimLedgerEntry,
    PMTrade,
    settlement_pnl_cents,
)
from dynamo.execution.errors import BrokerRejection

PM_SIM_SCOPE_PREFIX = "POLYMARKET:"

# Reject reasons (terminal; the order manager records them on the order).
REJECT_NO_QUOTE = "no_quote"
REJECT_STALE_QUOTE = "stale_quote"
REJECT_MARKET_CLOSED = "market_closed"
REJECT_DEGENERATE_PRICE = "degenerate_price"
REJECT_PRICE_AT_PAYOUT = "price_at_or_above_payout"
REJECT_INSUFFICIENT_CASH = "insufficient_sim_cash"

MAX_QUOTE_AGE = timedelta(minutes=30)


def pm_sim_scope(market_id: str) -> str:
    """Gate/DecisionContext scope for one shadow-traded Polymarket market."""
    return f"{PM_SIM_SCOPE_PREFIX}{market_id}"


@dataclass(frozen=True)
class SimQuote:
    """A fresh Gamma read the caller hands to the broker (never fetched here)."""

    yes_price: float  # YES outcome price in [0, 1]
    ts: datetime
    closed: bool = False


class PMSimLedgerStore(Protocol):
    """Ledger persistence surface (SQL implementation in persistence/repositories/pm)."""

    def append(self, entry: PMSimLedgerEntry) -> bool: ...  # False -> duplicate, no-op

    def cash_cents(self) -> float: ...  # seed + signed sum of movements


class SimTradeStore(Protocol):
    def insert(self, trade: PMTrade) -> None: ...


class SimPositionStore(Protocol):
    def close(self, venue: str, market_id: str) -> None: ...


@dataclass
class _SimBrokerOrder:
    client_order_id: str
    broker_order_id: str
    market_id: str
    side: str
    contracts: int
    limit_price_cents: int
    status: str  # filled | canceled
    filled_contracts: int = 0
    filled_avg_price_cents: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "market_id": self.market_id,
            "side": self.side,
            "contracts": self.contracts,
            "filled_contracts": self.filled_contracts,
            "filled_avg_price_cents": self.filled_avg_price_cents,
            "status": self.status,
        }


@dataclass
class PolymarketSimBroker:
    """``PMBroker`` port over an internal simulated book — SIMULATED ONLY.

    ``quotes`` maps market_id -> the caller's fresh :class:`SimQuote`;
    ``cash_cents`` is the sim bankroll available at construction. Every
    accepted order fills immediately and in full at quote + slippage; the
    broker tracks its own spend so a batch of orders can never overdraw the
    bankroll within one run. There is no transport, no auth, and no way to
    reach a trading endpoint from here.
    """

    quotes: Mapping[str, SimQuote]
    cash_cents: float
    now: datetime
    slippage_cents: int = 1
    max_quote_age: timedelta = MAX_QUOTE_AGE
    _orders: dict[str, _SimBrokerOrder] = field(default_factory=dict, init=False, repr=False)
    _positions: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _spent_cents: float = field(default=0.0, init=False, repr=False)
    _seq: int = field(default=0, init=False, repr=False)

    # --- PMBroker port ---

    def submit(self, order: PMOrder) -> str:
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            # Same idempotency semantics as Kalshi: known intent, same id back.
            return existing.broker_order_id

        fill_price_cents = self._fill_price(order)
        cost = order.contracts * fill_price_cents
        if cost > self.cash_cents - self._spent_cents:
            raise BrokerRejection(order.client_order_id, REJECT_INSUFFICIENT_CASH)

        self._seq += 1
        broker_order = _SimBrokerOrder(
            client_order_id=order.client_order_id,
            broker_order_id=f"polymarket-sim-{self._seq}",
            market_id=order.market_id,
            side=order.side.value,
            contracts=order.contracts,
            limit_price_cents=order.limit_price_cents,
            status="filled",
            filled_contracts=order.contracts,
            filled_avg_price_cents=fill_price_cents,
        )
        self._orders[order.client_order_id] = broker_order
        self._spent_cents += cost
        self._apply_position(order.market_id, order.side.value, order.contracts, fill_price_cents)
        return broker_order.broker_order_id

    def cancel(self, client_order_id: str) -> None:
        bo = self._require(client_order_id)
        if bo.status == "filled":
            return  # everything fills immediately; nothing is ever resting
        bo.status = "canceled"

    def get_order(self, client_order_id: str) -> dict[str, Any]:
        return self._require(client_order_id).as_dict()

    def open_orders(self) -> list[dict[str, Any]]:
        return []  # immediate fills: no resting orders, ever

    def positions(self) -> list[dict[str, Any]]:
        return [
            {
                "market_id": market_id,
                "side": p["side"],
                "contracts": p["contracts"],
                "exposure_cents": p["contracts"] * p["avg_price_cents"],
            }
            for market_id, p in self._positions.items()
            if p["contracts"] > 0
        ]

    def balance(self) -> dict[str, Any]:
        return {"balance_cents": self.cash_cents - self._spent_cents}

    # --- fill model (module docstring) ---

    def _fill_price(self, order: PMOrder) -> float:
        quote = self.quotes.get(order.market_id)
        if quote is None:
            raise BrokerRejection(order.client_order_id, REJECT_NO_QUOTE)
        if quote.closed:
            raise BrokerRejection(order.client_order_id, REJECT_MARKET_CLOSED)
        if self.now - quote.ts > self.max_quote_age:
            raise BrokerRejection(order.client_order_id, REJECT_STALE_QUOTE)
        yes = quote.yes_price
        if not 0.0 < yes < 1.0:
            raise BrokerRejection(order.client_order_id, REJECT_DEGENERATE_PRICE)
        base_cents = yes * 100.0 if order.side is PMSide.YES else (1.0 - yes) * 100.0
        fill = base_cents + float(self.slippage_cents)  # worse for us, always
        if fill >= CONTRACT_PAYOUT_CENTS:
            raise BrokerRejection(order.client_order_id, REJECT_PRICE_AT_PAYOUT)
        return fill

    # --- internals ---

    def _require(self, client_order_id: str) -> _SimBrokerOrder:
        bo = self._orders.get(client_order_id)
        if bo is None:
            raise BrokerRejection(client_order_id, "unknown client_order_id")
        return bo

    def _apply_position(
        self, market_id: str, side: str, contracts: int, price_cents: float
    ) -> None:
        pos = self._positions.setdefault(
            market_id, {"side": side, "contracts": 0, "avg_price_cents": 0.0}
        )
        new_contracts = pos["contracts"] + contracts
        pos["avg_price_cents"] = (
            pos["avg_price_cents"] * pos["contracts"] + price_cents * contracts
        ) / new_contracts
        pos["contracts"] = new_contracts


def fill_debit_entry(trade: PMTrade, now: datetime) -> PMSimLedgerEntry:
    """The ledger debit for one simulated fill (idempotent on the order id)."""
    cost_cents = trade.contracts * trade.price_cents + trade.fees_cents
    return PMSimLedgerEntry(
        venue=trade.venue,
        kind="fill_debit",
        ref=trade.client_order_id or "",
        market_id=trade.market_id,
        amount_cents=-cost_cents,
        note=f"simulated fill {trade.contracts}x {trade.side.value} @ {trade.price_cents:.1f}c",
        ts=now,
    )


def settle_sim_positions(
    positions: Sequence[PMPosition],
    resolved: Mapping[str, bool],
    trades: SimTradeStore,
    position_store: SimPositionStore,
    ledger: PMSimLedgerStore,
    now: datetime,
) -> tuple[int, float]:
    """Settle resolved SIMULATED positions; returns (settlements, realized cents).

    Win -> ``contracts * 100c`` credited to the sim bankroll; loss -> a 0c
    credit row. The settlement-credit ledger row (ref = market_id) is the
    idempotency marker: when ``append`` reports a duplicate the position was
    already settled by an earlier run and nothing else happens. Markets Gamma
    has not resolved yet are simply skipped (hold-to-resolution, matching the
    Kalshi sleeve's exit semantics).
    """
    settlements = 0
    realized_cents = 0.0
    for position in positions:
        outcome = resolved.get(position.market_id)
        if outcome is None:
            continue
        won = outcome if position.side is PMSide.YES else not outcome
        payout_cents = position.contracts * CONTRACT_PAYOUT_CENTS if won else 0.0
        pnl = settlement_pnl_cents(
            position.side, position.contracts, position.avg_price_cents, outcome
        )
        credited = ledger.append(
            PMSimLedgerEntry(
                venue=position.venue,
                kind="settlement_credit",
                ref=position.market_id,
                market_id=position.market_id,
                amount_cents=payout_cents,
                note=(
                    f"simulated settlement: {'won' if won else 'lost'} "
                    f"{position.contracts}x {position.side.value}"
                ),
                ts=now,
            )
        )
        if not credited:
            continue  # already settled — idempotent no-op
        trades.insert(
            PMTrade(
                client_order_id=None,
                venue=position.venue,
                market_id=position.market_id,
                side=position.side,
                contracts=position.contracts,
                price_cents=CONTRACT_PAYOUT_CENTS if won else 0.0,
                kind="settlement",
                realized_pnl_cents=pnl,
                filled_at=now,
            )
        )
        position_store.close(position.venue, position.market_id)
        settlements += 1
        realized_cents += pnl
    return settlements, realized_cents
