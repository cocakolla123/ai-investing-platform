"""Shared builders and in-memory fakes for options sleeve unit tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from dynamo.domain.models.options import (
    STRUCTURE_LEGS,
    OptionChainRow,
    OptionOrder,
    OptionPosition,
    OptionRight,
    OptionStructure,
    OptionTrade,
    OptionTradeDecision,
    occ_symbol,
)
from dynamo.domain.models.portfolio import TERMINAL_ORDER_STATES, OrderState
from dynamo.domain.models.research import Action, CommitteeVerdict, QuantAlignment
from dynamo.execution.mock_options_broker import MockOptionsBroker
from dynamo.execution.options_order_manager import OptionsOrderManager
from dynamo.risk.options_sizing import OptionsSizingInputs, OptionsSizingParams
from tests.unit.execution.fakes import EventRow, FakeClock

NOW = datetime(2026, 8, 4, 0, 45, tzinfo=UTC)
AS_OF = date(2026, 8, 4)
# Third Friday of September 2026 — a monthly expiry 45 DTE from NOW.
MONTHLY_EXPIRY = date(2026, 9, 18)
EQUITY = 100_000.0
PARAMS = OptionsSizingParams()


def make_verdict(
    symbol: str = "AAPL",
    action: Action = Action.BUY,
    conviction: float = 0.8,
    quant_alignment: QuantAlignment = QuantAlignment.AGREE,
) -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"cv-{symbol}-{AS_OF.isoformat()}",
        symbol=symbol,
        as_of=AS_OF,
        action=action,
        conviction=conviction,
        thesis=f"thesis for {symbol}",
        bull_case="bull",
        bear_case="bear",
        quant_alignment=quant_alignment,
    )


def make_chain_row(
    underlying: str = "AAPL",
    expiry: date = MONTHLY_EXPIRY,
    strike: float = 100.0,
    right: OptionRight = OptionRight.CALL,
    bid: float | None = 2.40,
    ask: float | None = 2.50,
    open_interest: int | None = 500,
    iv: float | None = 0.30,
    delta: float | None = None,
) -> OptionChainRow:
    return OptionChainRow(
        underlying=underlying,
        as_of=NOW,
        expiry=expiry,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        open_interest=open_interest,
        iv=iv,
        delta=delta,
    )


def default_chain(underlying: str = "AAPL", spot: float = 100.0) -> list[OptionChainRow]:
    """A liquid two-sided chain around spot at the monthly expiry."""
    rows: list[OptionChainRow] = []
    for pct in (0.90, 0.95, 1.00, 1.05, 1.10):
        strike = round(spot * pct, 2)
        for right in (OptionRight.CALL, OptionRight.PUT):
            rows.append(
                make_chain_row(underlying, MONTHLY_EXPIRY, strike, right)
            )
    return rows


def make_inputs(
    verdicts: Mapping[str, CommitteeVerdict] | None = None,
    chains: Mapping[str, list[OptionChainRow]] | None = None,
    spot_prices: Mapping[str, float] | None = None,
    iv_ranks: Mapping[str, float] | None = None,
    equity: float = EQUITY,
    cash_available: float = 50_000.0,
    shares_held: Mapping[str, float] | None = None,
    long_term_holdings: frozenset[str] = frozenset(),
    long_term_eligible: frozenset[str] = frozenset(),
    open_positions: tuple[OptionPosition, ...] = (),
    now: datetime = NOW,
) -> OptionsSizingInputs:
    verdicts = verdicts if verdicts is not None else {"AAPL": make_verdict()}
    chains = (
        chains if chains is not None else {sym: default_chain(sym) for sym in verdicts}
    )
    spot_prices = (
        spot_prices if spot_prices is not None else dict.fromkeys(verdicts, 100.0)
    )
    iv_ranks = iv_ranks if iv_ranks is not None else dict.fromkeys(verdicts, 0.4)
    return OptionsSizingInputs(
        verdicts=verdicts,
        chains=chains,
        spot_prices=spot_prices,
        iv_ranks=iv_ranks,
        equity=equity,
        cash_available=cash_available,
        shares_held=shares_held if shares_held is not None else {},
        long_term_holdings=long_term_holdings,
        long_term_eligible=long_term_eligible,
        open_positions=open_positions,
        now=now,
    )


def make_decision(
    structure: OptionStructure = OptionStructure.LONG_CALL,
    underlying: str = "AAPL",
    strike: float = 95.0,
    contracts: int = 4,
    limit_price: float = 2.50,
    collateral_shares: float = 0.0,
    collateral_cash: float = 0.0,
) -> OptionTradeDecision:
    side, right = STRUCTURE_LEGS[structure]
    premium = (
        contracts * limit_price * 100
        if structure in (OptionStructure.LONG_CALL, OptionStructure.LONG_PUT)
        else (contracts * strike * 100 if structure is OptionStructure.CASH_SECURED_PUT else 0.0)
    )
    return OptionTradeDecision(
        structure=structure,
        side=side,
        right=right,
        underlying=underlying,
        occ_symbol=occ_symbol(underlying, MONTHLY_EXPIRY, right, strike),
        expiry=MONTHLY_EXPIRY,
        strike=strike,
        contracts=contracts,
        limit_price=limit_price,
        premium_at_risk=premium,
        collateral_shares=collateral_shares,
        collateral_cash=collateral_cash,
        conviction=0.8,
        iv_rank=0.4,
        dte=45,
        rationale="test decision",
        verdict_ref=f"cv-{underlying}-{AS_OF.isoformat()}",
    )


def make_option_position(
    structure: OptionStructure = OptionStructure.LONG_CALL,
    underlying: str = "AAPL",
    strike: float = 95.0,
    contracts: int = 4,
    avg_price: float = 2.50,
    expiry: date = MONTHLY_EXPIRY,
) -> OptionPosition:
    side, right = STRUCTURE_LEGS[structure]
    return OptionPosition(
        occ_symbol=occ_symbol(underlying, expiry, right, strike),
        underlying=underlying,
        structure=structure,
        side=side,
        right=right,
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        avg_price=avg_price,
    )


class FakeOptionOrderStore:
    """Satisfies the OptionOrderStore protocol; one event row per transition."""

    def __init__(self) -> None:
        self.orders: dict[str, OptionOrder] = {}
        self.events: list[EventRow] = []

    def insert(self, order: OptionOrder) -> None:
        if order.client_order_id in self.orders:
            raise ValueError(f"duplicate insert {order.client_order_id}")
        self.orders[order.client_order_id] = order
        self.events.append(
            EventRow(
                order.client_order_id,
                None,
                order.state.value,
                "created",
                order.created_at or datetime.now(UTC),
            )
        )

    def get(self, client_order_id: str) -> OptionOrder | None:
        return self.orders.get(client_order_id)

    def open_orders(self) -> list[OptionOrder]:
        return [o for o in self.orders.values() if o.state not in TERMINAL_ORDER_STATES]

    def set_broker_order_id(self, client_order_id: str, broker_order_id: str) -> None:
        order = self.orders[client_order_id]
        self.orders[client_order_id] = order.model_copy(
            update={"broker_order_id": broker_order_id}
        )

    def set_decision_context(self, client_order_id: str, decision_context_id: str) -> None:
        order = self.orders[client_order_id]
        self.orders[client_order_id] = order.model_copy(
            update={"decision_context_id": decision_context_id}
        )

    def transition_with_event(
        self,
        client_order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str,
        ts: datetime,
        payload: Mapping[str, Any] | None = None,
    ) -> OptionOrder:
        order = self.orders[client_order_id]
        updated = order.model_copy(update={"state": to_state})
        self.orders[client_order_id] = updated
        self.events.append(
            EventRow(client_order_id, from_state.value, to_state.value, reason, ts, payload)
        )
        return updated


class FakeOptionTradeStore:
    def __init__(self) -> None:
        self.trades: list[OptionTrade] = []

    def insert(self, trade: OptionTrade) -> None:
        self.trades.append(trade)

    def filled_contracts(self, client_order_id: str) -> int:
        return sum(
            t.contracts
            for t in self.trades
            if t.client_order_id == client_order_id and t.kind == "fill"
        )


class PermissiveGate:
    permitted = True
    reason = "paper_permitted"


class RefusingGate:
    permitted = False
    reason = "approval_missing"


def make_options_manager(
    broker: MockOptionsBroker | None = None,
    *,
    gate: Any = None,
    shares: float | None = None,
    cash: float | None = None,
) -> tuple[
    OptionsOrderManager, FakeOptionOrderStore, FakeOptionTradeStore, FakeClock, MockOptionsBroker
]:
    broker = broker if broker is not None else MockOptionsBroker()
    store = FakeOptionOrderStore()
    trade_store = FakeOptionTradeStore()
    clock = FakeClock()
    manager = OptionsOrderManager(
        broker,
        store,
        trade_store,
        clock,
        gate=(lambda _order: gate) if gate is not None else None,
        shares_lookup=(lambda _u: shares) if shares is not None else None,
        cash_lookup=(lambda: cash) if cash is not None else None,
    )
    return manager, store, trade_store, clock, broker
