"""Polymarket shadow-trading sleeve — SIMULATED broker + settlement (ADR-0011).

Covers the fill model (Gamma price + slippage against us, fail-closed
rejects), bankroll accounting (overdraw rejection, never-negative cash),
settlement (win/loss/idempotent), the order-manager integration, and the
structural guarantee that the sim module cannot reach a network.
"""

from __future__ import annotations

import inspect
from datetime import timedelta

import pytest

from dynamo.domain.models.pm import (
    PMSide,
    PMSimLedgerEntry,
    PMTrade,
    is_simulated_venue,
)
from dynamo.domain.models.portfolio import OrderState
from dynamo.execution.errors import BrokerRejection
from dynamo.execution.pm_order_manager import PMOrderManager
from dynamo.execution.polymarket_sim import (
    REJECT_INSUFFICIENT_CASH,
    REJECT_MARKET_CLOSED,
    REJECT_NO_QUOTE,
    REJECT_PRICE_AT_PAYOUT,
    REJECT_STALE_QUOTE,
    PolymarketSimBroker,
    SimQuote,
    fill_debit_entry,
    pm_sim_scope,
    settle_sim_positions,
)
from tests.unit.execution.fakes import FakeClock
from tests.unit.pm.helpers import (
    NOW,
    FakePMOrderStore,
    FakePMTradeStore,
    PermissiveGate,
    make_pm_order,
    make_position,
)

MARKET = "514729"


def make_broker(
    yes: float = 0.43,
    *,
    cash_cents: float = 100_000.0,  # $1,000 sim bankroll
    slippage_cents: int = 1,
    closed: bool = False,
    quote_age: timedelta = timedelta(minutes=1),
    quotes: dict[str, SimQuote] | None = None,
) -> PolymarketSimBroker:
    if quotes is None:
        quotes = {MARKET: SimQuote(yes_price=yes, ts=NOW - quote_age, closed=closed)}
    return PolymarketSimBroker(
        quotes=quotes, cash_cents=cash_cents, now=NOW, slippage_cents=slippage_cents
    )


def sim_order(coid: str = "sim-1", side: PMSide = PMSide.YES, contracts: int = 10):
    return make_pm_order(coid=coid, market_id=MARKET, side=side, contracts=contracts)


class FakeLedger:
    """PMSimLedgerStore fake: (venue, kind, ref) idempotency, signed-sum cash."""

    def __init__(self) -> None:
        self.entries: list[PMSimLedgerEntry] = []
        self._keys: set[tuple[str, str, str]] = set()

    def append(self, entry: PMSimLedgerEntry) -> bool:
        key = (entry.venue, entry.kind, entry.ref)
        if key in self._keys:
            return False
        self._keys.add(key)
        self.entries.append(entry)
        return True

    def cash_cents(self) -> float:
        return sum(e.amount_cents for e in self.entries)


class FakePositionStore:
    def __init__(self) -> None:
        self.closed: list[tuple[str, str]] = []

    def close(self, venue: str, market_id: str) -> None:
        self.closed.append((venue, market_id))


class TestFillModel:
    def test_yes_fill_at_gamma_price_plus_slippage(self) -> None:
        broker = make_broker(yes=0.43, slippage_cents=1)
        broker.submit(sim_order(contracts=10))
        remote = broker.get_order("sim-1")
        assert remote["status"] == "filled"
        assert remote["filled_contracts"] == 10
        assert remote["filled_avg_price_cents"] == pytest.approx(44.0)  # 43 + 1 against us

    def test_no_side_prices_the_no_contract_plus_slippage(self) -> None:
        broker = make_broker(yes=0.43, slippage_cents=2)
        broker.submit(sim_order(side=PMSide.NO))
        remote = broker.get_order("sim-1")
        assert remote["filled_avg_price_cents"] == pytest.approx(59.0)  # (100-43) + 2

    def test_fill_debits_the_balance(self) -> None:
        broker = make_broker(yes=0.43, cash_cents=10_000.0)
        broker.submit(sim_order(contracts=10))  # 10 x 44c = 440c
        assert broker.balance()["balance_cents"] == pytest.approx(10_000.0 - 440.0)

    def test_duplicate_submit_is_idempotent(self) -> None:
        broker = make_broker()
        first = broker.submit(sim_order())
        again = broker.submit(sim_order())
        assert first == again
        assert broker.balance()["balance_cents"] == pytest.approx(100_000.0 - 440.0)

    def test_missing_quote_rejects(self) -> None:
        broker = make_broker(quotes={})
        with pytest.raises(BrokerRejection) as exc:
            broker.submit(sim_order())
        assert exc.value.reason == REJECT_NO_QUOTE

    def test_closed_market_rejects(self) -> None:
        broker = make_broker(closed=True)
        with pytest.raises(BrokerRejection) as exc:
            broker.submit(sim_order())
        assert exc.value.reason == REJECT_MARKET_CLOSED

    def test_stale_quote_rejects(self) -> None:
        broker = make_broker(quote_age=timedelta(hours=2))
        with pytest.raises(BrokerRejection) as exc:
            broker.submit(sim_order())
        assert exc.value.reason == REJECT_STALE_QUOTE

    def test_degenerate_price_rejects(self) -> None:
        for yes in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(BrokerRejection):
                make_broker(yes=yes).submit(sim_order())

    def test_price_at_or_above_payout_rejects(self) -> None:
        broker = make_broker(yes=0.995, slippage_cents=1)  # 99.5 + 1 >= 100
        with pytest.raises(BrokerRejection) as exc:
            broker.submit(sim_order())
        assert exc.value.reason == REJECT_PRICE_AT_PAYOUT

    def test_no_resting_orders_ever(self) -> None:
        broker = make_broker()
        broker.submit(sim_order())
        assert broker.open_orders() == []


class TestBankroll:
    def test_overdraw_rejected(self) -> None:
        broker = make_broker(yes=0.43, cash_cents=400.0)  # $4 cash, order costs 440c
        with pytest.raises(BrokerRejection) as exc:
            broker.submit(sim_order(contracts=10))
        assert exc.value.reason == REJECT_INSUFFICIENT_CASH
        assert broker.balance()["balance_cents"] == pytest.approx(400.0)

    def test_cumulative_spend_never_goes_negative(self) -> None:
        broker = make_broker(yes=0.43, cash_cents=500.0)
        broker.submit(sim_order("sim-1", contracts=10))  # 440c: fits
        with pytest.raises(BrokerRejection) as exc:  # second 440c would overdraw
            broker.submit(sim_order("sim-2", contracts=10))
        assert exc.value.reason == REJECT_INSUFFICIENT_CASH
        assert broker.balance()["balance_cents"] >= 0.0

    def test_fill_debit_entry_shape(self) -> None:
        trade = PMTrade(
            client_order_id="sim-1",
            venue="polymarket",
            market_id=MARKET,
            side=PMSide.YES,
            contracts=10,
            price_cents=44.0,
            kind="fill",
            filled_at=NOW,
        )
        entry = fill_debit_entry(trade, NOW)
        assert entry.kind == "fill_debit"
        assert entry.ref == "sim-1"  # idempotency key = the deterministic order id
        assert entry.amount_cents == pytest.approx(-440.0)


class TestManagerIntegration:
    """The sim broker rides the SAME doc 09 order manager as Kalshi."""

    def make_manager(self, broker: PolymarketSimBroker):
        store, trades = FakePMOrderStore(), FakePMTradeStore()
        manager = PMOrderManager(
            broker, store, trades, FakeClock(), gate=lambda _o: PermissiveGate()
        )
        return manager, store, trades

    def place(self, manager, coid: str = "sim-1", contracts: int = 10) -> None:
        order = sim_order(coid, contracts=contracts)
        manager._orders.insert(order)  # type: ignore[attr-defined]
        manager.attach_decision_context(coid, f"dc-{coid}")
        manager.request_approval(coid)
        manager.submit(coid)

    def test_submit_then_sync_records_simulated_fill(self) -> None:
        broker = make_broker(yes=0.43)
        manager, store, trades = self.make_manager(broker)
        self.place(manager)
        recorded = manager.sync_orders()
        assert len(recorded) == 1
        assert recorded[0].price_cents == pytest.approx(44.0)
        assert store.orders["sim-1"].state is OrderState.FILLED
        assert trades.filled_contracts("sim-1") == 10

    def test_overdraw_makes_the_order_terminal_rejected(self) -> None:
        broker = make_broker(yes=0.43, cash_cents=100.0)
        manager, store, _trades = self.make_manager(broker)
        self.place(manager)
        assert store.orders["sim-1"].state is OrderState.REJECTED
        events = [e for e in store.events if e.to_state == "rejected"]
        assert REJECT_INSUFFICIENT_CASH in events[0].reason

    def test_sync_twice_records_nothing_new(self) -> None:
        broker = make_broker()
        manager, _store, trades = self.make_manager(broker)
        self.place(manager)
        assert len(manager.sync_orders()) == 1
        assert manager.sync_orders() == []  # idempotent by cumulative count
        assert trades.filled_contracts("sim-1") == 10


class TestSettlement:
    def test_win_credits_payout_and_realizes_pnl(self) -> None:
        position = make_position(MARKET, side=PMSide.YES, contracts=10, avg_price_cents=43.0)
        position = position.model_copy(update={"venue": "polymarket"})
        trades, closer, ledger = FakePMTradeStore(), FakePositionStore(), FakeLedger()
        settlements, realized = settle_sim_positions(
            [position], {MARKET: True}, trades, closer, ledger, NOW
        )
        assert (settlements, realized) == (1, pytest.approx(10 * (100.0 - 43.0)))
        assert ledger.cash_cents() == pytest.approx(1_000.0)  # 10 x 100c payout
        assert closer.closed == [("polymarket", MARKET)]
        settlement = trades.trades[0]
        assert settlement.kind == "settlement"
        assert settlement.realized_pnl_cents == pytest.approx(570.0)

    def test_loss_credits_nothing(self) -> None:
        position = make_position(
            MARKET, side=PMSide.YES, contracts=10, avg_price_cents=43.0
        ).model_copy(update={"venue": "polymarket"})
        trades, closer, ledger = FakePMTradeStore(), FakePositionStore(), FakeLedger()
        settlements, realized = settle_sim_positions(
            [position], {MARKET: False}, trades, closer, ledger, NOW
        )
        assert (settlements, realized) == (1, pytest.approx(-430.0))
        assert ledger.cash_cents() == pytest.approx(0.0)
        assert closer.closed == [("polymarket", MARKET)]

    def test_no_side_wins_when_market_resolves_no(self) -> None:
        position = make_position(
            MARKET, side=PMSide.NO, contracts=5, avg_price_cents=58.0
        ).model_copy(update={"venue": "polymarket"})
        trades, closer, ledger = FakePMTradeStore(), FakePositionStore(), FakeLedger()
        _n, realized = settle_sim_positions(
            [position], {MARKET: False}, trades, closer, ledger, NOW
        )
        assert realized == pytest.approx(5 * (100.0 - 58.0))
        assert ledger.cash_cents() == pytest.approx(500.0)

    def test_settling_twice_is_a_noop(self) -> None:
        position = make_position(
            MARKET, side=PMSide.YES, contracts=10, avg_price_cents=43.0
        ).model_copy(update={"venue": "polymarket"})
        trades, closer, ledger = FakePMTradeStore(), FakePositionStore(), FakeLedger()
        settle_sim_positions([position], {MARKET: True}, trades, closer, ledger, NOW)
        again, realized = settle_sim_positions(
            [position], {MARKET: True}, trades, closer, ledger, NOW
        )
        assert (again, realized) == (0, 0.0)
        assert len(trades.trades) == 1
        assert len(closer.closed) == 1
        assert ledger.cash_cents() == pytest.approx(1_000.0)

    def test_unresolved_markets_are_held(self) -> None:
        position = make_position(MARKET).model_copy(update={"venue": "polymarket"})
        trades, closer, ledger = FakePMTradeStore(), FakePositionStore(), FakeLedger()
        settlements, realized = settle_sim_positions([position], {}, trades, closer, ledger, NOW)
        assert (settlements, realized) == (0, 0.0)
        assert trades.trades == [] and closer.closed == []


class TestNoRealExecutionSurface:
    """ADR-0011 hard constraint: the sim module cannot reach a trading venue."""

    def test_module_has_no_transport(self) -> None:
        import dynamo.execution.polymarket_sim as sim

        source = inspect.getsource(sim)
        assert "httpx" not in source
        assert "requests" not in source
        assert not hasattr(sim, "httpx")

    def test_broker_holds_no_client_or_credentials(self) -> None:
        broker = make_broker()
        for attr in vars(broker):
            assert "client" not in attr
            assert "key" not in attr
            assert "url" not in attr

    def test_polymarket_is_a_simulated_venue(self) -> None:
        assert is_simulated_venue("polymarket")
        assert not is_simulated_venue("kalshi")

    def test_sim_scope_is_distinct_from_kalshi(self) -> None:
        assert pm_sim_scope(MARKET) == f"POLYMARKET:{MARKET}"
