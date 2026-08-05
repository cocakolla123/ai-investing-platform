"""PM domain entities: state-machine reuse, model guards, settlement math."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dynamo.domain.models.pm import (
    PMSide,
    settlement_pnl_cents,
)
from dynamo.domain.models.portfolio import ORDER_TRANSITIONS, OrderState
from dynamo.execution.mock_pm_broker import MockPMBroker, PMScriptStep
from tests.unit.pm.helpers import make_pm_order


class TestModels:
    def test_pm_order_reuses_the_equity_state_machine(self) -> None:
        order = make_pm_order()
        assert order.state is OrderState.PROPOSED
        assert OrderState.RISK_CHECKED in ORDER_TRANSITIONS[order.state]
        assert ORDER_TRANSITIONS[OrderState.FILLED] == frozenset()  # absorbing

    def test_contracts_and_price_bounds(self) -> None:
        with pytest.raises(ValidationError):
            make_pm_order(contracts=0)
        with pytest.raises(ValidationError):
            make_pm_order(limit_price_cents=0)
        with pytest.raises(ValidationError):
            make_pm_order(limit_price_cents=100)

    def test_frozen(self) -> None:
        order = make_pm_order()
        with pytest.raises(ValidationError):
            order.contracts = 5  # type: ignore[misc]


class TestSettlementPnl:
    def test_yes_wins(self) -> None:
        # 100 contracts at 43c: payout $100, cost $43 -> +$57 (5700 cents).
        assert settlement_pnl_cents(PMSide.YES, 100, 43.0, True) == 5700.0

    def test_yes_loses(self) -> None:
        assert settlement_pnl_cents(PMSide.YES, 100, 43.0, False) == -4300.0

    def test_no_wins_when_market_resolves_no(self) -> None:
        assert settlement_pnl_cents(PMSide.NO, 100, 57.0, False) == 4300.0

    def test_no_loses_when_market_resolves_yes(self) -> None:
        assert settlement_pnl_cents(PMSide.NO, 100, 57.0, True) == -5700.0

    def test_zero_contracts(self) -> None:
        assert settlement_pnl_cents(PMSide.YES, 0, 43.0, True) == 0.0


class TestMockPMBrokerIdempotency:
    def test_duplicate_client_order_id_returns_existing_order(self) -> None:
        broker = MockPMBroker(script=[PMScriptStep(behavior="accept")])
        order = make_pm_order()
        first = broker.submit(order)
        second = broker.submit(order)
        assert first == second
        assert broker.submit_calls == 2
        assert len(broker.open_orders()) == 1  # no duplicate order created

    def test_partial_then_force_fill_accumulates_position(self) -> None:
        broker = MockPMBroker(
            script=[PMScriptStep(behavior="partial", fill_contracts=40, fill_price_cents=43.0)]
        )
        order = make_pm_order(contracts=100)
        broker.submit(order)
        assert broker.get_order(order.client_order_id)["status"] == "partially_filled"
        broker.force_fill(order.client_order_id, price_cents=45.0)
        remote = broker.get_order(order.client_order_id)
        assert remote["status"] == "filled"
        assert remote["filled_contracts"] == 100
        position = broker.positions()[0]
        assert position["contracts"] == 100
        # Weighted average: (40*43 + 60*45) / 100 = 44.2
        assert position["exposure_cents"] == pytest.approx(4420.0)

    def test_balance_decreases_with_fills(self) -> None:
        broker = MockPMBroker(balance_cents=10_000.0)
        broker.submit(make_pm_order(contracts=100, limit_price_cents=43))  # default: fill
        assert broker.balance() == {"balance_cents": 10_000.0 - 4300.0}

    def test_cancel_is_idempotent_for_terminal_orders(self) -> None:
        broker = MockPMBroker()  # default script fills immediately
        order = make_pm_order()
        broker.submit(order)
        broker.cancel(order.client_order_id)  # no-op on a filled order
        assert broker.get_order(order.client_order_id)["status"] == "filled"
