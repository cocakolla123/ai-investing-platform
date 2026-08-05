"""Chain adapter parsing + broker request shaping (ADR-0010).

The installed alpaca-py surface is probed once at import; these tests inject
fake clients so they run without keys and pin the shapes we rely on.
"""

from __future__ import annotations

from typing import Any

from dynamo.data.options_chain import AlpacaOptionsChain, atm_iv
from dynamo.domain.models.options import OptionRight, occ_symbol
from dynamo.execution.options_broker import AlpacaOptionsBroker
from tests.unit.options.helpers import MONTHLY_EXPIRY, NOW, make_chain_row, make_decision


def test_installed_sdk_has_the_options_surface() -> None:
    """alpaca-py >= 0.43 ships options; the guarded import must have found it."""
    from dynamo.data import options_chain

    assert options_chain._SDK_AVAILABLE


class _FakeTradingClient:
    def __init__(self, contracts: list[dict[str, Any]]) -> None:
        self._contracts = contracts
        self.requests: list[Any] = []

    def get_option_contracts(self, request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {"option_contracts": self._contracts}


class _FakeDataClient:
    def __init__(self, snapshots: dict[str, dict[str, Any]]) -> None:
        self._snapshots = snapshots

    def get_option_snapshot(self, request: Any) -> dict[str, Any]:
        return {"snapshots": self._snapshots}


class TestChainSnapshot:
    def test_rows_parsed_from_contracts_and_snapshots(self) -> None:
        occ = occ_symbol("AAPL", MONTHLY_EXPIRY, OptionRight.CALL, 95.0)
        chain = AlpacaOptionsChain(
            "key",
            "secret",
            trading_client=_FakeTradingClient(
                [{"symbol": occ, "open_interest": 512}, {"symbol": "not-an-occ"}]
            ),
            data_client=_FakeDataClient(
                {
                    occ: {
                        "latestQuote": {"bp": 2.40, "ap": 2.50},
                        "impliedVolatility": 0.31,
                        "greeks": {"delta": 0.62},
                    }
                }
            ),
        )
        rows = chain.chain_snapshot("AAPL", 100.0, NOW.date(), MONTHLY_EXPIRY, NOW)
        assert len(rows) == 1  # the malformed OCC symbol is dropped
        row = rows[0]
        assert (row.underlying, row.expiry, row.right, row.strike) == (
            "AAPL",
            MONTHLY_EXPIRY,
            OptionRight.CALL,
            95.0,
        )
        assert (row.bid, row.ask, row.open_interest) == (2.40, 2.50, 512)
        assert row.iv == 0.31 and row.delta == 0.62

    def test_unavailable_without_clients_returns_empty(self) -> None:
        chain = AlpacaOptionsChain("", "")
        assert not chain.available
        assert chain.chain_snapshot("AAPL", 100.0, NOW.date(), MONTHLY_EXPIRY, NOW) == []

    def test_bad_spot_fails_closed(self) -> None:
        chain = AlpacaOptionsChain(
            "key",
            "secret",
            trading_client=_FakeTradingClient([]),
            data_client=_FakeDataClient({}),
        )
        assert chain.chain_snapshot("AAPL", float("nan"), NOW.date(), MONTHLY_EXPIRY, NOW) == []


class TestAtmIv:
    def test_median_of_nearest_strikes(self) -> None:
        rows = [
            make_chain_row(strike=95.0, iv=0.28),
            make_chain_row(strike=100.0, iv=0.30),
            make_chain_row(strike=105.0, iv=0.34),
        ]
        value = atm_iv(rows, 100.0)
        assert value is not None and 0.28 <= value <= 0.34

    def test_no_iv_fails_closed(self) -> None:
        assert atm_iv([make_chain_row(iv=None)], 100.0) is None
        assert atm_iv([], 100.0) is None
        assert atm_iv([make_chain_row(iv=0.3)], float("nan")) is None


class _RecordingTradingClient:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def submit_order(self, order_data: Any) -> dict[str, Any]:
        self.submitted.append(order_data)
        return {"id": "broker-1"}

    def get_all_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "AAPL",
                "qty": "100",
                "avg_entry_price": "180.0",
                "asset_class": "us_equity",
            },
            {
                "symbol": "AAPL260918C00095000",
                "qty": "-2",
                "avg_entry_price": "2.4",
                "asset_class": "us_option",
            },
        ]


class TestOptionsBroker:
    def test_submit_builds_a_day_limit_order(self) -> None:
        client = _RecordingTradingClient()
        broker = AlpacaOptionsBroker(client=client)  # type: ignore[arg-type]
        decision = make_decision(limit_price=2.505)
        from dynamo.domain.models.options import OptionOrder
        from dynamo.domain.models.portfolio import OrderState

        order = OptionOrder(
            client_order_id="opt-1",
            structure=decision.structure,
            side=decision.side,
            right=decision.right,
            occ_symbol=decision.occ_symbol,
            underlying=decision.underlying,
            expiry=decision.expiry,
            strike=decision.strike,
            contracts=decision.contracts,
            limit_price=decision.limit_price,
            state=OrderState.PROPOSED,
        )
        assert broker.submit(order) == "broker-1"
        request = client.submitted[0]
        assert request.symbol == decision.occ_symbol
        assert request.qty == 4
        assert request.limit_price == 2.5  # rounded to cents
        assert request.time_in_force.value == "day"  # options are DAY-only
        assert request.client_order_id == "opt-1"

    def test_positions_returns_only_options(self) -> None:
        broker = AlpacaOptionsBroker(client=_RecordingTradingClient())  # type: ignore[arg-type]
        positions = broker.positions()
        assert positions == [
            {"occ_symbol": "AAPL260918C00095000", "qty": -2.0, "avg_entry_price": 2.4}
        ]
