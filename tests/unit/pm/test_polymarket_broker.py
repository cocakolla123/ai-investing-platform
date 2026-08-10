"""PolymarketBroker (ADR-0012): configured chain, order/token mapping, the
per-order USD cap, typed error mapping, redaction, positions/cancel mapping.
All CLOB traffic is a fake client — no network, no py-clob-client needed."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from dynamo.config.settings import Settings
from dynamo.domain.models.pm import PMSide
from dynamo.execution.errors import (
    BrokerRejection,
    BrokerUnavailable,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)
from dynamo.execution.polymarket_broker import PolymarketBroker
from tests.unit.pm.helpers import make_pm_order

YES_TOKEN = "111111"
NO_TOKEN = "222222"
SECRETS = {"k3y-secret", "s3cret-secret", "p4ss-secret", "0xdeadbeefwalletkey"}


def resolver(market_id: str) -> tuple[str, str] | None:
    return (YES_TOKEN, NO_TOKEN) if market_id == "500001" else None


class FakeClobClient:
    """Records calls; scripted responses per method."""

    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Any]] = []

    def _respond(self, name: str, arg: Any = None) -> Any:
        self.calls.append((name, arg))
        value = self.responses.get(name)
        if callable(value):
            return value()
        return value

    def create_order(self, args: Any) -> Any:
        self.calls.append(("create_order", args))
        value = self.responses.get("create_order", {"signed": True})
        if callable(value):
            return value()
        return value

    def post_order(self, signed: Any, order_type: Any) -> Any:
        return self._respond("post_order", (signed, order_type))

    def cancel(self, order_id: str) -> Any:
        return self._respond("cancel", order_id)

    def get_order(self, order_id: str) -> Any:
        return self._respond("get_order", order_id)

    def get_orders(self, params: Any) -> Any:
        return self._respond("get_orders", params)

    def get_balance_allowance(self, params: Any) -> Any:
        return self._respond("get_balance_allowance", params)


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "DYNAMO_ENV": "LIVE",
        "POLYMARKET_API_KEY": "k3y-secret",
        "POLYMARKET_API_SECRET": "s3cret-secret",
        "POLYMARKET_PASSPHRASE": "p4ss-secret",
        "POLYMARKET_WALLET_KEY_PATH": "/tmp/nonexistent-wallet-key",
        "POLYMARKET_FUNDER_ADDRESS": "0xfunder",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[call-arg]


def make_broker(
    client: FakeClobClient | None = None,
    *,
    lookup: Any = None,
    positions_payload: Any = None,
    market_ids_hint: tuple[str, ...] = (),
    **setting_overrides: Any,
) -> PolymarketBroker:
    def positions_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=positions_payload or [])

    return PolymarketBroker(
        settings(**setting_overrides),
        token_resolver=resolver,
        broker_order_lookup=lookup,
        market_ids_hint=market_ids_hint,
        client=client if client is not None else FakeClobClient(),
        positions_client=httpx.Client(transport=httpx.MockTransport(positions_handler)),
        sleep=lambda _s: None,
    )


class TestConfigured:
    def test_fully_configured_with_injected_client(self, tmp_path) -> None:
        key_file = tmp_path / "wallet.key"
        key_file.write_text("0xdeadbeefwalletkey")
        broker = make_broker(POLYMARKET_WALLET_KEY_PATH=str(key_file))
        assert broker.configured

    @pytest.mark.parametrize(
        "override, expected",
        [
            ({"POLYMARKET_API_KEY": ""}, "credentials"),
            ({"POLYMARKET_API_SECRET": ""}, "credentials"),
            ({"POLYMARKET_PASSPHRASE": ""}, "credentials"),
            ({"POLYMARKET_WALLET_KEY_PATH": ""}, "wallet_key_path"),
            ({"POLYMARKET_FUNDER_ADDRESS": ""}, "funder"),
        ],
    )
    def test_each_missing_credential_unconfigures(self, override, expected) -> None:
        broker = make_broker(**override)
        assert not broker.configured
        assert expected in str(broker.configured_detail())

    def test_direct_eoa_needs_no_funder(self) -> None:
        # signature_type 0 = direct EOA; the funder proxy is a 1/2 concept.
        broker = make_broker(POLYMARKET_FUNDER_ADDRESS="", polymarket_signature_type=0)
        assert broker.configured

    def test_missing_wallet_file_unconfigures_without_injected_client(self) -> None:
        broker = PolymarketBroker(settings(), token_resolver=resolver)
        assert not broker.configured  # extra missing and/or wallet file absent

    def test_missing_extra_reported(self, monkeypatch, tmp_path) -> None:
        key_file = tmp_path / "wallet.key"
        key_file.write_text("0xdeadbeefwalletkey")
        import dynamo.execution.polymarket_broker as mod

        monkeypatch.setattr(mod, "_clob_import_error", lambda: "No module named 'py_clob_client'")
        broker = PolymarketBroker(
            settings(POLYMARKET_WALLET_KEY_PATH=str(key_file)), token_resolver=resolver
        )
        assert not broker.configured
        assert "py-clob-client" in str(broker.configured_detail())


class TestSubmit:
    def test_maps_order_to_clob_token_price_size(self) -> None:
        client = FakeClobClient(post_order={"success": True, "orderID": "clob-1"})
        broker = make_broker(client)
        order = make_pm_order(market_id="500001", side=PMSide.YES, contracts=10,
                              limit_price_cents=43)
        assert broker.submit(order) == "clob-1"
        name, args = client.calls[0]
        assert name == "create_order"
        payload = args if isinstance(args, dict) else vars(args)
        assert payload["token_id"] == YES_TOKEN
        assert payload["price"] == pytest.approx(0.43)  # cents → 0-1
        assert payload["size"] == 10.0

    def test_no_side_uses_no_token(self) -> None:
        client = FakeClobClient(post_order={"success": True, "orderID": "clob-2"})
        broker = make_broker(client)
        order = make_pm_order(market_id="500001", side=PMSide.NO, contracts=5,
                              limit_price_cents=57)
        broker.submit(order)
        _name, args = client.calls[0]
        payload = args if isinstance(args, dict) else vars(args)
        assert payload["token_id"] == NO_TOKEN

    def test_unknown_market_token_rejects(self) -> None:
        broker = make_broker(FakeClobClient())
        order = make_pm_order(market_id="unknown", side=PMSide.YES)
        with pytest.raises(BrokerRejection, match="no_clob_token_id"):
            broker.submit(order)

    def test_per_order_usd_cap_rejects_inside_broker(self) -> None:
        client = FakeClobClient(post_order={"success": True, "orderID": "x"})
        broker = make_broker(client)  # default cap $25
        # 60 contracts * 43c = $25.80 > $25 → reject before any CLOB call.
        order = make_pm_order(market_id="500001", contracts=60, limit_price_cents=43)
        with pytest.raises(BrokerRejection, match="polymarket_live_max_order_usd"):
            broker.submit(order)
        assert client.calls == []

    def test_at_cap_is_allowed(self) -> None:
        client = FakeClobClient(post_order={"success": True, "orderID": "y"})
        broker = make_broker(client)
        order = make_pm_order(market_id="500001", contracts=50, limit_price_cents=50)
        assert broker.submit(order) == "y"  # exactly $25.00

    def test_known_client_order_id_raises_duplicate_not_reorder(self) -> None:
        client = FakeClobClient()
        broker = make_broker(client, lookup=lambda coid: "clob-known")
        order = make_pm_order(market_id="500001")
        with pytest.raises(DuplicateClientOrderId):
            broker.submit(order)
        assert client.calls == []  # never resubmitted: a second real order

    def test_clob_reject_is_terminal(self) -> None:
        client = FakeClobClient(post_order={"success": False, "errorMsg": "not enough balance"})
        broker = make_broker(client)
        order = make_pm_order(market_id="500001")
        with pytest.raises(BrokerRejection, match="not enough balance"):
            broker.submit(order)


class _ApiError(Exception):
    def __init__(self, status_code: int | None, msg: str = "boom") -> None:
        super().__init__(msg)
        self.status_code = status_code


class TestErrorDiscipline:
    def _submit(self, error: Exception, attempts: dict) -> Any:
        def failing() -> Any:
            attempts["n"] += 1
            raise error

        client = FakeClobClient(create_order=failing)
        broker = make_broker(client)
        return broker.submit(make_pm_order(market_id="500001"))

    def test_429_is_rate_limited_no_retry(self) -> None:
        attempts = {"n": 0}
        with pytest.raises(RateLimited):
            self._submit(_ApiError(429), attempts)
        assert attempts["n"] == 1

    def test_4xx_terminal_no_retry(self) -> None:
        attempts = {"n": 0}
        with pytest.raises(BrokerRejection):
            self._submit(_ApiError(400, "bad order"), attempts)
        assert attempts["n"] == 1

    def test_5xx_one_retry_then_halt(self) -> None:
        attempts = {"n": 0}
        with pytest.raises(HaltSignal):
            self._submit(_ApiError(500), attempts)
        assert attempts["n"] == 2  # exactly one jittered retry

    def test_transport_fault_one_retry_then_unavailable(self) -> None:
        attempts = {"n": 0}
        with pytest.raises(BrokerUnavailable):
            self._submit(ConnectionError("reset"), attempts)
        assert attempts["n"] == 2


class TestCancelAndOrders:
    def test_cancel_resolves_broker_id_from_lookup(self) -> None:
        client = FakeClobClient(cancel={"canceled": True})
        broker = make_broker(client, lookup=lambda coid: "clob-9")
        broker.cancel("pm-1")
        assert ("cancel", "clob-9") in client.calls

    def test_cancel_unknown_order_rejects(self) -> None:
        broker = make_broker(FakeClobClient())
        with pytest.raises(BrokerRejection, match="unknown order"):
            broker.cancel("pm-unknown")

    def test_get_order_normalizes_clob_payload(self) -> None:
        client = FakeClobClient(
            get_order={
                "id": "clob-5",
                "status": "LIVE",
                "market": "0xcondition",
                "side": "BUY",
                "original_size": "10",
                "size_matched": "4",
                "price": "0.43",
            }
        )
        broker = make_broker(client, lookup=lambda coid: "clob-5")
        normalized = broker.get_order("pm-1")
        assert normalized["broker_order_id"] == "clob-5"
        assert normalized["status"] == "new"
        assert normalized["contracts"] == 10
        assert normalized["filled_contracts"] == 4
        assert normalized["filled_avg_price_cents"] == pytest.approx(43.0)

    def test_get_order_matched_maps_to_filled(self) -> None:
        client = FakeClobClient(
            get_order={"id": "c", "status": "MATCHED", "original_size": 3,
                       "size_matched": 3, "price": 0.5}
        )
        broker = make_broker(client, lookup=lambda coid: "c")
        assert broker.get_order("pm-1")["status"] == "filled"

    def test_open_orders_normalized(self) -> None:
        client = FakeClobClient(
            get_orders=[{"id": "a", "status": "LIVE", "original_size": 1,
                         "size_matched": 0, "price": 0.2}]
        )
        broker = make_broker(client)
        orders = broker.open_orders()
        assert len(orders) == 1
        assert orders[0]["status"] == "new"


class TestPositionsAndBalance:
    def test_positions_reverse_map_tokens_to_markets(self) -> None:
        payload = [
            {"asset": YES_TOKEN, "size": 10, "avgPrice": 0.4, "outcome": "Yes"},
            {"asset": "999", "size": 3, "avgPrice": 0.6, "outcome": "No"},
            {"asset": NO_TOKEN, "size": 0, "avgPrice": 0.5, "outcome": "No"},  # flat
        ]
        broker = make_broker(
            FakeClobClient(), positions_payload=payload, market_ids_hint=("500001",)
        )
        positions = broker.positions()
        assert len(positions) == 2  # zero-size dropped
        mapped = positions[0]
        assert mapped["market_id"] == "500001"
        assert mapped["side"] == "yes"
        assert mapped["contracts"] == 10
        assert mapped["exposure_cents"] == pytest.approx(400.0)
        unknown = positions[1]
        assert unknown["market_id"] == "token:999"  # surfaced, not dropped
        assert unknown["side"] == "no"

    def test_balance_usdc_base_units_to_cents(self) -> None:
        client = FakeClobClient(get_balance_allowance={"balance": "12500000"})  # $12.50
        broker = make_broker(client)
        assert broker.balance()["balance_cents"] == pytest.approx(1250.0)


class _RecordingStore:
    def __init__(self) -> None:
        self.runs: list[tuple] = []

    def record_run(self, ts, scope, result, divergences) -> int:
        self.runs.append((ts, scope, result, list(divergences)))
        return len(self.runs)

    def record_ack(self, run_id, acked_by, acked_at, note) -> None:  # pragma: no cover
        pass


class TestLiveReconciliation:
    """CLOB truth vs internal positions: mismatch → sticky POLYMARKET: halt."""

    def _reconcile(self, positions_payload, internal):
        from datetime import UTC, datetime

        from dynamo.execution.pm_order_manager import PMReconciler
        from dynamo.execution.polymarket_sim import pm_sim_scope

        broker = make_broker(
            FakeClobClient(),
            positions_payload=positions_payload,
            market_ids_hint=("500001",),
        )
        alerts: list[tuple[str, str]] = []
        store = _RecordingStore()
        reconciler = PMReconciler(
            broker,
            store,
            lambda name, msg: alerts.append((name, msg)),
            scope_fn=pm_sim_scope,
            run_scope="pm_live",
        )
        result = reconciler.reconcile(datetime(2026, 8, 10, tzinfo=UTC), internal)
        return result, reconciler, store, alerts

    def test_matching_positions_are_clean(self) -> None:
        from tests.unit.pm.helpers import make_position

        payload = [{"asset": YES_TOKEN, "size": 10, "avgPrice": 0.4, "outcome": "Yes"}]
        internal = [make_position(market_id="500001", contracts=10)]
        result, _reconciler, store, alerts = self._reconcile(payload, internal)
        assert result.clean
        assert alerts == []
        assert store.runs[0][1] == "pm_live"

    def test_mismatch_engages_sticky_polymarket_halt(self) -> None:
        from dynamo.execution.polymarket_sim import pm_sim_scope
        from tests.unit.pm.helpers import make_position

        payload = [{"asset": YES_TOKEN, "size": 25, "avgPrice": 0.4, "outcome": "Yes"}]
        internal = [make_position(market_id="500001", contracts=10)]
        result, reconciler, store, alerts = self._reconcile(payload, internal)
        assert not result.clean
        assert reconciler.halts.is_halted(pm_sim_scope("500001"))
        assert result.divergences[0].scope == "POLYMARKET:500001"
        # Run scope stays under the sticky pm* umbrella that blocks new orders.
        assert store.runs[0][1] == "pm_live:markets"
        assert store.runs[0][1].startswith("pm")
        assert alerts and alerts[0][0] == "pm_reconciliation_divergence"


class TestRedaction:
    def test_repr_and_detail_contain_no_secrets(self, tmp_path) -> None:
        key_file = tmp_path / "wallet.key"
        key_file.write_text("0xdeadbeefwalletkey")
        broker = make_broker(POLYMARKET_WALLET_KEY_PATH=str(key_file))
        surfaces = repr(broker) + str(broker.configured_detail())
        for secret in SECRETS:
            assert secret not in surfaces
