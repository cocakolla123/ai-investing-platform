"""KalshiBroker: env-based base URL, signed requests, typed error mapping
(rejects terminal, one jittered retry on 5xx then HaltSignal), normalization."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from dynamo.config.settings import Settings
from dynamo.domain.models.pm import PMSide
from dynamo.execution.errors import (
    BrokerRejection,
    DuplicateClientOrderId,
    HaltSignal,
    RateLimited,
)
from dynamo.execution.kalshi_broker import KalshiBroker
from tests.unit.pm.helpers import make_pm_order

DEMO = "https://demo-api.kalshi.co"
PROD = "https://api.elections.kalshi.com"


class FakeSigner:
    def __init__(self) -> None:
        self.messages: list[bytes] = []

    def sign(self, message: bytes) -> bytes:
        self.messages.append(message)
        return b"sig"


def make_broker(
    handler: Callable[[httpx.Request], httpx.Response],
    env: str = "PAPER",
    signer: FakeSigner | None = None,
) -> KalshiBroker:
    settings = Settings(
        DYNAMO_ENV=env,  # type: ignore[call-arg]
        KALSHI_API_KEY_ID="kid",  # type: ignore[call-arg]
        KALSHI_PRIVATE_KEY_PATH="/tmp/unused.pem",  # type: ignore[call-arg]
        _env_file=None,  # type: ignore[call-arg]
    )
    return KalshiBroker(
        settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        signer=signer or FakeSigner(),
        sleep=lambda _s: None,
    )


def ok(payload: dict) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(200, json=payload)


class TestEnvironmentSelection:
    def test_demo_unless_live(self) -> None:
        for env in ("DEV", "TEST", "PAPER"):
            assert make_broker(ok({}), env=env).base_url == DEMO

    def test_prod_only_in_live(self) -> None:
        assert make_broker(ok({}), env="LIVE").base_url == PROD

    def test_unconfigured_without_signer(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        broker = KalshiBroker(settings, client=httpx.Client())
        assert not broker.configured


class TestSubmit:
    def test_signed_limit_order_in_cents(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"order": {"order_id": "ord-1"}})

        signer = FakeSigner()
        broker = make_broker(handler, signer=signer)
        order = make_pm_order(contracts=100, limit_price_cents=43)
        assert broker.submit(order) == "ord-1"
        assert captured["url"].startswith(f"{DEMO}/trade-api/v2/portfolio/orders")
        body = captured["body"]
        assert body["client_order_id"] == order.client_order_id  # idempotency key
        assert body["type"] == "limit"
        assert body["yes_price"] == 43 and "no_price" not in body
        assert captured["headers"]["kalshi-access-key"] == "kid"
        # Signed message covers METHOD + full path.
        assert signer.messages[0].endswith(b"POST/trade-api/v2/portfolio/orders")

    def test_no_side_uses_no_price(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"order": {"order_id": "ord-2"}})

        broker = make_broker(handler)
        broker.submit(make_pm_order(side=PMSide.NO, limit_price_cents=57))
        assert captured["body"]["no_price"] == 57 and "yes_price" not in captured["body"]


class TestErrorDiscipline:
    def test_reject_is_terminal_no_retry(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": {"message": "market closed"}})

        broker = make_broker(handler)
        with pytest.raises(BrokerRejection):
            broker.submit(make_pm_order())
        assert calls["n"] == 1  # never auto-retried

    def test_duplicate_client_order_id(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"error": {"message": "duplicate client_order_id"}})

        broker = make_broker(handler)
        with pytest.raises(DuplicateClientOrderId):
            broker.submit(make_pm_order())

    def test_rate_limit_is_typed(self) -> None:
        broker = make_broker(lambda _r: httpx.Response(429, text="slow down"))
        with pytest.raises(RateLimited):
            broker.balance()

    def test_5xx_single_jittered_retry_then_halt(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="unavailable")

        broker = make_broker(handler)
        with pytest.raises(HaltSignal):
            broker.balance()
        assert calls["n"] == 2  # exactly one retry

    def test_5xx_then_success_recovers(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, text="blip")
            return httpx.Response(200, json={"balance": 123456})

        broker = make_broker(handler)
        assert broker.balance() == {"balance_cents": 123456.0}


class TestReads:
    def test_get_order_normalization(self) -> None:
        broker = make_broker(
            ok(
                {
                    "orders": [
                        {
                            "order_id": "ord-9",
                            "client_order_id": "c-9",
                            "ticker": "FED-26DEC",
                            "side": "yes",
                            "status": "executed",
                            "initial_count": 100,
                            "remaining_count": 0,
                            "yes_price": 43,
                        }
                    ]
                }
            )
        )
        remote = broker.get_order("c-9")
        assert remote["broker_order_id"] == "ord-9"
        assert remote["status"] == "filled"
        assert remote["filled_contracts"] == 100
        assert remote["filled_avg_price_cents"] == 43.0

    def test_get_order_unknown_raises_rejection(self) -> None:
        broker = make_broker(ok({"orders": []}))
        with pytest.raises(BrokerRejection):
            broker.get_order("nope")

    def test_positions_normalization(self) -> None:
        broker = make_broker(
            ok(
                {
                    "market_positions": [
                        {"ticker": "A", "position": 100, "market_exposure": 4300},
                        {"ticker": "B", "position": -50, "market_exposure": 2850},
                        {"ticker": "C", "position": 0},
                    ]
                }
            )
        )
        positions = broker.positions()
        assert positions == [
            {"market_id": "A", "side": "yes", "contracts": 100, "exposure_cents": 4300.0},
            {"market_id": "B", "side": "no", "contracts": 50, "exposure_cents": 2850.0},
        ]
