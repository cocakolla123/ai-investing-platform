"""Shared HTTP plumbing tests: typed errors, retry-once, 429 handling."""

from __future__ import annotations

import httpx
import pytest

from dynamo.data.errors import ProviderDown, RateLimited
from dynamo.data.http import get_json


def test_ok_json() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    )
    assert get_json(client, "https://x.test/a", provider="p") == {"ok": True}


def test_429_raises_rate_limited_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "7"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RateLimited) as exc_info:
        get_json(client, "https://x.test/a", provider="p")
    assert calls["n"] == 1  # a 429 is an incident, not a retry storm (doc 02 §1)
    assert exc_info.value.retry_after_s == 7.0


def test_5xx_retried_once_then_provider_down() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderDown):
        get_json(client, "https://x.test/a", provider="p")
    assert calls["n"] == 2  # exactly 1 retry (doc 01 §6)


def test_5xx_then_success_recovers() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] == 1 else httpx.Response(200, json=[1])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert get_json(client, "https://x.test/a", provider="p") == [1]


def test_4xx_raises_provider_down_immediately() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
    with pytest.raises(ProviderDown):
        get_json(client, "https://x.test/a", provider="p")
