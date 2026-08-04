"""Kill-switch round-trip through the API with a fake gate."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dynamo.api.app import create_app

from .conftest import FakeAdminStore, FakeKillSwitch, make_deps


@pytest.fixture
def kill() -> FakeKillSwitch:
    return FakeKillSwitch()


@pytest.fixture
def admin() -> FakeAdminStore:
    return FakeAdminStore()


@pytest.fixture
def client(kill: FakeKillSwitch, admin: FakeAdminStore) -> TestClient:
    return TestClient(create_app(deps=make_deps(kill=kill, admin=admin)))


def _toggle(client: TestClient, engage: bool, note: str = "") -> dict[str, Any]:
    resp = client.post(
        "/killswitch",
        headers={"X-Dynamo-Confirm": "killswitch"},
        json={"engage": engage, "note": note},
    )
    assert resp.status_code == 200
    return resp.json()


def test_round_trip(client: TestClient, kill: FakeKillSwitch, admin: FakeAdminStore) -> None:
    assert _toggle(client, True, "manual halt") == {"engaged": True, "note": "manual halt"}
    assert kill.engaged()
    assert _toggle(client, False, "all clear") == {"engaged": False, "note": "all clear"}
    assert not kill.engaged()
    assert kill.log == ["engage:owner:manual halt", "disengage:all clear"]
    assert [a for a, _ in admin.audits] == ["killswitch", "killswitch"]


def test_engage_is_idempotent(client: TestClient, kill: FakeKillSwitch) -> None:
    _toggle(client, True)
    _toggle(client, True)
    assert kill.log == ["engage:owner:"]  # second engage is a no-op
    _toggle(client, False, "ok")
    _toggle(client, False, "ok")
    assert kill.log == ["engage:owner:", "disengage:ok"]
