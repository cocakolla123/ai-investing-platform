"""Admin actions: confirm-header enforcement, idempotency, audit rows."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from dynamo.api.app import create_app

from .conftest import FakeAdminStore, FakeKillSwitch, make_deps


@pytest.fixture
def admin() -> FakeAdminStore:
    store = FakeAdminStore()
    store.alerts[1] = {
        "id": 1,
        "name": "universe_change_recommendation",
        "message": "ADD NVDA to core universe: sustained tier-2 interest",
        "severity": "info",
        "ackable": True,
        "acked_at": None,
    }
    store.alerts[2] = {
        "id": 2,
        "name": "universe_change_recommendation",
        "message": "DEMOTE IBM from core universe: liquidity floor failed 60d",
        "severity": "info",
        "ackable": True,
        "acked_at": None,
    }
    store.alerts[3] = {
        "id": 3,
        "name": "budget_70pct",
        "message": "Claude spend at 70%",
        "severity": "warn",
        "ackable": True,
        "acked_at": None,
    }
    store.alerts[4] = {
        "id": 4,
        "name": "kill_switch_engaged",
        "message": "engaged",
        "severity": "page",
        "ackable": False,
        "acked_at": None,
    }
    store.target_portfolios["tp-1"] = {
        "id": "tp-1",
        "payload": {"weights": []},
        "approved_at": None,
        "approved_by": None,
    }
    store.improvements["ip-1"] = {"id": "ip-1", "status": "open"}
    return store


@pytest.fixture
def kill() -> FakeKillSwitch:
    return FakeKillSwitch()


@pytest.fixture
def client(admin: FakeAdminStore, kill: FakeKillSwitch) -> TestClient:
    return TestClient(create_app(deps=make_deps(admin=admin, kill=kill)))


def _post(client: TestClient, url: str, confirm: str | None, body: dict[str, Any] | None = None):
    headers = {"X-Dynamo-Confirm": confirm} if confirm else {}
    return client.post(url, headers=headers, json=body)


# --- Confirm header ---


@pytest.mark.parametrize(
    ("url", "confirm"),
    [
        ("/universe/queue/1/approve", "universe-approve"),
        ("/universe/queue/1/reject", "universe-reject"),
        ("/rebalance/tp-1/approve", "rebalance-approve"),
        ("/rebalance/tp-1/reject", "rebalance-reject"),
        ("/improvements/ip-1/approve", "improvement-approve"),
        ("/improvements/ip-1/reject", "improvement-reject"),
        ("/alerts/3/ack", "alert-ack"),
    ],
)
def test_confirm_header_enforced(client: TestClient, url: str, confirm: str) -> None:
    missing = _post(client, url, None)
    assert missing.status_code == 400
    assert confirm in missing.json()["detail"]
    wrong = _post(client, url, "something-else")
    assert wrong.status_code == 400
    ok = _post(client, url, confirm)
    assert ok.status_code == 200


def test_killswitch_confirm_header_enforced(client: TestClient) -> None:
    resp = _post(client, "/killswitch", None, {"engage": True, "note": "n"})
    assert resp.status_code == 400


# --- Universe queue ---


def test_universe_approve_add(client: TestClient, admin: FakeAdminStore) -> None:
    resp = _post(client, "/universe/queue/1/approve", "universe-approve")
    assert resp.json()["status"] == "approved"
    assert admin.universe_calls == [{"symbol": "NVDA", "change": "add", "by": "owner"}]
    assert admin.alerts[1]["acked_at"] is not None
    assert ("universe_approve", {"alert_id": 1, "symbol": "NVDA", "change": "add"}) in admin.audits


def test_universe_approve_demote(client: TestClient, admin: FakeAdminStore) -> None:
    _post(client, "/universe/queue/2/approve", "universe-approve")
    assert admin.universe_calls[0]["change"] == "demote"
    assert admin.universe_calls[0]["symbol"] == "IBM"


def test_universe_approve_is_idempotent(client: TestClient, admin: FakeAdminStore) -> None:
    _post(client, "/universe/queue/1/approve", "universe-approve")
    again = _post(client, "/universe/queue/1/approve", "universe-approve")
    assert again.json()["status"] == "already_processed"
    assert len(admin.universe_calls) == 1
    assert len(admin.audits) == 1


def test_universe_reject_acks_without_universe_write(
    client: TestClient, admin: FakeAdminStore
) -> None:
    resp = _post(client, "/universe/queue/1/reject", "universe-reject")
    assert resp.json()["status"] == "rejected"
    assert admin.universe_calls == []
    assert admin.alerts[1]["acked_at"] is not None


def test_universe_approve_404_for_non_recommendation_alert(client: TestClient) -> None:
    assert _post(client, "/universe/queue/3/approve", "universe-approve").status_code == 404
    assert _post(client, "/universe/queue/99/approve", "universe-approve").status_code == 404


# --- Rebalance ---


def test_rebalance_approve_then_idempotent(client: TestClient, admin: FakeAdminStore) -> None:
    first = _post(client, "/rebalance/tp-1/approve", "rebalance-approve")
    assert first.json()["status"] == "approved"
    assert admin.target_portfolios["tp-1"]["approved_by"] == "owner"
    again = _post(client, "/rebalance/tp-1/approve", "rebalance-approve")
    assert again.json()["status"] == "already_approved"
    assert len(admin.audits) == 1


def test_rebalance_reject_conflicts_with_prior_approve(client: TestClient) -> None:
    _post(client, "/rebalance/tp-1/approve", "rebalance-approve")
    assert _post(client, "/rebalance/tp-1/reject", "rebalance-reject").status_code == 409


def test_rebalance_reject_marks_payload(client: TestClient, admin: FakeAdminStore) -> None:
    resp = _post(client, "/rebalance/tp-1/reject", "rebalance-reject")
    assert resp.json()["status"] == "rejected"
    assert "rejected" in admin.target_portfolios["tp-1"]["payload"]
    # approve after reject is a conflict; reject again is idempotent
    assert _post(client, "/rebalance/tp-1/approve", "rebalance-approve").status_code == 409
    again = _post(client, "/rebalance/tp-1/reject", "rebalance-reject")
    assert again.json()["status"] == "already_rejected"


# --- Improvements ---


def test_improvement_approve_and_idempotency(client: TestClient, admin: FakeAdminStore) -> None:
    assert (
        _post(client, "/improvements/ip-1/approve", "improvement-approve").json()["status"]
        == "accepted"
    )
    assert admin.improvements["ip-1"]["status"] == "accepted"
    again = _post(client, "/improvements/ip-1/approve", "improvement-approve")
    assert again.json()["note"] == "already_processed"
    # cross-action after terminal status conflicts
    assert _post(client, "/improvements/ip-1/reject", "improvement-reject").status_code == 409
    assert len(admin.audits) == 1


def test_improvement_unknown_404(client: TestClient) -> None:
    assert _post(client, "/improvements/nope/approve", "improvement-approve").status_code == 404


# --- Alert ack ---


def test_alert_ack_and_idempotency(client: TestClient, admin: FakeAdminStore) -> None:
    assert _post(client, "/alerts/3/ack", "alert-ack").json()["status"] == "acked"
    assert admin.alerts[3]["acked_at"] is not None
    assert _post(client, "/alerts/3/ack", "alert-ack").json()["status"] == "already_acked"
    assert admin.audits == [("alert_ack", {"alert_id": 3})]


def test_page_class_alert_not_ackable(client: TestClient) -> None:
    resp = _post(client, "/alerts/4/ack", "alert-ack")
    assert resp.status_code == 409  # clears only when the condition clears (doc 11 §4)
