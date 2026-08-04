"""Basic-auth behavior (doc 11 §1.4): single credential, constant-time."""

from __future__ import annotations

from fastapi.testclient import TestClient

from dynamo.api.app import create_app
from dynamo.config.settings import DynamoEnv

from .conftest import make_deps


def _client(**kw) -> TestClient:
    return TestClient(create_app(deps=make_deps(**kw)))


def test_401_without_credentials_when_password_set() -> None:
    client = _client(password="hunter2")
    resp = client.get("/watchlist")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.headers["www-authenticate"] == "Basic"


def test_401_with_wrong_password() -> None:
    client = _client(password="hunter2")
    assert client.get("/watchlist", auth=("dynamo", "nope")).status_code == 401


def test_401_with_wrong_username() -> None:
    client = _client(password="hunter2")
    assert client.get("/watchlist", auth=("admin", "hunter2")).status_code == 401


def test_200_with_correct_credentials() -> None:
    client = _client(password="hunter2")
    assert client.get("/watchlist", auth=("dynamo", "hunter2")).status_code == 200


def test_auth_disabled_when_password_unset_in_dev_and_test() -> None:
    for env in (DynamoEnv.DEV, DynamoEnv.TEST):
        client = _client(password=None, env=env)
        assert client.get("/watchlist").status_code == 200


def test_fails_closed_when_password_unset_in_paper() -> None:
    client = _client(password=None, env=DynamoEnv.PAPER)
    assert client.get("/watchlist").status_code == 401


def test_health_needs_no_auth() -> None:
    client = _client(password="hunter2")
    assert client.get("/health").status_code == 200
