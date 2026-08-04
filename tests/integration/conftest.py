"""Integration-test configuration: marker registration + gated DB engine."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires a live Postgres (set INTEGRATION_DB_URL to run)",
    )


@pytest.fixture(scope="session")
def integration_db_url() -> str:
    url = os.environ.get("INTEGRATION_DB_URL")
    if not url:
        pytest.skip("INTEGRATION_DB_URL not set")
    return url
