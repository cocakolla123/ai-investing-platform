"""Learning test config: registers the integration marker."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires a real database (INTEGRATION_DB_URL)")
