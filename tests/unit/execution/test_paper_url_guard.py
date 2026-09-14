"""Non-LIVE environments must never construct a live Alpaca client.

Paper vs live is decided by ``alpaca_base_url``, but the environment out-votes
the URL: outside ``DYNAMO_ENV=LIVE`` a non-paper base URL is a refused
misconfiguration, not a routing choice. Injected-client construction (the test
path) is untouched — the guard sits exactly where the real ``TradingClient``
would be built.
"""

from __future__ import annotations

import pytest

from dynamo.config.settings import Settings
from dynamo.execution.alpaca_broker import AlpacaBroker
from dynamo.execution.options_broker import AlpacaOptionsBroker

LIVE_URL = "https://api.alpaca.markets"


def make_settings(env: str, base_url: str) -> Settings:
    values: dict[str, object] = {
        "DYNAMO_ENV": env,
        "ALPACA_BASE_URL": base_url,
        "ALPACA_API_KEY": "key",
        "ALPACA_API_SECRET": "secret",
        "_env_file": None,
    }
    return Settings(**values)  # type: ignore[call-arg]


@pytest.mark.parametrize("env", ["DEV", "TEST", "PAPER"])
@pytest.mark.parametrize("broker_cls", [AlpacaBroker, AlpacaOptionsBroker])
def test_non_live_env_refuses_live_base_url(env: str, broker_cls: type) -> None:
    with pytest.raises(ValueError, match="not the paper endpoint"):
        broker_cls(make_settings(env, LIVE_URL))


@pytest.mark.parametrize("broker_cls", [AlpacaBroker, AlpacaOptionsBroker])
def test_paper_env_accepts_paper_base_url(broker_cls: type) -> None:
    broker = broker_cls(make_settings("PAPER", "https://paper-api.alpaca.markets"))
    assert broker is not None
