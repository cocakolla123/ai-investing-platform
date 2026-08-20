"""Crypto symbol convention helpers (ADR-0013, dynamo.domain.symbols)."""

from __future__ import annotations

import pytest

from dynamo.domain.symbols import is_crypto_symbol, normalize_crypto_symbol


@pytest.mark.parametrize("symbol", ["BTC/USD", "ETH/USD", "SOL/USDT", "AAVE/BTC"])
def test_canonical_crypto_pairs_are_crypto(symbol: str) -> None:
    assert is_crypto_symbol(symbol)


@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "SPY", "BTCUSD", "AAPL260918C00230000"])
def test_non_slash_symbols_are_not_crypto(symbol: str) -> None:
    # NOTE: "BTCUSD" is the broker form, NOT canonical — adapters must
    # normalize before anything downstream sees it.
    assert not is_crypto_symbol(symbol)


@pytest.mark.parametrize(
    ("broker", "canonical"),
    [
        ("BTCUSD", "BTC/USD"),
        ("ETHUSD", "ETH/USD"),
        ("LINKUSD", "LINK/USD"),
        ("SOLUSDT", "SOL/USDT"),
        ("AAVEUSDC", "AAVE/USDC"),
        ("ETHBTC", "ETH/BTC"),
        ("BTC/USD", "BTC/USD"),  # already canonical -> unchanged
    ],
)
def test_normalize_broker_form(broker: str, canonical: str) -> None:
    assert normalize_crypto_symbol(broker) == canonical


def test_normalize_leaves_non_crypto_unchanged() -> None:
    # Only called on rows known to be crypto, but must not mangle a bare
    # quote-currency string or a symbol that IS a quote currency.
    assert normalize_crypto_symbol("USD") == "USD"
    assert normalize_crypto_symbol("AAPL") == "AAPL"
