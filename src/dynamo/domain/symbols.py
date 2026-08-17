"""Symbol conventions shared across the platform (ADR-0013 crypto sleeve).

Canonical internal representation for crypto pairs is the slash form used by
Alpaca's trading and crypto-data APIs: ``"BTC/USD"``. Equities and OCC option
symbols never contain a slash, so ``"/" in symbol`` is the discriminator.

Alpaca's *positions* and some order payloads return crypto symbols without
the slash (``"BTCUSD"``); adapters normalize those back to canonical form at
the edge via :func:`normalize_crypto_symbol` — nothing outside an adapter
should ever see the slashless form.

Domain-pure on purpose: the broker, order manager, risk sizing, portfolio
constructor, and reporting all need this test, and everything may import
domain.
"""

from __future__ import annotations

# Quote currencies Alpaca supports for crypto pairs; used to re-insert the
# slash when a broker payload returns the concatenated form.
CRYPTO_QUOTE_CURRENCIES: tuple[str, ...] = ("USDT", "USDC", "USD", "BTC")


def is_crypto_symbol(symbol: str) -> bool:
    """True for canonical crypto pairs (``"BTC/USD"``).

    Equities (``"AAPL"``, ``"BRK.B"``) and OCC option symbols never contain
    ``"/"``, so the slash is a sufficient discriminator for canonical symbols.
    """
    return "/" in symbol


def normalize_crypto_symbol(symbol: str) -> str:
    """Broker form -> canonical: ``"BTCUSD"`` -> ``"BTC/USD"``.

    Already-canonical input is returned unchanged. Input that does not end in
    a known quote currency (i.e. is not a crypto pair) is returned unchanged —
    call this only on symbols known to be crypto (e.g. rows whose
    ``asset_class`` is ``"crypto"``); equity symbols that merely end in "USD"
    would otherwise be mangled.
    """
    if is_crypto_symbol(symbol):
        return symbol
    for quote in CRYPTO_QUOTE_CURRENCIES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[: -len(quote)]}/{quote}"
    return symbol
