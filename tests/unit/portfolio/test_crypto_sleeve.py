"""Crypto sleeve in the constructor (ADR-0013): tagging + hard caps."""

from __future__ import annotations

from dynamo.domain.models.research import Action, Sleeve
from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import RuleBasedConstructor
from tests.unit.portfolio.builders import (
    advisory,
    make_cverdict,
    make_row,
    make_state,
    make_verdict,
)

RCFG = RiskConfig()
CONFIG = ConstructionConfig()
PAIRS = ("AVAX/USD", "BTC/USD", "ETH/USD", "LINK/USD", "SOL/USD")


def test_crypto_entries_tagged_crypto_and_collectively_capped() -> None:
    inputs = advisory(
        watchlist=[make_row(s, i + 1) for i, s in enumerate(PAIRS)],
        verdicts=[make_cverdict(s, Action.BUY, conviction=0.9) for s in PAIRS],
        sectors=dict.fromkeys(PAIRS, "crypto"),
    )
    target = RuleBasedConstructor().construct(
        inputs, make_verdict(), make_state({}), CONFIG, risk_config=RCFG
    )
    crypto_weights = [w for w in target.weights if "/" in w.symbol]
    assert crypto_weights, "crypto entries should be constructible"
    assert all(w.sleeve is Sleeve.CRYPTO for w in crypto_weights)
    assert all(w.weight <= RCFG.crypto_max_position + 1e-9 for w in crypto_weights)
    assert sum(w.weight for w in crypto_weights) <= RCFG.crypto_sleeve_cap + 1e-9


def test_retained_crypto_holding_is_kept_not_liquidated() -> None:
    """A held crypto position with no fresh verdict must be retained exactly
    like a long-term equity holding — not diffed to an exit."""
    state = make_state({"BTC/USD": 0.04}, sleeves={"BTC/USD": Sleeve.CRYPTO})
    inputs = advisory(symbols=("BTC/USD",), sectors={"BTC/USD": "crypto"})
    target = RuleBasedConstructor().construct(
        inputs, make_verdict(), state, CONFIG, risk_config=RCFG
    )
    by_symbol = {w.symbol: w for w in target.weights}
    assert "BTC/USD" in by_symbol
    assert by_symbol["BTC/USD"].sleeve is Sleeve.CRYPTO
    assert by_symbol["BTC/USD"].weight > 0.0


def test_oversized_retained_crypto_clamped_by_compliance_pass() -> None:
    """Belt-and-suspenders: even a RETAINED crypto lot above the per-pair cap
    is clamped in the verdict-compliance pass (crypto caps are hard)."""
    state = make_state({"BTC/USD": 0.09}, sleeves={"BTC/USD": Sleeve.CRYPTO})
    inputs = advisory(symbols=("BTC/USD",), sectors={"BTC/USD": "crypto"})
    target = RuleBasedConstructor().construct(
        inputs, make_verdict(), state, CONFIG, risk_config=RCFG
    )
    by_symbol = {w.symbol: w for w in target.weights}
    assert by_symbol["BTC/USD"].weight <= RCFG.crypto_max_position + 1e-9


def test_lotless_crypto_position_defaults_to_crypto_sleeve() -> None:
    """Positions synced from the broker may carry no lots; the sleeve default
    keys off the symbol form so crypto holdings stay in their sleeve."""
    from dynamo.domain.models.portfolio import PortfolioState, Position
    from dynamo.portfolio.constructor import _held_sleeves

    state = PortfolioState(
        as_of=make_state({}).as_of,
        equity=100_000.0,
        cash=90_000.0,
        positions=(
            Position(symbol="ETH/USD", qty=1.0, avg_entry_price=4_000.0),
            Position(symbol="AAPL", qty=10.0, avg_entry_price=230.0),
        ),
    )
    held = _held_sleeves(state)
    assert held["ETH/USD"] == {Sleeve.CRYPTO}
    assert held["AAPL"] == {Sleeve.LONG_TERM}
