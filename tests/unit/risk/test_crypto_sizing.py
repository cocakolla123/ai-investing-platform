"""Crypto sleeve caps in the sizing cascade + PDT exemption (ADR-0013)."""

from __future__ import annotations

from datetime import date

from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.intraday import count_day_trades, pdt_relevant_fills
from dynamo.risk.sizing import (
    CLAMP_CRYPTO_POSITION,
    CLAMP_CRYPTO_SLEEVE,
    SizingCandidate,
    SizingContext,
    size_positions,
)


def crypto_candidate(symbol: str, sigma: float = 0.2) -> SizingCandidate:
    # Low sigma + high conviction on purpose: the vol-parity base would be
    # LARGE, so the crypto caps must be what binds.
    return SizingCandidate(
        symbol=symbol,
        sleeve=Sleeve.CRYPTO,
        conviction=0.9,
        sigma_annual=sigma,
        adv_dollar_20d=1e9,
    )


CFG = RiskConfig()
CTX = SizingContext(equity=100_000.0)


def test_single_crypto_position_capped_at_crypto_max_position() -> None:
    [sized] = size_positions([crypto_candidate("BTC/USD")], CFG, CTX)
    assert sized.weight <= CFG.crypto_max_position + 1e-12
    assert CLAMP_CRYPTO_POSITION in sized.clamps_applied


def test_crypto_positions_collectively_capped_at_sleeve_cap() -> None:
    candidates = [
        crypto_candidate(s) for s in ("AVAX/USD", "BTC/USD", "ETH/USD", "LINK/USD", "SOL/USD")
    ]
    sized = size_positions(candidates, CFG, CTX)
    total = sum(p.weight for p in sized)
    assert total <= CFG.crypto_sleeve_cap + 1e-9
    # 5 pairs x 5% per-pair would be 25% — the 10% sleeve cap must bind.
    assert any(CLAMP_CRYPTO_SLEEVE in p.clamps_applied for p in sized)


def test_existing_crypto_holdings_consume_the_sleeve_budget() -> None:
    ctx = SizingContext(
        equity=100_000.0,
        existing_weights={"BTC/USD": 0.08},  # 8% already held in crypto
    )
    [sized] = size_positions([crypto_candidate("ETH/USD")], CFG, ctx)
    assert sized.weight <= CFG.crypto_sleeve_cap - 0.08 + 1e-9
    assert CLAMP_CRYPTO_SLEEVE in sized.clamps_applied


def test_equity_candidates_never_hit_crypto_clamps() -> None:
    equity_candidate = SizingCandidate(
        symbol="AAPL",
        sleeve=Sleeve.LONG_TERM,
        conviction=0.9,
        sigma_annual=0.2,
        adv_dollar_20d=1e9,
    )
    [sized] = size_positions([equity_candidate], CFG, CTX)
    assert CLAMP_CRYPTO_POSITION not in sized.clamps_applied
    assert CLAMP_CRYPTO_SLEEVE not in sized.clamps_applied
    assert sized.weight > CFG.crypto_max_position  # equity cap (8%) > crypto cap


def test_higher_sigma_still_shrinks_crypto_naturally() -> None:
    # Vol parity: 4x sigma -> smaller base before any crypto cap applies.
    calm, wild = size_positions(
        [crypto_candidate("BTC/USD", sigma=0.2), crypto_candidate("ETH/USD", sigma=0.8)],
        CFG,
        CTX,
    )
    assert wild.base_weight < calm.base_weight


# --- PDT exemption (ADR-0013: crypto never counts as a day trade) ----------


def test_pdt_relevant_fills_drops_crypto() -> None:
    day = date(2026, 8, 17)
    fills = [
        ("BTC/USD", "buy", day),
        ("BTC/USD", "sell", day),
        ("AAPL", "buy", day),
        ("AAPL", "sell", day),
    ]
    relevant = pdt_relevant_fills(fills)
    assert all(sym == "AAPL" for sym, _, _ in relevant)
    # Same-day crypto round trip contributes zero day trades.
    assert count_day_trades(relevant) == 1
    assert count_day_trades(fills) == 2  # unfiltered would over-count
