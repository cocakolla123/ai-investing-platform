"""pm_favorites: band selection, fail-closed skips, the assumed-edge gate,
determinism. The selector may never discover an edge — it only stamps the
caller's study-demonstrated assumption onto band members."""

from __future__ import annotations

import math

from tests.unit.pm.helpers import NOW

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.risk.pm_arbitrage import kalshi_fee_per_contract
from dynamo.risk.pm_favorites import (
    SKIP_DEGENERATE_COST,
    SKIP_INVALID_INPUT,
    SKIP_INVALID_PRICE,
    SKIP_MARKET_RESOLVED,
    SKIP_NO_ASSUMED_EDGE,
    SKIP_NO_PRICE,
    SKIP_OUTSIDE_BAND,
    SKIP_UNTRUSTED,
    select_favorites,
)


def market(market_id: str, resolved: bool | None = None) -> PredictionMarket:
    return PredictionMarket(
        venue="kalshi", market_id=market_id, question=f"{market_id}?", resolved_outcome=resolved
    )


def price(market_id: str, implied: float, trusted: bool = True) -> PredictionMarketPrice:
    return PredictionMarketPrice(
        venue="kalshi", market_id=market_id, ts=NOW, implied_prob=implied, trusted=trusted
    )


def no_fee(_price: float) -> float:
    return 0.0


BAND = {"min_price": 0.90, "max_price": 0.99}


def test_selects_band_members_and_stamps_the_assumed_edge() -> None:
    markets = [market("M-A"), market("M-B"), market("M-C")]
    prices = {"M-A": price("M-A", 0.95), "M-B": price("M-B", 0.50), "M-C": price("M-C", 0.92)}
    result = select_favorites(markets, prices, **BAND, fee_fn=no_fee, min_net_edge_assumed=0.008)
    assert [c.market_id for c in result.candidates] == ["M-A", "M-C"]  # price descending
    assert ("M-B", SKIP_OUTSIDE_BAND) in result.skipped
    a = result.candidates[0]
    assert a.price == 0.95 and a.fee == 0.0 and a.cost == 0.95
    assert math.isclose(a.win_profit, 0.05)
    assert a.breakeven_prob == a.cost
    assert a.net_edge_assumed == 0.008  # stamped, not discovered
    assert a.rationale.startswith("fav:harvest M-A")


def test_kalshi_fee_raises_cost_and_breakeven() -> None:
    result = select_favorites(
        [market("M-A")],
        {"M-A": price("M-A", 0.95)},
        **BAND,
        fee_fn=kalshi_fee_per_contract,
        min_net_edge_assumed=0.01,
    )
    (c,) = result.candidates
    assert c.fee == kalshi_fee_per_contract(0.95) == 0.01
    assert math.isclose(c.cost, 0.96) and math.isclose(c.breakeven_prob, 0.96)


def test_zero_or_negative_assumed_edge_selects_nothing() -> None:
    markets = [market("M-A")]
    prices = {"M-A": price("M-A", 0.95)}
    for edge in (0.0, -0.02):
        result = select_favorites(markets, prices, **BAND, fee_fn=no_fee, min_net_edge_assumed=edge)
        assert result.candidates == ()
        assert result.skipped == (("*", SKIP_NO_ASSUMED_EDGE),)


def test_fail_closed_skip_reasons() -> None:
    markets = [
        market("M-RES", resolved=True),
        market("M-NOPRICE"),
        market("M-UNTRUSTED"),
        market("M-NAN"),
        market("M-FEEEATS"),
    ]
    prices = {
        "M-UNTRUSTED": price("M-UNTRUSTED", 0.95, trusted=False),
        "M-NAN": price("M-NAN", float("nan")),
        "M-FEEEATS": price("M-FEEEATS", 0.95),
    }
    result = select_favorites(
        markets,
        prices,
        **BAND,
        fee_fn=lambda _p: 0.06,  # 95c + 6c fee -> cost > $1 payout
        min_net_edge_assumed=0.01,
    )
    assert result.candidates == ()
    assert set(result.skipped) == {
        ("M-RES", SKIP_MARKET_RESOLVED),
        ("M-NOPRICE", SKIP_NO_PRICE),
        ("M-UNTRUSTED", SKIP_UNTRUSTED),
        ("M-NAN", SKIP_INVALID_PRICE),
        ("M-FEEEATS", SKIP_DEGENERATE_COST),
    }


def test_misbehaving_fee_fn_and_bad_band_fail_closed() -> None:
    markets = [market("M-A")]
    prices = {"M-A": price("M-A", 0.95)}
    bad_fee = select_favorites(
        markets, prices, **BAND, fee_fn=lambda _p: float("nan"), min_net_edge_assumed=0.01
    )
    assert bad_fee.candidates == ()
    assert bad_fee.skipped == (("M-A", SKIP_INVALID_INPUT),)
    for band in ({"min_price": 0.99, "max_price": 0.90}, {"min_price": 0.0, "max_price": 0.99}):
        bad_band = select_favorites(
            markets, prices, **band, fee_fn=no_fee, min_net_edge_assumed=0.01
        )
        assert bad_band.candidates == ()
        assert bad_band.skipped == (("*", SKIP_INVALID_INPUT),)


def test_band_edges_are_inclusive_and_first_row_per_market_wins() -> None:
    markets = [market("M-LO"), market("M-HI"), market("M-DUP"), market("M-DUP")]
    prices = {
        "M-LO": price("M-LO", 0.90),
        "M-HI": price("M-HI", 0.99),
        "M-DUP": price("M-DUP", 0.95),
    }
    result = select_favorites(markets, prices, **BAND, fee_fn=no_fee, min_net_edge_assumed=0.01)
    assert [c.market_id for c in result.candidates] == ["M-HI", "M-DUP", "M-LO"]


def test_determinism_over_input_order() -> None:
    markets = [market(f"M-{i}") for i in range(6)]
    prices = {m.market_id: price(m.market_id, 0.91 + 0.01 * i) for i, m in enumerate(markets)}
    forward = select_favorites(markets, prices, **BAND, fee_fn=no_fee, min_net_edge_assumed=0.01)
    backward = select_favorites(
        list(reversed(markets)), prices, **BAND, fee_fn=no_fee, min_net_edge_assumed=0.01
    )
    assert forward.candidates == backward.candidates
