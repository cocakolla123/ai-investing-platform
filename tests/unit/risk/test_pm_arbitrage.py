"""pm_arbitrage: dutch books, cross-venue locks, the fee model, determinism.

Most tests run with ``slippage=0`` (and sometimes a zero fee fn) so the
arithmetic is exact; dedicated tests pin the haircut's and the fee's
direction — each must only ever make a basket LESS attractive.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

from tests.unit.pm.helpers import NOW

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import CrossVenueSpread, PMSide
from dynamo.risk.pm_arbitrage import (
    BOOK_BUY,
    BOOK_SELL,
    SKIP_INVALID_INPUT,
    SKIP_INVALID_PRICE,
    SKIP_MARKET_RESOLVED,
    SKIP_NET_EDGE_BELOW_FLOOR,
    SKIP_NO_EVENT_KEY,
    SKIP_NO_PRICE,
    SKIP_SINGLE_OUTCOME,
    SKIP_UNTRUSTED,
    cross_venue_pair_id,
    find_cross_venue_arbs,
    find_dutch_books,
    kalshi_event_key,
    kalshi_fee_per_contract,
)

EVENT = "KXEV-26SEP"


def no_fee(_price: float) -> float:
    return 0.0


def make_market(
    market_id: str, venue: str = "kalshi", resolved: bool | None = None
) -> PredictionMarket:
    return PredictionMarket(
        venue=venue, market_id=market_id, question=f"{market_id}?", resolved_outcome=resolved
    )


def make_price(
    market_id: str, implied: float, trusted: bool = True, venue: str = "kalshi"
) -> PredictionMarketPrice:
    return PredictionMarketPrice(
        venue=venue, market_id=market_id, ts=NOW, implied_prob=implied, trusted=trusted
    )


def event_inputs(
    implieds: Mapping[str, float], event: str = EVENT
) -> tuple[list[PredictionMarket], dict[str, PredictionMarketPrice]]:
    """({suffix: implied}) -> (markets, prices) for one multi-outcome event."""
    markets = [make_market(f"{event}-{suffix}") for suffix in implieds]
    prices = {
        f"{event}-{suffix}": make_price(f"{event}-{suffix}", implied)
        for suffix, implied in implieds.items()
    }
    return markets, prices


def make_spread(
    kalshi_yes: float,
    polymarket_yes: float,
    kalshi_id: str = "KXPAIR-26SEP-A",
    polymarket_id: str = "poly-a",
) -> CrossVenueSpread:
    return CrossVenueSpread(
        question=f"{kalshi_id}?",
        kalshi_market_id=kalshi_id,
        kalshi_yes=kalshi_yes,
        polymarket_id=polymarket_id,
        polymarket_yes=polymarket_yes,
        yes_diff=kalshi_yes - polymarket_yes,
        as_of=NOW,
    )


class TestKalshiFee:
    def test_formula_golden(self) -> None:
        # ceil(7 * 0.5 * 0.5) = ceil(1.75) = 2 cents
        assert kalshi_fee_per_contract(0.5) == 0.02
        # ceil(7 * 0.05 * 0.95) = ceil(0.3325) = 1 cent
        assert kalshi_fee_per_contract(0.05) == 0.01

    def test_midrange_price_costs_more_than_tail(self) -> None:
        assert kalshi_fee_per_contract(0.5) > kalshi_fee_per_contract(0.05)

    def test_degenerate_price_fails_closed_to_worst_case(self) -> None:
        worst = kalshi_fee_per_contract(0.5)
        assert kalshi_fee_per_contract(float("nan")) == worst
        assert kalshi_fee_per_contract(0.0) == worst
        assert kalshi_fee_per_contract(1.5) == worst


class TestEventKey:
    def test_kalshi_ticker_drops_the_outcome_suffix(self) -> None:
        assert kalshi_event_key(make_market("KXHIGHNY-25AUG21-B87.5")) == "KXHIGHNY-25AUG21"

    def test_polymarket_and_flat_tickers_are_ungroupable(self) -> None:
        assert kalshi_event_key(make_market("poly-slug-here", venue="polymarket")) is None
        assert kalshi_event_key(make_market("INXD-23DEC29")) is None


class TestDutchBooks:
    def test_buy_book_found_when_asks_sum_below_one_minus_fees(self) -> None:
        # asks 3 x 0.30 = 0.90; fees 3 x ceil(7*0.3*0.7)=2c = 0.06;
        # net = 1 - 0.90 - 0.06 = 0.04 >= 0.02 floor.
        markets, prices = event_inputs({"A": 0.30, "B": 0.30, "C": 0.30})
        result = find_dutch_books(markets, prices, min_net_edge=0.02, slippage=0.0)
        assert len(result.candidates) == 1
        book = result.candidates[0]
        assert book.kind == BOOK_BUY
        assert book.event_key == EVENT
        assert [leg.side for leg in book.legs] == [PMSide.YES] * 3
        assert math.isclose(book.gross_cost, 0.90)
        assert math.isclose(book.total_fees, 0.06)
        assert math.isclose(book.net_profit, 0.04)
        assert math.isclose(book.profit_per_dollar, 0.04 / 0.96)
        assert book.rationale.startswith("arb:dutch buy_book")

    def test_not_found_when_fees_eat_the_gap(self) -> None:
        # asks sum 0.96 < 1, but 0.06 of fees push the net to -0.02.
        markets, prices = event_inputs({"A": 0.32, "B": 0.32, "C": 0.32})
        with_fees = find_dutch_books(markets, prices, min_net_edge=0.02, slippage=0.0)
        assert with_fees.candidates == ()
        assert (EVENT, SKIP_NET_EDGE_BELOW_FLOOR) in with_fees.skipped
        # Counterfactual: the SAME book clears once fees are removed.
        without_fees = find_dutch_books(
            markets, prices, fee_per_contract_fn=no_fee, min_net_edge=0.02, slippage=0.0
        )
        assert len(without_fees.candidates) == 1
        assert math.isclose(without_fees.candidates[0].net_profit, 0.04)

    def test_sell_book_buys_no_on_every_outcome(self) -> None:
        # bids 3 x 0.45 = 1.35 > 1: buy NO at 0.55 each (cost 1.65), fees
        # 3 x ceil(7*0.55*0.45)=2c = 0.06, payout n-1 = 2.00, net 0.29.
        markets, prices = event_inputs({"A": 0.45, "B": 0.45, "C": 0.45})
        result = find_dutch_books(markets, prices, min_net_edge=0.02, slippage=0.0)
        assert len(result.candidates) == 1
        book = result.candidates[0]
        assert book.kind == BOOK_SELL
        assert [leg.side for leg in book.legs] == [PMSide.NO] * 3
        assert all(math.isclose(leg.price, 0.55) for leg in book.legs)
        assert math.isclose(book.payout, 2.0)
        assert math.isclose(book.net_profit, 2.0 - 1.65 - 0.06)

    def test_slippage_haircut_can_only_hurt(self) -> None:
        markets, prices = event_inputs({"A": 0.31, "B": 0.31, "C": 0.31})
        clean = find_dutch_books(
            markets, prices, fee_per_contract_fn=no_fee, min_net_edge=0.02, slippage=0.0
        )
        assert len(clean.candidates) == 1
        assert math.isclose(clean.candidates[0].net_profit, 0.07)
        # +0.02 per leg -> asks sum 0.99 -> net 0.01 < the 0.02 floor.
        haircut = find_dutch_books(
            markets, prices, fee_per_contract_fn=no_fee, min_net_edge=0.02, slippage=0.02
        )
        assert haircut.candidates == ()
        assert (EVENT, SKIP_NET_EDGE_BELOW_FLOOR) in haircut.skipped

    def test_one_bad_leg_poisons_the_whole_book(self) -> None:
        markets, prices = event_inputs({"A": 0.30, "B": 0.30, "C": 0.30})
        untrusted = dict(prices)
        untrusted[f"{EVENT}-B"] = make_price(f"{EVENT}-B", 0.30, trusted=False)
        result = find_dutch_books(markets, untrusted, min_net_edge=0.02, slippage=0.0)
        assert result.candidates == ()
        assert (EVENT, SKIP_UNTRUSTED) in result.skipped

        missing = {k: v for k, v in prices.items() if not k.endswith("-C")}
        result = find_dutch_books(markets, missing, min_net_edge=0.02, slippage=0.0)
        assert result.candidates == ()
        assert (EVENT, SKIP_NO_PRICE) in result.skipped

    def test_resolved_outcome_skips_the_event(self) -> None:
        markets, prices = event_inputs({"A": 0.30, "B": 0.30, "C": 0.30})
        markets[0] = make_market(markets[0].market_id, resolved=True)
        result = find_dutch_books(markets, prices, min_net_edge=0.02, slippage=0.0)
        assert result.candidates == ()
        assert (EVENT, SKIP_MARKET_RESOLVED) in result.skipped

    def test_ungroupable_and_single_outcome_are_recorded(self) -> None:
        markets = [make_market("poly-thing", venue="polymarket"), make_market("KXSOLO-26SEP-A")]
        prices = {"KXSOLO-26SEP-A": make_price("KXSOLO-26SEP-A", 0.30)}
        result = find_dutch_books(markets, prices, min_net_edge=0.02, slippage=0.0)
        assert result.candidates == ()
        assert ("poly-thing", SKIP_NO_EVENT_KEY) in result.skipped
        assert ("KXSOLO-26SEP", SKIP_SINGLE_OUTCOME) in result.skipped

    def test_invalid_params_fail_closed(self) -> None:
        markets, prices = event_inputs({"A": 0.30, "B": 0.30})
        result = find_dutch_books(markets, prices, min_net_edge=float("nan"), slippage=0.0)
        assert result.candidates == ()
        assert result.skipped == (("*", SKIP_INVALID_INPUT),)

    def test_deterministic_ordering_by_profit_then_key(self) -> None:
        rich_m, rich_p = event_inputs({"A": 0.20, "B": 0.20}, event="KXRICH-26SEP")
        thin_m, thin_p = event_inputs({"A": 0.40, "B": 0.40}, event="KXTHIN-26SEP")
        tie_m, tie_p = event_inputs({"A": 0.40, "B": 0.40}, event="KXALSO-26SEP")
        markets = rich_m + thin_m + tie_m
        prices = {**rich_p, **thin_p, **tie_p}
        expected = ["KXRICH-26SEP", "KXALSO-26SEP", "KXTHIN-26SEP"]
        for seed in range(3):
            shuffled = list(markets)
            random.Random(seed).shuffle(shuffled)
            result = find_dutch_books(
                shuffled, prices, fee_per_contract_fn=no_fee, min_net_edge=0.02, slippage=0.0
            )
            assert [c.event_key for c in result.candidates] == expected


class TestCrossVenueArbs:
    def test_verified_pair_emits_a_tradable_candidate(self) -> None:
        spread = make_spread(kalshi_yes=0.30, polymarket_yes=0.45)
        pair = cross_venue_pair_id(spread)
        assert pair == "KXPAIR-26SEP-A|poly-a"
        result = find_cross_venue_arbs(
            [spread],
            min_net_edge=0.02,
            verified_pairs={pair},
            fee_per_contract_fn=no_fee,
            slippage=0.0,
        )
        assert result.unverified == ()
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        yes_leg, no_leg = candidate.legs
        assert (yes_leg.venue, yes_leg.side) == ("kalshi", PMSide.YES)
        assert (no_leg.venue, no_leg.side) == ("polymarket", PMSide.NO)
        assert math.isclose(yes_leg.price, 0.30)
        assert math.isclose(no_leg.price, 0.55)
        assert math.isclose(candidate.net_profit, 0.15)
        assert candidate.rationale.startswith("arb:xvenue")

    def test_unverified_pair_is_never_a_trade(self) -> None:
        # THE hard guard: same profitable pair, but nobody verified that the
        # two venues resolve by the same criteria -> research row only.
        spread = make_spread(kalshi_yes=0.30, polymarket_yes=0.45)
        result = find_cross_venue_arbs(
            [spread],
            min_net_edge=0.02,
            verified_pairs=frozenset(),
            fee_per_contract_fn=no_fee,
            slippage=0.0,
        )
        assert result.candidates == ()
        assert len(result.unverified) == 1
        assert result.unverified[0].pair_id == cross_venue_pair_id(spread)

    def test_fees_eat_the_gap(self) -> None:
        # YES kalshi 0.48 (fee ceil(7*.48*.52)=2c) + NO poly 0.50 = 1.00 cost
        # -> net 0.00 < 0.02 floor; the same pair clears without the fee.
        spread = make_spread(kalshi_yes=0.48, polymarket_yes=0.50)
        pair = cross_venue_pair_id(spread)
        with_fees = find_cross_venue_arbs(
            [spread], min_net_edge=0.02, verified_pairs={pair}, slippage=0.0
        )
        assert with_fees.candidates == () and with_fees.unverified == ()
        assert (pair, SKIP_NET_EDGE_BELOW_FLOOR) in with_fees.skipped
        without_fees = find_cross_venue_arbs(
            [spread],
            min_net_edge=0.02,
            verified_pairs={pair},
            fee_per_contract_fn=no_fee,
            slippage=0.0,
        )
        assert len(without_fees.candidates) == 1
        assert math.isclose(without_fees.candidates[0].net_profit, 0.02)

    def test_direction_buys_yes_on_the_cheap_venue(self) -> None:
        spread = make_spread(kalshi_yes=0.45, polymarket_yes=0.30)
        result = find_cross_venue_arbs(
            [spread],
            min_net_edge=0.02,
            verified_pairs={cross_venue_pair_id(spread)},
            fee_per_contract_fn=no_fee,
            slippage=0.0,
        )
        yes_leg, no_leg = result.candidates[0].legs
        assert yes_leg.venue == "polymarket" and math.isclose(yes_leg.price, 0.30)
        assert no_leg.venue == "kalshi" and math.isclose(no_leg.price, 0.55)

    def test_kalshi_fee_lands_on_the_kalshi_leg_only(self) -> None:
        spread = make_spread(kalshi_yes=0.30, polymarket_yes=0.45)
        result = find_cross_venue_arbs(
            [spread], min_net_edge=0.02, verified_pairs={cross_venue_pair_id(spread)}, slippage=0.0
        )
        yes_leg, no_leg = result.candidates[0].legs
        assert yes_leg.venue == "kalshi" and yes_leg.fee == kalshi_fee_per_contract(0.30)
        assert no_leg.venue == "polymarket" and no_leg.fee == 0.0

    def test_degenerate_quote_skips_the_pair(self) -> None:
        spread = make_spread(kalshi_yes=1.0, polymarket_yes=0.45)
        result = find_cross_venue_arbs(
            [spread], min_net_edge=0.02, verified_pairs=frozenset(), slippage=0.0
        )
        assert result.candidates == () and result.unverified == ()
        assert (cross_venue_pair_id(spread), SKIP_INVALID_PRICE) in result.skipped

    def test_deterministic_ordering_by_profit_then_pair_id(self) -> None:
        rich = make_spread(0.20, 0.50, kalshi_id="KXR-26SEP-A", polymarket_id="poly-r")
        thin = make_spread(0.40, 0.50, kalshi_id="KXT-26SEP-A", polymarket_id="poly-t")
        tie = make_spread(0.40, 0.50, kalshi_id="KXA-26SEP-A", polymarket_id="poly-a")
        pairs = {cross_venue_pair_id(s) for s in (rich, thin, tie)}
        expected = ["KXR-26SEP-A|poly-r", "KXA-26SEP-A|poly-a", "KXT-26SEP-A|poly-t"]
        for order in ([rich, thin, tie], [tie, rich, thin], [thin, tie, rich]):
            result = find_cross_venue_arbs(
                order,
                min_net_edge=0.02,
                verified_pairs=pairs,
                fee_per_contract_fn=no_fee,
                slippage=0.0,
            )
            assert [c.pair_id for c in result.candidates] == expected
