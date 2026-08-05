"""PM sizing: hypothesis invariants + Kelly golden cases (ADR-0009).

The sleeve cap and per-market cap are never exceeded, untrusted markets and
non-Kalshi venues are never selected, floors bind, output is deterministic,
and NaN/missing inputs skip fail-closed instead of raising.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.pm import PMSide
from dynamo.risk.pm_sizing import (
    SKIP_AT_PER_MARKET_CAP,
    SKIP_CONFIDENCE_BELOW_FLOOR,
    SKIP_EDGE_BELOW_FLOOR,
    SKIP_INVALID_EQUITY,
    SKIP_MARKET_CLOSED,
    SKIP_RESOLUTION_TOO_FAR,
    SKIP_UNTRUSTED,
    SKIP_VENUE_NOT_TRADABLE,
    PMSizingParams,
    kelly_fraction,
    select_and_size_pm_trades,
)
from tests.unit.pm.helpers import NOW, make_market, make_position, make_price, make_view

EQUITY = 100_000.0
PARAMS = PMSizingParams()


def run(views, prices=None, markets=None, equity=EQUITY, positions=(), params=None):
    prices = (
        prices
        if prices is not None
        else {v.market_id: make_price(v.market_id, v.market_implied_prob) for v in views}
    )
    markets = (
        markets if markets is not None else {v.market_id: make_market(v.market_id) for v in views}
    )
    return select_and_size_pm_trades(
        views, prices, markets, equity, list(positions), NOW, params or PARAMS
    )


class TestKellyGolden:
    def test_yes_side(self) -> None:
        # f* = (q - p) / (1 - p) = (0.6 - 0.5) / 0.5 = 0.2
        assert math.isclose(kelly_fraction(0.6, 0.5, PMSide.YES), 0.2)

    def test_no_side(self) -> None:
        # f* = (p - q) / p = (0.5 - 0.4) / 0.5 = 0.2
        assert math.isclose(kelly_fraction(0.4, 0.5, PMSide.NO), 0.2)

    def test_negative_edge_is_zero(self) -> None:
        assert kelly_fraction(0.4, 0.5, PMSide.YES) == 0.0
        assert kelly_fraction(0.6, 0.5, PMSide.NO) == 0.0

    def test_degenerate_prices_are_zero(self) -> None:
        assert kelly_fraction(0.6, 0.0, PMSide.YES) == 0.0
        assert kelly_fraction(0.6, 1.0, PMSide.YES) == 0.0
        assert kelly_fraction(float("nan"), 0.5, PMSide.YES) == 0.0

    def test_sized_contracts_golden(self) -> None:
        # q=0.60, p=0.43: kelly = 0.17/0.57 = 0.2982; stake = 0.25x = 0.07456 of
        # equity = $7,456 -> clamped to the 1% per-market cap = $1,000 at 43c
        # -> floor(1000/0.43) = 2325 contracts.
        result = run([make_view(estimated=0.60, implied=0.43)])
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.side is PMSide.YES
        assert d.price_cents == 43
        assert d.contracts == 2325
        assert math.isclose(d.kelly_fraction, 0.25 * (0.17 / 0.57), rel_tol=1e-9)
        assert d.contracts * d.price_cents / 100.0 <= PARAMS.per_market_cap * EQUITY

    def test_no_side_prices_the_no_contract(self) -> None:
        result = run([make_view(estimated=0.30, implied=0.43)])
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.side is PMSide.NO
        assert d.price_cents == 57  # 100 - 43
        assert d.edge < 0


class TestFilters:
    def test_untrusted_never_selected(self) -> None:
        view = make_view()
        result = run([view], prices={view.market_id: make_price(view.market_id, 0.43, False)})
        assert result.decisions == ()
        assert (view.market_id, SKIP_UNTRUSTED) in result.skipped

    def test_polymarket_view_never_tradable(self) -> None:
        view = make_view(venue="polymarket")
        result = run([view])
        assert result.decisions == ()
        assert (view.market_id, SKIP_VENUE_NOT_TRADABLE) in result.skipped

    def test_edge_floor(self) -> None:
        view = make_view(estimated=0.50, implied=0.43)  # edge 0.07 < 0.08
        result = run([view])
        assert (view.market_id, SKIP_EDGE_BELOW_FLOOR) in result.skipped

    def test_confidence_floor(self) -> None:
        view = make_view(confidence=0.59)
        result = run([view])
        assert (view.market_id, SKIP_CONFIDENCE_BELOW_FLOOR) in result.skipped

    def test_resolution_window(self) -> None:
        view = make_view()
        far = {view.market_id: make_market(view.market_id, NOW + timedelta(days=120))}
        assert (view.market_id, SKIP_RESOLUTION_TOO_FAR) in run([view], markets=far).skipped
        closed = {view.market_id: make_market(view.market_id, NOW - timedelta(days=1))}
        assert (view.market_id, SKIP_MARKET_CLOSED) in run([view], markets=closed).skipped

    def test_at_per_market_cap_skips(self) -> None:
        view = make_view()
        held = make_position(view.market_id, contracts=2400, avg_price_cents=43.0)  # $1,032
        result = run([view], positions=[held])
        assert (view.market_id, SKIP_AT_PER_MARKET_CAP) in result.skipped

    def test_invalid_equity_fails_closed(self) -> None:
        for equity in (0.0, -1.0, float("nan"), float("inf")):
            result = run([make_view()], equity=equity)
            assert result.decisions == ()
            assert ("*", SKIP_INVALID_EQUITY) in result.skipped


class TestGreedyAllocation:
    def test_sleeve_cap_binds_across_markets(self) -> None:
        views = [make_view(f"M{i}", estimated=0.60, implied=0.43) for i in range(8)]
        result = run(views)
        total = sum(d.contracts * d.price_cents / 100.0 for d in result.decisions)
        assert total <= PARAMS.sleeve_cap * EQUITY
        # 8 markets want ~$1,000 each but the sleeve only holds $5,000.
        assert len(result.decisions) == 6  # 5 full + 1 remainder-sized
        assert any(reason == "sleeve_cap_exhausted" for _, reason in result.skipped)

    def test_greedy_prefers_edge_times_confidence(self) -> None:
        weak = make_view("WEAK", estimated=0.55, implied=0.43, confidence=0.65)
        strong = make_view("STRONG", estimated=0.70, implied=0.43, confidence=0.95)
        result = run([weak, strong], params=PMSizingParams(sleeve_cap=0.01))
        assert [d.market_id for d in result.decisions] == ["STRONG"]

    def test_deterministic_tiebreak_and_order_independence(self) -> None:
        views = [make_view(f"M{i}", estimated=0.60, implied=0.43) for i in range(6)]
        a = run(views)
        b = run(list(reversed(views)))
        assert a == b
        assert [d.market_id for d in a.decisions] == sorted(d.market_id for d in a.decisions)

    def test_max_open_markets(self) -> None:
        views = [make_view(f"M{i}") for i in range(4)]
        held = make_position("M0", contracts=10, avg_price_cents=43.0)
        result = run(views, positions=[held], params=PMSizingParams(max_open_markets=2))
        markets_selected = {d.market_id for d in result.decisions}
        # M0 is already open (tops up without consuming a new-market slot);
        # only ONE brand-new market may be added on top of it.
        assert "M0" in markets_selected
        assert len(markets_selected - {"M0"}) == 1
        assert sum(1 for _, r in result.skipped if r == "max_open_markets") == 2


# --- hypothesis invariants -------------------------------------------------

probs = st.floats(min_value=0.0, max_value=1.0)
# estimated_prob is unconstrained in the domain model, so the sizer must
# survive adversarial values; confidence is model-guarded to [0, 1].
wild_probs = st.one_of(
    probs,
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(-1.5),
    st.just(2.0),
)
confidences = probs


@st.composite
def scenario(draw: st.DrawFn):
    n = draw(st.integers(min_value=1, max_value=8))
    views = []
    prices = {}
    markets = {}
    for i in range(n):
        market_id = f"M{i}"
        implied = draw(st.floats(min_value=0.01, max_value=0.99))
        views.append(
            make_view(
                market_id,
                estimated=draw(wild_probs),
                implied=implied,
                confidence=draw(confidences),
                venue=draw(st.sampled_from(["kalshi", "polymarket"])),
            )
        )
        if draw(st.booleans()):
            prices[market_id] = make_price(market_id, implied, trusted=draw(st.booleans()))
        if draw(st.booleans()):
            markets[market_id] = make_market(
                market_id,
                close_ts=draw(
                    st.one_of(
                        st.none(),
                        st.datetimes(
                            min_value=datetime(2026, 1, 1),
                            max_value=datetime(2027, 8, 1),
                        ).map(lambda d: d.replace(tzinfo=UTC)),
                    )
                ),
            )
    positions = []
    for i in range(draw(st.integers(min_value=0, max_value=3))):
        positions.append(
            make_position(
                f"M{i}",
                contracts=draw(st.integers(min_value=0, max_value=5_000)),
                avg_price_cents=draw(st.floats(min_value=1.0, max_value=99.0)),
            )
        )
    equity = draw(
        st.one_of(
            st.floats(min_value=1_000.0, max_value=1e7),
            st.just(float("nan")),
            st.just(0.0),
        )
    )
    return views, prices, markets, equity, positions


@given(scenario())
@settings(max_examples=200, deadline=None)
def test_invariants(data) -> None:
    views, prices, markets, equity, positions = data
    result = select_and_size_pm_trades(views, prices, markets, equity, positions, NOW, PARAMS)

    exposure = {p.market_id: p.exposure_dollars for p in positions}
    new_total = 0.0
    seen: set[str] = set()
    for d in result.decisions:
        # Contracts are positive integers at valid prices, on Kalshi only.
        assert isinstance(d.contracts, int) and d.contracts >= 1
        assert 1 <= d.price_cents <= 99
        assert d.venue == "kalshi"
        assert d.market_id not in seen  # at most one decision per market
        seen.add(d.market_id)
        # Floors held.
        assert abs(d.edge) >= PARAMS.min_edge
        assert d.confidence >= PARAMS.min_confidence
        # Selected markets had a trusted price and an in-window close.
        assert prices[d.market_id].trusted
        assert markets[d.market_id].close_ts is not None
        dollars = d.contracts * d.price_cents / 100.0
        new_total += dollars
        # Per-market cap: existing + new never exceeds it.
        assert exposure.get(d.market_id, 0.0) + dollars <= (
            PARAMS.per_market_cap * equity + 1e-6
        )
    if result.decisions:
        held_markets = {p.market_id for p in positions}
        new_markets = seen - held_markets
        assert len(new_markets) <= max(0, PARAMS.max_open_markets - len(held_markets))
        # Sleeve cap: existing + new never exceeds it.
        assert sum(exposure.values()) + new_total <= PARAMS.sleeve_cap * equity + 1e-6

    # Deterministic under input reordering.
    again = select_and_size_pm_trades(
        list(reversed(views)), prices, markets, equity, positions, NOW, PARAMS
    )
    assert again.decisions == result.decisions
