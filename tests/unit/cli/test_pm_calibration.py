"""pm_calibration: bucket math, the binomial CI, verdict thresholds, and the
observation assembly for both venues — fakes only, no network."""

from __future__ import annotations

import math

from dynamo.cli.pm_calibration import (
    MIN_POWERED_N,
    VERDICT_EDGE,
    VERDICT_NO_EDGE,
    VERDICT_UNDERPOWERED,
    BucketReport,
    ResolvedObservation,
    binomial_ci,
    bucket_reports,
    kalshi_observations,
    polymarket_observations,
    render_report,
    venue_verdict,
)
from dynamo.data.kalshi import KalshiSettlement
from dynamo.data.polymarket import PolymarketResolution
from dynamo.risk.pm_arbitrage import kalshi_fee_per_contract


def obs(price: float, won: bool, venue: str = "kalshi", n: str = "M") -> ResolvedObservation:
    return ResolvedObservation(venue=venue, market_id=n, price=price, won=won, price_field="test")


def no_fee(_price: float) -> float:
    return 0.0


# -- bucket math ------------------------------------------------------------


def test_bucket_reports_math_is_exact() -> None:
    # 10 markets at 0.94, 9 winners -> realized 0.9, gross edge -0.04.
    observations = [obs(0.94, i < 9) for i in range(10)]
    rows = bucket_reports(observations, "kalshi", kalshi_fee_per_contract)
    assert len(rows) == 5  # one row per bucket, empty ones included
    row = next(r for r in rows if r.lo == 0.93)
    assert row.n == 10
    assert math.isclose(row.mean_price, 0.94)
    assert math.isclose(row.realized, 0.9)
    assert math.isclose(row.gross_edge, -0.04)
    assert math.isclose(row.mean_fee, kalshi_fee_per_contract(0.94))
    assert math.isclose(row.net_edge, row.gross_edge - row.mean_fee)
    assert all(r.n == 0 for r in rows if r.lo != 0.93)


def test_buckets_are_half_open_and_venue_scoped() -> None:
    observations = [
        obs(0.85, True),  # lands in [0.85, 0.90), not [0.80, 0.85)
        obs(0.90, True),  # lands in [0.90, 0.93)
        obs(0.99, True),  # above the study band entirely
        obs(0.95, True, venue="polymarket"),  # other venue, ignored here
    ]
    rows = bucket_reports(observations, "kalshi", no_fee)
    by_lo = {r.lo: r.n for r in rows}
    assert by_lo == {0.80: 0, 0.85: 1, 0.90: 1, 0.93: 0, 0.96: 0}


def test_binomial_ci_normal_approximation() -> None:
    lo, hi = binomial_ci(90, 100)
    half = 1.959963984540054 * math.sqrt(0.9 * 0.1 / 100)
    assert math.isclose(lo, 0.9 - half) and math.isclose(hi, 0.9 + half)
    assert binomial_ci(0, 0) == (0.0, 1.0)  # empty bucket -> vacuous interval
    assert binomial_ci(100, 100) == (1.0, 1.0)  # clamped into [0, 1]


# -- verdict thresholds -----------------------------------------------------


def make_row(n: int, mean_price: float, realized: float, fee: float = 0.0) -> BucketReport:
    ci_lo, ci_hi = binomial_ci(round(realized * n), n)
    return BucketReport(
        venue="kalshi",
        lo=0.93,
        hi=0.96,
        n=n,
        mean_price=mean_price,
        realized=realized,
        gross_edge=realized - mean_price,
        mean_fee=fee,
        net_edge=realized - mean_price - fee,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )


def test_verdict_edge_needs_ci_floor_above_breakeven() -> None:
    # 990/1000 at mean 0.94: CI floor ~0.984 > 0.94 + 0.005 breakeven.
    verdict, sentence = venue_verdict([make_row(1000, 0.94, 0.99, fee=0.005)])
    assert verdict == VERDICT_EDGE
    assert "bottom of the 95% CI" in sentence


def test_verdict_no_edge_needs_ci_ceiling_below_breakeven() -> None:
    # 900/1000 at mean 0.94: CI ceiling ~0.919 < 0.94 breakeven.
    verdict, _ = venue_verdict([make_row(1000, 0.94, 0.90)])
    assert verdict == VERDICT_NO_EDGE


def test_verdict_underpowered_on_small_n_or_straddling_ci() -> None:
    # Too few markets, however good they look.
    verdict, _ = venue_verdict([make_row(MIN_POWERED_N - 1, 0.94, 1.0)])
    assert verdict == VERDICT_UNDERPOWERED
    # Enough markets but the CI straddles breakeven: 29/30 at mean 0.94.
    verdict, _ = venue_verdict([make_row(30, 0.94, 29 / 30)])
    assert verdict == VERDICT_UNDERPOWERED


def test_verdict_fee_moves_breakeven() -> None:
    # 970/1000 at mean 0.94: edge without fees (CI floor ~0.959 > 0.94)...
    assert venue_verdict([make_row(1000, 0.94, 0.97)])[0] == VERDICT_EDGE
    # ...but a 2c fee moves breakeven to 0.96, inside the CI -> underpowered.
    assert venue_verdict([make_row(1000, 0.94, 0.97, fee=0.02)])[0] == VERDICT_UNDERPOWERED


# -- observation assembly (fake adapters, both venues) ----------------------


class FakeKalshi:
    def __init__(self, rows: list[KalshiSettlement]) -> None:
        self.rows = rows
        self.pages: int | None = None

    def settled_markets(self, *, max_pages: int) -> list[KalshiSettlement]:
        self.pages = max_pages
        return self.rows


class FakePolymarket:
    def __init__(self, rows: list[PolymarketResolution]) -> None:
        self.rows = rows

    def resolved_markets(self, *, max_pages: int) -> list[PolymarketResolution]:
        return self.rows


def test_kalshi_observations_prefer_previous_price() -> None:
    rows = [
        KalshiSettlement(
            "K-A", True, last_price=0.999, previous_price=0.94, volume=10.0, settled_ts=None
        ),
        KalshiSettlement(
            "K-B", False, last_price=0.02, previous_price=None, volume=10.0, settled_ts=None
        ),
        KalshiSettlement(
            "K-C", True, last_price=None, previous_price=None, volume=10.0, settled_ts=None
        ),
        KalshiSettlement(
            "K-D", True, last_price=0.95, previous_price=0.94, volume=0.0, settled_ts=None
        ),
    ]
    observations, dropped = kalshi_observations(FakeKalshi(rows), max_pages=2, min_volume=1.0)
    assert [(o.market_id, o.price, o.price_field) for o in observations] == [
        ("K-A", 0.94, "previous_price"),  # previous preferred over converged last
        ("K-B", 0.02, "last_price"),  # fallback when no previous exists
    ]
    assert dropped == {"no_usable_price": 1, "below_min_volume": 1}


def test_polymarket_observations_require_derived_prior() -> None:
    rows = [
        PolymarketResolution("P-A", "a?", True, final_price=1.0, prior_price=0.94, volume=10.0),
        PolymarketResolution("P-B", "b?", False, final_price=0.0, prior_price=None, volume=10.0),
    ]
    observations, dropped = polymarket_observations(
        FakePolymarket(rows), max_pages=2, min_volume=1.0
    )
    assert [(o.market_id, o.price, o.won) for o in observations] == [("P-A", 0.94, True)]
    assert observations[0].price_field == "final_minus_oneDayPriceChange"
    assert dropped == {"no_derivable_prior_price": 1}


def test_render_report_prints_verdicts_and_caveats() -> None:
    observations = [obs(0.94, i < 9, n=f"M{i}") for i in range(10)]
    report = render_report(observations, {})
    assert "VERDICT (kalshi):" in report
    assert "VERDICT (polymarket):" in report
    assert VERDICT_UNDERPOWERED in report  # 10 markets can demonstrate nothing
    assert "biased toward 1.0" in report  # the convergence caveat prints every run
    assert "previous_price" in report
