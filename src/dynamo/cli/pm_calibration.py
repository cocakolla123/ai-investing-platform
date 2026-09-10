"""Favorite-longshot calibration study over SETTLED prediction markets.

    python -m dynamo.cli.pm_calibration
    python -m dynamo.cli.pm_calibration --kalshi-pages 8 --polymarket-pages 8
    python -m dynamo.cli.pm_calibration --min-volume 100

Why
---
The hypothesis: high-probability contracts may be UNDERPRICED (the
favorite-longshot bias), so buying 90-99c favorites could be +EV even after
fees. This CLI is the empirical test — it fetches RESOLVED markets from both
venues, buckets them by their pre-settlement YES price, and compares each
bucket's realized YES rate against price + fees with a binomial 95% CI. Its
printed per-venue verdict is the ONLY thing allowed to flip
``pm_favorites_enabled`` (see ``dynamo.config.settings``): the harvester
sleeve stays gated off until this study says EDGE EXISTS.

Price fields, stated rather than discovered later
-------------------------------------------------
Neither venue serves a clean "price one week before resolution" — so the
study uses the least-settlement-adjacent field each payload offers and says
so on every run (the table footer repeats these caveats with live counts):

* **Kalshi** — prefers ``previous_price`` (Kalshi's previous-period reference
  price, further from settlement); falls back to ``last_price`` (the final
  trade, heavily converged toward the outcome) when no previous price exists,
  which is common for markets that open and settle the same day.
* **Polymarket** — Gamma's closed-market payload stores NO pre-settlement
  snapshot (``outcomePrices`` is the final 0/1), so the price is DERIVED as
  ``final - oneDayPriceChange`` (~24h before the final tick). Gamma does not
  document whether that change field freezes at close. Markets without the
  field are dropped.

THE BIAS, PLAINLY: every usable price is settlement-adjacent, and prices
near resolution have already converged toward the outcome. High buckets are
therefore enriched with markets that were about to win, so realized YES
rates are biased UPWARD (toward 1.0) and any measured edge OVERSTATES what a
tradeable earlier entry would earn. A NO-EDGE verdict here is strong
evidence against the sleeve; an EDGE-EXISTS verdict is necessary but NOT
sufficient — it must survive an entry-time price study before real sizing.

Persists nothing; this is a report. Network: Kalshi (signed reads) +
Polymarket Gamma (keyless), a handful of paginated list calls each, paced by
the adapters' shared token buckets. ZERO LLM calls.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from dynamo.risk.pm_arbitrage import kalshi_fee_per_contract

# The favorite band under study, [lo, hi) per bucket.
BUCKETS: tuple[tuple[float, float], ...] = (
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.93),
    (0.93, 0.96),
    (0.96, 0.99),
)
# Below this many resolved markets a bucket's CI is too wide to mean anything.
MIN_POWERED_N = 30
Z95 = 1.959963984540054  # two-sided 95% normal quantile

VERDICT_EDGE = "EDGE EXISTS"
VERDICT_NO_EDGE = "NO EDGE"
VERDICT_UNDERPOWERED = "UNDERPOWERED"


@dataclass(frozen=True)
class ResolvedObservation:
    """One resolved market: its pre-settlement price and what happened."""

    venue: str
    market_id: str
    price: float  # pre-settlement YES price, dollars in (0, 1) — see module docstring
    won: bool  # resolved YES
    price_field: str  # which payload field produced the price (caveat bookkeeping)


@dataclass(frozen=True)
class BucketReport:
    """One (venue, bucket) row of the calibration table."""

    venue: str
    lo: float
    hi: float
    n: int
    mean_price: float
    realized: float  # realized YES rate
    gross_edge: float  # realized - mean_price
    mean_fee: float  # mean per-contract fee at each observed price (0 for Polymarket)
    net_edge: float  # gross_edge - mean_fee
    ci_lo: float  # binomial 95% CI on the realized rate (normal approx)
    ci_hi: float


def binomial_ci(wins: int, n: int) -> tuple[float, float]:
    """Normal-approximation 95% CI on a binomial proportion, clamped to [0, 1]."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    half = Z95 * math.sqrt(p * (1.0 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


def bucket_reports(
    observations: Iterable[ResolvedObservation],
    venue: str,
    fee_fn: Callable[[float], float],
) -> list[BucketReport]:
    """The per-bucket calibration rows for ONE venue (empty buckets included)."""
    rows: list[BucketReport] = []
    obs = [o for o in observations if o.venue == venue and math.isfinite(o.price)]
    for lo, hi in BUCKETS:
        hits = [o for o in obs if lo <= o.price < hi]
        n = len(hits)
        if n == 0:
            rows.append(BucketReport(venue, lo, hi, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
            continue
        wins = sum(1 for o in hits if o.won)
        mean_price = sum(o.price for o in hits) / n
        realized = wins / n
        mean_fee = sum(fee_fn(o.price) for o in hits) / n
        ci_lo, ci_hi = binomial_ci(wins, n)
        rows.append(
            BucketReport(
                venue=venue,
                lo=lo,
                hi=hi,
                n=n,
                mean_price=mean_price,
                realized=realized,
                gross_edge=realized - mean_price,
                mean_fee=mean_fee,
                net_edge=realized - mean_price - mean_fee,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
        )
    return rows


def venue_verdict(rows: Sequence[BucketReport]) -> tuple[str, str]:
    """(verdict, plain-language sentence) for one venue's bucket rows.

    Breakeven realized rate for a bucket is ``mean_price + mean_fee`` — buy
    YES at the price, pay the fee, collect $1 on a win. Thresholds:

    - EDGE EXISTS: some adequately-sampled bucket (n >= MIN_POWERED_N) keeps
      its realized rate above breakeven even at the BOTTOM of the 95% CI.
    - NO EDGE: every adequately-sampled bucket's realized rate sits below
      breakeven at the TOP of the 95% CI.
    - UNDERPOWERED: everything else — no bucket has enough resolved markets,
      or the CIs straddle breakeven.
    """
    powered = [r for r in rows if r.n >= MIN_POWERED_N]
    if not powered:
        return (
            VERDICT_UNDERPOWERED,
            f"no bucket has {MIN_POWERED_N}+ resolved markets — fetch more pages "
            "before reading anything into these numbers.",
        )
    winners = [r for r in powered if r.ci_lo > r.mean_price + r.mean_fee]
    if winners:
        names = ", ".join(f"[{r.lo:.2f},{r.hi:.2f})" for r in winners)
        return (
            VERDICT_EDGE,
            f"buckets {names} keep realized YES rate above price+fees even at the "
            "bottom of the 95% CI. Remember the convergence bias: this OVERSTATES "
            "a tradeable edge — confirm with entry-time prices before enabling "
            "any sleeve.",
        )
    if all(r.ci_hi < r.mean_price + r.mean_fee for r in powered):
        return (
            VERDICT_NO_EDGE,
            "every adequately-sampled bucket's realized rate sits below price+fees "
            "at the top of the 95% CI — and that is DESPITE the convergence bias "
            "flattering favorites. Buying these favorites loses after fees.",
        )
    return (
        VERDICT_UNDERPOWERED,
        "the 95% CIs straddle breakeven in the adequately-sampled buckets — "
        "neither edge nor its absence is demonstrated at this sample size.",
    )


# -- fetch (adapter-backed; injectable for tests) ---------------------------


def kalshi_observations(
    client: object, *, max_pages: int, min_volume: float
) -> tuple[list[ResolvedObservation], Counter[str]]:
    """Settled Kalshi markets → observations, preferring previous_price.

    ``client`` needs only ``settled_markets(max_pages=...)`` yielding
    :class:`dynamo.data.kalshi.KalshiSettlement` rows; dropped rows (no usable
    price, volume below the floor) are counted in the returned Counter.
    """
    observations: list[ResolvedObservation] = []
    dropped: Counter[str] = Counter()
    for s in client.settled_markets(max_pages=max_pages):  # type: ignore[attr-defined]
        if s.volume < min_volume:
            dropped["below_min_volume"] += 1
            continue
        if s.previous_price is not None and 0.0 < s.previous_price < 1.0:
            price, field = s.previous_price, "previous_price"
        elif s.last_price is not None and 0.0 < s.last_price < 1.0:
            price, field = s.last_price, "last_price"
        else:
            dropped["no_usable_price"] += 1
            continue
        observations.append(
            ResolvedObservation(
                venue="kalshi",
                market_id=s.market_id,
                price=price,
                won=s.result_yes,
                price_field=field,
            )
        )
    return observations, dropped


def polymarket_observations(
    client: object, *, max_pages: int, min_volume: float
) -> tuple[list[ResolvedObservation], Counter[str]]:
    """Resolved Gamma markets → observations from the DERIVED prior price.

    ``client`` needs only ``resolved_markets(max_pages=...)`` yielding
    :class:`dynamo.data.polymarket.PolymarketResolution` rows. Markets without
    a derivable ~24h-prior price are dropped and counted — the final price is
    0/1 and carries no calibration information.
    """
    observations: list[ResolvedObservation] = []
    dropped: Counter[str] = Counter()
    for r in client.resolved_markets(max_pages=max_pages):  # type: ignore[attr-defined]
        if r.volume < min_volume:
            dropped["below_min_volume"] += 1
            continue
        if r.prior_price is None:
            dropped["no_derivable_prior_price"] += 1
            continue
        observations.append(
            ResolvedObservation(
                venue="polymarket",
                market_id=r.market_id,
                price=r.prior_price,
                won=r.resolved_yes,
                price_field="final_minus_oneDayPriceChange",
            )
        )
    return observations, dropped


# -- report -----------------------------------------------------------------


def render_report(
    observations: Sequence[ResolvedObservation],
    dropped: dict[str, Counter[str]],
) -> str:
    """The full study as one printable string (tables, caveats, verdicts)."""
    fee_fns: dict[str, Callable[[float], float]] = {
        "kalshi": kalshi_fee_per_contract,  # charged per contract on execution
        "polymarket": lambda _price: 0.0,  # Gamma CLOB charges no trading fee
    }
    lines: list[str] = []
    lines.append("Favorite-longshot calibration study — resolved markets, both venues")
    lines.append("=" * 78)
    header = (
        f"{'venue':<11} {'bucket':<13} {'n':>5} {'mean_px':>8} {'realized':>9} "
        f"{'gross':>7} {'fee':>6} {'net':>7} {'95% CI':>15}"
    )
    for venue in ("kalshi", "polymarket"):
        rows = bucket_reports(observations, venue, fee_fns[venue])
        n_venue = sum(r.n for r in rows)
        fields = Counter(o.price_field for o in observations if o.venue == venue)
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for r in rows:
            if r.n == 0:
                lines.append(f"{r.venue:<11} [{r.lo:.2f},{r.hi:.2f}) {0:>5} {'—':>8}")
                continue
            lines.append(
                f"{r.venue:<11} [{r.lo:.2f},{r.hi:.2f}) {r.n:>5} {r.mean_price:>8.3f} "
                f"{r.realized:>9.3f} {r.gross_edge:>+7.3f} {r.mean_fee:>6.3f} "
                f"{r.net_edge:>+7.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]"
            )
        verdict, sentence = venue_verdict(rows)
        lines.append(
            f"{venue}: {n_venue} in-band observations "
            f"(price fields: {dict(sorted(fields.items())) or '{}'}; "
            f"dropped: {dict(sorted(dropped.get(venue, Counter()).items())) or '{}'})"
        )
        lines.append(f"VERDICT ({venue}): {verdict} — {sentence}")
    lines.append("")
    lines.append("Caveats (read before acting):")
    lines.append("- Prices are settlement-adjacent. Kalshi rows use previous_price where it")
    lines.append("  exists (previous-period reference) and otherwise last_price, the final")
    lines.append("  trade; Polymarket rows are DERIVED as final - oneDayPriceChange (~24h")
    lines.append("  out; Gamma does not document whether the field freezes at close). All of")
    lines.append("  these have already converged toward the outcome, so realized YES rates in")
    lines.append("  high buckets are biased toward 1.0 and any edge here is an OVERSTATEMENT.")
    lines.append("- Gross edge = realized - mean price; net subtracts the Kalshi per-contract")
    lines.append("  fee at each observed price (Polymarket trading fee: 0). Fees at 90-99c")
    lines.append("  are small; the bias above is the bigger number.")
    lines.append("- The pm_favorites sleeve gate (pm_favorites_enabled) may only flip on an")
    lines.append("  EDGE EXISTS verdict here, and even that needs an entry-time-price study")
    lines.append("  before it means anything tradeable.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.cli.pm_calibration",
        description="Favorite-longshot calibration over settled Kalshi + Polymarket markets.",
    )
    parser.add_argument(
        "--kalshi-pages",
        type=int,
        default=5,
        help="settled-market pages to fetch from Kalshi, 200/page (default 5)",
    )
    parser.add_argument(
        "--polymarket-pages",
        type=int,
        default=8,
        help="closed-market pages to fetch from Gamma, 100/page (default 8)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1.0,
        help="drop markets that traded less than this (contracts/$; default 1)",
    )
    args = parser.parse_args(argv)

    from dynamo.config.settings import Settings
    from dynamo.data.kalshi import KalshiClient
    from dynamo.data.polymarket import PolymarketClient

    settings = Settings()
    observations: list[ResolvedObservation] = []
    dropped: dict[str, Counter[str]] = {}

    kalshi = KalshiClient(
        settings.kalshi_api_key_id,
        settings.kalshi_private_key_path,
        min_volume_usd=settings.pm_min_volume_usd,
        max_spread=settings.pm_max_spread,
    )
    if kalshi.configured:
        obs, dropped["kalshi"] = kalshi_observations(
            kalshi, max_pages=args.kalshi_pages, min_volume=args.min_volume
        )
        observations.extend(obs)
    else:
        print("kalshi: not configured (no key/signer) — venue skipped", file=sys.stderr)

    polymarket = PolymarketClient(
        min_volume_usd=settings.pm_min_volume_usd, max_spread=settings.pm_max_spread
    )
    obs, dropped["polymarket"] = polymarket_observations(
        polymarket, max_pages=args.polymarket_pages, min_volume=args.min_volume
    )
    observations.extend(obs)

    print(render_report(observations, dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
