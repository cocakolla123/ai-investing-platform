"""Favorite-harvester selection — SIM-ONLY sleeve, pure logic, GATED OFF.

The hypothesis under test (favorite-longshot bias): high-probability
contracts may be UNDERPRICED, so systematically buying YES on favorites in a
90-99c band could be +EV even after fees. Whether that is true for our
venues is an EMPIRICAL question answered by the calibration study
(``python -m dynamo.cli.pm_calibration``) — not by this module.

:func:`select_favorites` is therefore deliberately dumb: it deterministically
selects every tradable favorite in the caller's price band and attaches the
caller-ASSUMED per-contract net edge. It discovers nothing. The assumed edge
must come from the calibration study's verdict, and the function fails closed
on the null: ``min_net_edge_assumed <= 0`` — "the study has not demonstrated
a positive net edge" — selects NOTHING, no matter how attractive the band
looks. The orchestration gate (``pm_favorites_enabled``, default False) sits
in front of this for the same reason.

Same contract as :mod:`dynamo.risk.pm_arbitrage`: deterministic pure
functions, domain models in, candidates plus string skip reasons out, no I/O,
no sizing (the caller owns capital), and every rule fails closed — a missing,
untrusted, non-finite, resolved, or out-of-band input is skipped with a
recorded reason, never guessed at.

This module is CORE LOGIC under the import-linter contracts: it imports
domain models only and never fetches, stores, or executes anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.risk.pm_arbitrage import FeePerContractFn

SKIP_INVALID_INPUT = "invalid_input"
SKIP_NO_ASSUMED_EDGE = "no_demonstrated_edge"
SKIP_MARKET_RESOLVED = "market_resolved"
SKIP_NO_PRICE = "no_price"
SKIP_UNTRUSTED = "untrusted_quote"
SKIP_INVALID_PRICE = "invalid_price"
SKIP_OUTSIDE_BAND = "outside_band"
SKIP_DEGENERATE_COST = "degenerate_cost"


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class FavoriteCandidate:
    """One buy-YES-the-favorite candidate (unsized; the caller owns capital)."""

    venue: str
    market_id: str
    price: float  # assumed fill = the stored implied prob, dollars in (0, 1)
    fee: float  # per-contract fee at that price (0 for fee-free venues)
    cost: float  # price + fee, per contract
    win_profit: float  # 1 - cost, per contract, when the favorite resolves YES
    breakeven_prob: float  # = cost: the realized YES rate needed to break even
    net_edge_assumed: float  # the study-supplied assumption, NOT a discovery
    rationale: str


@dataclass(frozen=True)
class FavoritesResult:
    candidates: tuple[FavoriteCandidate, ...]  # price descending, ties by market_id
    skipped: tuple[tuple[str, str], ...]  # (market_id or "*", reason)


def select_favorites(
    markets: Sequence[PredictionMarket],
    prices: Mapping[str, PredictionMarketPrice],
    *,
    min_price: float,
    max_price: float,
    fee_fn: FeePerContractFn,
    min_net_edge_assumed: float,
) -> FavoritesResult:
    """Deterministically select every tradable favorite in the price band.

    A candidate is an unresolved market whose stored price is trusted, finite,
    and inside ``[min_price, max_price]``, whose fee (from ``fee_fn`` at the
    assumed fill = the stored implied prob) is sane, and whose all-in cost
    stays below the $1 payout. Everything else is skipped with a reason.

    ``min_net_edge_assumed`` is the per-contract net edge (realized YES rate
    minus price minus fees) that the calibration study DEMONSTRATED for the
    band — an input assumption stamped onto every candidate, not something
    this function measures. It doubles as the master gate: any value <= 0
    means the study has not demonstrated an edge, and the whole selection
    fails closed to zero candidates with one ``no_demonstrated_edge`` skip.

    Ordering is deterministic: candidates by price descending (deepest
    favorites first — the band where the studied bias is strongest), ties by
    market_id ascending.
    """
    if (
        not _finite(min_price, max_price, min_net_edge_assumed)
        or not 0.0 < min_price <= max_price < 1.0
    ):
        return FavoritesResult(candidates=(), skipped=(("*", SKIP_INVALID_INPUT),))
    if min_net_edge_assumed <= 0.0:
        return FavoritesResult(candidates=(), skipped=(("*", SKIP_NO_ASSUMED_EDGE),))

    candidates: list[FavoriteCandidate] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for market in markets:
        if market.market_id in seen:  # first row per market wins, like pm_sizing
            continue
        seen.add(market.market_id)
        if market.resolved_outcome is not None:
            skipped.append((market.market_id, SKIP_MARKET_RESOLVED))
            continue
        price = prices.get(market.market_id)
        if price is None:
            skipped.append((market.market_id, SKIP_NO_PRICE))
            continue
        if not price.trusted:
            skipped.append((market.market_id, SKIP_UNTRUSTED))
            continue
        implied = price.implied_prob
        if not _finite(implied) or not 0.0 < implied < 1.0:
            skipped.append((market.market_id, SKIP_INVALID_PRICE))
            continue
        if not min_price <= implied <= max_price:
            skipped.append((market.market_id, SKIP_OUTSIDE_BAND))
            continue
        fee = fee_fn(implied)
        if not _finite(fee) or fee < 0.0:
            skipped.append((market.market_id, SKIP_INVALID_INPUT))
            continue
        cost = implied + fee
        if cost >= 1.0:  # the fee ate the entire favorite payout — untradable
            skipped.append((market.market_id, SKIP_DEGENERATE_COST))
            continue
        candidates.append(
            FavoriteCandidate(
                venue=market.venue,
                market_id=market.market_id,
                price=implied,
                fee=fee,
                cost=cost,
                win_profit=1.0 - cost,
                breakeven_prob=cost,
                net_edge_assumed=min_net_edge_assumed,
                rationale=(
                    f"fav:harvest {market.market_id}: buy YES @ {implied:.3f} + fee "
                    f"{fee:.3f} = {cost:.3f}/contract; breakeven realized rate "
                    f"{cost:.3f}, study-assumed net edge {min_net_edge_assumed:+.4f}"
                ),
            )
        )

    candidates.sort(key=lambda c: (-c.price, c.market_id))
    return FavoritesResult(candidates=tuple(candidates), skipped=tuple(skipped))
