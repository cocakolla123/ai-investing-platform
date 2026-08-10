"""Prediction-market sleeve selection & sizing (Kalshi; ADR-0009).

Pure code with the doc 06 veto-contract philosophy: the risk engine — not the
research view — decides what trades, and every rule here fails CLOSED. A market
with any missing, non-finite, or out-of-range input is skipped with a recorded
reason, never sized.

Rules, in order:

1. Eligibility filters: Kalshi only, trusted quote, |edge| >= min_edge against
   the CURRENT price, confidence >= min_confidence, resolution within
   ``max_days_to_resolution``, not already at the per-market cap.
2. Binary-outcome Kelly from estimated vs current implied probability
   (YES: ``(q-p)/(1-p)``; NO: ``(p-q)/p``), scaled by ``kelly_fraction_cap``.
3. Dollar caps: per-market cap, then the hard sleeve cap over existing + new
   exposure, allocated greedily by ``|edge| * confidence`` with a
   deterministic market_id tiebreak. ``max_open_markets`` bounds distinct
   markets held.
4. Contracts = floor(dollars / price-per-contract); fewer than one contract
   means no trade.

Cross-venue corroboration (ADR-0009 addendum, additive): when a matched
Polymarket YES price is supplied via ``cross_venue`` (a plain mapping passed
IN by the caller — this module never fetches anything), a candidate whose
estimate the other venue *agrees* with is annotated as corroborated; one it
*disagrees* with beyond ``cross_venue_veto_gap`` (both venues against us) is
skipped fail-closed. Missing/NaN entries change nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import PM_VENUE_KALSHI, PMPosition, PMSide, PMTradeDecision
from dynamo.domain.models.research import PredictionMarketView

SKIP_VENUE_NOT_TRADABLE = "venue_not_tradable"
SKIP_NO_PRICE = "no_price"
SKIP_UNTRUSTED = "untrusted_quote"
SKIP_INVALID_INPUT = "invalid_input"
SKIP_NO_CLOSE_TS = "no_close_ts"
SKIP_RESOLUTION_TOO_FAR = "resolution_too_far"
SKIP_MARKET_CLOSED = "market_closed"
SKIP_EDGE_BELOW_FLOOR = "edge_below_floor"
SKIP_CONFIDENCE_BELOW_FLOOR = "confidence_below_floor"
SKIP_AT_PER_MARKET_CAP = "at_per_market_cap"
SKIP_BELOW_ONE_CONTRACT = "below_one_contract"
SKIP_MAX_OPEN_MARKETS = "max_open_markets"
SKIP_SLEEVE_CAP_EXHAUSTED = "sleeve_cap_exhausted"
SKIP_INVALID_EQUITY = "invalid_equity"
SKIP_CROSS_VENUE_DISAGREEMENT = "cross_venue_disagreement"


@dataclass(frozen=True)
class PMSizingParams:
    """Mirrors the ``pm_*`` settings; plain floats so this module stays pure."""

    sleeve_cap: float = 0.05
    per_market_cap: float = 0.01
    min_edge: float = 0.08
    min_confidence: float = 0.6
    max_days_to_resolution: int = 90
    kelly_fraction_cap: float = 0.25
    max_open_markets: int = 10
    cross_venue_veto_gap: float = 0.10  # |polymarket - estimated| veto threshold


@dataclass(frozen=True)
class PMSizingResult:
    decisions: tuple[PMTradeDecision, ...]
    skipped: tuple[tuple[str, str], ...]  # (market_id, reason)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def kelly_fraction(estimated_prob: float, implied_prob: float, side: PMSide) -> float:
    """Full-Kelly stake fraction for a binary contract bought at the implied price.

    YES at price ``p`` with win probability ``q``: ``f* = (q - p) / (1 - p)``.
    NO (price ``1-p``, win probability ``1-q``): ``f* = (p - q) / p``.
    Degenerate prices or a non-positive edge return 0 (fail closed).
    """
    q, p = estimated_prob, implied_prob
    if not _finite(q, p) or not 0.0 < p < 1.0 or not 0.0 <= q <= 1.0:
        return 0.0
    raw = (q - p) / (1.0 - p) if side is PMSide.YES else (p - q) / p
    if not math.isfinite(raw) or raw <= 0.0:
        return 0.0
    return raw


@dataclass(frozen=True)
class _Candidate:
    view: PredictionMarketView
    side: PMSide
    edge: float
    implied: float
    price_cents: int
    stake_fraction: float  # capped fractional Kelly, as fraction of equity
    dollars: float  # after the per-market cap
    days_to_resolution: int
    corroboration: str | None = None  # cross-venue agreement note, if any


def _eligible(
    view: PredictionMarketView,
    prices: Mapping[str, PredictionMarketPrice],
    markets: Mapping[str, PredictionMarket],
    equity: float,
    exposure_by_market: Mapping[str, float],
    now: datetime,
    params: PMSizingParams,
    cross_venue: Mapping[str, float] | None,
) -> _Candidate | str:
    """One market through the eligibility + per-market sizing rules.

    Returns a sized candidate, or the skip reason (fail closed on anything
    missing or non-finite).
    """
    if view.venue != PM_VENUE_KALSHI:
        return SKIP_VENUE_NOT_TRADABLE
    price = prices.get(view.market_id)
    if price is None:
        return SKIP_NO_PRICE
    if not price.trusted:
        return SKIP_UNTRUSTED

    market = markets.get(view.market_id)
    if market is None or market.close_ts is None:
        return SKIP_NO_CLOSE_TS
    remaining = market.close_ts - now
    days = remaining.days
    if remaining.total_seconds() <= 0:
        return SKIP_MARKET_CLOSED
    if days > params.max_days_to_resolution:
        return SKIP_RESOLUTION_TOO_FAR

    q, p, confidence = view.estimated_prob, price.implied_prob, view.confidence
    if not _finite(q, p, confidence) or not 0.0 <= q <= 1.0 or not 0.0 < p < 1.0:
        return SKIP_INVALID_INPUT
    edge = q - p
    if abs(edge) < params.min_edge:
        return SKIP_EDGE_BELOW_FLOOR
    if not 0.0 <= confidence <= 1.0 or confidence < params.min_confidence:
        return SKIP_CONFIDENCE_BELOW_FLOOR

    # Cross-venue corroboration (additive; module docstring). ``polymarket``
    # is the matched venue's YES price for the SAME question. Agreement with
    # our direction vs the Kalshi price -> note; disagreement beyond the veto
    # gap (both venues against our estimate) -> stand down. Anything missing,
    # non-finite, or out of range changes nothing.
    corroboration: str | None = None
    polymarket = None if cross_venue is None else cross_venue.get(view.market_id)
    if polymarket is not None and _finite(polymarket) and 0.0 <= polymarket <= 1.0:
        ours, theirs = edge, polymarket - p
        if ours * theirs > 0.0:
            corroboration = f"cross-venue corroborated (polymarket yes {polymarket:.2f})"
        elif theirs != 0.0 and abs(polymarket - q) > params.cross_venue_veto_gap:
            return SKIP_CROSS_VENUE_DISAGREEMENT

    side = PMSide.YES if edge > 0.0 else PMSide.NO
    full_kelly = kelly_fraction(q, p, side)
    if full_kelly <= 0.0:
        return SKIP_INVALID_INPUT
    stake_fraction = params.kelly_fraction_cap * full_kelly

    used = exposure_by_market.get(view.market_id, 0.0)
    if not math.isfinite(used):
        return SKIP_INVALID_INPUT
    per_market_available = params.per_market_cap * equity - used
    if per_market_available <= 0.0:
        return SKIP_AT_PER_MARKET_CAP
    dollars = min(stake_fraction * equity, per_market_available)

    entry_prob = p if side is PMSide.YES else 1.0 - p
    price_cents = min(max(round(entry_prob * 100.0), 1), 99)
    if _contracts(dollars, price_cents) < 1:
        return SKIP_BELOW_ONE_CONTRACT
    return _Candidate(
        view=view,
        side=side,
        edge=edge,
        implied=p,
        price_cents=price_cents,
        stake_fraction=stake_fraction,
        dollars=dollars,
        days_to_resolution=days,
        corroboration=corroboration,
    )


def _contracts(dollars: float, price_cents: int) -> int:
    if not math.isfinite(dollars) or dollars <= 0.0:
        return 0
    return math.floor(dollars * 100.0 / price_cents)


def _decision(candidate: _Candidate, contracts: int) -> PMTradeDecision:
    view = candidate.view
    rationale = (
        f"{candidate.side.value} {contracts}x {view.market_id} @ {candidate.price_cents}c: "
        f"estimated {view.estimated_prob:.2f} vs implied {candidate.implied:.2f} "
        f"(edge {candidate.edge:+.2f}, confidence {view.confidence:.2f}, "
        f"resolves in {candidate.days_to_resolution}d); "
        f"fractional-Kelly stake {candidate.stake_fraction:.4f} of equity, "
        f"capped to ${contracts * candidate.price_cents / 100.0:.2f} "
        f"by the per-market and sleeve caps"
    )
    if candidate.corroboration is not None:
        rationale += f"; {candidate.corroboration}"
    return PMTradeDecision(
        venue=view.venue,
        market_id=view.market_id,
        side=candidate.side,
        edge=candidate.edge,
        estimated_prob=view.estimated_prob,
        implied_prob=candidate.implied,
        confidence=view.confidence,
        kelly_fraction=candidate.stake_fraction,
        contracts=contracts,
        price_cents=candidate.price_cents,
        rationale=rationale,
        view_ref=f"{view.venue}:{view.market_id}:{view.as_of.isoformat()}",
    )


def select_and_size_pm_trades(
    views: Sequence[PredictionMarketView],
    prices: Mapping[str, PredictionMarketPrice],
    markets: Mapping[str, PredictionMarket],
    equity: float,
    open_positions: Sequence[PMPosition],
    now: datetime,
    params: PMSizingParams | None = None,
    cross_venue: Mapping[str, float] | None = None,
) -> PMSizingResult:
    """Filter, size, and greedily allocate the Kalshi sleeve (module docstring).

    Deterministic: ranked by ``|edge| * confidence`` descending, ties broken by
    ``market_id`` ascending. The sleeve total (existing exposure + new orders)
    never exceeds ``sleeve_cap * equity``; no market exceeds
    ``per_market_cap * equity``; fewer than one affordable contract → no trade.

    ``cross_venue`` optionally maps a Kalshi market_id to the matched
    Polymarket YES price (corroboration input, passed in by the caller);
    ``None`` or a missing entry leaves behaviour exactly unchanged.
    """
    params = params if params is not None else PMSizingParams()
    if not _finite(equity) or equity <= 0.0:
        return PMSizingResult(decisions=(), skipped=((("*", SKIP_INVALID_EQUITY)),))

    exposure_by_market: dict[str, float] = {}
    for position in open_positions:
        exposure_by_market[position.market_id] = (
            exposure_by_market.get(position.market_id, 0.0) + position.exposure_dollars
        )
    open_market_ids = frozenset(exposure_by_market)

    skipped: list[tuple[str, str]] = []
    candidates: list[_Candidate] = []
    seen: set[str] = set()
    for view in views:
        if view.market_id in seen:  # one decision per market — first view wins
            continue
        seen.add(view.market_id)
        outcome = _eligible(
            view, prices, markets, equity, exposure_by_market, now, params, cross_venue
        )
        if isinstance(outcome, str):
            skipped.append((view.market_id, outcome))
        else:
            candidates.append(outcome)

    candidates.sort(key=lambda c: (-abs(c.edge) * c.view.confidence, c.view.market_id))

    sleeve_used = sum(exposure_by_market.values())
    sleeve_remaining = params.sleeve_cap * equity - sleeve_used
    new_markets = 0
    decisions: list[PMTradeDecision] = []
    for candidate in candidates:
        market_id = candidate.view.market_id
        is_new_market = market_id not in open_market_ids
        if is_new_market and len(open_market_ids) + new_markets >= params.max_open_markets:
            skipped.append((market_id, SKIP_MAX_OPEN_MARKETS))
            continue
        dollars = min(candidate.dollars, sleeve_remaining)
        contracts = _contracts(dollars, candidate.price_cents)
        if contracts < 1:
            skipped.append((market_id, SKIP_SLEEVE_CAP_EXHAUSTED))
            continue
        decisions.append(_decision(candidate, contracts))
        sleeve_remaining -= contracts * candidate.price_cents / 100.0
        if is_new_market:
            new_markets += 1
    return PMSizingResult(decisions=tuple(decisions), skipped=tuple(skipped))
