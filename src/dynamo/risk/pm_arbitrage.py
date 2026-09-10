"""Prediction-market arbitrage detection — SIM-ONLY sleeve, pure logic.

Two finders in the :mod:`dynamo.risk.pm_sizing` mold — deterministic pure
functions, domain models in, candidates plus string skip reasons out, no I/O
and no sizing:

- :func:`find_dutch_books`: within ONE venue, group a multi-outcome event's
  markets and look for books whose YES legs sum below $1 (buy YES on every
  outcome) or above $1 (buy NO on every outcome) after fees and a slippage
  haircut. Exactly one outcome resolving YES locks the payout.
- :func:`find_cross_venue_arbs`: for matched Kalshi/Polymarket pairs, look
  for a YES-on-one-venue / NO-on-the-other lock whose combined cost is below
  the $1 payout after fees. HARD-gated on caller-verified resolution
  criteria — see the function docstring; an unverified pair is research, not
  a trade.

Candidates carry per-book economics and per-dollar locked profit; the CALLER
sizes them (capital awareness deliberately stays out, exactly as pm_sizing
leaves equity to its caller). Every rule fails closed: a missing, untrusted,
non-finite, or degenerate input skips the whole basket with a recorded
reason — one bad leg poisons the entire book, never just the leg.

Price realism (documented assumption): :class:`PredictionMarketPrice`
carries a single ``implied_prob`` (last trade, else bid/ask mid — see
``dynamo.data.kalshi``), not a bid/ask pair. Fills are therefore assumed
conservatively at ``implied ± slippage``: buys fill at ``implied +
slippage`` and sells at ``implied - slippage``, with ``slippage`` a
configurable haircut. A basket only profitable inside that haircut is not
emitted.

This module is CORE LOGIC under the import-linter contracts: it imports
domain models only and never fetches, stores, or executes anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import PM_VENUE_KALSHI, PM_VENUE_POLYMARKET, CrossVenueSpread, PMSide

SKIP_INVALID_INPUT = "invalid_input"
SKIP_NO_EVENT_KEY = "no_event_key"
SKIP_SINGLE_OUTCOME = "single_outcome_event"
SKIP_MARKET_RESOLVED = "market_resolved"
SKIP_NO_PRICE = "no_price"
SKIP_UNTRUSTED = "untrusted_quote"
SKIP_INVALID_PRICE = "invalid_price"
SKIP_DEGENERATE_FILL = "degenerate_fill_price"
SKIP_NET_EDGE_BELOW_FLOOR = "net_edge_below_floor"

BOOK_BUY = "buy_book"  # buy YES on every outcome; costs < $1, pays exactly $1
BOOK_SELL = "sell_book"  # buy NO on every outcome; n-leg book pays exactly $(n-1)

FeePerContractFn = Callable[[float], float]


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def kalshi_fee_per_contract(price: float) -> float:
    """Kalshi trading fee for ONE contract at ``price`` (dollars in (0, 1)).

    Kalshi's fee schedule (https://kalshi.com/docs/kalshi-fee-schedule.pdf)
    charges ``fees = ceil_cents(0.07 * C * P * (1 - P))`` on a trade of ``C``
    contracts at price ``P`` — 7% of price times counter-price, rounded UP to
    the next cent, collected on execution. Per contract that is
    ``ceil(7 * P * (1 - P))`` cents: the single-contract fee with the same
    ceiling, which over-counts a multi-contract batch slightly (the ceiling
    of a sum never exceeds the sum of ceilings), so sizing with it is
    conservative — never optimistic.

    Fail closed: a non-finite or out-of-range price is charged the WORST-case
    fee (the maximum at p=0.5) instead of raising or under-charging.
    """
    if not _finite(price) or not 0.0 < price < 1.0:
        price = 0.5
    return math.ceil(7.0 * price * (1.0 - price)) / 100.0


def polymarket_fee_per_contract(price: float) -> float:
    """Polymarket trading fee for one contract: $0 at any price.

    Polymarket's CLOB charges no trading fee on standard markets (the same
    assumption :func:`find_cross_venue_arbs` already makes for its Polymarket
    legs), so the injectable fee for a Polymarket dutch-book scan is a
    documented zero rather than an anonymous lambda. Expect such scans to
    come back EMPTY almost always: on negative-risk (mutually exclusive)
    events, Polymarket's protocol-level neg-risk conversion lets anyone swap
    a full set of NO shares across outcomes for cash, so resting dutch books
    are vacuumed by automated converters almost immediately. Checking is one
    cheap keyless read; finding one would be the anomaly worth investigating.
    """
    return 0.0


def kalshi_event_key(market: PredictionMarket) -> str | None:
    """Event grouping key for one Kalshi market; None when ungroupable.

    :class:`PredictionMarket` carries no explicit event/series field, so the
    key is derived from Kalshi's ticker convention: a market ticker is
    ``{SERIES}-{EVENT}-{OUTCOME}`` (e.g. ``KXHIGHNY-25AUG21-B87.5``) and the
    event ticker is everything before the LAST hyphen. Polymarket ids carry
    no such structure, so any non-Kalshi market returns None — dutch-booking
    under the default key function is Kalshi-only (see
    :func:`find_dutch_books`).
    """
    if market.venue != PM_VENUE_KALSHI:
        return None
    if market.market_id.count("-") < 2:
        return None
    return market.market_id.rsplit("-", 1)[0]


def cross_venue_pair_id(spread: CrossVenueSpread) -> str:
    """The stable identity of one matched pair: ``<kalshi_id>|<polymarket_id>``."""
    return f"{spread.kalshi_market_id}|{spread.polymarket_id}"


@dataclass(frozen=True)
class ArbLeg:
    """One buy order in a basket: ``side`` at the assumed fill ``price``."""

    venue: str
    market_id: str
    side: PMSide
    price: float  # assumed fill price per contract, dollars in (0, 1)
    fee: float  # per-contract fee in dollars (0 for fee-free venues)


@dataclass(frozen=True)
class DutchBookCandidate:
    """One executable book over every outcome of one event (unsized)."""

    event_key: str
    kind: str  # BOOK_BUY | BOOK_SELL
    legs: tuple[ArbLeg, ...]  # one leg per outcome, market_id ascending
    gross_cost: float  # sum of leg prices for a one-contract-per-leg book
    total_fees: float  # sum of leg fees for that book
    payout: float  # guaranteed settlement value of the book
    net_profit: float  # payout - gross_cost - total_fees (locked, per book)
    profit_per_dollar: float  # net_profit / (gross_cost + total_fees)
    rationale: str


@dataclass(frozen=True)
class DutchBookResult:
    candidates: tuple[DutchBookCandidate, ...]
    skipped: tuple[tuple[str, str], ...]  # (event_key or market_id, reason)


@dataclass(frozen=True)
class CrossVenueArbCandidate:
    """YES on one venue + NO on the other for the SAME question (unsized)."""

    pair_id: str
    question: str
    legs: tuple[ArbLeg, ...]  # exactly two: the YES leg, then the NO leg
    gross_gap: float  # |kalshi_yes - polymarket_yes| as quoted
    gross_cost: float  # sum of leg prices for one contract per leg
    total_fees: float
    payout: float  # 1.0 — exactly one leg pays out when criteria match
    net_profit: float  # payout - gross_cost - total_fees (locked, per pair)
    profit_per_dollar: float
    rationale: str


@dataclass(frozen=True)
class CrossVenueArbResult:
    candidates: tuple[CrossVenueArbCandidate, ...]  # verified pairs ONLY
    unverified: tuple[CrossVenueArbCandidate, ...]  # research rows, never trades
    skipped: tuple[tuple[str, str], ...]  # (pair_id, reason)


def _leg_fees(prices: Sequence[float], fee_fn: FeePerContractFn) -> tuple[float, ...] | None:
    """Per-leg fees from the injected fn; None when the fn misbehaves."""
    fees: list[float] = []
    for price in prices:
        fee = fee_fn(price)
        if not _finite(fee) or fee < 0.0:
            return None
        fees.append(fee)
    return tuple(fees)


def _book_candidate(
    event_key: str,
    outcomes: Sequence[PredictionMarket],
    implieds: Sequence[float],
    fee_fn: FeePerContractFn,
    min_net_edge: float,
    slippage: float,
) -> DutchBookCandidate | str:
    """One priced event through both book directions; candidate or skip reason.

    At most one direction can clear the floor (asks sit above bids), so the
    buy side is tried first and the sell side only when the buy side fails.
    """
    count = len(outcomes)

    asks = tuple(p + slippage for p in implieds)
    if all(0.0 < a < 1.0 for a in asks):
        fees = _leg_fees(asks, fee_fn)
        if fees is None:
            return SKIP_INVALID_INPUT
        gross_cost, total_fees = sum(asks), sum(fees)
        net = 1.0 - gross_cost - total_fees
        if net >= min_net_edge:
            legs = tuple(
                ArbLeg(venue=m.venue, market_id=m.market_id, side=PMSide.YES, price=a, fee=f)
                for m, a, f in zip(outcomes, asks, fees, strict=True)
            )
            return DutchBookCandidate(
                event_key=event_key,
                kind=BOOK_BUY,
                legs=legs,
                gross_cost=gross_cost,
                total_fees=total_fees,
                payout=1.0,
                net_profit=net,
                profit_per_dollar=net / (gross_cost + total_fees),
                rationale=(
                    f"arb:dutch {BOOK_BUY} {event_key}: buy YES on {count} outcomes, "
                    f"cost {gross_cost:.3f} + fees {total_fees:.3f} vs $1.00 payout "
                    f"locks {net:.3f}/book (fills assumed at implied+{slippage:.2f})"
                ),
            )

    bids = tuple(p - slippage for p in implieds)
    if all(0.0 < b < 1.0 for b in bids):
        no_prices = tuple(1.0 - b for b in bids)
        fees = _leg_fees(no_prices, fee_fn)
        if fees is None:
            return SKIP_INVALID_INPUT
        gross_cost, total_fees = sum(no_prices), sum(fees)
        payout = float(count - 1)  # all outcomes but the winner settle NO at $1
        net = payout - gross_cost - total_fees
        if net >= min_net_edge:
            legs = tuple(
                ArbLeg(venue=m.venue, market_id=m.market_id, side=PMSide.NO, price=n, fee=f)
                for m, n, f in zip(outcomes, no_prices, fees, strict=True)
            )
            return DutchBookCandidate(
                event_key=event_key,
                kind=BOOK_SELL,
                legs=legs,
                gross_cost=gross_cost,
                total_fees=total_fees,
                payout=payout,
                net_profit=net,
                profit_per_dollar=net / (gross_cost + total_fees),
                rationale=(
                    f"arb:dutch {BOOK_SELL} {event_key}: buy NO on {count} outcomes "
                    f"(= sell the YES book), cost {gross_cost:.3f} + fees "
                    f"{total_fees:.3f} vs ${payout:.2f} payout locks {net:.3f}/book "
                    f"(fills assumed at implied-{slippage:.2f})"
                ),
            )

    return SKIP_NET_EDGE_BELOW_FLOOR


def find_dutch_books(
    markets: Sequence[PredictionMarket],
    prices: Mapping[str, PredictionMarketPrice],
    *,
    fee_per_contract_fn: FeePerContractFn = kalshi_fee_per_contract,
    min_net_edge: float = 0.02,
    slippage: float = 0.01,
    event_key_fn: Callable[[PredictionMarket], str | None] = kalshi_event_key,
) -> DutchBookResult:
    """Scan one venue's events for dutch books (module docstring).

    Grouping: :class:`PredictionMarket` has no explicit event field, so
    outcomes are grouped by ``event_key_fn`` — the default derives Kalshi's
    event ticker from the market ticker (``{SERIES}-{EVENT}-{OUTCOME}``) and
    returns None for every other venue, so **only Kalshi markets are
    grouped**; Polymarket has no equivalent event structure in our domain
    model and its markets are skipped with ``no_event_key``.

    Mutual exclusivity is ASSUMED, not proven: Kalshi events group outcomes
    that resolve one-and-only-one YES in the common case, but the ticker
    convention cannot guarantee it (some events allow multiple or zero YES
    resolutions). That assumption — and the last±slippage fill assumption
    above — is exactly why this sleeve is SIM-ONLY.

    A candidate is emitted when, over ALL of an event's >= 2 outcomes:

    - buy-the-book: ``sum(ask_i) < 1 - fees - min_net_edge`` → buy YES on
      every outcome; exactly one resolves YES, the book pays exactly $1;
    - sell-the-book: ``sum(bid_i) > 1 + fees + min_net_edge`` → buy NO on
      every outcome (Kalshi-native equivalent of selling every YES); all but
      one resolve NO, an n-leg book pays exactly $(n-1).

    Fees come from ``fee_per_contract_fn`` (default
    :func:`kalshi_fee_per_contract`) evaluated at each leg's assumed fill.
    Ordering is deterministic: candidates by locked profit per dollar
    descending, ties by event key; legs by market_id ascending.
    """
    if not _finite(min_net_edge, slippage) or min_net_edge < 0.0 or slippage < 0.0:
        return DutchBookResult(candidates=(), skipped=(("*", SKIP_INVALID_INPUT),))

    groups: dict[str, list[PredictionMarket]] = {}
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for market in markets:
        if market.market_id in seen:  # first row per market wins, like pm_sizing
            continue
        seen.add(market.market_id)
        key = event_key_fn(market)
        if key is None:
            skipped.append((market.market_id, SKIP_NO_EVENT_KEY))
            continue
        groups.setdefault(key, []).append(market)

    candidates: list[DutchBookCandidate] = []
    for key in sorted(groups):
        outcomes = sorted(groups[key], key=lambda m: m.market_id)
        if len(outcomes) < 2:
            skipped.append((key, SKIP_SINGLE_OUTCOME))
            continue
        implieds: list[float] = []
        reason: str | None = None
        for market in outcomes:
            if market.resolved_outcome is not None:
                reason = SKIP_MARKET_RESOLVED
                break
            price = prices.get(market.market_id)
            if price is None:
                reason = SKIP_NO_PRICE
                break
            if not price.trusted:
                reason = SKIP_UNTRUSTED
                break
            implied = price.implied_prob
            if not _finite(implied) or not 0.0 < implied < 1.0:
                reason = SKIP_INVALID_PRICE
                break
            implieds.append(implied)
        if reason is not None:  # one bad leg poisons the whole book — fail closed
            skipped.append((key, reason))
            continue
        outcome = _book_candidate(
            key, outcomes, implieds, fee_per_contract_fn, min_net_edge, slippage
        )
        if isinstance(outcome, str):
            skipped.append((key, outcome))
        else:
            candidates.append(outcome)

    candidates.sort(key=lambda c: (-c.profit_per_dollar, c.event_key))
    return DutchBookResult(candidates=tuple(candidates), skipped=tuple(skipped))


def _pair_candidate(
    spread: CrossVenueSpread,
    pair_id: str,
    fee_fn: FeePerContractFn,
    min_net_edge: float,
    slippage: float,
) -> CrossVenueArbCandidate | str:
    """Both YES/NO directions for one pair; the better one, or a skip reason.

    Fees: ``fee_fn`` is charged on the KALSHI leg only — Polymarket's CLOB
    charges no trading fee on standard markets, so its legs carry fee 0.
    """
    directions: list[tuple[float, tuple[ArbLeg, ArbLeg]]] = []
    quotes = {PM_VENUE_KALSHI: spread.kalshi_yes, PM_VENUE_POLYMARKET: spread.polymarket_yes}
    ids = {PM_VENUE_KALSHI: spread.kalshi_market_id, PM_VENUE_POLYMARKET: spread.polymarket_id}
    for yes_venue, no_venue in (
        (PM_VENUE_KALSHI, PM_VENUE_POLYMARKET),
        (PM_VENUE_POLYMARKET, PM_VENUE_KALSHI),
    ):
        yes_price = quotes[yes_venue] + slippage
        no_price = (1.0 - quotes[no_venue]) + slippage
        if not (0.0 < yes_price < 1.0 and 0.0 < no_price < 1.0):
            continue
        fees = _leg_fees(
            [yes_price if yes_venue == PM_VENUE_KALSHI else no_price],
            fee_fn,
        )
        if fees is None:
            return SKIP_INVALID_INPUT
        kalshi_fee = fees[0]
        yes_leg = ArbLeg(
            venue=yes_venue,
            market_id=ids[yes_venue],
            side=PMSide.YES,
            price=yes_price,
            fee=kalshi_fee if yes_venue == PM_VENUE_KALSHI else 0.0,
        )
        no_leg = ArbLeg(
            venue=no_venue,
            market_id=ids[no_venue],
            side=PMSide.NO,
            price=no_price,
            fee=kalshi_fee if no_venue == PM_VENUE_KALSHI else 0.0,
        )
        net = 1.0 - (yes_price + no_price) - kalshi_fee
        directions.append((net, (yes_leg, no_leg)))
    if not directions:
        return SKIP_DEGENERATE_FILL
    net, legs = max(directions, key=lambda d: d[0])
    if net < min_net_edge:
        return SKIP_NET_EDGE_BELOW_FLOOR
    gross_cost = legs[0].price + legs[1].price
    total_fees = legs[0].fee + legs[1].fee
    return CrossVenueArbCandidate(
        pair_id=pair_id,
        question=spread.question,
        legs=legs,
        gross_gap=abs(spread.kalshi_yes - spread.polymarket_yes),
        gross_cost=gross_cost,
        total_fees=total_fees,
        payout=1.0,
        net_profit=net,
        profit_per_dollar=net / (gross_cost + total_fees),
        rationale=(
            f"arb:xvenue {pair_id}: YES@{legs[0].venue} {legs[0].price:.3f} + "
            f"NO@{legs[1].venue} {legs[1].price:.3f} + fees {total_fees:.3f} vs "
            f"$1.00 payout locks {net:.3f}/pair (quoted gap "
            f"{spread.kalshi_yes - spread.polymarket_yes:+.3f}, fills assumed at "
            f"quote+{slippage:.2f})"
        ),
    )


def find_cross_venue_arbs(
    spread_pairs: Sequence[CrossVenueSpread],
    *,
    min_net_edge: float,
    verified_pairs: AbstractSet[str],
    fee_per_contract_fn: FeePerContractFn = kalshi_fee_per_contract,
    slippage: float = 0.01,
) -> CrossVenueArbResult:
    """Scan matched Kalshi/Polymarket pairs for cross-venue YES/NO locks.

    For each pair, buying YES on the cheaper venue and NO (the YES-equivalent
    hedge) on the other locks a $1 payout when — and ONLY when — both markets
    resolve the same question by the same criteria. When they do, exactly one
    leg pays $1, so a combined cost below ``1 - fees - min_net_edge`` is a
    locked profit regardless of the outcome.

    HARD GUARD — ``verified_pairs``: a candidate is emitted as tradable only
    when its :func:`cross_venue_pair_id` is in ``verified_pairs``, the
    caller-supplied set of pairs whose resolution criteria a human has
    verified to match (same question, same resolution source, same fine
    print, same expiry). Everything else lands in ``result.unverified`` and
    must NEVER be traded. This guard is the entire difference between an
    arbitrage and two uncorrelated bets: matched titles with mismatched
    resolution criteria can resolve OPPOSITE ways, turning the "locked"
    profit into a loss on both legs. No price gap, however wide, overrides
    it.

    Fees: ``fee_per_contract_fn`` (default :func:`kalshi_fee_per_contract`)
    is charged on the Kalshi leg; Polymarket legs are fee-free (its CLOB
    charges no trading fee on standard markets). Fills are assumed at the
    stored aggregator quote plus the ``slippage`` haircut on both legs.
    Ordering is deterministic: both lists by locked profit per dollar
    descending, ties by pair id.
    """
    if not _finite(min_net_edge, slippage) or min_net_edge < 0.0 or slippage < 0.0:
        return CrossVenueArbResult(
            candidates=(), unverified=(), skipped=(("*", SKIP_INVALID_INPUT),)
        )

    verified: list[CrossVenueArbCandidate] = []
    unverified: list[CrossVenueArbCandidate] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spread in spread_pairs:
        pair_id = cross_venue_pair_id(spread)
        if pair_id in seen:  # first row per pair wins, like pm_sizing
            continue
        seen.add(pair_id)
        kalshi_yes, polymarket_yes = spread.kalshi_yes, spread.polymarket_yes
        if (
            not _finite(kalshi_yes, polymarket_yes)
            or not 0.0 < kalshi_yes < 1.0
            or not 0.0 < polymarket_yes < 1.0
        ):
            skipped.append((pair_id, SKIP_INVALID_PRICE))
            continue
        outcome = _pair_candidate(spread, pair_id, fee_per_contract_fn, min_net_edge, slippage)
        if isinstance(outcome, str):
            skipped.append((pair_id, outcome))
        elif pair_id in verified_pairs:
            verified.append(outcome)
        else:
            unverified.append(outcome)

    def order(candidate: CrossVenueArbCandidate) -> tuple[float, str]:
        return (-candidate.profit_per_dollar, candidate.pair_id)

    verified.sort(key=order)
    unverified.sort(key=order)
    return CrossVenueArbResult(
        candidates=tuple(verified), unverified=tuple(unverified), skipped=tuple(skipped)
    )
