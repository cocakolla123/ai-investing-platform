"""Deep-dive selection — CORE, pure. Design: doc 03 section 1.

Nightly priority: new promotions > event-triggered (earnings within 2 sessions,
new filing, |move| > 2 sigma, quant-rank jump >= 20 percentiles) > staleness
(verdict older than 14 days, or never reviewed). Hard cap enforced in code;
ordering fully deterministic.

Two things this module used to get wrong
----------------------------------------
**Ties broke alphabetically.** ``sorted(..., key=(priority, symbol))`` meant
that on a night with more candidates than slots — every night — the eight
dives went to whichever qualifying names sort earliest. A name beginning
with A was structurally more likely to be analyzed than one beginning with
W, for no reason connected to the market. Within a priority band the
candidates now rank by :func:`urgency`, a bounded score built from the very
signals that qualified them (how close earnings are, how large the move,
how big the quant-rank jump, how stale the last verdict). The symbol
remains the FINAL tiebreak — determinism is a hard requirement, and two
genuinely identical candidates have to break somehow — but it is no longer
the primary one.

**Promotions could take every slot.** Priority 0 with a hard cap means a
night with 8+ new promotions leaves zero dives for the existing universe.
Measured, that was every night: ~18 promotions against 8 slots, so Tier-1
names — the ones actually held in the portfolio — were never re-examined.
``promotion_slots`` caps how much of the nightly budget brand-new
promotions may claim. Slots are never wasted: leftover promotion slots
fall through to events and staleness, and leftover event/stale slots fall
back to promotions.

Diversity (sector round-robin, per-name cooldown, rotation) is delegated to
:func:`dynamo.research.shortlist.select_shortlist` — the same layer the
replay deep-dive slate uses — rather than reimplemented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dynamo.domain.models.market import Frozen
from dynamo.research.shortlist import SessionHistory, select_shortlist

EARNINGS_SESSIONS_TRIGGER = 2
MOVE_SIGMA_TRIGGER = 2.0
RANK_JUMP_TRIGGER_PCTILES = 20.0
STALENESS_DAYS = 14

_PRIORITY_PROMOTION = 0
_PRIORITY_EVENT = 1
_PRIORITY_STALE = 2


class DeepDiveCandidate(Frozen):
    symbol: str
    newly_promoted: bool = False
    earnings_in_sessions: int | None = None  # sessions until earnings; None = none scheduled
    has_new_filing: bool = False
    move_sigma: float = 0.0
    quant_rank_jump_pctiles: float = 0.0
    last_verdict_age_days: int | None = None  # None = never reviewed


def is_event_triggered(candidate: DeepDiveCandidate) -> bool:
    return (
        (
            candidate.earnings_in_sessions is not None
            and candidate.earnings_in_sessions <= EARNINGS_SESSIONS_TRIGGER
        )
        or candidate.has_new_filing
        or abs(candidate.move_sigma) > MOVE_SIGMA_TRIGGER
        or candidate.quant_rank_jump_pctiles >= RANK_JUMP_TRIGGER_PCTILES
    )


def is_stale(candidate: DeepDiveCandidate) -> bool:
    age = candidate.last_verdict_age_days
    return age is None or age > STALENESS_DAYS


def urgency(candidate: DeepDiveCandidate) -> float:
    """How badly this candidate needs tonight's dive, in [0, ~4.5].

    A sum of bounded components, each normalized against its own trigger
    threshold so no single signal can dominate by unit choice — the same
    incomparable-scales mistake the surveillance severities made. Every
    component is monotone in "more urgent".
    """
    score = 0.0
    if candidate.earnings_in_sessions is not None:
        # 1.0 the session before earnings, decaying to 0 outside the window.
        horizon = EARNINGS_SESSIONS_TRIGGER + 1
        remaining = max(0, horizon - max(candidate.earnings_in_sessions, 0))
        score += remaining / horizon
    if candidate.has_new_filing:
        score += 0.5
    score += min(abs(candidate.move_sigma) / (2.0 * MOVE_SIGMA_TRIGGER), 1.0)
    score += min(candidate.quant_rank_jump_pctiles / (2.0 * RANK_JUMP_TRIGGER_PCTILES), 1.0)
    age = candidate.last_verdict_age_days
    # Never reviewed outranks ANY finite age — no opinion at all is a bigger
    # gap than an old one — so it sits above where finite ages saturate
    # (2x the staleness trigger).
    score += 1.5 if age is None else min(age / (2.0 * STALENESS_DAYS), 1.0)
    return score


def _priority(candidate: DeepDiveCandidate) -> int | None:
    if candidate.newly_promoted:
        return _PRIORITY_PROMOTION
    if is_event_triggered(candidate):
        return _PRIORITY_EVENT
    if is_stale(candidate):
        return _PRIORITY_STALE
    return None


def _ranked_band(candidates: Sequence[DeepDiveCandidate]) -> list[str]:
    """Symbols of one priority band, most urgent first, symbol as last resort."""
    return [
        c.symbol
        for c in sorted(candidates, key=lambda c: (-urgency(c), c.symbol))
    ]


def select_deep_dives(
    candidates: Sequence[DeepDiveCandidate],
    cap: int,
    *,
    promotion_slots: int | None = None,
    sectors: Mapping[str, str] | None = None,
    history: SessionHistory = (),
    cooldown_sessions: int = 0,
    rotation_slots: int = 0,
    rotation_lookback_sessions: int = 20,
    crypto_slots: int = 1,
) -> list[DeepDiveCandidate]:
    """Pure, deterministic: dedupe by symbol (best priority wins), rank, cap.

    ``promotion_slots`` bounds how many of ``cap`` newly-promoted names may
    claim (``None`` = the whole cap, the pre-reserve behavior). Bands are
    filled promotions → events → staleness, each through the shared
    diversity layer, then any slots the bands left unused are backfilled in
    the same priority order.
    """
    if cap <= 0:
        return []
    best: dict[str, tuple[int, DeepDiveCandidate]] = {}
    for candidate in candidates:
        priority = _priority(candidate)
        if priority is None:
            continue
        current = best.get(candidate.symbol)
        if current is None or priority < current[0]:
            best[candidate.symbol] = (priority, candidate)
    if not best:
        return []

    by_symbol = {symbol: candidate for symbol, (_, candidate) in best.items()}
    bands: dict[int, list[DeepDiveCandidate]] = {}
    for priority, candidate in best.values():
        bands.setdefault(priority, []).append(candidate)

    sectors = sectors or {}
    picked: list[str] = []
    picked_set: set[str] = set()

    def fill(band: int, slots: int) -> None:
        if slots <= 0:
            return
        pool = [s for s in _ranked_band(bands.get(band, [])) if s not in picked_set]
        chosen = select_shortlist(
            pool,
            slots,
            sectors=sectors,
            history=history,
            crypto_slots=crypto_slots,
            cooldown_sessions=cooldown_sessions,
            rotation_slots=rotation_slots,
            rotation_lookback_sessions=rotation_lookback_sessions,
        )
        picked.extend(chosen)
        picked_set.update(chosen)

    promotion_cap = cap if promotion_slots is None else min(cap, promotion_slots)
    fill(_PRIORITY_PROMOTION, promotion_cap)
    for band in (_PRIORITY_EVENT, _PRIORITY_STALE):
        fill(band, cap - len(picked))
    # Slots the reserve held back but nothing else claimed go to promotions.
    fill(_PRIORITY_PROMOTION, cap - len(picked))

    order = {symbol: index for index, symbol in enumerate(picked)}
    ranked = sorted(picked, key=lambda s: (best[s][0], order[s]))
    return [by_symbol[symbol] for symbol in ranked[:cap]]
