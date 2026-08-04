"""Deep-dive selection — CORE, pure. Design: doc 03 section 1.

Nightly priority: new promotions > event-triggered (earnings within 2 sessions,
new filing, |move| > 2 sigma, quant-rank jump >= 20 percentiles) > staleness
(verdict older than 14 days, or never reviewed). Hard cap enforced in code;
ordering fully deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

from dynamo.domain.models.market import Frozen

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


def _priority(candidate: DeepDiveCandidate) -> int | None:
    if candidate.newly_promoted:
        return _PRIORITY_PROMOTION
    if is_event_triggered(candidate):
        return _PRIORITY_EVENT
    if is_stale(candidate):
        return _PRIORITY_STALE
    return None


def select_deep_dives(
    candidates: Sequence[DeepDiveCandidate], cap: int
) -> list[DeepDiveCandidate]:
    """Pure, deterministic: dedupe by symbol (best priority wins), rank, hard cap."""
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
    ranked = sorted(best.values(), key=lambda item: (item[0], item[1].symbol))
    return [candidate for _, candidate in ranked[:cap]]
