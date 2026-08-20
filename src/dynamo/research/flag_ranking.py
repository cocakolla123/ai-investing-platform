"""Rank tonight's surveillance flags before triage — CORE, pure.

The problem this fixes
----------------------
Triage took ``sorted(flags, key=lambda f: -f.severity)`` and kept the first
60 symbols. That ordering is meaningless, because the detectors do not
share a severity scale:

- ``range_break`` emits a **hardcoded 0.6** for every breakout, whatever
  its size;
- ``volume_spike`` emits ``ratio / (2 * multiple)``, clamped to 1.0 — so a
  day at 10x its 60-day median dollar volume scores 1.0, and *any* spike
  past 6x outranks *every* 52-week breakout that exists;
- ``price_zscore`` emits ``z / (2 * k)``, a third scale again;
- ``range_bound`` emits ``0.4 + 0.3*oscillation + 0.3*depth``, i.e. a
  floor of 0.4 and a ceiling of 1.0 that is not comparable to either.

A routine volume blip therefore outranked a genuine 52-week breakout,
every night, structurally. The cap was selecting the LOUDEST 60 symbols,
not the best 60.

The fix, in three parts
-----------------------
1. **Cross-sectional normalization.** Each flag's severity is replaced by
   its percentile *within its own detector's flags for that night*. This is
   the part that makes detectors comparable: "the biggest volume spike
   tonight" and "the biggest breakout tonight" both become 1.0, and a
   middling one of either becomes 0.5, regardless of the raw scales. It
   also neutralizes ``range_break``'s constant 0.6 honestly — every
   breakout ties, so they all sit at the same percentile and are separated
   by the corroboration term and the weights rather than by an accident of
   which detector happened to pick a bigger constant.

2. **Weighted composite.** Per-detector weights say what a top-percentile
   flag from each detector is worth; a firing floor
   (:data:`DEFAULT_FIRING_FLOOR`) says how much of that a detector earns
   just by firing, so a detector with no severity resolution is not
   punished for its own formula. A symbol's score is its best weighted
   contribution, plus a corroboration term for independent detectors
   agreeing — two different detectors firing on one name is real evidence,
   ten flags from one detector is not.

3. **Diversity and cooldown**, delegated to
   :func:`dynamo.research.shortlist.select_shortlist` — the same
   sector-bucketed round-robin, cooldown, and rotation floor the replay
   deep-dive slate already uses, rather than a second implementation of
   the same idea. Twenty regional banks flagging on one macro move now
   take a few slots between them instead of twenty.

Determinism is a hard requirement, as everywhere else in selection: no
randomness, no wall clock, no set-iteration order. Every tie breaks by
score, then by symbol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from dynamo.research.shortlist import SessionHistory, select_shortlist


class HasSeverity(Protocol):
    """The slice of ``SurveillanceFlag`` this module reads."""

    @property
    def symbol(self) -> str: ...

    @property
    def detector(self) -> str: ...

    @property
    def severity(self) -> float: ...


#: How much a TOP-percentile flag from each detector is worth, before
#: corroboration. These are judgement calls and are meant to be tuned, so
#: here is the reasoning rather than just the numbers.
#:
#: - ``range_break`` (1.0) and ``range_bound`` (0.9): structural,
#:   slow-moving facts about where price sits in a year of its own range.
#:   They are the detectors the funnel exists to surface, and the ones the
#:   old ordering suppressed entirely.
#: - ``filing_event`` (1.0) and ``insider_cluster`` (0.95): discrete
#:   corporate events with known information content.
#: - ``earnings_surprise`` (0.9).
#: - ``price_zscore`` (0.7) and ``rel_strength`` (0.7): real signals, but
#:   frequently just beta or a sector move.
#: - ``news_burst`` (0.6): headline counts are the noisiest input here, and
#:   the social/news surface is the one most easily manufactured.
#: - ``volume_spike`` (0.5): the loudest and least specific detector. It
#:   dominated the old ordering purely because its severity formula
#:   saturates fastest; on its own it is the weakest reason to spend a
#:   triage call.
DEFAULT_DETECTOR_WEIGHTS: Mapping[str, float] = {
    "range_break": 1.0,
    "filing_event": 1.0,
    "insider_cluster": 0.95,
    "range_bound": 0.9,
    "earnings_surprise": 0.9,
    "price_zscore": 0.7,
    "rel_strength": 0.7,
    "news_burst": 0.6,
    "volume_spike": 0.5,
}

#: Weight of an unknown detector. Deliberately mid-scale, not 0: a new
#: detector should be visible in the ranking on the night it ships rather
#: than silently ranking last forever.
DEFAULT_UNKNOWN_WEIGHT = 0.7

#: How much of a detector's weight is earned simply by FIRING, with the
#: percentile earning the rest: ``weight * (floor + (1 - floor) * pctile)``.
#:
#: Without this, a detector whose severity carries no resolution is
#: punished for its own formula. ``range_break`` emits a constant 0.6, so
#: every breakout ties at percentile 0.5 — and a bare percentile composite
#: would then score a top-of-the-night volume spike (weight 0.5 x 1.0)
#: exactly level with a genuine 52-week breakout (weight 1.0 x 0.5),
#: reintroducing the failure in a subtler form. A detector firing at all is
#: information; where the flag sits among its peers is additional
#: information. 0.5 splits them evenly.
DEFAULT_FIRING_FLOOR = 0.5

#: Maximum bonus for independent corroboration, reached asymptotically as
#: the number of DISTINCT detectors firing on a symbol grows. Two detectors
#: earn half of it, three earn two thirds. Repeated flags from the SAME
#: detector earn nothing — they are one observation, not two.
DEFAULT_CORROBORATION_BONUS = 0.30


def detector_percentiles(flags: Sequence[HasSeverity]) -> list[float]:
    """Each flag's severity as a percentile within its own detector.

    Ties share the midpoint of the ranks they span, so ``range_break``'s
    constant 0.6 maps every breakout to the same value instead of ordering
    them by dict insertion. A detector with a single flag scores 0.5:
    one observation carries no cross-sectional information, and calling it
    "the best" would hand a lone volume spike the top of the ranking —
    precisely the failure being repaired.
    """
    by_detector: dict[str, list[float]] = {}
    for flag in flags:
        by_detector.setdefault(flag.detector, []).append(float(flag.severity))

    ranks: dict[str, dict[float, float]] = {}
    for detector, severities in by_detector.items():
        ordered = sorted(severities)
        n = len(ordered)
        table: dict[float, float] = {}
        for value in set(ordered):
            if n == 1:
                table[value] = 0.5
                continue
            below = sum(1 for s in ordered if s < value)
            equal = sum(1 for s in ordered if s == value)
            # Midpoint of the span this value occupies, normalized to [0, 1].
            table[value] = (below + (equal - 1) / 2.0) / (n - 1)
        ranks[detector] = table

    return [ranks[f.detector][float(f.severity)] for f in flags]


def score_symbols(
    flags: Sequence[HasSeverity],
    *,
    weights: Mapping[str, float] = DEFAULT_DETECTOR_WEIGHTS,
    unknown_weight: float = DEFAULT_UNKNOWN_WEIGHT,
    firing_floor: float = DEFAULT_FIRING_FLOOR,
    corroboration_bonus: float = DEFAULT_CORROBORATION_BONUS,
) -> dict[str, float]:
    """Composite score per symbol from tonight's flags.

    ``score = max_d(weight_d * (floor + (1 - floor) * pctile_d))
              + bonus * (1 - 1/n_detectors)``

    The max (not the sum) is deliberate: a symbol should be ranked by its
    single strongest reason, so that ten weak volume flags cannot add up to
    one strong breakout. Corroboration is then a bounded bonus on top,
    counting DISTINCT detectors only.
    """
    percentiles = detector_percentiles(flags)
    best: dict[str, float] = {}
    detectors_seen: dict[str, set[str]] = {}
    for flag, percentile in zip(flags, percentiles, strict=True):
        weight = float(weights.get(flag.detector, unknown_weight))
        contribution = weight * (firing_floor + (1.0 - firing_floor) * percentile)
        symbol = flag.symbol
        best[symbol] = max(best.get(symbol, 0.0), contribution)
        detectors_seen.setdefault(symbol, set()).add(flag.detector)

    return {
        symbol: score + corroboration_bonus * (1.0 - 1.0 / len(detectors_seen[symbol]))
        for symbol, score in best.items()
    }


def rank_symbols(
    flags: Sequence[HasSeverity],
    *,
    weights: Mapping[str, float] = DEFAULT_DETECTOR_WEIGHTS,
    unknown_weight: float = DEFAULT_UNKNOWN_WEIGHT,
    firing_floor: float = DEFAULT_FIRING_FLOOR,
    corroboration_bonus: float = DEFAULT_CORROBORATION_BONUS,
) -> list[str]:
    """Tonight's flagged symbols, best first. Ties break by symbol."""
    scores = score_symbols(
        flags,
        weights=weights,
        unknown_weight=unknown_weight,
        firing_floor=firing_floor,
        corroboration_bonus=corroboration_bonus,
    )
    return sorted(scores, key=lambda symbol: (-scores[symbol], symbol))


def select_for_triage(
    flags: Sequence[HasSeverity],
    cap: int,
    *,
    sectors: Mapping[str, str] | None = None,
    history: SessionHistory = (),
    weights: Mapping[str, float] = DEFAULT_DETECTOR_WEIGHTS,
    unknown_weight: float = DEFAULT_UNKNOWN_WEIGHT,
    firing_floor: float = DEFAULT_FIRING_FLOOR,
    corroboration_bonus: float = DEFAULT_CORROBORATION_BONUS,
    cooldown_sessions: int = 3,
    rotation_slots: int = 0,
    rotation_lookback_sessions: int = 20,
    crypto_slots: int | None = None,
) -> list[str]:
    """The best ``cap`` symbols to spend triage calls on tonight.

    Ranking first, then the diversity layer:

    - **Cooldown** (default 3 sessions) stops the same name consuming a
      triage call every night off the same slow-moving flag. It is shorter
      than the deep-dive slate's 5 because triage is cheap and its job is
      breadth; it is not zero because a range_break flag persists for days.
    - **Sector round-robin** stops one macro move — twenty regional banks
      on a rate print — from eating twenty of the sixty slots.
    - **Rotation** defaults OFF (0 slots): the deep-dive slate needs
      forced sweeping because its ranking is near-frozen, whereas triage
      already sees a fresh flag set nightly.
    - The **crypto cap** defaults to a tenth of the cap (at least one).
      Tier-2 is equities today, so this is a bound, not a live constraint.

    ``sectors`` maps symbol -> GICS sector (``instruments.sector``);
    symbols missing from it share the ``unknown`` bucket, which is never
    merged with crypto. Returns picks in rank order.
    """
    if cap <= 0 or not flags:
        return []
    ranked = rank_symbols(
        flags,
        weights=weights,
        unknown_weight=unknown_weight,
        firing_floor=firing_floor,
        corroboration_bonus=corroboration_bonus,
    )
    return select_shortlist(
        ranked,
        cap,
        sectors=sectors or {},
        history=history,
        crypto_slots=max(1, cap // 10) if crypto_slots is None else crypto_slots,
        cooldown_sessions=cooldown_sessions,
        rotation_slots=rotation_slots,
        rotation_lookback_sessions=rotation_lookback_sessions,
    )


def group_by_symbol(flags: Sequence[Any], symbols: Sequence[str]) -> dict[str, list[Any]]:
    """``symbol -> its flags``, in ``symbols`` order, strongest flag first.

    The ordering inside a symbol is by raw severity, which is fine: within
    one symbol the reader is a prompt, not a ranking, and the incomparable
    scales do no harm there.
    """
    wanted = list(symbols)
    out: dict[str, list[Any]] = {symbol: [] for symbol in wanted}
    for flag in flags:
        if flag.symbol in out:
            out[flag.symbol].append(flag)
    for symbol in wanted:
        out[symbol].sort(key=lambda f: (-float(f.severity), f.detector))
    return out
