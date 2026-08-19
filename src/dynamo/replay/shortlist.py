"""Deep-dive shortlist selection: diversity controls over a ranked band.

Campaign findings item 19. The control run (`replay-e5a49317bbee`, 74
sessions) picked each day's 4-name slate with a plain
``argmax_4(expected_excess_return)`` and nothing else. Measured result:

- WMT was debated on **52 of 74 days** (70%), JPM 37, ABBV 33, SOL/USD 29,
  NVDA 28 — **20 distinct symbols in total**, while AAPL/GOOGL/AMZN/MSFT
  were never debated once despite complete data;
- day-over-day rank correlation 0.88 (still 0.55 at lag 40): the ranking is
  near-frozen, so argmax is near-constant;
- crypto took **29% of the slots while being 9% of the universe**.

Repairing the feature pipeline (Phase 1) does NOT fix this — the Phase-1
counterfactual showed crypto dominance getting *worse* once their
volatility/momentum features come back. Diversity therefore has to be a
selection-layer control, independent of model quality, which is what this
module is: a pure, deterministic function over a ranked candidate list.

Determinism is a hard requirement (the run id is a hash of the config, and
the same config + same data must reproduce the same slate): no randomness,
no wall clock, no set iteration order leaking into the output — every tie
breaks by candidate rank, then by symbol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from dynamo.domain.symbols import is_crypto_symbol

#: Bucket assigned to every crypto pair, regardless of what ``sectors()``
#: says — crypto is capped as one bucket AND by its own hard slot cap.
CRYPTO_BUCKET = "crypto"
#: Bucket for names the gateway has no sector for (never merged with crypto).
UNKNOWN_BUCKET = "unknown"

#: One prior session: its date and the symbols debated on it (possibly none).
SessionHistory = Sequence[tuple[date, Sequence[str]]]


def bucket_of(symbol: str, sectors: Mapping[str, str]) -> str:
    """Diversity bucket for one symbol: crypto pairs first, else the GICS
    sector (``gw.sectors()`` also buckets funds/ETFs), else ``unknown``."""
    if is_crypto_symbol(symbol):
        return CRYPTO_BUCKET
    return (sectors.get(symbol) or "").strip() or UNKNOWN_BUCKET


def _symbols_in_last(history: SessionHistory, sessions: int) -> frozenset[str]:
    """Every symbol debated in the last ``sessions`` entries of ``history``."""
    if sessions <= 0:
        return frozenset()
    window = list(history)[-sessions:]
    return frozenset(sym for _, symbols in window for sym in symbols)


def _sessions_since_last_debate(history: SessionHistory) -> dict[str, int]:
    """Sessions elapsed since each symbol was last debated (1 = yesterday).

    Symbols absent from ``history`` are absent from the mapping — callers
    treat "never debated" as maximally stale.
    """
    entries = list(history)
    out: dict[str, int] = {}
    for offset, (_, symbols) in enumerate(reversed(entries), start=1):
        for sym in symbols:
            out.setdefault(sym, offset)
    return out


def select_shortlist(
    candidates: Sequence[str],
    slots: int,
    *,
    sectors: Mapping[str, str],
    history: SessionHistory = (),
    crypto_slots: int = 1,
    cooldown_sessions: int = 5,
    rotation_slots: int = 1,
    rotation_lookback_sessions: int = 20,
) -> list[str]:
    """The day's deep-dive slate, best-first, at most ``slots`` names.

    ``candidates`` is the ranked candidate band (best first, ties already
    broken by symbol — see
    :func:`dynamo.replay.runner.shortlist_candidates`). ``history`` is this
    run's prior sessions in ascending order, each carrying the symbols
    debated that day (empty tuples for sessions that debated nothing), so
    "the last K sessions" means calendar sessions, not "the last K days that
    happened to produce dives".

    Rules, applied in this order:

    1. **Cooldown** — symbols debated in the last ``cooldown_sessions``
       sessions are removed from the pool. Over the control window the
       top-band held 31 distinct symbols, so even K=5 forces substantially
       more names through.
    2. **Rotation floor** — up to ``rotation_slots`` of the slate go to the
       highest-ranked candidates NOT debated in the last
       ``rotation_lookback_sessions`` sessions, so the universe gets swept
       instead of orbiting the same head of the ranking.
    3. **Sector-bucketed round-robin** — the rest fill one name per bucket
       per pass, buckets ordered by their best remaining candidate's rank
       (ties by bucket name), so one hot sector cannot monopolize the slate.
    4. **Crypto cap** — at most ``crypto_slots`` crypto pairs per day, in
       total, counting the rotation pick. This cap NEVER relaxes.
    5. **Graceful degradation** — if cooldown leaves fewer eligible names
       than slots (tiny universe, or every candidate recently debated), the
       remaining slots go to the candidates whose last debate is OLDEST
       (ties by rank), so relaxing the cooldown still favors rotation
       instead of snapping back to the head of the ranking. Only the
       cooldown relaxes; the crypto cap still binds. A day is never left
       artificially short of deep dives.

    Returns the picks in candidate-rank order (not selection order).
    """
    ranked: list[str] = []
    rank: dict[str, int] = {}
    for sym in candidates:  # dedupe, preserve rank order
        if sym not in rank:
            rank[sym] = len(ranked)
            ranked.append(sym)
    if slots <= 0 or not ranked:
        return []

    cooled = _symbols_in_last(history, cooldown_sessions)
    seen_recently = _symbols_in_last(history, rotation_lookback_sessions)
    buckets = {sym: bucket_of(sym, sectors) for sym in ranked}

    picked: list[str] = []
    picked_set: set[str] = set()
    crypto_used = 0

    def take(sym: str) -> None:
        nonlocal crypto_used
        picked.append(sym)
        picked_set.add(sym)
        if buckets[sym] == CRYPTO_BUCKET:
            crypto_used += 1

    def blocked(sym: str) -> bool:
        return buckets[sym] == CRYPTO_BUCKET and crypto_used >= crypto_slots

    eligible = [sym for sym in ranked if sym not in cooled]

    # 1./2. rotation floor, over the cooled-down pool.
    rotation_target = min(rotation_slots, slots)
    for sym in eligible:
        if len(picked) >= rotation_target:
            break
        if sym in seen_recently or blocked(sym):
            continue
        take(sym)

    # 3. sector-bucketed round-robin over what is left.
    queues: dict[str, list[str]] = {}
    for sym in eligible:
        if sym not in picked_set:
            queues.setdefault(buckets[sym], []).append(sym)
    while len(picked) < slots and queues:
        took = False
        for bucket, queue in sorted(queues.items(), key=lambda kv: (rank[kv[1][0]], kv[0])):
            if len(picked) >= slots:
                break
            if bucket == CRYPTO_BUCKET and crypto_used >= crypto_slots:
                continue
            take(queue.pop(0))
            took = True
        queues = {bucket: q for bucket, q in queues.items() if q}
        if not took:
            break  # only capped buckets remain

    # 5. graceful degradation: relax the cooldown (never the crypto cap),
    # stalest name first so the relaxation still rotates.
    if len(picked) < slots:
        sessions_since = _sessions_since_last_debate(history)
        stale_first = sorted(
            (sym for sym in ranked if sym not in picked_set),
            key=lambda sym: (-sessions_since.get(sym, len(history) + 1), rank[sym]),
        )
        for sym in stale_first:
            if len(picked) >= slots:
                break
            if blocked(sym):
                continue
            take(sym)

    return sorted(picked, key=lambda sym: rank[sym])
