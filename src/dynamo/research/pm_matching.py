"""Kalshi <-> Polymarket title matcher — pure core logic, no I/O.

Replaces OddsPipe as the SOURCE of matched cross-venue pairs when the
aggregator returns an empty catalog: both venues' own adapters supply their
market lists, and :func:`match_markets` pairs them by title similarity.

A TITLE MATCH IS NOT RESOLUTION EQUIVALENCE. Two markets can share almost
every word and still resolve by different criteria, different sources, or
different deadlines — which turns an apparent cross-venue arbitrage into two
uncorrelated bets. Every pair this module emits is therefore a CANDIDATE
only: it must be human/committee-verified (both venues' full rules read side
by side) before its pair id is added to ``Settings.pm_arb_verified_pairs``,
and :func:`dynamo.risk.pm_arbitrage.find_cross_venue_arbs` keeps hard-gating
trades on that verified set exactly as before.

Matching rules (deterministic, pure python, no new dependencies):

- titles are normalized: lowercased, punctuation stripped, stopwords dropped,
  digit-group commas removed ("100,000" -> "100000");
- the base score is Jaccard similarity over the two normalized token sets;
- HARD entity requirement: every year (1900-2099) and every other number
  appearing in one title must appear in the other — a "2027" vs "2028" or a
  "$3.50" vs "$4.00" mismatch rejects the pair outright no matter how much
  text overlaps (wrong-year matches are the classic cross-venue killer);
- matching non-empty entity sets add a small boost (shared years +0.10,
  shared non-year numbers +0.05, capped at 1.0);
- mutual exclusivity: a Kalshi market clearing ``min_score`` against two or
  more Polymarket markets — or a Polymarket market claimed by two or more
  Kalshi markets — is reported in ``ambiguous`` and never auto-paired.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from dynamo.domain.models.market import PredictionMarket

# Deliberately small: words like "before"/"after"/"above"/"below" carry
# resolution-relevant meaning in market titles and are NOT stopwords.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_YEAR = re.compile(r"^(19|20)\d{2}$")


@dataclass(frozen=True)
class TitleFeatures:
    """Normalized token set + extracted entities for one title."""

    tokens: frozenset[str]
    years: frozenset[str]
    numbers: frozenset[str]  # non-year numerics, canonicalized ("3.50" -> "3.5")


@dataclass(frozen=True)
class MatchedPair:
    """One unambiguous title match — a VERIFICATION CANDIDATE, not an arb."""

    kalshi_market_id: str
    polymarket_id: str
    kalshi_question: str
    polymarket_question: str
    score: float

    @property
    def pair_id(self) -> str:
        """The id format the arb sleeve's verified_pairs guard consumes."""
        return f"{self.kalshi_market_id}|{self.polymarket_id}"


@dataclass(frozen=True)
class AmbiguousMatch:
    """One Kalshi market with 2+ plausible Polymarket matches (never paired)."""

    kalshi_market_id: str
    kalshi_question: str
    polymarket_ids: tuple[str, ...]  # score descending, ties by id
    scores: tuple[float, ...]


@dataclass(frozen=True)
class UnmatchedCounts:
    kalshi: int
    polymarket: int


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[MatchedPair, ...]  # score descending, ties by pair id
    ambiguous: tuple[AmbiguousMatch, ...]  # kalshi_market_id ascending
    unmatched_counts: UnmatchedCounts


def title_features(title: str) -> TitleFeatures:
    """Normalize one title into tokens + hard-matched entities (pure)."""
    lowered = _DIGIT_COMMA.sub("", title.lower())
    raw = _TOKEN.findall(lowered)
    tokens: set[str] = set()
    years: set[str] = set()
    numbers: set[str] = set()
    for token in raw:
        if token[0].isdigit():
            if _YEAR.match(token):
                years.add(token)
                tokens.add(token)
            else:
                canonical = format(float(token), "g")
                numbers.add(canonical)
                tokens.add(canonical)
        elif token not in _STOPWORDS:
            tokens.add(token)
    return TitleFeatures(
        tokens=frozenset(tokens), years=frozenset(years), numbers=frozenset(numbers)
    )


def pair_score(a: TitleFeatures, b: TitleFeatures) -> float:
    """Similarity in [0, 1]; 0.0 whenever the hard entity requirement fails.

    Jaccard over token sets, plus +0.10 when both share at least one year and
    +0.05 when both share at least one non-year number — but ANY year or
    number present on one side and missing on the other rejects the pair.
    """
    if a.years != b.years or a.numbers != b.numbers:
        return 0.0
    if not a.tokens or not b.tokens:
        return 0.0
    union = len(a.tokens | b.tokens)
    score = len(a.tokens & b.tokens) / union if union else 0.0
    if score <= 0.0:
        return 0.0
    if a.years:  # equal by the guard above
        score += 0.10
    if a.numbers:
        score += 0.05
    return min(score, 1.0)


def _dedupe_sorted(markets: Iterable[PredictionMarket]) -> list[PredictionMarket]:
    """Unique-by-id, market_id ascending — order-independent by construction."""
    by_id: dict[str, PredictionMarket] = {}
    for market in sorted(markets, key=lambda m: (m.market_id, m.question)):
        by_id.setdefault(market.market_id, market)
    return [by_id[key] for key in sorted(by_id)]


def match_markets(
    kalshi_markets: Iterable[PredictionMarket],
    polymarket_markets: Iterable[PredictionMarket],
    *,
    min_score: float,
) -> MatchResult:
    """Pair Kalshi and Polymarket markets by title similarity (module rules).

    A pair is emitted only when exactly ONE Polymarket market clears
    ``min_score`` for a Kalshi market AND that Polymarket market is not also
    the sole match of another Kalshi market — every contested candidate goes
    to ``ambiguous`` instead. Output ordering is deterministic under input
    shuffle (inputs are de-duplicated by market_id and sorted first).

    A TITLE MATCH IS NOT RESOLUTION EQUIVALENCE: emitted pairs still require
    human/committee verification of both venues' resolution criteria before
    trading — the existing ``Settings.pm_arb_verified_pairs`` hard guard in
    :func:`dynamo.risk.pm_arbitrage.find_cross_venue_arbs` stays in force.
    """
    kalshi = _dedupe_sorted(kalshi_markets)
    polymarket = _dedupe_sorted(polymarket_markets)
    pm_features = [(m, title_features(m.question)) for m in polymarket]

    tentative: list[tuple[PredictionMarket, PredictionMarket, float]] = []
    ambiguous: list[AmbiguousMatch] = []
    unmatched_kalshi = 0
    for km in kalshi:
        kf = title_features(km.question)
        candidates = [(m, s) for m, f in pm_features if (s := pair_score(kf, f)) >= min_score]
        if not candidates:
            unmatched_kalshi += 1
            continue
        candidates.sort(key=lambda c: (-c[1], c[0].market_id))
        if len(candidates) > 1:
            ambiguous.append(
                AmbiguousMatch(
                    kalshi_market_id=km.market_id,
                    kalshi_question=km.question,
                    polymarket_ids=tuple(m.market_id for m, _ in candidates),
                    scores=tuple(s for _, s in candidates),
                )
            )
            continue
        pm, score = candidates[0]
        tentative.append((km, pm, score))

    # Reverse exclusivity: one Polymarket market claimed by 2+ Kalshi markets
    # is contested — none of the claimants is auto-paired.
    claims: dict[str, int] = {}
    for _, pm, _score in tentative:
        claims[pm.market_id] = claims.get(pm.market_id, 0) + 1
    pairs: list[MatchedPair] = []
    for km, pm, score in tentative:
        if claims[pm.market_id] > 1:
            ambiguous.append(
                AmbiguousMatch(
                    kalshi_market_id=km.market_id,
                    kalshi_question=km.question,
                    polymarket_ids=(pm.market_id,),
                    scores=(score,),
                )
            )
            continue
        pairs.append(
            MatchedPair(
                kalshi_market_id=km.market_id,
                polymarket_id=pm.market_id,
                kalshi_question=km.question,
                polymarket_question=pm.question,
                score=score,
            )
        )

    pairs.sort(key=lambda p: (-p.score, p.pair_id))
    ambiguous.sort(key=lambda a: a.kalshi_market_id)
    matched_pm = {p.polymarket_id for p in pairs}
    return MatchResult(
        pairs=tuple(pairs),
        ambiguous=tuple(ambiguous),
        unmatched_counts=UnmatchedCounts(
            kalshi=unmatched_kalshi, polymarket=len(polymarket) - len(matched_pm)
        ),
    )
