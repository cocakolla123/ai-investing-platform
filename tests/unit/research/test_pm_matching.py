"""Unit tests for the Kalshi<->Polymarket title matcher (pure core logic)."""

from __future__ import annotations

import random

from dynamo.domain.models.market import PredictionMarket
from dynamo.research.pm_matching import MatchResult, match_markets, pair_score, title_features


def _kalshi(market_id: str, question: str) -> PredictionMarket:
    return PredictionMarket(venue="kalshi", market_id=market_id, question=question)


def _poly(market_id: str, question: str) -> PredictionMarket:
    return PredictionMarket(venue="polymarket", market_id=market_id, question=question)


class TestTitleFeatures:
    def test_normalization_strips_stopwords_and_punctuation(self) -> None:
        f = title_features("Will the Fed cut rates in December 2026?")
        assert f.tokens == frozenset({"fed", "cut", "rates", "december", "2026"})
        assert f.years == frozenset({"2026"})
        assert f.numbers == frozenset()

    def test_numbers_are_canonicalized_and_commas_dropped(self) -> None:
        f = title_features("Bitcoin above $100,000 or 3.50% by March?")
        assert "100000" in f.numbers
        assert "3.5" in f.numbers
        assert f.years == frozenset()

    def test_meaningful_words_like_before_are_kept(self) -> None:
        assert "before" in title_features("Ceasefire before July?").tokens


class TestPairScore:
    def test_year_mismatch_rejects_despite_high_overlap(self) -> None:
        a = title_features("Will Jane Doe win the 2027 mayoral election?")
        b = title_features("Will Jane Doe win the 2028 mayoral election?")
        assert pair_score(a, b) == 0.0

    def test_number_mismatch_rejects(self) -> None:
        a = title_features("Fed funds above 4.00% in December 2026?")
        b = title_features("Fed funds above 4.25% in December 2026?")
        assert pair_score(a, b) == 0.0

    def test_identical_entities_score_high(self) -> None:
        a = title_features("Will the Fed cut rates in December 2026?")
        b = title_features("Fed cut rates in December 2026")
        assert pair_score(a, b) == 1.0  # jaccard 1.0 + year boost, capped


class TestMatchMarkets:
    def test_exact_entity_match_pairs(self) -> None:
        result = match_markets(
            [_kalshi("KXFED-26DEC-T4.00", "Will the Fed cut rates in December 2026?")],
            [
                _poly("514729", "Fed cut rates in December 2026?"),
                _poly("999999", "Will aliens make contact in 2026?"),
            ],
            min_score=0.65,
        )
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.kalshi_market_id == "KXFED-26DEC-T4.00"
        assert pair.polymarket_id == "514729"
        assert pair.pair_id == "KXFED-26DEC-T4.00|514729"
        assert result.unmatched_counts.kalshi == 0
        assert result.unmatched_counts.polymarket == 1

    def test_year_mismatch_never_pairs(self) -> None:
        result = match_markets(
            [_kalshi("KXELECT-27", "Will Jane Doe win the 2027 mayoral election?")],
            [_poly("111", "Will Jane Doe win the 2028 mayoral election?")],
            min_score=0.3,
        )
        assert result.pairs == ()
        assert result.ambiguous == ()
        assert result.unmatched_counts.kalshi == 1

    def test_two_candidates_go_ambiguous_not_paired(self) -> None:
        result = match_markets(
            [_kalshi("KXCUT-26", "Fed cut rates December 2026")],
            [
                _poly("201", "Fed cut rates December 2026"),
                _poly("202", "Fed cut rates in December 2026?"),
            ],
            min_score=0.65,
        )
        assert result.pairs == ()
        assert len(result.ambiguous) == 1
        assert result.ambiguous[0].kalshi_market_id == "KXCUT-26"
        assert set(result.ambiguous[0].polymarket_ids) == {"201", "202"}

    def test_contested_polymarket_market_goes_ambiguous(self) -> None:
        # Two Kalshi markets whose single candidate is the SAME Polymarket
        # market: neither is auto-paired.
        result = match_markets(
            [
                _kalshi("KXA-26", "Fed cut rates December 2026"),
                _kalshi("KXB-26", "Fed cut rates in December 2026?"),
            ],
            [_poly("301", "Fed cut rates December 2026")],
            min_score=0.65,
        )
        assert result.pairs == ()
        assert {a.kalshi_market_id for a in result.ambiguous} == {"KXA-26", "KXB-26"}

    def test_min_score_respected(self) -> None:
        kalshi = [_kalshi("KXGDP-26", "US GDP growth positive in 2026?")]
        poly = [_poly("401", "Will US real GDP growth be positive for calendar year 2026?")]
        strict = match_markets(kalshi, poly, min_score=0.95)
        loose = match_markets(kalshi, poly, min_score=0.4)
        assert strict.pairs == ()
        assert len(loose.pairs) == 1

    def test_deterministic_under_shuffle(self) -> None:
        kalshi = [
            _kalshi("KXFED-26DEC-T4.00", "Will the Fed cut rates in December 2026?"),
            _kalshi("KXBTC-26-100K", "Bitcoin above $100,000 in 2026?"),
            _kalshi("KXGDP-26", "US GDP growth positive in 2026?"),
            _kalshi("KXNOPE-26", "Completely unrelated question about llamas?"),
        ]
        poly = [
            _poly("501", "Fed cut rates in December 2026?"),
            _poly("502", "Bitcoin above $100,000 in 2026?"),
            _poly("503", "US GDP growth positive in 2026?"),
            _poly("504", "Will it rain in Paris tomorrow?"),
        ]
        baseline: MatchResult = match_markets(kalshi, poly, min_score=0.65)
        assert len(baseline.pairs) == 3
        for seed in range(5):
            rng = random.Random(seed)
            k, p = list(kalshi), list(poly)
            rng.shuffle(k)
            rng.shuffle(p)
            assert match_markets(k, p, min_score=0.65) == baseline

    def test_duplicate_ids_are_deduplicated(self) -> None:
        kalshi = [
            _kalshi("KXFED-26DEC-T4.00", "Will the Fed cut rates in December 2026?"),
            _kalshi("KXFED-26DEC-T4.00", "Will the Fed cut rates in December 2026?"),
        ]
        poly = [_poly("601", "Fed cut rates in December 2026?")]
        result = match_markets(kalshi, poly, min_score=0.65)
        assert len(result.pairs) == 1
