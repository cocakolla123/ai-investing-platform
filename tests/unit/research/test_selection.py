"""Deep-dive selection priority + hard cap (doc 03 section 1)."""

from __future__ import annotations

from dynamo.research.selection import DeepDiveCandidate, select_deep_dives


def promoted(symbol: str) -> DeepDiveCandidate:
    return DeepDiveCandidate(symbol=symbol, newly_promoted=True, last_verdict_age_days=1)


def stale(symbol: str, age: int | None = 30) -> DeepDiveCandidate:
    return DeepDiveCandidate(symbol=symbol, last_verdict_age_days=age)


def fresh(symbol: str) -> DeepDiveCandidate:
    return DeepDiveCandidate(symbol=symbol, last_verdict_age_days=3)


class TestPriority:
    def test_promotions_beat_events_beat_staleness(self) -> None:
        candidates = [
            stale("STL"),
            DeepDiveCandidate(symbol="EVT", has_new_filing=True, last_verdict_age_days=2),
            promoted("PRM"),
        ]
        selected = select_deep_dives(candidates, cap=10)
        assert [c.symbol for c in selected] == ["PRM", "EVT", "STL"]

    def test_event_triggers(self) -> None:
        triggers = [
            DeepDiveCandidate(symbol="ERN", earnings_in_sessions=2, last_verdict_age_days=1),
            DeepDiveCandidate(symbol="FIL", has_new_filing=True, last_verdict_age_days=1),
            DeepDiveCandidate(symbol="MOV", move_sigma=-2.5, last_verdict_age_days=1),
            DeepDiveCandidate(
                symbol="RNK", quant_rank_jump_pctiles=20.0, last_verdict_age_days=1
            ),
        ]
        assert len(select_deep_dives(triggers, cap=10)) == 4

    def test_non_triggers_excluded(self) -> None:
        duds = [
            fresh("AAA"),
            DeepDiveCandidate(symbol="BBB", earnings_in_sessions=5, last_verdict_age_days=1),
            DeepDiveCandidate(symbol="CCC", move_sigma=1.9, last_verdict_age_days=1),
            DeepDiveCandidate(
                symbol="DDD", quant_rank_jump_pctiles=19.9, last_verdict_age_days=1
            ),
        ]
        assert select_deep_dives(duds, cap=10) == []

    def test_never_reviewed_counts_as_stale(self) -> None:
        assert select_deep_dives([stale("NEW", age=None)], cap=10)
        assert select_deep_dives([stale("OLD", age=15)], cap=10)
        assert not select_deep_dives([stale("OK", age=14)], cap=10)


class TestCapAndDeterminism:
    def test_hard_cap_enforced(self) -> None:
        candidates = [promoted(f"P{i}") for i in range(5)] + [stale(f"S{i}") for i in range(5)]
        selected = select_deep_dives(candidates, cap=8)
        assert len(selected) == 8
        assert select_deep_dives(candidates, cap=0) == []

    def test_cap_prefers_higher_priority(self) -> None:
        candidates = [stale("S1"), stale("S2"), promoted("P1")]
        selected = select_deep_dives(candidates, cap=1)
        assert [c.symbol for c in selected] == ["P1"]

    def test_deterministic_ordering_and_dedupe(self) -> None:
        candidates = [stale("ZZZ"), promoted("ZZZ"), promoted("AAA")]
        first = select_deep_dives(candidates, cap=10)
        second = select_deep_dives(list(reversed(candidates)), cap=10)
        assert [c.symbol for c in first] == ["AAA", "ZZZ"]
        assert [c.symbol for c in second] == ["AAA", "ZZZ"]
        # dedupe kept the higher-priority (promoted) row
        assert first[1].newly_promoted


class TestCryptoEligibility:
    """ADR-0013: crypto pairs must be selectable for committee deep dives.

    Selection never keys off fundamentals/filings existence — a crypto pair
    with no earnings, no filings, and no fundamentals reaches the committee
    through the same staleness and price-move triggers as any core name, so
    no special floor is needed.
    """

    def test_never_reviewed_crypto_is_selected_via_staleness(self) -> None:
        candidates = [
            DeepDiveCandidate(symbol="BTC/USD", last_verdict_age_days=None),
            fresh("AAPL"),
        ]
        selected = select_deep_dives(candidates, cap=10)
        assert [c.symbol for c in selected] == ["BTC/USD"]

    def test_crypto_price_move_is_an_event_trigger(self) -> None:
        candidates = [
            DeepDiveCandidate(symbol="ETH/USD", move_sigma=3.1, last_verdict_age_days=2),
            stale("STL"),
        ]
        selected = select_deep_dives(candidates, cap=1)
        assert [c.symbol for c in selected] == ["ETH/USD"]  # event beats staleness


class TestTiebreakIsNoLongerAlphabetical:
    """The old `sorted(..., key=(priority, symbol))` meant that on any night
    with more candidates than slots — every night — a name beginning with A
    was structurally more likely to be analyzed than one beginning with W."""

    def test_within_a_band_urgency_beats_the_alphabet(self) -> None:
        candidates = [
            # Alphabetically first, but barely qualifying on every axis.
            DeepDiveCandidate(symbol="AAAA", move_sigma=2.01, last_verdict_age_days=1),
            # Alphabetically last, reporting earnings tomorrow after a 6-sigma
            # move and a 60-percentile quant-rank jump.
            DeepDiveCandidate(
                symbol="ZZZZ",
                earnings_in_sessions=1,
                move_sigma=6.0,
                quant_rank_jump_pctiles=60.0,
                has_new_filing=True,
                last_verdict_age_days=1,
            ),
        ]
        assert [c.symbol for c in select_deep_dives(candidates, cap=1)] == ["ZZZZ"]

    def test_staleness_band_takes_the_stalest_not_the_earliest_letter(self) -> None:
        candidates = [stale("AAAA", age=15), stale("ZZZZ", age=400)]
        assert [c.symbol for c in select_deep_dives(candidates, cap=1)] == ["ZZZZ"]
        # Never reviewed is maximally stale and beats any finite age.
        never = [stale("AAAA", age=999), stale("ZZZZ", age=None)]
        assert [c.symbol for c in select_deep_dives(never, cap=1)] == ["ZZZZ"]

    def test_earnings_proximity_orders_the_event_band(self) -> None:
        candidates = [
            DeepDiveCandidate(symbol="AAAA", earnings_in_sessions=2, last_verdict_age_days=1),
            DeepDiveCandidate(symbol="ZZZZ", earnings_in_sessions=0, last_verdict_age_days=1),
        ]
        assert [c.symbol for c in select_deep_dives(candidates, cap=1)] == ["ZZZZ"]

    def test_symbol_remains_the_final_deterministic_tiebreak(self) -> None:
        """Two genuinely identical candidates still have to break somehow."""
        candidates = [stale("ZZZZ", age=30), stale("AAAA", age=30)]
        assert [c.symbol for c in select_deep_dives(candidates, cap=1)] == ["AAAA"]


class TestPromotionReserve:
    def test_promotions_cannot_claim_the_whole_nightly_budget(self) -> None:
        """Measured before the fix: ~18 promotions a night against 8 slots,
        so Tier-1 names — the ones actually held — got zero dives."""
        candidates = [promoted(f"P{i:02d}") for i in range(18)]
        candidates += [stale(f"CORE{i:02d}", age=30) for i in range(10)]
        selected = select_deep_dives(candidates, cap=8, promotion_slots=5)
        symbols = [c.symbol for c in selected]
        assert len(symbols) == 8
        assert len([s for s in symbols if s.startswith("P")]) == 5
        assert len([s for s in symbols if s.startswith("CORE")]) == 3

    def test_unused_reserve_falls_back_to_promotions(self) -> None:
        """A reserve must not waste slots on a night with nothing else to do."""
        candidates = [promoted(f"P{i}") for i in range(8)]
        selected = select_deep_dives(candidates, cap=8, promotion_slots=5)
        assert len(selected) == 8

    def test_unused_promotion_slots_fall_through_to_the_rest(self) -> None:
        candidates = [promoted("P1"), *[stale(f"S{i}", age=30) for i in range(10)]]
        selected = select_deep_dives(candidates, cap=8, promotion_slots=5)
        assert len(selected) == 8
        assert selected[0].symbol == "P1"

    def test_no_reserve_is_the_old_behavior(self) -> None:
        candidates = [promoted(f"P{i}") for i in range(10)] + [stale("S", age=30)]
        selected = select_deep_dives(candidates, cap=8)
        assert all(c.newly_promoted for c in selected)


class TestDiversityLayer:
    def test_sector_round_robin_stops_one_sector_taking_the_slate(self) -> None:
        candidates = [stale(f"BANK{i:02d}", age=100 - i) for i in range(10)]
        candidates += [stale("MINER", age=20), stale("BIOTECH", age=20)]
        sectors = {f"BANK{i:02d}": "Financials" for i in range(10)}
        sectors |= {"MINER": "Materials", "BIOTECH": "Health Care"}
        selected = [c.symbol for c in select_deep_dives(candidates, cap=3, sectors=sectors)]
        assert len([s for s in selected if s.startswith("BANK")]) == 1
        assert {"MINER", "BIOTECH"} <= set(selected)

    def test_cooldown_excludes_recently_debated_names(self) -> None:
        from datetime import date

        candidates = [stale("RECENT", age=100), stale("NEVER", age=20)]
        history = [(date(2026, 8, 16), ["RECENT"]), (date(2026, 8, 17), ["RECENT"])]
        selected = select_deep_dives(
            candidates, cap=1, history=history, cooldown_sessions=2
        )
        assert [c.symbol for c in selected] == ["NEVER"]

    def test_still_deterministic_with_the_diversity_layer_on(self) -> None:
        candidates = [
            promoted("PRM"),
            stale("S1", age=30),
            stale("S2", age=40),
            DeepDiveCandidate(symbol="EVT", move_sigma=3.0, last_verdict_age_days=1),
        ]
        sectors = {"PRM": "Energy", "S1": "Energy", "S2": "Financials", "EVT": "Materials"}
        first = select_deep_dives(candidates, cap=3, sectors=sectors, promotion_slots=1)
        second = select_deep_dives(
            list(reversed(candidates)), cap=3, sectors=sectors, promotion_slots=1
        )
        assert [c.symbol for c in first] == [c.symbol for c in second]
