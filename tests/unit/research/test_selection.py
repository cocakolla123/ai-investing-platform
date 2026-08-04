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
