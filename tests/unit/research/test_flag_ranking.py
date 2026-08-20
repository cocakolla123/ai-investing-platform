"""Flag ranking: the stage that makes incomparable detectors comparable."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.market import SurveillanceFlag
from dynamo.research.flag_ranking import (
    DEFAULT_DETECTOR_WEIGHTS,
    detector_percentiles,
    group_by_symbol,
    rank_symbols,
    score_symbols,
    select_for_triage,
)

AS_OF = datetime(2026, 8, 18, 22, 0, tzinfo=UTC)


def flag(symbol: str, detector: str, severity: float) -> SurveillanceFlag:
    return SurveillanceFlag(symbol=symbol, detector=detector, severity=severity, ts=AS_OF)


class TestPercentileNormalization:
    def test_a_loud_volume_spike_no_longer_outranks_every_breakout(self) -> None:
        """THE regression. volume_spike saturates at 1.0 past 6x median
        volume; range_break emits a hardcoded 0.6. Raw severity therefore
        put every routine blip above every 52-week breakout."""
        flags = [
            flag("BLIP", "volume_spike", 1.0),  # top of its own detector
            flag("QUIET1", "volume_spike", 0.9),
            flag("QUIET2", "volume_spike", 0.8),
            flag("BREAK", "range_break", 0.6),  # the hardcoded constant
            flag("BREAK2", "range_break", 0.6),
        ]
        # Raw severity would have ordered BLIP above BREAK.
        assert max(flags, key=lambda f: f.severity).symbol == "BLIP"
        ranked = rank_symbols(flags)
        assert ranked.index("BREAK") < ranked.index("BLIP")

    def test_top_of_each_detector_lands_at_the_same_percentile(self) -> None:
        flags = [
            flag("A", "volume_spike", 1.0),
            flag("B", "volume_spike", 0.1),
            flag("C", "price_zscore", 0.55),  # a different scale entirely
            flag("D", "price_zscore", 0.50),
        ]
        percentiles = detector_percentiles(flags)
        assert percentiles == [1.0, 0.0, 1.0, 0.0]

    def test_ties_share_the_midpoint_of_their_span(self) -> None:
        """range_break's constant severity must not be ordered by accident."""
        flags = [flag(s, "range_break", 0.6) for s in ("A", "B", "C")]
        assert detector_percentiles(flags) == [0.5, 0.5, 0.5]

    def test_a_lone_flag_scores_neutral_not_best(self) -> None:
        """One observation carries no cross-sectional information; calling
        it 'the best tonight' is exactly the failure being repaired."""
        assert detector_percentiles([flag("A", "volume_spike", 0.99)]) == [0.5]

    def test_percentiles_ignore_the_raw_scale_entirely(self) -> None:
        low = [flag("A", "x", 0.01), flag("B", "x", 0.02), flag("C", "x", 0.03)]
        high = [flag("A", "x", 100.0), flag("B", "x", 200.0), flag("C", "x", 300.0)]
        assert detector_percentiles(low) == detector_percentiles(high)


class TestComposite:
    def test_weights_order_detectors_at_equal_percentile(self) -> None:
        flags = [
            flag("VOL", "volume_spike", 1.0),
            flag("VOL2", "volume_spike", 0.0),
            flag("BRK", "range_break", 1.0),
            flag("BRK2", "range_break", 0.0),
        ]
        scores = score_symbols(flags)
        assert DEFAULT_DETECTOR_WEIGHTS["range_break"] > DEFAULT_DETECTOR_WEIGHTS["volume_spike"]
        assert scores["BRK"] > scores["VOL"]

    def test_corroboration_counts_distinct_detectors_only(self) -> None:
        many_same = [flag("MANY", "volume_spike", s) for s in (1.0, 0.9, 0.8, 0.7)]
        peers = [flag(f"P{i}", "volume_spike", 0.5) for i in range(3)]
        two_different = [flag("TWO", "volume_spike", 1.0), flag("TWO", "price_zscore", 1.0)]
        peers2 = [flag(f"Q{i}", "price_zscore", 0.5) for i in range(3)]
        scores = score_symbols([*many_same, *peers, *two_different, *peers2])
        assert scores["TWO"] > scores["MANY"]

    def test_score_is_the_best_reason_not_the_sum(self) -> None:
        """Ten weak flags must not add up to one strong one."""
        weak = [flag("WEAK", "volume_spike", 0.5) for _ in range(10)]
        weak += [flag(f"F{i}", "volume_spike", v) for i, v in enumerate((0.1, 0.9))]
        strong = [flag("STRONG", "volume_spike", 1.0)]
        scores = score_symbols([*weak, *strong])
        assert scores["STRONG"] > scores["WEAK"]

    def test_an_unknown_detector_is_visible_on_the_night_it_ships(self) -> None:
        flags = [
            flag("NEW", "brand_new_detector", 1.0),
            flag("NEW2", "brand_new_detector", 0.0),
            flag("VOL", "volume_spike", 1.0),
            flag("VOL2", "volume_spike", 0.0),
        ]
        assert score_symbols(flags)["NEW"] > 0.0
        assert rank_symbols(flags).index("NEW") < rank_symbols(flags).index("VOL")


class TestSelectForTriage:
    def test_cap_selects_the_best_not_the_loudest(self) -> None:
        flags = [flag(f"VOL{i}", "volume_spike", 1.0 - i * 0.01) for i in range(20)]
        flags += [flag(f"BRK{i}", "range_break", 0.6) for i in range(3)]
        picked = select_for_triage(flags, 3)
        assert set(picked) == {"BRK0", "BRK1", "BRK2"}

    def test_cooldown_stops_the_same_name_every_night(self) -> None:
        flags = [flag("STICKY", "range_break", 0.6), flag("FRESH", "range_break", 0.6)]
        flags += [flag("OTHER", "volume_spike", 0.9), flag("OTHER2", "volume_spike", 0.1)]
        history = [
            (date(2026, 8, 15), ["STICKY"]),
            (date(2026, 8, 16), ["STICKY"]),
            (date(2026, 8, 17), ["STICKY"]),
        ]
        picked = select_for_triage(flags, 1, history=history, cooldown_sessions=3)
        assert picked == ["FRESH"]

    def test_cooldown_ages_out_over_quiet_sessions(self) -> None:
        flags = [flag("STICKY", "range_break", 0.6), flag("FRESH", "range_break", 0.6)]
        history = [
            (date(2026, 8, 14), ["STICKY"]),
            (date(2026, 8, 15), []),
            (date(2026, 8, 16), []),
            (date(2026, 8, 17), []),
        ]
        picked = select_for_triage(flags, 1, history=history, cooldown_sessions=3)
        assert picked == ["FRESH"]

    def test_sector_dedup_stops_one_macro_move_eating_the_cap(self) -> None:
        """Twenty regional banks flagging on one rate print must not consume
        twenty slots."""
        banks = [flag(f"BANK{i:02d}", "price_zscore", 1.0 - i * 0.001) for i in range(20)]
        others = [
            flag("BIOTECH", "price_zscore", 0.5),
            flag("MINER", "price_zscore", 0.4),
            flag("UTILITY", "price_zscore", 0.3),
        ]
        sectors = {f"BANK{i:02d}": "Financials" for i in range(20)}
        sectors |= {
            "BIOTECH": "Health Care",
            "MINER": "Materials",
            "UTILITY": "Utilities",
        }
        picked = select_for_triage([*banks, *others], 4, sectors=sectors)
        assert len([s for s in picked if s.startswith("BANK")]) == 1
        assert {"BIOTECH", "MINER", "UTILITY"} <= set(picked)

    def test_names_without_a_sector_are_one_bucket_never_merged_with_crypto(self) -> None:
        flags = [flag(f"UNK{i}", "price_zscore", 1.0 - i * 0.01) for i in range(5)]
        flags += [flag("KNOWN", "price_zscore", 0.2)]
        picked = select_for_triage(flags, 2, sectors={"KNOWN": "Energy"})
        assert set(picked) == {"UNK0", "KNOWN"}

    def test_determinism_under_input_reordering(self) -> None:
        flags = [
            flag("A", "volume_spike", 0.9),
            flag("B", "range_break", 0.6),
            flag("C", "price_zscore", 0.7),
            flag("D", "range_break", 0.6),
            flag("E", "volume_spike", 0.2),
        ]
        sectors = {"A": "Energy", "B": "Energy", "C": "Financials", "D": "Materials"}
        first = select_for_triage(flags, 3, sectors=sectors)
        second = select_for_triage(list(reversed(flags)), 3, sectors=sectors)
        assert first == second

    def test_empty_and_degenerate_inputs(self) -> None:
        assert select_for_triage([], 10) == []
        assert select_for_triage([flag("A", "x", 1.0)], 0) == []

    def test_cap_larger_than_the_flag_set_returns_everything(self) -> None:
        flags = [flag("A", "x", 1.0), flag("B", "y", 1.0)]
        assert set(select_for_triage(flags, 50)) == {"A", "B"}


def test_group_by_symbol_preserves_selection_order() -> None:
    flags = [
        flag("A", "volume_spike", 0.2),
        flag("A", "range_break", 0.6),
        flag("B", "price_zscore", 0.9),
        flag("IGNORED", "volume_spike", 1.0),
    ]
    grouped = group_by_symbol(flags, ["B", "A"])
    assert list(grouped) == ["B", "A"]
    assert [f.detector for f in grouped["A"]] == ["range_break", "volume_spike"]
    assert "IGNORED" not in grouped
