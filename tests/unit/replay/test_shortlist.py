"""Deep-dive shortlist diversity (campaign findings item 19).

The control run's plain ``argmax_4(expected_excess_return)`` debated WMT on
52 of 74 days, produced only 20 distinct symbols, never once debated
AAPL/GOOGL/AMZN/MSFT, and gave crypto 29% of the slots on 9% of the
universe. These tests pin the selection-layer answer: sector-bucketed
round-robin, a hard crypto cap, a cooldown, a rotation floor — and, in
:func:`test_frozen_ranking_simulation_*`, quantify the effect on the exact
pathology that was measured (a near-frozen day-over-day ranking).

Pure functions, no gateway, no network, no LLM.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from dynamo.domain.models.research import Action, QuantScore
from dynamo.replay.runner import (
    BUY_PERCENTILE,
    shortlist_candidates,
    synthesize_verdicts,
)
from dynamo.replay.shortlist import CRYPTO_BUCKET, bucket_of, select_shortlist

DAY = date(2026, 3, 3)

# A miniature of the real universe's bucket profile: one dominant sector
# (IT), several thin ones, plus crypto and funds.
SECTORS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "ABBV": "Health Care",
    "JNJ": "Health Care",
    "WMT": "Consumer Staples",
    "SPY": "Fund/ETF",
    "QQQ": "Fund/ETF",
}
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]


def sessions(n: int, start: date = DAY) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def history_of(*days: tuple[date, list[str]]) -> tuple[tuple[date, list[str]], ...]:
    return tuple(days)


# --------------------------------------------------------------------------
# buckets
# --------------------------------------------------------------------------


def test_crypto_is_its_own_bucket_regardless_of_sector_metadata() -> None:
    assert bucket_of("BTC/USD", {"BTC/USD": "Information Technology"}) == CRYPTO_BUCKET
    assert bucket_of("AAPL", SECTORS) == "Information Technology"
    assert bucket_of("ZZZ", SECTORS) == "unknown"  # no sector != crypto


# --------------------------------------------------------------------------
# round-robin / crypto cap / cooldown / rotation
# --------------------------------------------------------------------------


def test_round_robin_fills_from_distinct_sector_buckets() -> None:
    """Four IT names lead the ranking; the slate still spans four buckets."""
    candidates = ["AAPL", "MSFT", "NVDA", "AVGO", "JPM", "ABBV", "WMT"]
    slate = select_shortlist(candidates, 4, sectors=SECTORS, rotation_slots=0)
    assert slate == ["AAPL", "JPM", "ABBV", "WMT"]
    assert len({bucket_of(s, SECTORS) for s in slate}) == 4


def test_round_robin_wraps_to_a_second_pass_when_buckets_run_out() -> None:
    """Fewer buckets than slots: the best remaining name in each bucket goes
    on the second pass rather than leaving the day short."""
    candidates = ["AAPL", "MSFT", "NVDA", "JPM"]
    slate = select_shortlist(candidates, 4, sectors=SECTORS, rotation_slots=0)
    assert slate == ["AAPL", "MSFT", "NVDA", "JPM"]


def test_crypto_capped_at_one_slot_even_when_it_sweeps_the_ranking() -> None:
    """The measured pathology: crypto held 29% of slots on 9% of the
    universe, and repairing the feature calendar makes that WORSE."""
    candidates = [*CRYPTO, "AAPL", "JPM", "ABBV"]
    slate = select_shortlist(candidates, 4, sectors=SECTORS, rotation_slots=0)
    assert [s for s in slate if s in CRYPTO] == ["BTC/USD"]
    assert slate == ["BTC/USD", "AAPL", "JPM", "ABBV"]


def test_crypto_slots_zero_excludes_crypto_entirely() -> None:
    slate = select_shortlist([*CRYPTO, "AAPL", "JPM"], 4, sectors=SECTORS, crypto_slots=0)
    assert slate == ["AAPL", "JPM"]


def test_crypto_cap_survives_the_cooldown_fallback() -> None:
    """Graceful degradation relaxes the cooldown — never the crypto cap."""
    history = history_of((DAY, ["BTC/USD", "ETH/USD", "SOL/USD"]))
    slate = select_shortlist(CRYPTO, 3, sectors=SECTORS, history=history, cooldown_sessions=5)
    assert slate == ["BTC/USD"]


def test_cooldown_excludes_recently_debated_names() -> None:
    days = sessions(3)
    history = history_of((days[0], ["AAPL", "JPM"]), (days[1], []), (days[2], ["ABBV"]))
    slate = select_shortlist(
        ["AAPL", "JPM", "ABBV", "WMT", "BAC"],
        2,
        sectors=SECTORS,
        history=history,
        cooldown_sessions=5,
        rotation_slots=0,
    )
    assert slate == ["WMT", "BAC"]


def test_cooldown_window_is_sessions_not_all_history() -> None:
    """A name debated 6 sessions ago is eligible again at K=5."""
    days = sessions(6)
    history = history_of((days[0], ["AAPL"]), *((d, []) for d in days[1:]))
    slate = select_shortlist(
        ["AAPL", "JPM"],
        1,
        sectors=SECTORS,
        history=history,
        cooldown_sessions=5,
        rotation_slots=0,
    )
    assert slate == ["AAPL"]


def test_rotation_slot_picks_the_highest_ranked_long_unseen_name() -> None:
    """One slot is reserved for a name not debated in the lookback window, so
    the universe gets swept instead of orbiting the head of the ranking."""
    days = sessions(3)
    # AAPL/MSFT/NVDA lead the ranking and were all debated recently-ish
    # (outside the 1-session cooldown, inside the 20-session rotation window).
    history = history_of((days[0], ["AAPL", "MSFT"]), (days[1], ["NVDA"]), (days[2], []))
    slate = select_shortlist(
        ["AAPL", "MSFT", "NVDA", "WMT"],
        2,
        sectors=SECTORS,
        history=history,
        cooldown_sessions=1,
        rotation_slots=1,
        rotation_lookback_sessions=20,
    )
    assert "WMT" in slate  # the only never-debated candidate
    assert slate == ["AAPL", "WMT"]


def test_degrades_gracefully_when_every_candidate_is_on_cooldown() -> None:
    """A tiny universe must still produce a full slate — the cooldown is a
    preference, not a reason to burn a session."""
    history = history_of((DAY, ["AAPL", "JPM"]))
    slate = select_shortlist(
        ["AAPL", "JPM"], 4, sectors=SECTORS, history=history, cooldown_sessions=5
    )
    assert slate == ["AAPL", "JPM"]


def test_empty_and_degenerate_inputs() -> None:
    assert select_shortlist([], 4, sectors=SECTORS) == []
    assert select_shortlist(["AAPL"], 0, sectors=SECTORS) == []
    # Unknown sectors bucket together but never with crypto.
    slate = select_shortlist(["ZZZ", "YYY"], 2, sectors={})
    assert slate == ["ZZZ", "YYY"]


def test_deterministic_same_inputs_same_slate() -> None:
    """Run-id reproducibility: same config + same data => same slate."""
    days = sessions(2)
    history = history_of((days[0], ["AAPL"]), (days[1], ["BTC/USD"]))
    args = dict(sectors=SECTORS, history=history, cooldown_sessions=3)
    first = select_shortlist([*CRYPTO, "MSFT", "JPM", "ABBV", "WMT"], 4, **args)  # type: ignore[arg-type]
    for _ in range(5):
        assert select_shortlist([*CRYPTO, "MSFT", "JPM", "ABBV", "WMT"], 4, **args) == first  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# candidate band vs BUY action
# --------------------------------------------------------------------------


def _score(symbol: str, eer: float, downside: float = 0.1) -> QuantScore:
    return QuantScore(
        symbol=symbol,
        as_of=DAY,
        model_version="test-v1",
        expected_excess_return=eer,
        downside_prob=downside,
        feature_coverage=1.0,
    )


def _ladder_scores(n: int = 10) -> dict[str, QuantScore]:
    # SYM00 best .. SYM09 worst; the bottom two have negative expectations.
    return {f"SYM{i:02d}": _score(f"SYM{i:02d}", 0.05 - 0.01 * i) for i in range(n)}


def test_widening_the_candidate_band_does_not_change_what_counts_as_a_buy() -> None:
    """The two thresholds are separate constants on purpose: the wide band
    feeds SELECTION, the BUY percentile still decides the ACTION."""
    scores = _ladder_scores()
    verdicts = synthesize_verdicts(scores, DAY, 20)
    buys = {s for s, v in verdicts.items() if v.action is Action.BUY}
    assert buys == {"SYM00", "SYM01", "SYM02"}  # percentile >= 0.8, unchanged

    wide = shortlist_candidates(scores, candidate_percentile=0.6)
    assert wide == ["SYM00", "SYM01", "SYM02", "SYM03", "SYM04"]  # percentile >= 0.6
    # …and synthesizing again over the same scores yields the same BUY set.
    assert {
        s for s, v in synthesize_verdicts(scores, DAY, 20).items() if v.action is Action.BUY
    } == buys


def test_candidate_band_at_the_buy_threshold_reproduces_the_buy_set() -> None:
    """Parity anchor: at candidate_percentile == BUY_PERCENTILE the band IS
    the old shortlist source (the BUY set in watchlist order)."""
    scores = _ladder_scores()
    verdicts = synthesize_verdicts(scores, DAY, 20)
    buys = sorted(
        (v for v in verdicts.values() if v.action is Action.BUY),
        key=lambda v: (-v.conviction, v.symbol),
    )
    assert shortlist_candidates(scores, candidate_percentile=BUY_PERCENTILE) == [
        v.symbol for v in buys
    ]


def test_candidate_band_excludes_negative_expectations_and_quant_avoids() -> None:
    scores = {
        "GOOD": _score("GOOD", 0.05),
        "FLAT": _score("FLAT", 0.0),
        "DOOMED": _score("DOOMED", 0.04, downside=0.7),
        "THIN": QuantScore(
            symbol="THIN",
            as_of=DAY,
            model_version="test-v1",
            expected_excess_return=0.06,
            downside_prob=0.1,
            feature_coverage=0.10,
        ),
    }
    band = shortlist_candidates(scores, min_feature_coverage=0.35, candidate_percentile=0.0)
    assert band == ["GOOD"]


# --------------------------------------------------------------------------
# the measured pathology, simulated
# --------------------------------------------------------------------------


def _frozen_ranking_universe() -> tuple[dict[str, float], dict[str, str]]:
    """55 names shaped like the real universe, with a near-frozen ranking.

    Base scores are fixed per symbol; the per-day perturbation is a
    deterministic sinusoid two orders of magnitude smaller than the spacing
    between adjacent names, which reproduces the measured pathology (rank
    correlation 0.88 day-over-day, 0.55 at lag 40): the ranking barely moves,
    so an unconstrained argmax picks the same handful of names forever.
    """
    sector_names = [
        "Information Technology",
        "Health Care",
        "Consumer Staples",
        "Financials",
        "Communication Services",
        "Consumer Discretionary",
        "Materials",
        "Energy",
        "Industrials",
    ]
    base: dict[str, float] = {}
    sectors: dict[str, str] = {}
    for i in range(40):  # 40 operating companies
        symbol = f"EQ{i:02d}"
        sectors[symbol] = sector_names[i % len(sector_names)]
        base[symbol] = 0.05 - 0.0005 * i
    for i in range(10):  # 10 funds/ETFs
        symbol = f"ETF{i:02d}"
        sectors[symbol] = "Fund/ETF"
        base[symbol] = 0.045 - 0.0005 * i
    for i, symbol in enumerate(["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]):
        sectors[symbol] = "crypto"  # overridden by bucket_of anyway
        base[symbol] = 0.052 - 0.0004 * i  # crypto leads, as measured
    return base, sectors


def _simulate(
    days: int = 40, slots: int = 4, diversified: bool = True
) -> tuple[list[list[str]], set[str]]:
    base, sectors = _frozen_ranking_universe()
    history: list[tuple[date, list[str]]] = []
    slates: list[list[str]] = []
    seen: set[str] = set()
    for d, day in enumerate(sessions(days)):
        scores = {
            symbol: _score(symbol, value + 1e-6 * math.sin(d * 0.7 + hash_index * 0.31))
            for hash_index, (symbol, value) in enumerate(sorted(base.items()))
        }
        if diversified:
            candidates = shortlist_candidates(scores, candidate_percentile=0.6)
            slate = select_shortlist(candidates, slots, sectors=sectors, history=history)
        else:  # the control run's behavior: top-20% band, then argmax_4
            slate = shortlist_candidates(scores, candidate_percentile=0.8)[:slots]
        history.append((day, slate))
        slates.append(slate)
        seen.update(slate)
    return slates, seen


def test_frozen_ranking_simulation_lifts_distinct_names_from_4_to_23() -> None:
    """Over 40 sessions on a near-frozen 55-name ranking: the old top-4 slice
    debates **4 distinct symbols** (the same slate, 40 days running — and all
    four are crypto, exactly the measured 29%-of-slots pathology); the
    diversity selector debates **23**, a 5.75x widening of the research
    surface off the SAME scores.

    23 is the honest ceiling here, not a shortfall: with a frozen ranking the
    top-40% candidate band IS ~22 names, so the selector sweeps essentially
    the whole eligible band instead of orbiting its head. Widening the band
    further is a config knob (``--candidate-percentile``), not a code change.
    This is the control run's "20 distinct symbols in 74 days, WMT on 52 of
    them" reproduced and fixed.
    """
    old_slates, old_seen = _simulate(diversified=False)
    _, new_seen = _simulate(diversified=True)
    assert len(old_seen) == 4
    assert all("/" in s for s in old_seen)  # unconstrained argmax = all crypto
    assert old_slates[0] == old_slates[-1]  # and the same slate every day
    assert len(new_seen) == 23
    assert len(new_seen) >= 5 * len(old_seen)


def test_frozen_ranking_simulation_caps_crypto_and_spreads_sectors() -> None:
    """Same simulation: no day exceeds the crypto cap, no symbol takes more
    than a small share of the 160 slots (the control run gave WMT 70% of
    days and crypto 29% of slots)."""
    slates, _ = _simulate()
    counts: dict[str, int] = {}
    for slate in slates:
        assert len([s for s in slate if "/" in s]) <= 1
        assert len(slate) == 4  # never short-changed
        for symbol in slate:
            counts[symbol] = counts.get(symbol, 0) + 1
    total = sum(counts.values())
    # No name is debated on more than 20% of days (WMT held 70% of days).
    assert max(counts.values()) / len(slates) <= 0.20
    # Crypto lands on exactly the configured ceiling — 1 of 4 slots — where
    # the unconstrained selector gave it 100% of the slates in this
    # simulation (29% in the real control run). `--crypto-slots 0` removes
    # it entirely.
    assert sum(c for s, c in counts.items() if "/" in s) / total == 0.25


def test_simulation_is_deterministic() -> None:
    assert _simulate()[0] == _simulate()[0]
