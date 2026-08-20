"""GO/NO-GO #3: does the discovery report measure what it claims?

The assertions that matter, in order:

1. **It differences against the same day.** A detector that fires only on
   days the whole market rose must NOT score positive. This is the failure
   the pilot's raw flagged-average would have walked into.
2. **It refuses to be confident on overlapping windows.** Effective n is
   days/H, so a 20-session horizon over 60 days is 3 observations and the
   t-statistic must reflect that, not 60.
3. **It calls a real negative detector negative.** volume_spike measured
   -3.03%; the tool has to be able to say so.
4. **It is loud when its own inputs are wrong** — a closes frame holding
   only the flagged names makes every spread zero, and that must surface as
   a warning rather than as "no edge found".
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from dynamo.replay.discovery_report import (
    SURVIVORSHIP_BAND,
    DiscoveryReportError,
    build_discovery_report,
    render_discovery_report,
)

HORIZON = 5
START = date(2026, 3, 2)
N_SESSIONS = 60
UNIVERSE = [f"SYM{i:02d}" for i in range(10)]


def _sessions(n: int = N_SESSIONS) -> list[date]:
    return [d.date() for d in pd.bdate_range(START, periods=n)]


def _closes(paths: dict[str, list[float]], sessions: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": sym, "ts": pd.Timestamp(day), "close": px}
            for sym, series in paths.items()
            for day, px in zip(sessions, series, strict=True)
        ]
    )


def _flat_universe(sessions: list[date], level: float = 100.0) -> dict[str, list[float]]:
    return {sym: [level] * len(sessions) for sym in UNIVERSE}


def _drifting(sessions: list[date], per_session: float, *, seed: int = 0) -> list[float]:
    """A path with real session-to-session variation around a drift.

    Deliberately NOT a constant compounding rate. A constant rate makes every
    daily spread identical, the sample variance float noise, and the
    t-statistic meaningless-but-enormous — a fixture that would "prove" any
    detector significant. Deterministic (no RNG) so runs stay reproducible.
    """
    prices = [100.0]
    for i in range(1, len(sessions)):
        wobble = 0.03 * math.sin(float(seed) + i * 1.7)
        prices.append(prices[-1] * (1.0 + per_session + wobble))
    return prices


def _flags(
    detector: str, days: list[date], symbols: list[str], *, promoted: bool = False
) -> list[dict[str, object]]:
    return [
        {
            "as_of": day,
            "symbol": sym,
            "detector": detector,
            "severity": 0.5,
            "promoted": promoted,
        }
        for day in days
        for sym in symbols
    ]


def test_a_detector_that_only_fires_in_a_rally_scores_zero() -> None:
    """The load-bearing one. Every name rises 1%/session, so a flagged name's
    RAW forward return is strongly positive while it has predicted nothing.
    Differencing against the same session's unflagged names must show that."""
    sessions = _sessions()
    rising = {sym: [100.0 * (1.01**i) for i in range(len(sessions))] for sym in UNIVERSE}
    bench = {day: rising[UNIVERSE[0]][i] for i, day in enumerate(sessions)}

    report = build_discovery_report(
        "replay-rally",
        _flags("rally_rider", sessions[:-HORIZON], [UNIVERSE[0], UNIVERSE[1]]),
        _closes(rising, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.mean_spread == pytest.approx(0.0, abs=1e-9), (
        "a detector firing on names that move exactly like everything else "
        "has no edge, however much the market rose"
    )
    assert stat.verdict == "inconclusive"


def test_a_genuinely_predictive_detector_is_measured_positive() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    winner = UNIVERSE[0]
    # The flagged name compounds; everything else is flat.
    paths[winner] = _drifting(sessions, 0.02, seed=1)
    bench = {day: 100.0 for day in sessions}

    report = build_discovery_report(
        "replay-edge",
        _flags("real_edge", sessions[:-HORIZON], [winner], promoted=True),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.mean_spread > SURVIVORSHIP_BAND
    assert stat.verdict == "positive"
    assert report.distinct_promoted == 1


def test_a_contra_indicative_detector_is_named_as_one() -> None:
    """volume_spike measured -3.03% in the pilot. The tool must be able to
    return that verdict, not merely 'not positive'."""
    sessions = _sessions()
    paths = _flat_universe(sessions)
    loser = UNIVERSE[0]
    paths[loser] = _drifting(sessions, -0.02, seed=2)
    bench = {day: 100.0 for day in sessions}

    report = build_discovery_report(
        "replay-contra",
        _flags("volume_spike", sessions[:-HORIZON], [loser]),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    assert report.detectors[0].verdict == "CONTRA-INDICATIVE"
    assert "volume_spike" in render_discovery_report(report)


def test_effective_n_discounts_overlapping_forward_windows() -> None:
    """55 scored days at a 5-session horizon is ~11 independent observations,
    and the t-statistic must be computed on 11, not on 55."""
    sessions = _sessions()
    paths = _flat_universe(sessions)
    paths[UNIVERSE[0]] = _drifting(sessions, 0.02, seed=1)
    bench = {day: 100.0 for day in sessions}

    report = build_discovery_report(
        "replay-overlap",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]]),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.effective_n == pytest.approx(stat.n_days / HORIZON)
    assert stat.effective_n < stat.n_days / 4.0

    # ...and the discount is real: the naive t would be sqrt(H) times larger.
    naive_scale = (stat.n_days / stat.effective_n) ** 0.5
    assert naive_scale == pytest.approx(HORIZON**0.5)


def test_truncated_flags_are_dropped_and_counted() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    bench = {day: 100.0 for day in sessions}
    # Flag EVERY session, including the last H which have no forward window.
    report = build_discovery_report(
        "replay-truncated",
        _flags("d", sessions, [UNIVERSE[0]]),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.n_flags == len(sessions)
    assert stat.n_scored == len(sessions) - HORIZON
    assert any("unscored" in w for w in report.warnings)


def test_a_closes_frame_of_only_flagged_names_warns_instead_of_reading_zero() -> None:
    """The silent-failure mode: with no unflagged baseline every spread
    collapses to zero, which looks exactly like 'discovery found nothing'."""
    sessions = _sessions()
    only_flagged = {UNIVERSE[0]: [100.0] * len(sessions)}
    bench = {day: 100.0 for day in sessions}

    report = build_discovery_report(
        "replay-selfref",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]]),
        _closes(only_flagged, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    assert any("unflagged baseline" in w for w in report.warnings)


def test_promoted_share_of_decisions() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    bench = {day: 100.0 for day in sessions}
    report = build_discovery_report(
        "replay-share",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]], promoted=True),
        _closes(paths, sessions),
        bench,
        decisions=[
            {"symbol": UNIVERSE[0]},
            {"symbol": UNIVERSE[0]},
            {"symbol": "CORE"},
            {"symbol": "CORE"},
        ],
        horizon_sessions=HORIZON,
    )
    assert report.promoted_share_of_decisions == pytest.approx(0.5)


def test_a_run_with_no_flags_refuses_rather_than_reporting_zeroes() -> None:
    with pytest.raises(DiscoveryReportError, match="no flags"):
        build_discovery_report("replay-empty", [], pd.DataFrame(), {})


def test_per_symbol_forward_windows_survive_a_missing_session() -> None:
    """A symbol that does not trade one session must be measured over ITS
    next H sessions, not a calendar window — the calendar-mixing bug that
    NaN'd 16 features for every equity in the catalog."""
    sessions = _sessions()
    paths = _flat_universe(sessions)
    gappy = UNIVERSE[0]
    paths[gappy] = _drifting(sessions, 0.02, seed=1)

    frame = _closes(paths, sessions)
    hole = pd.Timestamp(sessions[10])
    frame = frame[~((frame["symbol"] == gappy) & (frame["ts"] == hole))]

    report = build_discovery_report(
        "replay-gap",
        _flags("d", sessions[:-HORIZON], [gappy]),
        frame,
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    # 2%/session compounded over 5 of the symbol's own sessions.
    assert stat.mean_excess == pytest.approx(1.02**HORIZON - 1.0, rel=0.02)


def test_render_is_stable_and_shows_the_pass_bands() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    report = build_discovery_report(
        "replay-render",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]]),
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    text = render_discovery_report(report)
    assert "pass: 30-120/day" in text
    assert "pass: 250-800" in text
    assert "survivorship" in text


def test_window_bounds_come_from_the_flags() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    days = sessions[3:-HORIZON]
    report = build_discovery_report(
        "replay-window",
        _flags("d", days, [UNIVERSE[0]]),
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    assert report.window_start == days[0]
    assert report.window_end == days[-1]
    assert report.n_sessions_with_flags == len(days)


def test_detectors_are_ordered_best_spread_first() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    paths[UNIVERSE[0]] = _drifting(sessions, 0.02, seed=1)
    paths[UNIVERSE[1]] = _drifting(sessions, -0.02, seed=2)
    bench = {day: 100.0 for day in sessions}
    days = sessions[:-HORIZON]

    report = build_discovery_report(
        "replay-order",
        _flags("bad", days, [UNIVERSE[1]]) + _flags("good", days, [UNIVERSE[0]]),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    assert [d.detector for d in report.detectors] == ["good", "bad"]


def test_a_flag_on_a_symbol_with_no_closes_is_unscored_not_crashed() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    report = build_discovery_report(
        "replay-missing",
        _flags("d", sessions[:-HORIZON], ["NOT_IN_CLOSES"]),
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.n_scored == 0
    assert stat.mean_spread == 0.0
    assert stat.verdict == "inconclusive"


def test_horizon_beyond_the_data_scores_nothing_rather_than_inventing_returns() -> None:
    sessions = _sessions(10)
    paths = {sym: [100.0] * len(sessions) for sym in UNIVERSE}
    report = build_discovery_report(
        "replay-short",
        _flags("d", sessions[:2], [UNIVERSE[0]]),
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=50,
    )
    assert report.detectors[0].n_scored == 0


def test_benchmark_gap_drops_the_day_rather_than_treating_it_as_zero() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    paths[UNIVERSE[0]] = _drifting(sessions, 0.02, seed=1)
    # Benchmark missing entirely for the first 20 sessions.
    bench = {day: 100.0 for day in sessions[20:]}

    report = build_discovery_report(
        "replay-benchgap",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]]),
        _closes(paths, sessions),
        bench,
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.n_scored < len(sessions) - HORIZON
    assert stat.n_days > 0, "the days that DO have a benchmark must still score"


def test_promotion_flag_does_not_change_the_measured_spread() -> None:
    """Promotion is a routing decision, not an outcome. Two identical runs
    differing only in `promoted` must measure the same edge."""
    sessions = _sessions()
    paths = _flat_universe(sessions)
    paths[UNIVERSE[0]] = _drifting(sessions, 0.02, seed=1)
    bench = {day: 100.0 for day in sessions}
    days = sessions[:-HORIZON]
    closes = _closes(paths, sessions)

    plain = build_discovery_report(
        "a", _flags("d", days, [UNIVERSE[0]]), closes, bench, horizon_sessions=HORIZON
    )
    promoted = build_discovery_report(
        "b",
        _flags("d", days, [UNIVERSE[0]], promoted=True),
        closes,
        bench,
        horizon_sessions=HORIZON,
    )
    assert plain.detectors[0].mean_spread == promoted.detectors[0].mean_spread
    assert plain.distinct_promoted == 0 and promoted.distinct_promoted == 1


def test_flags_per_day_summary() -> None:
    sessions = _sessions()
    paths = _flat_universe(sessions)
    days = sessions[:-HORIZON]
    flags = _flags("d", days, UNIVERSE[:3])
    flags += _flags("e", [days[0]], UNIVERSE[:5])  # one busy day

    report = build_discovery_report(
        "replay-perday",
        flags,
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    assert report.flags_per_day_max == 8
    assert report.flags_per_day_median == 3.0
    assert report.distinct_flagged == 5



def test_a_constant_spread_is_inconclusive_not_infinitely_confident() -> None:
    """The guard that makes the tests above mean something.

    A perfectly constant compounding path gives every day the identical
    spread, so the sample variance is float noise (~1e-34) and the naive
    t-statistic lands around 1e17 — a fixture could 'prove' any detector
    significant. Degenerate variance must read as no evidence.
    """
    sessions = _sessions()
    paths = _flat_universe(sessions)
    paths[UNIVERSE[0]] = [100.0 * (1.02**i) for i in range(len(sessions))]

    report = build_discovery_report(
        "replay-degenerate",
        _flags("d", sessions[:-HORIZON], [UNIVERSE[0]]),
        _closes(paths, sessions),
        {day: 100.0 for day in sessions},
        horizon_sessions=HORIZON,
    )
    stat = report.detectors[0]
    assert stat.mean_spread > SURVIVORSHIP_BAND, "the spread itself is still large"
    assert stat.t_stat == 0.0
    assert stat.verdict == "inconclusive"
