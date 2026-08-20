"""Guard: the Tier-2 scan window must cover every detector's lookback.

The failure this protects against is silent. ``tier2_scan`` hands ONE bars
frame to every bar-based detector; when the frame is shorter than a
detector's declared lookback, the detector either computes a wrong answer
(``range_break`` calling a 3-month high a 52-week breakout) or hits its
minimum-bars gate and ``continue``s forever (``range_bound`` produced zero
flags market-wide at the old 100-day window). Neither raises, neither logs.

So the requirement is introspected from :mod:`dynamo.data.detectors` rather
than hardcoded here: add a detector with a longer lookback than the scan
window allows and this test fails, instead of the detector quietly never
firing in production.
"""

from __future__ import annotations

import inspect

import pytest

from dynamo.data import detectors
from dynamo.data.detector_windows import (
    COMBINATION_ALLOWANCE_SESSIONS,
    TIER2_SCAN_LOOKBACK_DAYS,
    bar_detectors,
    calendar_days_for_sessions,
    declared_windows,
    detector_session_requirements,
    max_detector_sessions,
    scan_window_sessions,
    sessions_in_calendar_days,
)


def test_discovers_every_bar_based_detector() -> None:
    """Introspection, not a hardcoded list — a new detector is picked up."""
    found = set(bar_detectors())
    # Every detector ``run_detectors`` feeds bars to must be discovered.
    assert {
        "price_zscore",
        "volume_spike",
        "rel_strength",
        "range_break",
        "range_bound",
    } <= found
    # Event detectors take sequences, not bars, and must NOT be included:
    # their ``window_days`` / ``baseline_days`` are calendar days over event
    # tables and say nothing about how much price history the scan needs.
    assert not found & {"earnings_surprise", "filing_event", "insider_cluster", "news_burst"}
    for name in found:
        params = list(inspect.signature(getattr(detectors, name)).parameters)
        assert params[0] == "bars"


def test_scan_window_covers_the_longest_detector_lookback() -> None:
    """THE guard. If this fails, either widen the window or shorten the
    detector — never just delete the assertion."""
    required = max_detector_sessions()
    available = scan_window_sessions()
    assert available >= required, (
        f"tier2_scan reads {TIER2_SCAN_LOOKBACK_DAYS} calendar days "
        f"(~{available} sessions) but the detector catalog needs {required} "
        f"sessions: {detector_session_requirements()}. Raise "
        f"TIER2_SCAN_LOOKBACK_DAYS to at least "
        f"{calendar_days_for_sessions(required)}."
    )


def test_the_old_hundred_day_window_would_now_fail_the_guard() -> None:
    """The regression this exists to catch, stated as a test."""
    assert sessions_in_calendar_days(100) < max_detector_sessions()


@pytest.mark.parametrize(
    ("detector", "minimum_bars_gate"),
    [
        # The real gates, read off dynamo.data.detectors. Each is the number
        # of sessions below which the detector is wrong or inert.
        ("price_zscore", 60 + 1),  # returns[:-1].tail(60) needs 60 prior returns
        ("volume_spike", 60 + 1),
        ("rel_strength", 20 + 1),  # closes.iloc[-(window + 1)]
        ("range_break", 252 + 1),  # highs[:-1].tail(252) — a TRUE 52-week high
        ("range_bound", 120 + 10 + 1),  # vr_window + vr_lag + 1, its hard gate
    ],
)
def test_allowance_covers_each_detectors_real_minimum(
    detector: str, minimum_bars_gate: int
) -> None:
    """``largest declared window + allowance`` must reach the real gate.

    This is what keeps COMBINATION_ALLOWANCE_SESSIONS honest: it is not a
    vibe, it is the additive overhead each detector layers on its headline
    window, checked against the source.
    """
    requirement = detector_session_requirements()[detector]
    assert requirement >= minimum_bars_gate
    assert scan_window_sessions() >= minimum_bars_gate


def test_requirement_is_derived_from_declared_defaults() -> None:
    """Requirements track the detectors' own signatures, not a copy of them."""
    assert declared_windows(detectors.range_break) == {
        "lookback": 252,
        "volume_window": 60,
    }
    assert detector_session_requirements()["range_break"] == 252 + COMBINATION_ALLOWANCE_SESSIONS
    # Non-window numeric knobs are not mistaken for lookbacks.
    assert "min_band_touches" not in declared_windows(detectors.range_bound)
    assert "band_sigma" not in declared_windows(detectors.range_bound)


def test_calendar_conversion_is_conservative() -> None:
    """A calendar window must never be credited with more sessions than a
    real exchange calendar delivers (measured: 420d -> 288 sessions)."""
    assert sessions_in_calendar_days(420) <= 288
    assert sessions_in_calendar_days(365) <= 252
    assert calendar_days_for_sessions(max_detector_sessions()) <= TIER2_SCAN_LOOKBACK_DAYS
