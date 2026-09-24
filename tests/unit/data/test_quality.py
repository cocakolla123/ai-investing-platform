"""Data-quality check tests (doc 02 §6) — pure functions + runner."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from dynamo.data.quality import (
    QualityIssue,
    corp_action_sanity,
    dual_source_close_check,
    extreme_move_census,
    gap_check,
    nearest_split_ratio,
    run_quality_checks,
    staleness_check,
)

AS_OF = datetime(2026, 7, 31, 22, 0, tzinfo=UTC)


def closes_frame(symbol: str, values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": [pd.Timestamp(d, tz="UTC") for d in values],
            "close": list(values.values()),
        }
    )


class FakeStore:
    def __init__(self) -> None:
        self.quarantines: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []

    def add_quarantine(
        self,
        *,
        scope: str,
        key: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.quarantines.append({"scope": scope, "key": key, "reason": reason})

    def add_alert(
        self,
        *,
        severity: str,
        name: str,
        message: str,
        ackable: bool = True,
        created_at: datetime | None = None,
    ) -> None:
        self.alerts.append({"severity": severity, "name": name, "message": message})


def test_dual_source_close_flags_divergence_over_50bps() -> None:
    alpaca = closes_frame("ACME", {"2026-07-30": 100.0, "2026-07-31": 101.0})
    tiingo = closes_frame("ACME", {"2026-07-30": 100.0, "2026-07-31": 100.0})  # 100 bps
    issues = dual_source_close_check(alpaca, tiingo)
    assert len(issues) == 1
    assert issues[0].check == "dual_source_close"
    assert issues[0].symbol == "ACME"
    assert issues[0].details["max_diff_bps"] == 100.0


def test_dual_source_close_passes_within_threshold() -> None:
    alpaca = closes_frame("ACME", {"2026-07-31": 100.02})  # 2 bps
    tiingo = closes_frame("ACME", {"2026-07-31": 100.0})
    assert dual_source_close_check(alpaca, tiingo) == []


def test_gap_check_vs_xnys_calendar() -> None:
    # Week of 2026-07-27: Mon-Fri are all XNYS sessions; Wednesday missing.
    bars = closes_frame(
        "ACME",
        {"2026-07-27": 1.0, "2026-07-28": 1.0, "2026-07-30": 1.0, "2026-07-31": 1.0},
    )
    issues = gap_check(bars, date(2026, 7, 27), date(2026, 7, 31))
    assert len(issues) == 1
    assert issues[0].check == "gap"
    assert issues[0].details["missing_days"] == ["2026-07-29"]

    complete = closes_frame(
        "ACME",
        {
            "2026-07-27": 1.0,
            "2026-07-28": 1.0,
            "2026-07-29": 1.0,
            "2026-07-30": 1.0,
            "2026-07-31": 1.0,
        },
    )
    assert gap_check(complete, date(2026, 7, 27), date(2026, 7, 31)) == []


def test_gap_check_ignores_a_session_that_has_not_closed_yet() -> None:
    """The nightly check runs at 22:30 ET, when the UTC date is already
    tomorrow. Asking for tomorrow's bar quarantined every symbol every night."""
    bars = closes_frame("ACME", {"2026-09-21": 1.0, "2026-09-22": 1.0, "2026-09-23": 1.0})
    # 22:30 ET on Wed 09-23 == 02:30 UTC on Thu 09-24.
    tonight = datetime(2026, 9, 24, 2, 30, tzinfo=UTC)
    assert gap_check(bars, date(2026, 9, 21), date(2026, 9, 24), now=tonight) == []
    # Without the cutoff the unfinished session is (wrongly) reported.
    legacy = gap_check(bars, date(2026, 9, 21), date(2026, 9, 24))
    assert legacy[0].details["missing_days"] == ["2026-09-24"]


def test_gap_check_still_flags_a_closed_session_that_is_missing() -> None:
    """Tonight's own session HAS closed (16:00 ET) — if eod ingest failed, the
    cutoff must not hide that."""
    bars = closes_frame("ACME", {"2026-09-21": 1.0, "2026-09-22": 1.0})
    tonight = datetime(2026, 9, 24, 2, 30, tzinfo=UTC)
    issues = gap_check(bars, date(2026, 9, 21), date(2026, 9, 24), now=tonight)
    assert issues[0].details["missing_days"] == ["2026-09-23"]


def test_gap_check_respects_an_early_close() -> None:
    """2026-11-27 (day after Thanksgiving) closes at 13:00 ET. At 14:00 ET it
    has closed and must be required; at 12:00 ET it has not."""
    bars = closes_frame("ACME", {"2026-11-25": 1.0})
    after = datetime(2026, 11, 27, 19, 0, tzinfo=UTC)  # 14:00 ET
    before = datetime(2026, 11, 27, 17, 0, tzinfo=UTC)  # 12:00 ET
    assert gap_check(bars, date(2026, 11, 25), date(2026, 11, 27), now=after)[0].details[
        "missing_days"
    ] == ["2026-11-27"]
    assert gap_check(bars, date(2026, 11, 25), date(2026, 11, 27), now=before) == []


def test_corp_action_sanity_flags_unexplained_jump() -> None:
    bars = closes_frame("ACME", {"2026-07-30": 100.0, "2026-07-31": 50.0})  # -50%
    no_actions = pd.DataFrame({"symbol": [], "ex_date": []})
    issues = corp_action_sanity(bars, no_actions)
    assert len(issues) == 1
    assert issues[0].check == "corp_action_sanity"

    explained = pd.DataFrame([{"symbol": "ACME", "ex_date": "2026-07-31"}])  # 2:1 split
    assert corp_action_sanity(bars, explained) == []


def test_staleness_check() -> None:
    latest = {"FRESH": AS_OF, "STALE": AS_OF - timedelta(days=3)}
    issues = staleness_check(latest, ["FRESH", "STALE", "MISSING"], as_of=AS_OF)
    assert {i.symbol for i in issues} == {"STALE", "MISSING"}
    assert all(i.check == "staleness" for i in issues)


def test_runner_writes_quarantine_rows() -> None:
    store = FakeStore()
    issues = [
        QualityIssue(check="gap", symbol="ACME", reason="1 missing day", details={}),
        QualityIssue(check="staleness", symbol="XYZ", reason="no bar", details={}),
    ]
    flagged = run_quality_checks(store, issues, now=AS_OF)
    assert flagged == 2
    assert len(store.quarantines) == 2
    assert store.quarantines[0]["scope"] == "symbol"
    assert store.alerts == []  # below systemic threshold


def test_runner_raises_systemic_alert_over_threshold() -> None:
    store = FakeStore()
    issues = [
        QualityIssue(check="gap", symbol=f"SYM{i}", reason="missing", details={}) for i in range(21)
    ]
    flagged = run_quality_checks(store, issues, now=AS_OF, systemic_threshold=20)
    assert flagged == 21
    assert len(store.alerts) == 1
    assert store.alerts[0]["name"] == "data_quality_systemic"
    assert store.alerts[0]["severity"] == "warn"


# --- Split-artifact census (D1) -------------------------------------------


def series_frame(symbol: str, closes: list[float], *, start: str = "2026-01-05") -> pd.DataFrame:
    """One symbol, consecutive daily closes."""
    stamps = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"symbol": symbol, "ts": stamps, "close": closes})


def test_nearest_split_ratio_matches_within_relative_tolerance() -> None:
    assert nearest_split_ratio(2.0) == 2.0
    assert nearest_split_ratio(1.995) == 2.0  # 0.25% off a 2-for-1
    assert nearest_split_ratio(0.1004) == 0.1  # a 1-for-10, price 10x'd
    # An ordinary -47% crash is NOT a 2-for-1: ratio 1.887 is 5.7% away.
    assert nearest_split_ratio(1.887) is None
    assert nearest_split_ratio(0.0) is None
    assert nearest_split_ratio(float("nan")) is None


def test_census_finds_an_unadjusted_two_for_one_split() -> None:
    """The synthetic defect: a flat $100 stock that halves overnight and then
    carries on flat. Unadjusted, that is a -50% "crash" on a 2.000x ratio."""
    raw = series_frame("ACME", [100.0, 100.0, 50.0, 50.0, 50.0])
    census = extreme_move_census(raw)
    assert census.extreme_events == 1
    assert census.split_artifact_events == 1
    assert census.split_artifact_symbols == 1
    event = census.events.iloc[0]
    assert event["symbol"] == "ACME"
    assert event["split_ratio"] == 2.0
    assert abs(float(event["ret"]) + 0.5) < 1e-9


def test_census_is_clean_once_the_same_series_is_adjusted() -> None:
    """Same corporate event, back-adjusted the way Alpaca's split-adjusted
    feed returns it: the pre-split closes are halved and the jump is gone."""
    adjusted = series_frame("ACME", [50.0, 50.0, 50.0, 50.0, 50.0])
    census = extreme_move_census(adjusted)
    assert census.extreme_events == 0
    assert census.split_artifact_events == 0
    assert census.events.empty


def test_census_finds_an_unadjusted_reverse_split() -> None:
    """1-for-10: the price is multiplied tenfold, ratio 0.1, a +900% "squeeze"."""
    raw = series_frame("PENNY", [0.40, 0.40, 4.00, 4.00])
    census = extreme_move_census(raw)
    assert census.extreme_events == 1
    assert census.split_artifact_events == 1
    assert float(census.events.iloc[0]["split_ratio"]) == 0.1


def test_census_counts_a_real_crash_as_extreme_but_not_a_split() -> None:
    """The distinction the whole census rests on: a -40% crash is extreme and
    must stay counted, but it is not a split ratio and must not be blamed on
    adjustment."""
    raw = series_frame("BUST", [100.0, 100.0, 60.0, 60.0])
    census = extreme_move_census(raw)
    assert census.extreme_events == 1
    assert census.extreme_symbols == 1
    assert census.split_artifact_events == 0


def test_census_min_median_close_drops_penny_names() -> None:
    frame = pd.concat(
        [
            series_frame("PENNY", [0.40, 0.40, 4.00, 4.00]),
            series_frame("BIG", [100.0, 100.0, 50.0, 50.0]),
        ]
    )
    unfiltered = extreme_move_census(frame)
    assert unfiltered.symbols == 2
    assert unfiltered.split_artifact_events == 2
    floored = extreme_move_census(frame, min_median_close=5.0)
    assert floored.symbols == 1
    assert floored.split_artifact_events == 1
    assert set(floored.events["symbol"]) == {"BIG"}


def test_census_does_not_cross_symbol_boundaries() -> None:
    """Two flat series at very different price levels: the census must never
    difference the last bar of one symbol against the first of the next."""
    frame = pd.concat(
        [
            series_frame("AAA", [100.0, 100.0, 100.0]),
            series_frame("BBB", [1.0, 1.0, 1.0]),
        ]
    )
    census = extreme_move_census(frame)
    assert census.extreme_events == 0
    assert census.symbols == 2
    assert census.sessions == 4  # 2 usable per symbol


def test_census_sorts_an_unordered_frame() -> None:
    ordered = series_frame("ACME", [100.0, 100.0, 50.0, 50.0])
    shuffled = ordered.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    assert extreme_move_census(shuffled).as_dict() == extreme_move_census(ordered).as_dict()


def test_census_on_an_empty_frame() -> None:
    census = extreme_move_census(pd.DataFrame({"symbol": [], "ts": [], "close": []}))
    assert census.as_dict()["extreme_events"] == 0
    assert census.events.empty
    assert "split_ratio" in census.events.columns
