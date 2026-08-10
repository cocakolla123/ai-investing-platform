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
    gap_check,
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
            "2026-07-27": 1.0, "2026-07-28": 1.0, "2026-07-29": 1.0,
            "2026-07-30": 1.0, "2026-07-31": 1.0,
        },
    )
    assert gap_check(complete, date(2026, 7, 27), date(2026, 7, 31)) == []


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
        QualityIssue(check="gap", symbol=f"SYM{i}", reason="missing", details={})
        for i in range(21)
    ]
    flagged = run_quality_checks(store, issues, now=AS_OF, systemic_threshold=20)
    assert flagged == 21
    assert len(store.alerts) == 1
    assert store.alerts[0]["name"] == "data_quality_systemic"
    assert store.alerts[0]["severity"] == "warn"
