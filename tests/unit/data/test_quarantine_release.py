"""A quarantine must lift once the data-quality check passes again.

The nightly checks re-examine every tier-1 symbol from scratch and quarantine
what fails, but nothing ever closed a quarantine afterwards — while the risk
layer blocks on ANY unresolved one, regardless of age. The 2026-09-14/15 host
outage left every core name two sessions stale; the check correctly
quarantined all 55, and they stayed frozen after the data was repaired.
``new_risk_allowed`` was true and the target portfolio was still 100% cash.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from dynamo.data.quality import QUALITY_CHECKS, QualityIssue, release_cleared_quarantines

NOW = datetime(2026, 9, 23, 22, 0, tzinfo=UTC)


class FakeReleaser:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def release_quarantines(
        self,
        *,
        scope: str,
        keys: Sequence[str],
        reason_prefixes: Sequence[str],
        resolved_at: datetime,
    ) -> int:
        self.calls.append(
            {
                "scope": scope,
                "keys": list(keys),
                "prefixes": tuple(reason_prefixes),
                "at": resolved_at,
            }
        )
        return len(keys)


def _issue(symbol: str, check: str = "staleness") -> QualityIssue:
    return QualityIssue(symbol=symbol, check=check, reason="x", details={})


def test_symbols_that_passed_tonight_are_released() -> None:
    store = FakeReleaser()
    released = release_cleared_quarantines(store, ["AAPL", "MSFT", "NVDA"], [], now=NOW)
    assert released == 3
    assert store.calls[0]["keys"] == ["AAPL", "MSFT", "NVDA"]
    assert store.calls[0]["scope"] == "symbol"
    assert store.calls[0]["at"] == NOW


def test_a_symbol_that_still_fails_stays_quarantined() -> None:
    store = FakeReleaser()
    release_cleared_quarantines(store, ["AAPL", "MSFT"], [_issue("MSFT", "gap")], now=NOW)
    assert store.calls[0]["keys"] == ["AAPL"]


def test_only_symbols_the_check_covered_are_released() -> None:
    """A symbol outside tonight's checked set was not examined, so passing
    silence says nothing about it."""
    store = FakeReleaser()
    release_cleared_quarantines(store, ["AAPL"], [], now=NOW)
    assert "TSLA" not in store.calls[0]["keys"]


def test_release_is_scoped_to_this_modules_own_quarantines() -> None:
    """A quarantine placed for some other reason must never be undone by a
    passing data-quality night."""
    store = FakeReleaser()
    release_cleared_quarantines(store, ["AAPL"], [], now=NOW)
    assert set(store.calls[0]["prefixes"]) == {f"{c}:" for c in QUALITY_CHECKS}


def test_every_check_this_module_runs_is_releasable() -> None:
    """If a new check is added without registering its name, its quarantines
    would silently become permanent again. Keep the registry complete."""
    import inspect

    import dynamo.data.quality as quality

    source = inspect.getsource(quality)
    emitted = {
        line.split('check="')[1].split('"')[0] for line in source.splitlines() if 'check="' in line
    }
    assert emitted <= set(QUALITY_CHECKS)


def test_nothing_to_release_makes_no_call() -> None:
    store = FakeReleaser()
    assert release_cleared_quarantines(store, ["AAPL"], [_issue("AAPL")], now=NOW) == 0
    assert store.calls == []
