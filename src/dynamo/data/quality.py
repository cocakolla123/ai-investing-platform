"""Data-quality checks — pure functions + a quarantine-writing runner.

Design: doc 02 §6. Checks run between ingestion and features; failures
quarantine the *symbol*, not the pipeline. Quarantined symbols are excluded
from signal generation and the suppression is logged, never silent.

Pure checks (unit-testable on small frames):

1. :func:`dual_source_close_check` — |Alpaca close - Tiingo close| > 50 bps.
2. :func:`gap_check`                — missing trading days vs XNYS calendar.
3. :func:`corp_action_sanity`      — overnight return > 40% without an action.
4. :func:`staleness_check`         — Tier-1 symbol without a fresh EOD bar.

Runner: :func:`run_quality_checks` writes quarantine rows and raises a
systemic alert when more than ``systemic_threshold`` symbols are flagged
(default 20, doc 02 §6.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, cast

import exchange_calendars as xcals
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def _to_date(value: Any) -> date:
    """Coerce a timestamp-like value to a plain date (typing-safe)."""
    return cast(pd.Timestamp, pd.Timestamp(value)).date()


@dataclass(frozen=True)
class QualityIssue:
    """One failed check for one symbol; becomes a quarantine row."""

    check: str
    symbol: str
    reason: str
    details: dict[str, Any]


def dual_source_close_check(
    alpaca: pd.DataFrame,
    tiingo: pd.DataFrame,
    *,
    threshold_bps: float = 50.0,
) -> list[QualityIssue]:
    """Doc 02 §6.1 — dual-source close divergence > ``threshold_bps``.

    Inputs are per-source frames with columns ``symbol, ts, close`` holding
    *raw* (unadjusted) closes; Tiingo's adjusted series is unwound upstream by
    passing its raw ``close`` here.
    """
    if alpaca.empty or tiingo.empty:
        return []
    merged = alpaca.merge(tiingo, on=["symbol", "ts"], suffixes=("_alpaca", "_tiingo"))
    if merged.empty:
        return []
    merged = merged[merged["close_tiingo"] > 0]
    close_alpaca = cast(pd.Series, merged["close_alpaca"])
    close_tiingo = cast(pd.Series, merged["close_tiingo"])
    merged["diff_bps"] = close_alpaca.sub(close_tiingo).abs() / close_tiingo * 10_000.0
    issues: list[QualityIssue] = []
    for symbol, grp in merged.groupby("symbol"):
        worst = float(grp["diff_bps"].max())
        if worst > threshold_bps:
            issues.append(
                QualityIssue(
                    check="dual_source_close",
                    symbol=str(symbol),
                    reason=f"alpaca/tiingo close diverge {worst:.1f} bps (> {threshold_bps})",
                    details={"max_diff_bps": round(worst, 1), "days": len(grp)},
                )
            )
    return issues


def gap_check(
    bars: pd.DataFrame,
    start: date,
    end: date,
    *,
    calendar_name: str = "XNYS",
) -> list[QualityIssue]:
    """Doc 02 §6.2 — missing trading days vs the exchange calendar."""
    if bars.empty:
        return []
    calendar = xcals.get_calendar(calendar_name)
    sessions = {
        s.date()
        for s in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    }
    issues: list[QualityIssue] = []
    for symbol, grp in bars.groupby("symbol"):
        have = {_to_date(t) for t in grp["ts"]}
        missing = sorted(sessions - have)
        if missing:
            issues.append(
                QualityIssue(
                    check="gap",
                    symbol=str(symbol),
                    reason=f"{len(missing)} missing trading day(s) vs {calendar_name}",
                    details={"missing_days": [d.isoformat() for d in missing[:20]]},
                )
            )
    return issues


def corp_action_sanity(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    threshold: float = 0.40,
) -> list[QualityIssue]:
    """Doc 02 §6.3 — overnight return > 40% without a corporate action → hold.

    ``bars``: columns ``symbol, ts, close``; ``actions``: columns
    ``symbol, ex_date`` (empty frame allowed).
    """
    if bars.empty:
        return []
    action_days: set[tuple[str, date]] = set()
    if not actions.empty:
        action_days = {
            (str(s), _to_date(d))
            for s, d in zip(actions["symbol"], actions["ex_date"], strict=True)
        }
    issues: list[QualityIssue] = []
    for symbol, grp in bars.groupby("symbol"):
        grp = grp.sort_values("ts")
        returns = grp["close"].pct_change()
        for ts, ret in zip(grp["ts"], returns, strict=True):
            if pd.isna(ret) or abs(float(ret)) <= threshold:
                continue
            day = _to_date(ts)
            if (str(symbol), day) in action_days:
                continue
            issues.append(
                QualityIssue(
                    check="corp_action_sanity",
                    symbol=str(symbol),
                    reason=f"overnight return {float(ret):+.1%} on {day} without corp action",
                    details={"date": day.isoformat(), "return": round(float(ret), 4)},
                )
            )
            break  # one issue per symbol is enough to quarantine
    return issues


def staleness_check(
    latest_bar_ts: Mapping[str, datetime],
    expected_symbols: Sequence[str],
    *,
    as_of: datetime,
    max_age_days: int = 1,
) -> list[QualityIssue]:
    """Doc 02 §6.4 — Tier-1 symbol without a fresh EOD bar by the deadline."""
    cutoff = as_of - timedelta(days=max_age_days)
    issues: list[QualityIssue] = []
    for symbol in expected_symbols:
        last = latest_bar_ts.get(symbol)
        if last is None or last < cutoff:
            issues.append(
                QualityIssue(
                    check="staleness",
                    symbol=symbol,
                    reason="no fresh EOD bar" if last is None else f"last bar {last.isoformat()}",
                    details={"last_bar": last.isoformat() if last else None},
                )
            )
    return issues


class QualityStore(Protocol):
    """The slice of ``MarketDataRepository`` the runner writes through."""

    def add_quarantine(
        self,
        *,
        scope: str,
        key: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None: ...

    def add_alert(
        self,
        *,
        severity: str,
        name: str,
        message: str,
        ackable: bool = True,
        created_at: datetime | None = None,
    ) -> None: ...


def run_quality_checks(
    store: QualityStore,
    issues: Sequence[QualityIssue],
    *,
    now: datetime | None = None,
    systemic_threshold: int = 20,
) -> int:
    """Write quarantine rows; raise a systemic alert if > threshold symbols.

    Returns the number of distinct symbols quarantined. Suppression is logged,
    never silent (doc 02 §6).
    """
    ts = now or datetime.now(UTC)
    symbols: set[str] = set()
    for issue in issues:
        symbols.add(issue.symbol)
        store.add_quarantine(
            scope="symbol",
            key=issue.symbol,
            reason=f"{issue.check}: {issue.reason}",
            payload=issue.details,
            created_at=ts,
        )
        log.warning(
            "symbol_quarantined", symbol=issue.symbol, check=issue.check, reason=issue.reason
        )
    if len(symbols) > systemic_threshold:
        store.add_alert(
            severity="warn",
            name="data_quality_systemic",
            message=(
                f"{len(symbols)} symbols failed quality checks "
                f"(> {systemic_threshold}) — possible systemic data issue"
            ),
            created_at=ts,
        )
        log.error("data_quality_systemic", flagged=len(symbols))
    return len(symbols)
