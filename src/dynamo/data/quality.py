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

Census (a *measurement*, not a per-symbol check): :func:`extreme_move_census`
counts single-session moves large enough to be corporate actions rather than
prices, and how many of those land on a near-exact split ratio. It exists
because "is this panel adjusted?" was, before it, a question nobody could
answer with a number — see its docstring for the measurement that motivated it.
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
    sessions = {s.date() for s in calendar.sessions_in_range(start.isoformat(), end.isoformat())}
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


# --- Split-artifact census (D1) -------------------------------------------
#
# A price series that has not been adjusted for corporate actions still looks
# like a price series. Nothing raises; the split simply appears as a return.
# The only way to know is to count the moves that are too large to be prices
# and ask how many of them land on a split ratio.

#: Single-session simple return below this is "extreme" (a crash, or a split).
EXTREME_DOWN_RETURN = -0.30
#: Single-session simple return above this is "extreme" (a squeeze, or a
#: reverse split). Asymmetric because an N-for-1 reverse split multiplies the
#: price, so its upside tail is unbounded while a split's downside is not.
EXTREME_UP_RETURN = 0.45

#: ``prev_close / close`` values that a split produces. A 2-for-1 forward
#: split halves the price (ratio 2.0); a 1-for-10 reverse split multiplies it
#: tenfold (ratio 0.1). These five are the ones the wide panel was measured
#: against; the parameter is open so a wider audit can pass more.
SPLIT_RATIOS: tuple[float, ...] = (2.0, 3.0, 4.0, 0.5, 0.1)
#: Relative tolerance for calling a ratio "near-exact". 1% is tight enough
#: that an ordinary -50% crash does not masquerade as a 2-for-1 split, and
#: loose enough to absorb the day's genuine price move on top of the split.
SPLIT_RATIO_TOLERANCE = 0.01


def nearest_split_ratio(
    ratio: float,
    *,
    ratios: Sequence[float] = SPLIT_RATIOS,
    tolerance: float = SPLIT_RATIO_TOLERANCE,
) -> float | None:
    """The split ratio ``ratio`` matches within ``tolerance``, else ``None``.

    ``tolerance`` is *relative* to the candidate, so 1% means 1% of 0.1 for a
    1-for-10 and 1% of 4.0 for a 4-for-1 — a split leaves a much tighter
    fingerprint on a cheap stock than an absolute tolerance would allow.
    """
    if not (ratio > 0.0) or ratio != ratio:  # non-positive or NaN
        return None
    best: float | None = None
    best_error = float("inf")
    for candidate in ratios:
        error = abs(ratio - candidate)
        if error <= tolerance * abs(candidate) and error < best_error:
            best, best_error = float(candidate), error
    return best


@dataclass(frozen=True)
class ExtremeMoveCensus:
    """How much of a bar panel is corporate action wearing a price's clothes.

    ``split_artifact_events`` is the headline: sessions whose move is extreme
    AND whose ``prev_close / close`` is a near-exact split ratio. On an
    adjusted panel it should be ~0 — real single-day moves do not land on
    2.000x, 3.000x or 0.100x to within 1%.
    """

    symbols: int
    """Distinct symbols in the panel."""
    sessions: int
    """Symbol-days with a computable previous close."""
    extreme_events: int
    extreme_symbols: int
    split_artifact_events: int
    split_artifact_symbols: int
    events: pd.DataFrame
    """One row per extreme event: ``symbol, ts, prev_close, close, ret,
    ratio, split_ratio`` (``split_ratio`` is NaN when it is not a match)."""

    def summary(self) -> str:
        pct = 100.0 * self.extreme_symbols / self.symbols if self.symbols else 0.0
        return (
            f"{self.symbols} symbols / {self.sessions} sessions: "
            f"{self.extreme_events} extreme moves across {self.extreme_symbols} "
            f"symbols ({pct:.1f}%); "
            f"{self.split_artifact_events} of them are near-exact split ratios "
            f"across {self.split_artifact_symbols} symbols"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "sessions": self.sessions,
            "extreme_events": self.extreme_events,
            "extreme_symbols": self.extreme_symbols,
            "split_artifact_events": self.split_artifact_events,
            "split_artifact_symbols": self.split_artifact_symbols,
        }


def extreme_move_census(
    bars: pd.DataFrame,
    *,
    down: float = EXTREME_DOWN_RETURN,
    up: float = EXTREME_UP_RETURN,
    ratios: Sequence[float] = SPLIT_RATIOS,
    tolerance: float = SPLIT_RATIO_TOLERANCE,
    min_median_close: float = 0.0,
) -> ExtremeMoveCensus:
    """Count extreme single-session moves and the split artifacts among them.

    ``bars``: columns ``symbol, ts, close`` (extra columns ignored). Rows are
    sorted per symbol here, so an unsorted frame is fine.

    ``min_median_close`` drops symbols whose median close is below the floor
    before counting. Sub-$5 names dominate percentage-move detectors and are
    where unadjusted reverse splits live, so the census is usually worth
    reading twice: once unfiltered, once at the floor the live screen uses.

    WHY THIS EXISTS. Measured on this repo's ``bars_1d`` (source=alpaca,
    2024-01-02..2026-08-18, 6,039,736 rows) while every row was still
    ``Adjustment.RAW``: 9,153 extreme moves across 2,459 of 11,264 symbols
    (21.8%), of which **217 events across 211 symbols** landed on a near-exact
    2:1/3:1/4:1/1:2/1:10 ratio. Those are splits, indistinguishable from
    crashes, and they land disproportionately in the surveillance detectors'
    flagged set because a crash is exactly what those detectors hunt.
    """
    columns = ["symbol", "ts", "prev_close", "close", "ret", "ratio", "split_ratio"]
    empty = ExtremeMoveCensus(0, 0, 0, 0, 0, 0, pd.DataFrame({c: [] for c in columns}))
    if bars.empty:
        return empty

    frame = cast(pd.DataFrame, bars[["symbol", "ts", "close"]]).copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if min_median_close > 0.0:
        medians = cast(pd.Series, frame.groupby("symbol")["close"].median())
        liquid = cast(pd.Series, medians[medians >= min_median_close])
        keep = [str(s) for s in liquid.index]
        frame = cast(pd.DataFrame, frame[cast(pd.Series, frame["symbol"]).isin(keep)])
        if frame.empty:
            return empty

    frame = frame.sort_values(by=["symbol", "ts"], kind="mergesort")
    close = cast(pd.Series, frame["close"])
    frame["prev_close"] = frame.groupby("symbol")["close"].shift(1)
    prev = cast(pd.Series, frame["prev_close"])
    # Only a positive previous close yields a meaningful return or ratio.
    valid = prev > 0.0
    frame["ret"] = close / prev - 1.0
    frame["ratio"] = prev / close.where(close > 0.0)

    ret = cast(pd.Series, frame["ret"])
    extreme = valid & ((ret < down) | (ret > up))
    events = cast(pd.DataFrame, frame[extreme]).copy()
    events["split_ratio"] = [
        nearest_split_ratio(float(r), ratios=ratios, tolerance=tolerance) if pd.notna(r) else None
        for r in cast(pd.Series, events["ratio"])
    ]
    artifacts = cast(pd.DataFrame, events[cast(pd.Series, events["split_ratio"]).notna()])
    return ExtremeMoveCensus(
        symbols=int(cast(pd.Series, frame["symbol"]).nunique()),
        sessions=int(valid.sum()),
        extreme_events=len(events),
        extreme_symbols=int(cast(pd.Series, events["symbol"]).nunique()),
        split_artifact_events=len(artifacts),
        split_artifact_symbols=int(cast(pd.Series, artifacts["symbol"]).nunique()),
        events=cast(pd.DataFrame, events[columns]).reset_index(drop=True),
    )


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
