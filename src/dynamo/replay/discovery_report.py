"""Did the discovery layer actually find anything? (campaign GO/NO-GO #3)

``docs/runbook_wide_campaign.md`` gates the campaign on five numbers —
flags/day, distinct promoted names, promoted share of the deep-dive slots,
per-detector forward return, and whether any single detector is significantly
negative. This module computes them from a finished run's ``replay_flags``,
so the gate is a command rather than an invitation to eyeball a table.

The measurement is *forward N-session excess return vs the benchmark*, and
the comparison that matters is flagged vs unflagged ON THE SAME DAY. A raw
flagged average measures the market: if every flag fires in a month SPY rose
9%, the flags look brilliant and have told you nothing. Differencing against
the same session's unflagged names removes that, which is the same reason the
campaign runs a discovery-off control arm.

Three ways this number can lie, all reported rather than hidden:

1. **Overlapping horizons.** Consecutive sessions share H-1 of their H
   forward days, so 115 daily spreads are nowhere near 115 independent
   observations. :attr:`DetectorStat.effective_n` is ``n_days / H`` and the
   t-statistic uses it. The naive t would be ~5x too confident.
2. **Truncation.** Flags in the last H sessions have no full forward window.
   They are dropped, and ``n_flags - n_scored`` says how many.
3. **Survivorship.** The panel holds ~99.87% survivors, and delisted names
   are disproportionately what detectors flag, so the bias is optimistic and
   lands hardest on exactly the flagged set. :data:`SURVIVORSHIP_BAND` is the
   measured band; a spread inside it is reported as inconclusive no matter
   what its t-statistic says.

Post-hoc evaluation, NOT a decision path: it reads outcomes from after each
flag's ``as_of`` by construction. That is the whole point, and it is why this
module is never imported by the runner. Nothing here may feed a replay
decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import pandas as pd

#: Forward window, in sessions. Matches the quant label horizon so a
#: detector's measured edge is on the same clock as the model's.
DEFAULT_HORIZON_SESSIONS = 20

#: Measured survivorship distortion per 20-session window (see
#: ``docs/wide_campaign_design.md``): 99.87% of symbols in the panel survive
#: to the last bar, against ~1% of expected 2024-2026 delistings present. A
#: flagged-vs-unflagged spread smaller than this is not evidence.
SURVIVORSHIP_BAND = 0.01


class DiscoveryReportError(RuntimeError):
    """The run cannot be analysed (unknown, or it recorded no flags)."""


@dataclass(frozen=True)
class DetectorStat:
    """One detector's measured forward edge, with its honesty attached."""

    detector: str
    n_flags: int
    n_scored: int
    n_promoted: int
    mean_excess: float
    """Mean forward excess return of flagged names vs the benchmark."""
    mean_spread: float
    """Mean of the DAILY (flagged - unflagged) spread. The real number."""
    n_days: int
    effective_n: float
    t_stat: float

    @property
    def conclusive(self) -> bool:
        """Whether the spread clears both the noise and the bias floors.

        Requires |t| > 2 on the EFFECTIVE sample and a spread outside
        :data:`SURVIVORSHIP_BAND`. Both, because either alone is routinely
        cleared by an artifact.
        """
        return abs(self.t_stat) > 2.0 and abs(self.mean_spread) > SURVIVORSHIP_BAND

    @property
    def verdict(self) -> str:
        if not self.conclusive:
            return "inconclusive"
        return "positive" if self.mean_spread > 0 else "CONTRA-INDICATIVE"


@dataclass(frozen=True)
class DiscoveryReport:
    """Everything GO/NO-GO #3 asks for, in one object."""

    run_id: str
    window_start: date
    window_end: date
    horizon_sessions: int
    n_sessions_with_flags: int
    n_flags: int
    flags_per_day_mean: float
    flags_per_day_median: float
    flags_per_day_max: int
    distinct_flagged: int
    distinct_promoted: int
    promoted_share_of_decisions: float | None
    detectors: tuple[DetectorStat, ...] = ()
    warnings: tuple[str, ...] = field(default=())


def _forward_excess(
    closes: pd.DataFrame,
    benchmark: Mapping[date, float],
    horizon: int,
) -> pd.DataFrame:
    """Per (symbol, session) forward H-session return minus the benchmark's.

    Indexed positionally within each symbol's own session list, so a symbol
    that does not trade every session is measured over ITS next H sessions
    rather than a calendar window — the same per-symbol rule the feature
    pipeline learned the hard way (see ``catalog.py::_apply_per_symbol``).
    """
    bench = pd.Series(dict(benchmark)).sort_index()
    bench_fwd = bench.shift(-horizon) / bench - 1.0

    frames: list[pd.DataFrame] = []
    for symbol, group in closes.groupby("symbol", sort=True):
        grp = group.sort_values("ts")
        fwd = grp["close"].shift(-horizon) / grp["close"] - 1.0
        sessions = pd.to_datetime(grp["ts"]).dt.date
        frames.append(
            pd.DataFrame(
                {
                    "symbol": str(symbol),
                    "as_of": sessions.to_numpy(),
                    "fwd": fwd.to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame({"symbol": [], "as_of": [], "excess": []})

    out = pd.concat(frames, ignore_index=True)
    out["bench"] = out["as_of"].map(bench_fwd.to_dict())
    out["excess"] = out["fwd"] - out["bench"]
    scored = cast("pd.DataFrame", out.dropna(subset=["excess"]))
    return cast("pd.DataFrame", scored[["symbol", "as_of", "excess"]])


def _t_stat(spreads: Sequence[float], effective_n: float) -> float:
    """Paired t on the EFFECTIVE sample, not the count of overlapping days.

    A degenerate (constant) spread returns 0.0, not a huge number. Taken
    literally, zero sample variance is infinite confidence; in practice it
    means the input is synthetic or the sample is one repeated observation,
    and float noise on a "constant" series lands the naive formula around
    t=1e17. Real detector spreads are never constant, so treating this as
    "no evidence" costs nothing and stops a fixture from proving a detector
    works.
    """
    n = len(spreads)
    if n < 2 or effective_n < 2.0:
        return 0.0
    mean = sum(spreads) / n
    var = sum((s - mean) ** 2 for s in spreads) / (n - 1)
    degenerate = (1e-9 * max(abs(mean), 1e-6)) ** 2
    if var <= degenerate:
        return 0.0
    return mean / math.sqrt(var / effective_n)


def _detector_stat(
    detector: str,
    rows: pd.DataFrame,
    excess: pd.DataFrame,
    horizon: int,
) -> DetectorStat:
    n_flags = len(rows)
    n_promoted = int(rows["promoted"].sum())

    flagged = rows.merge(excess, on=["symbol", "as_of"], how="inner")
    n_scored = len(flagged)

    daily_spreads: list[float] = []
    for day, day_flagged in flagged.groupby("as_of"):
        universe = cast("pd.DataFrame", excess[excess["as_of"] == day])
        names = sorted(set(day_flagged["symbol"]))
        unflagged = cast("pd.DataFrame", universe[~universe["symbol"].isin(names)])
        if unflagged.empty:
            continue
        daily_spreads.append(
            float(day_flagged["excess"].mean()) - float(unflagged["excess"].mean())
        )

    n_days = len(daily_spreads)
    # Overlapping forward windows: H consecutive days share H-1 of their
    # outcome. Independent observations ~ n_days / H, floored at 1.
    effective_n = max(1.0, n_days / float(horizon))
    return DetectorStat(
        detector=detector,
        n_flags=n_flags,
        n_scored=n_scored,
        n_promoted=n_promoted,
        mean_excess=float(flagged["excess"].mean()) if n_scored else 0.0,
        mean_spread=(sum(daily_spreads) / n_days) if n_days else 0.0,
        n_days=n_days,
        effective_n=effective_n,
        t_stat=_t_stat(daily_spreads, effective_n),
    )


def build_discovery_report(
    run_id: str,
    flags: Sequence[Mapping[str, Any]],
    closes: pd.DataFrame,
    benchmark_closes: Mapping[date, float],
    *,
    decisions: Sequence[Mapping[str, Any]] = (),
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
) -> DiscoveryReport:
    """Compute GO/NO-GO #3 from a finished run's persisted flags.

    ``closes`` must cover the CANDIDATE universe, not just the flagged names:
    the unflagged half of every daily spread comes from it, and a ``closes``
    holding only flagged symbols silently reduces every spread to zero.
    """
    if not flags:
        raise DiscoveryReportError(
            f"run {run_id!r} recorded no flags — it ran with discovery off, "
            "or the gate filtered everything"
        )

    frame = pd.DataFrame(list(flags))
    frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.date
    frame["promoted"] = frame["promoted"].fillna(False).astype(bool)

    per_day = frame.groupby("as_of").size()
    excess = _forward_excess(closes, benchmark_closes, horizon_sessions)

    promoted_symbols = set(frame.loc[frame["promoted"], "symbol"])
    share: float | None = None
    if decisions:
        hits = sum(1 for d in decisions if d.get("symbol") in promoted_symbols)
        share = hits / len(decisions)

    detectors = tuple(
        sorted(
            (
                _detector_stat(str(name), rows, excess, horizon_sessions)
                for name, rows in frame.groupby("detector", sort=False)
            ),
            key=lambda s: -s.mean_spread,
        )
    )

    warnings: list[str] = []
    dropped = sum(d.n_flags - d.n_scored for d in detectors)
    if dropped:
        warnings.append(
            f"{dropped} of {len(frame)} flags have no full {horizon_sessions}-session "
            "forward window (window end, or the symbol stopped trading) and are unscored"
        )
    if excess.empty:
        warnings.append(
            "no forward returns could be computed — closes/benchmark do not "
            "overlap the flag dates, so every spread below is zero"
        )
    universe_size = int(closes["symbol"].nunique()) if not closes.empty else 0
    n_flagged_names = int(frame["symbol"].nunique())
    if universe_size and universe_size <= n_flagged_names:
        warnings.append(
            f"closes cover {universe_size} symbols but {n_flagged_names} were "
            "flagged — the unflagged baseline is empty or near-empty, so the "
            "spreads are not measuring what they claim"
        )
    warnings.append(
        f"survivorship: spreads inside +/-{SURVIVORSHIP_BAND:.1%} are inside the "
        "measured bias band and are reported inconclusive regardless of t"
    )

    return DiscoveryReport(
        run_id=run_id,
        window_start=min(frame["as_of"]),
        window_end=max(frame["as_of"]),
        horizon_sessions=horizon_sessions,
        n_sessions_with_flags=int(per_day.size),
        n_flags=len(frame),
        flags_per_day_mean=float(per_day.mean()),
        flags_per_day_median=float(per_day.median()),
        flags_per_day_max=int(per_day.max()),
        distinct_flagged=n_flagged_names,
        distinct_promoted=len(promoted_symbols),
        promoted_share_of_decisions=share,
        detectors=detectors,
        warnings=tuple(warnings),
    )


def render_discovery_report(report: DiscoveryReport) -> str:
    """Human-readable GO/NO-GO #3, with the runbook's pass bands inline."""
    lines = [
        f"discovery report for {report.run_id}",
        f"  window:            {report.window_start} -> {report.window_end} "
        f"({report.n_sessions_with_flags} sessions with flags)",
        f"  flags:             {report.n_flags} total, "
        f"{report.flags_per_day_mean:.1f}/day mean, "
        f"{report.flags_per_day_median:.0f} median, {report.flags_per_day_max} max"
        f"   [pass: 30-120/day]",
        f"  distinct flagged:  {report.distinct_flagged}",
        f"  distinct promoted: {report.distinct_promoted}   [pass: 250-800]",
    ]
    if report.promoted_share_of_decisions is None:
        lines.append("  promoted share:    n/a (no decisions passed in)")
    else:
        lines.append(
            f"  promoted share:    {report.promoted_share_of_decisions:.1%} "
            "of decisions   [pass: 30-70%]"
        )
    lines.append("")
    lines.append(
        f"  {'detector':<18}{'flags':>7}{'scored':>8}{'spread':>10}"
        f"{'eff n':>8}{'t':>7}  verdict"
    )
    for d in report.detectors:
        lines.append(
            f"  {d.detector:<18}{d.n_flags:>7}{d.n_scored:>8}"
            f"{d.mean_spread * 100.0:>9.2f}%{d.effective_n:>8.1f}"
            f"{d.t_stat:>7.2f}  {d.verdict}"
        )
    if report.warnings:
        lines.append("")
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
    return "\n".join(lines)
