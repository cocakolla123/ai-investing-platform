"""Replay preflight — LOUD data-readiness checks (backtest-campaign Phase 1).

An empty universe or missing history must never again "succeed" with a flat
equity curve: :func:`run_preflight` runs before every replay and raises
:class:`PreflightError` (listing every failed check) when the run would be
dishonest. The full :class:`PreflightReport` — including warnings — is
stored into ``replay_runs.config_json`` under the ``"preflight"`` key by the
CLI (NOT inside ``ReplayConfig``, so run ids stay stable).

Checks:

- universe non-empty at ``cfg.start`` (for the configured tiers);
- daily-bar coverage >= 95% of the universe both at ``cfg.start`` and at
  ``cfg.start - 750d`` (the runner's returns lookback) — presence measured
  in a +/-5-session window around each date;
- regime-critical macro series (HY OAS, curve slope) reach back to at least
  ``cfg.start - 900d`` (the runner's regime lookback) with vintages knowable
  by ``cfg.start``;
- quant-model resolvability (Phase-2 hook): ``quant_model_version`` set but
  not loadable -> error; unset -> warning recorded (baseline-v0 abstention);
- portfolio-mode source run (Phase-3 hook): ``source_run_id`` set -> the run
  must exist, be committee-mode, and cover the window (checked against
  ``replay_runs`` now, since the table already exists).

Zero LLM calls, zero writes: preflight only reads.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Protocol, cast

from sqlalchemy import Engine, func, select

from dynamo.persistence import schema
from dynamo.replay.runner import (
    REGIME_MACRO_LOOKBACK_DAYS,
    RETURNS_LOOKBACK_DAYS,
    ReplayConfig,
)

BAR_COVERAGE_MIN = 0.95
BAR_WINDOW_SESSIONS = 5
# The two macro series the runner's regime history actually consumes
# (``_regime_history``: hy_oas + curve_slope).
REQUIRED_MACRO_SERIES: tuple[str, ...] = ("BAMLH0A0HYM2", "T10Y3M")

OK = "ok"
WARNING = "warning"
ERROR = "error"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str  # ok | warning | error
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    """Every check that ran, in order; serializable for config_json."""

    checks: tuple[PreflightCheck, ...]

    @property
    def errors(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if c.status == ERROR)

    @property
    def warnings(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if c.status == WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


class PreflightError(RuntimeError):
    """At least one preflight check failed; ``report`` holds all of them."""

    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        lines = [f"{c.name}: {c.detail}" for c in report.errors]
        super().__init__(
            "replay preflight failed "
            f"({len(report.errors)} error(s)):\n  " + "\n  ".join(lines)
        )


class PreflightReads(Protocol):
    """The read slice preflight needs (satisfied by :class:`EngineReads`)."""

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...]) -> list[str]: ...

    def symbols_with_bars(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[str]: ...

    def macro_first_observation(
        self, series_ids: Sequence[str], vintage_cutoff: date
    ) -> dict[str, date | None]: ...

    def get_run(self, run_id: str) -> Mapping[str, Any] | None: ...


class EngineReads:
    """Concrete :class:`PreflightReads` over the platform database."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...]) -> list[str]:
        from dynamo.persistence.repositories.market import MarketDataRepository

        members = MarketDataRepository(self._engine).universe_at(as_of)
        return sorted({m.symbol for m in members if int(m.tier) in tiers})

    def symbols_with_bars(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[str]:
        t = schema.bars_1d
        stmt = (
            select(t.c.symbol)
            .where(
                t.c.symbol.in_(list(symbols)),
                func.date(t.c.ts) >= start,
                func.date(t.c.ts) <= end,
            )
            .distinct()
        )
        with self._engine.connect() as cx:
            return {str(r.symbol) for r in cx.execute(stmt)}

    def macro_first_observation(
        self, series_ids: Sequence[str], vintage_cutoff: date
    ) -> dict[str, date | None]:
        m = schema.macro_series
        stmt = (
            select(m.c.series_id, func.min(m.c.ts).label("first_ts"))
            .where(
                m.c.series_id.in_(list(series_ids)),
                func.date(m.c.vintage_ts) <= vintage_cutoff,
            )
            .group_by(m.c.series_id)
        )
        with self._engine.connect() as cx:
            found = {str(r.series_id): cast(date, r.first_ts) for r in cx.execute(stmt)}
        return {sid: found.get(sid) for sid in series_ids}

    def get_run(self, run_id: str) -> Mapping[str, Any] | None:
        from dynamo.persistence.repositories.replay import ReplayRepository

        return ReplayRepository(self._engine).get_run(run_id)


def session_window(center: date, n_sessions: int = BAR_WINDOW_SESSIONS) -> tuple[date, date]:
    """[first, last] dates of the +/-``n_sessions`` NYSE-session window
    around ``center`` (calendar fallback of +/-``2 * n_sessions`` days when
    the calendar has no sessions nearby)."""
    import exchange_calendars as xcals
    import pandas as pd

    fallback = (
        center - timedelta(days=2 * n_sessions),
        center + timedelta(days=2 * n_sessions),
    )
    calendar = xcals.get_calendar("XNYS")
    lo = center - timedelta(days=3 * n_sessions)
    hi = center + timedelta(days=3 * n_sessions)
    try:
        sessions = [
            cast(date, pd.Timestamp(s).date())
            for s in calendar.sessions_in_range(lo.isoformat(), hi.isoformat())
        ]
    except ValueError:  # outside the calendar's bounds
        return fallback
    before = [d for d in sessions if d <= center][-n_sessions:]
    after = [d for d in sessions if d >= center][:n_sessions]
    return (
        before[0] if before else fallback[0],
        after[-1] if after else fallback[1],
    )


# -- individual checks ------------------------------------------------------


def _check_universe(
    reads: PreflightReads, cfg: ReplayConfig
) -> tuple[PreflightCheck, list[str]]:
    symbols = reads.universe_symbols(cfg.start, tuple(cfg.universe_tiers))
    if not symbols:
        return (
            PreflightCheck(
                "universe",
                ERROR,
                f"universe is EMPTY at {cfg.start} for tiers {cfg.universe_tiers} — "
                "run `dynamo.cli.backfill universe --as-of <date>` to backdate the seed",
            ),
            [],
        )
    return (
        PreflightCheck(
            "universe", OK, f"{len(symbols)} symbols active at {cfg.start}"
        ),
        symbols,
    )


def _check_bar_coverage(
    reads: PreflightReads, symbols: list[str], center: date, label: str
) -> PreflightCheck:
    lo, hi = session_window(center)
    present = reads.symbols_with_bars(symbols, lo, hi)
    covered = sum(1 for s in symbols if s in present)
    coverage = covered / len(symbols) if symbols else 0.0
    name = f"bar_coverage_{label}"
    detail = (
        f"{covered}/{len(symbols)} symbols ({coverage:.1%}) have bars within "
        f"[{lo}, {hi}] around {center}"
    )
    if coverage < BAR_COVERAGE_MIN:
        missing = sorted(set(symbols) - present)[:10]
        return PreflightCheck(
            name,
            ERROR,
            f"{detail} — below the {BAR_COVERAGE_MIN:.0%} floor "
            f"(missing e.g. {missing}); backfill bars first",
        )
    return PreflightCheck(name, OK, detail)


def _check_macro(reads: PreflightReads, cfg: ReplayConfig) -> PreflightCheck:
    required_reach = cfg.start - timedelta(days=REGIME_MACRO_LOOKBACK_DAYS)
    first = reads.macro_first_observation(REQUIRED_MACRO_SERIES, cfg.start)
    bad: list[str] = []
    for sid in REQUIRED_MACRO_SERIES:
        earliest = first.get(sid)
        if earliest is None:
            bad.append(f"{sid} (no observations)")
        elif earliest > required_reach:
            bad.append(f"{sid} (starts {earliest}, need <= {required_reach})")
    if bad:
        return PreflightCheck(
            "macro_history",
            ERROR,
            f"macro series missing before {required_reach} "
            f"(start - {REGIME_MACRO_LOOKBACK_DAYS}d): {', '.join(bad)} — "
            "run `dynamo.cli.backfill macro --start <earlier date>`",
        )
    return PreflightCheck(
        "macro_history",
        OK,
        f"required series {list(REQUIRED_MACRO_SERIES)} reach back to {required_reach}",
    )


def _check_quant_model(cfg: ReplayConfig) -> PreflightCheck:
    """Phase-2 hook: works before AND after the trained-model loader lands."""
    version = getattr(cfg, "quant_model_version", None)
    if not version:
        return PreflightCheck(
            "quant_model",
            WARNING,
            "quant_model_version unset — replay runs the baseline-v0 heuristic "
            "(all-None scores => honest abstention, likely zero candidates)",
        )
    try:
        module = importlib.import_module("dynamo.research.quant.training")
        loader = getattr(module, "bundle_from_version", None)
    except ImportError:
        loader = None
    if loader is None:
        return PreflightCheck(
            "quant_model",
            ERROR,
            f"quant_model_version={version!r} is set but no model loader exists "
            "(research.quant.training.bundle_from_version — Phase 2 not landed)",
        )
    try:
        loader(version)
    except Exception as exc:
        return PreflightCheck(
            "quant_model",
            ERROR,
            f"quant_model_version={version!r} is not loadable: {exc}",
        )
    return PreflightCheck("quant_model", OK, f"model version {version!r} loads")


def _check_source_run(reads: PreflightReads, cfg: ReplayConfig) -> PreflightCheck:
    """Phase-3 hook: portfolio mode replays a committee run's verdicts."""
    source_run_id = getattr(cfg, "source_run_id", None)
    if not source_run_id:
        if cfg.mode == "portfolio":
            return PreflightCheck(
                "source_run",
                ERROR,
                "portfolio mode requires source_run_id (a completed committee run)",
            )
        return PreflightCheck("source_run", OK, "no source run required")
    run = reads.get_run(str(source_run_id))
    if run is None:
        return PreflightCheck(
            "source_run", ERROR, f"source run {source_run_id!r} does not exist"
        )
    problems: list[str] = []
    if run.get("mode") != "committee":
        problems.append(f"mode is {run.get('mode')!r}, need 'committee'")
    start_as_of = cast(date | None, run.get("start_as_of"))
    end_as_of = cast(date | None, run.get("end_as_of"))
    if start_as_of is None or start_as_of > cfg.start or end_as_of is None or end_as_of < cfg.end:
        problems.append(
            f"window [{start_as_of}, {end_as_of}] does not cover "
            f"[{cfg.start}, {cfg.end}]"
        )
    if problems:
        return PreflightCheck(
            "source_run",
            ERROR,
            f"source run {source_run_id!r}: " + "; ".join(problems),
        )
    return PreflightCheck(
        "source_run", OK, f"source run {source_run_id!r} exists, committee-mode, covering"
    )


# -- entry point ------------------------------------------------------------


def run_preflight(engine_or_reads: Any, config: ReplayConfig) -> PreflightReport:
    """All checks, always; raises :class:`PreflightError` when any errored.

    ``engine_or_reads``: a SQLAlchemy ``Engine`` (wrapped in
    :class:`EngineReads`) or any :class:`PreflightReads` implementation.
    """
    reads: PreflightReads = (
        EngineReads(engine_or_reads)
        if isinstance(engine_or_reads, Engine)
        else engine_or_reads
    )
    checks: list[PreflightCheck] = []

    universe_check, symbols = _check_universe(reads, config)
    checks.append(universe_check)
    if symbols:
        checks.append(_check_bar_coverage(reads, symbols, config.start, "start"))
        checks.append(
            _check_bar_coverage(
                reads,
                symbols,
                config.start - timedelta(days=RETURNS_LOOKBACK_DAYS),
                "lookback",
            )
        )
    checks.append(_check_macro(reads, config))
    checks.append(_check_quant_model(config))
    checks.append(_check_source_run(reads, config))

    report = PreflightReport(checks=tuple(checks))
    if not report.passed:
        raise PreflightError(report)
    return report
