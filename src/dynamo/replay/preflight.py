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
- bar ADJUSTMENT (wide-campaign D5): the run's bars must not contain
  unadjusted corporate actions. Asserted EMPIRICALLY — a single-session move
  large enough to be an action, landing on an ex-date recorded in
  ``corporate_actions``, is an unadjusted row and the run is refused. A
  declared per-source policy is checked too, but the empirical test carries
  the weight, because ``bars_1d`` records no per-row adjustment mode: the
  ~6.0M rows written under ``Adjustment.RAW`` were indistinguishable from
  adjusted ones by inspection, and only the price series gave them away. This
  is the check that stops D1 recurring silently;
- regime-critical macro series (HY OAS, curve slope) reach back to at least
  ``cfg.start - 900d`` (the runner's regime lookback) with vintages knowable
  by ``cfg.start``;
- quant-model resolvability (Phase-2 hook): ``quant_model_version`` set but
  not loadable -> error; unset -> warning recorded (baseline-v0 abstention);
  loadable but ``ships_ok=False`` on any task in its own training report ->
  error (downgradable to a warning via
  ``Settings.replay_require_shipping_bundle``). "The bundle loads" was the
  only thing this check ever asserted, which is how ``campaign-v3`` — losing
  to its own historic-average baseline on rank IC (0.049 vs 0.093) and
  downside AUC (0.519 vs 0.721) — powered a full 74-day campaign;
- portfolio-mode source run (backtest-campaign Phase 3): ``source_run_id``
  set -> the run must exist, be committee-mode, cover the window, and carry
  compatible ``universe_tiers`` (compared via its stored ``config_json``);
- discovery (wide campaign), only when ``discovery_mode != 'off'``: tier-2
  membership is non-empty at ``cfg.start``, and enough of it has bars at the
  START of the detectors' 420-day lookback. Discovery over an empty or
  history-less tier 2 reports zero promotions and looks identical to a run
  that genuinely found nothing. The adjustment audit is ALSO widened to the
  surveillance universe when discovery is on — an unadjusted split reads as a
  crash and the detectors hunt crashes, so auditing only the traded core
  while sweeping 11,500 names leaves the flagged set contaminated in exactly
  the population promotions are drawn from.

Zero LLM calls, zero writes: preflight only reads.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

from sqlalchemy import Engine, Text, bindparam, func, select, text
from sqlalchemy.dialects.postgresql import ARRAY

if TYPE_CHECKING:
    import pandas as pd

from dynamo.persistence import schema
from dynamo.replay.discovery import DISCOVERY_LOOKBACK_DAYS
from dynamo.replay.gateway import SURVEILLANCE_TIER
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

# --- Bar adjustment (D5) ---------------------------------------------------
#
# ``bars_1d`` carries no per-row record of the adjustment its prices were
# fetched under, so "is this row adjusted?" cannot be answered by looking at
# the row. It can only be answered by looking at the SERIES: an unadjusted
# corporate action leaves a step, and ``corporate_actions`` says where the
# steps should be. That is why the empirical check below exists and why it,
# not the declaration, is the one that fails a run.

#: Sources DECLARED to serve unadjusted prices. A symbol whose bars come only
#: from one of these and carry no ``adj_close`` is refused outright, without
#: needing a corporate action to prove it.
#:
#: Empty today, and that is a statement rather than an oversight: both live
#: feeds adjust (Tiingo fully, in ``adj_close``; Alpaca for splits, see
#: ``dynamo.data.alpaca_data.DAILY_BAR_ADJUSTMENT``). A declaration is only
#: as good as the day it was written — which is exactly how D1 happened, the
#: comment "Tiingo owns adjustment" having been true for 55 symbols and false
#: for 11,200 — so nothing here is trusted on its own.
UNADJUSTED_BAR_SOURCES: frozenset[str] = frozenset()

#: Days of slack when matching a large move to a split ex-date. A split's
#: step prints on the ex-date, but vendors disagree by a session about which
#: side of the boundary a bar belongs to, so +/-1 day.
SPLIT_MATCH_SLACK_DAYS = 1

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
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }


class PreflightError(RuntimeError):
    """At least one preflight check failed; ``report`` holds all of them."""

    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        lines = [f"{c.name}: {c.detail}" for c in report.errors]
        super().__init__(
            f"replay preflight failed ({len(report.errors)} error(s)):\n  " + "\n  ".join(lines)
        )


#: Bar coverage the SURVEILLANCE universe needs before discovery is honest.
#: Deliberately far below :data:`BAR_COVERAGE_MIN`: tier 2 is ~11,500 names
#: assembled from a whole-market symbol list, and a real fraction of it is
#: illiquid enough to have gaps. Discovery does not TRADE those names, it
#: scans them, and a name with no bars simply never flags. What must not
#: happen is discovery running against a universe that is mostly empty and
#: reporting "0 promotions" as a finding rather than as a data failure.
SURVEILLANCE_COVERAGE_MIN = 0.60


class PreflightReads(Protocol):
    """The read slice preflight needs (satisfied by :class:`EngineReads`)."""

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...]) -> list[str]: ...

    def symbols_with_bars(self, symbols: Sequence[str], start: date, end: date) -> set[str]: ...

    def macro_first_observation(
        self, series_ids: Sequence[str], vintage_cutoff: date
    ) -> dict[str, date | None]: ...

    def get_run(self, run_id: str) -> Mapping[str, Any] | None: ...

    def bars_for_adjustment_audit(
        self, symbols: Sequence[str], start: date, end: date
    ) -> pd.DataFrame: ...

    def split_ex_dates(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[tuple[str, date]]: ...

    def missing_tables(self, names: Sequence[str]) -> list[str]: ...


def _symbol_in(column: Any, symbols: Sequence[str]) -> Any:
    """``symbol = ANY(:array)`` rather than ``IN (:a, :b, ... x10,000)``.

    The wide universe is ~10,400 names. An expanded ``IN`` list of that size
    is one bind parameter per symbol, which pushed the planner into a
    parallel hash and blew the Postgres container's 64 MB ``/dev/shm``
    (``DiskFull: could not resize shared memory segment ... to 189140416
    bytes``) — measured, not hypothetical. An array is ONE parameter.
    """
    array = bindparam("preflight_symbols", value=list(symbols), type_=ARRAY(Text))
    return column == func.any(array)


@contextmanager
def _serial_connection(engine: Engine) -> Iterator[Any]:
    """A connection that will not plan a parallel scan.

    Preflight's wide-universe reads touch ~5.8M ``bars_1d`` rows. Postgres
    plans those in parallel and puts the workers' hash/sort in SHARED memory
    — 188 MB of it, measured — while the dockerised server has the Docker
    default 64 MB of ``/dev/shm``. The result is ``DiskFull: could not resize
    shared memory segment``, i.e. preflight failing for a reason that has
    nothing to do with the data it is checking.

    Serial is the right trade here regardless of the shm limit: this runs
    once, before a multi-hour replay, and a few extra seconds of scan buys a
    plan whose memory is bounded by ``work_mem`` in the backend's own
    address space.
    """
    with engine.connect() as cx:
        cx.execute(text("SET max_parallel_workers_per_gather = 0"))
        yield cx


class EngineReads:
    """Concrete :class:`PreflightReads` over the platform database."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...]) -> list[str]:
        from dynamo.persistence.repositories.market import MarketDataRepository

        members = MarketDataRepository(self._engine).universe_at(as_of)
        return sorted({m.symbol for m in members if int(m.tier) in tiers})

    def symbols_with_bars(self, symbols: Sequence[str], start: date, end: date) -> set[str]:
        t = schema.bars_1d
        stmt = (
            select(t.c.symbol)
            .where(
                _symbol_in(t.c.symbol, symbols),
                func.date(t.c.ts) >= start,
                func.date(t.c.ts) <= end,
            )
            .distinct()
        )
        with _serial_connection(self._engine) as cx:
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

    def bars_for_adjustment_audit(
        self, symbols: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """``symbol, ts, close, source, adj_close`` per STORED row.

        Not the canonical view: the question is what each feed stored, and a
        frame that substituted another feed's close would answer it wrong.

        Deliberately UNSORTED. A server-side ``ORDER BY symbol, ts`` over the
        wide universe's ~5.8M rows plans a parallel sort whose tuple store
        wanted 188 MB of shared memory and died on the container's 64 MB
        ``/dev/shm``. The caller sorts the frame in pandas, where the memory
        is the process's own and already accounted for.
        """
        import pandas as pd

        t = schema.bars_1d
        stmt = select(t.c.symbol, t.c.ts, t.c.close, t.c.source, t.c.adj_close).where(
            _symbol_in(t.c.symbol, symbols),
            func.date(t.c.ts) >= start,
            func.date(t.c.ts) <= end,
        )
        with _serial_connection(self._engine) as cx:
            return pd.read_sql(stmt, cx)

    def split_ex_dates(
        self, symbols: Sequence[str], start: date, end: date
    ) -> set[tuple[str, date]]:
        t = schema.corporate_actions
        stmt = select(t.c.symbol, t.c.ex_date).where(
            t.c.action_type == "split",
            _symbol_in(t.c.symbol, symbols),
            t.c.ex_date >= start,
            t.c.ex_date <= end,
        )
        with _serial_connection(self._engine) as cx:
            return {(str(r.symbol), cast(date, r.ex_date)) for r in cx.execute(stmt)}

    def missing_tables(self, names: Sequence[str]) -> list[str]:
        from sqlalchemy import inspect

        present = set(inspect(self._engine).get_table_names())
        return [n for n in names if n not in present]


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


def _check_universe(reads: PreflightReads, cfg: ReplayConfig) -> tuple[PreflightCheck, list[str]]:
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
        PreflightCheck("universe", OK, f"{len(symbols)} symbols active at {cfg.start}"),
        symbols,
    )


#: Tables every replay mode writes, regardless of config.
_ALWAYS_WRITTEN = ("replay_runs", "replay_decisions", "replay_equity", "replay_fills")
#: Tables only some configurations write, keyed by what turns them on.
_COMMITTEE_WRITTEN = ("replay_verdicts", "replay_contexts")
_DISCOVERY_WRITTEN = ("replay_flags",)


def _check_schema(reads: PreflightReads, cfg: ReplayConfig) -> PreflightCheck:
    """Do the tables this run will WRITE actually exist?

    Measured, not hypothetical: the dev database sat at migration 0020 while
    head was 0022, so ``replay_flags`` did not exist. Every other check here
    passed — the universe, the bars, the model were all fine — and the run
    would have died on its first flag insert, hours in, with a raw
    ``UndefinedTable`` traceback.

    Checking the tables rather than comparing ``alembic_version`` to head is
    deliberate: it names the thing that will break, it needs no alembic
    config discovery at runtime, and a database legitimately ahead of this
    checkout (a rollback in progress) is not an error for a run whose tables
    are all present.
    """
    wanted = list(_ALWAYS_WRITTEN)
    if cfg.mode == "committee":
        wanted += list(_COMMITTEE_WRITTEN)
    if cfg.discovery_mode != "off":
        wanted += list(_DISCOVERY_WRITTEN)

    missing = reads.missing_tables(wanted)
    if missing:
        return PreflightCheck(
            "schema",
            ERROR,
            f"{len(missing)} table(s) this run writes do not exist: "
            f"{', '.join(missing)} — the database is behind the code; "
            "run `alembic upgrade head`",
        )
    return PreflightCheck("schema", OK, f"all {len(wanted)} written tables present")


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


def _declared_raw_symbols(bars: pd.DataFrame) -> list[str]:
    """Symbols served ONLY by a source declared unadjusted, with no adj_close.

    "Only" matters: a symbol that also has a Tiingo row has an adjusted close
    available through the canonical view, so the raw row is harmless.
    """

    if bars.empty or not UNADJUSTED_BAR_SOURCES:
        return []
    source = cast("pd.Series", bars["source"])
    raw = source.isin(list(UNADJUSTED_BAR_SOURCES))
    grouped = bars.assign(_raw=raw, _adj=cast("pd.Series", bars["adj_close"]).notna()).groupby(
        "symbol"
    )
    flags = grouped.agg(all_raw=("_raw", "all"), any_adj=("_adj", "any"))
    bad = cast("pd.DataFrame", flags[flags["all_raw"] & ~flags["any_adj"]])
    return sorted(str(s) for s in bad.index)


def _check_bar_adjustment(
    reads: PreflightReads, symbols: list[str], cfg: ReplayConfig
) -> PreflightCheck:
    """D5 — refuse a run whose bars still contain unadjusted corporate actions.

    An unadjusted split is not a data gap; it is a fabricated return, and it
    is worse than a gap because the detectors are built to notice exactly
    that shape. Before the repair, 1,166 single-session moves in this panel
    landed on a recorded split ex-date. Nothing failed. The pilot's flagged
    set was contaminated and the run reported a number anyway.
    """
    import pandas as pd

    from dynamo.data.quality import EXTREME_DOWN_RETURN, EXTREME_UP_RETURN

    start = cfg.start - timedelta(days=RETURNS_LOOKBACK_DAYS)
    bars = reads.bars_for_adjustment_audit(symbols, start, cfg.end)
    if bars.empty:
        return PreflightCheck(
            "bar_adjustment",
            WARNING,
            f"no bars for the universe over [{start}, {cfg.end}] — adjustment could not be checked",
        )

    declared = _declared_raw_symbols(bars)
    splits = reads.split_ex_dates(symbols, start, cfg.end)

    # The EFFECTIVE close — adj_close where a feed supplies one, else close.
    # This is the series the canonical view hands the replay, and reading the
    # raw `close` column instead produces false alarms: Tiingo stores XLE's
    # unadjusted $92.22 -> $45.92 across its real 2025-12-05 two-for-one, and
    # its `adj_close` ($45.11 -> $44.92) is the continuous series the runner
    # actually sees. Auditing a column nobody trades is not auditing.
    frame = cast("pd.DataFrame", bars[["symbol", "ts", "close", "adj_close"]]).sort_values(
        by=["symbol", "ts"], kind="mergesort"
    )
    adj = cast("pd.Series", frame["adj_close"])
    frame["close"] = adj.where(adj.notna(), frame["close"])
    prev = cast("pd.Series", frame.groupby("symbol")["close"].shift(1))
    close = cast("pd.Series", frame["close"])
    ret = close / prev.where(prev > 0.0) - 1.0
    extreme = cast("pd.DataFrame", frame[(ret < EXTREME_DOWN_RETURN) | (ret > EXTREME_UP_RETURN)])
    slack = [timedelta(days=d) for d in range(-SPLIT_MATCH_SLACK_DAYS, SPLIT_MATCH_SLACK_DAYS + 1)]
    unadjusted = sorted(
        {
            str(sym)
            for sym, ts in zip(extreme["symbol"], extreme["ts"], strict=True)
            if any((str(sym), cast(date, pd.Timestamp(ts).date()) + d) in splits for d in slack)
        }
    )

    offenders = sorted(set(unadjusted) | set(declared))
    if offenders:
        why = []
        if unadjusted:
            why.append(
                f"{len(unadjusted)} with a large move on a recorded split "
                f"ex-date (e.g. {unadjusted[:10]})"
            )
        if declared:
            why.append(
                f"{len(declared)} served only by a source declared unadjusted "
                f"with no adj_close (e.g. {declared[:10]})"
            )
        return PreflightCheck(
            "bar_adjustment",
            ERROR,
            f"{len(offenders)} of {len(symbols)} universe symbols carry "
            f"UNADJUSTED prices over [{start}, {cfg.end}]: "
            + "; ".join(why)
            + ". A split reads as a crash and the detectors hunt crashes — "
            "re-backfill on an adjusted feed (`dynamo.cli.backfill bars "
            "--provider alpaca --force`), or drop these symbols from the "
            "universe. Audit with `python -m dynamo.cli.repair census "
            "--corporate-actions`",
        )
    if not splits:
        return PreflightCheck(
            "bar_adjustment",
            WARNING,
            f"no splits recorded in corporate_actions for this universe over "
            f"[{start}, {cfg.end}] — there was nothing to check against, so "
            "adjustment is UNVERIFIED (run `dynamo.cli.repair corporate-actions`)",
        )
    return PreflightCheck(
        "bar_adjustment",
        OK,
        f"no unadjusted corporate actions: {len(symbols)} symbols checked "
        f"against {len(splits)} recorded splits over [{start}, {cfg.end}]",
    )


def _check_surveillance_universe(
    reads: PreflightReads, cfg: ReplayConfig
) -> tuple[PreflightCheck, list[str]]:
    """Discovery has a universe to scan at ``cfg.start`` (wide campaign).

    Without this, ``--discovery code`` over a database whose tier-2 seed was
    never backdated runs 115 sessions of detectors over zero symbols, reports
    zero promotions, and looks exactly like a run where discovery genuinely
    found nothing. That is the same class of silent failure the universe and
    bar-coverage checks already exist to prevent, one tier over.
    """
    symbols = reads.universe_symbols(cfg.start, (SURVEILLANCE_TIER,))
    if not symbols:
        return (
            PreflightCheck(
                "discovery_universe",
                ERROR,
                f"discovery_mode={cfg.discovery_mode!r} but tier-2 membership is "
                f"EMPTY at {cfg.start} — the detectors would sweep nothing for "
                "the whole run and report zero promotions as a finding. Run "
                "`dynamo.cli.backfill universe-wide --backdate-from-bars` to "
                "backdate the surveillance seed",
            ),
            [],
        )
    return (
        PreflightCheck(
            "discovery_universe",
            OK,
            f"{len(symbols)} tier-2 symbols active at {cfg.start}",
        ),
        symbols,
    )


def _check_surveillance_coverage(
    reads: PreflightReads, symbols: list[str], cfg: ReplayConfig
) -> PreflightCheck:
    """Enough of the surveillance universe has bars at the START of the
    detector lookback, not just at ``cfg.start``.

    The window that matters is ``start - 420d``: ``range_break`` reads 252
    prior sessions and ``range_bound`` refuses to fire below 131. A universe
    with bars only from ``cfg.start`` onward produces a run whose first
    months are structurally incapable of flagging, silently.
    """
    center = cfg.start - timedelta(days=DISCOVERY_LOOKBACK_DAYS)
    lo, hi = session_window(center)
    present = reads.symbols_with_bars(symbols, lo, hi)
    covered = sum(1 for s in symbols if s in present)
    coverage = covered / len(symbols) if symbols else 0.0
    detail = (
        f"{covered}/{len(symbols)} tier-2 symbols ({coverage:.1%}) have bars "
        f"within [{lo}, {hi}] — the start of the detectors' "
        f"{DISCOVERY_LOOKBACK_DAYS}d lookback"
    )
    if coverage < SURVEILLANCE_COVERAGE_MIN:
        return PreflightCheck(
            "discovery_bar_coverage",
            ERROR,
            f"{detail} — below the {SURVEILLANCE_COVERAGE_MIN:.0%} floor. "
            "range_break needs 252 prior sessions and range_bound will not "
            "fire below 131, so the run's early months could not flag at all. "
            "Backfill bars for the wide universe first",
        )
    return PreflightCheck("discovery_bar_coverage", OK, detail)


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


def _read_bundle_metrics(version: str) -> Mapping[str, Any]:
    """The bundle manifest's ``metrics`` block (the trainer's report card)."""
    from dynamo.research.quant.bundle import default_bundle_dir, read_bundle_manifest

    manifest = read_bundle_manifest(default_bundle_dir(), version)
    metrics = manifest.get("metrics") or {}
    return cast(Mapping[str, Any], metrics)


def _model_names(pooled: Mapping[str, Any]) -> list[str]:
    """Trained model names in a task report (baselines are the comparison)."""
    return [str(n) for n in pooled if not str(n).startswith("baseline")]


def ship_gate_lines(metrics: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """``(failing task lines, all task lines)`` from a bundle's metrics.

    Reads the per-task ``ships_ok`` the trainer wrote alongside the pooled
    out-of-fold metric and its best baseline, so preflight reports the same
    go/no-go evidence the training run printed. An empty result means the
    manifest predates the ship gate (nothing to assert)."""
    tasks = metrics.get("tasks")
    if not isinstance(tasks, Mapping):
        return [], []
    failing: list[str] = []
    lines: list[str] = []
    for task in sorted(tasks):
        report = tasks[task]
        if not isinstance(report, Mapping):
            continue
        ships = report.get("ships_ok")
        if not isinstance(ships, Mapping) or not ships:
            continue
        pooled = report.get("pooled")
        pooled_map = pooled if isinstance(pooled, Mapping) else {}
        baseline = report.get("best_baseline_pooled")
        ok = any(bool(v) for v in ships.values())
        detail = f"{task}: ships_ok={ok}"
        for name in _model_names(pooled_map) or [str(n) for n in ships]:
            entry = pooled_map.get(name)
            if isinstance(entry, Mapping) and "fold_metric_mean" in entry:
                detail += (
                    f" ({name} pooled fold metric {float(entry['fold_metric_mean']):.4f}"
                    f", win share {float(entry.get('win_share', float('nan'))):.2f}"
                )
                if baseline is not None:
                    detail += f", best baseline {float(baseline):.4f}"
                detail += ")"
        lines.append(detail)
        if not ok:
            failing.append(detail)
    return failing, lines


def _check_quant_model(cfg: ReplayConfig, require_shipping: bool) -> PreflightCheck:
    """Phase-2 hook: works before AND after the trained-model loader lands.

    Beyond "does it load", the bundle must have PASSED its own ship gate:
    a model that loses to the historic-average baseline is worse than no
    model, because its rankings still get traded."""
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
    try:
        failing, lines = ship_gate_lines(_read_bundle_metrics(str(version)))
    except Exception as exc:  # unreadable manifest: loud, never fatal here
        return PreflightCheck(
            "quant_model",
            WARNING,
            f"model version {version!r} loads, but its ship-gate metrics could "
            f"not be read ({exc}) — the gate was NOT consulted",
        )
    if not lines:
        return PreflightCheck(
            "quant_model",
            WARNING,
            f"model version {version!r} loads, but its manifest carries no "
            "per-task ships_ok — the gate was NOT consulted (retrain to record it)",
        )
    if failing:
        return PreflightCheck(
            "quant_model",
            ERROR if require_shipping else WARNING,
            f"model version {version!r} loads but FAILED its own ship gate — "
            + "; ".join(failing)
            + ". A bundle that loses to its baseline still gets traded: retrain, "
            "or set REPLAY_REQUIRE_SHIPPING_BUNDLE=false to run it deliberately",
        )
    return PreflightCheck(
        "quant_model",
        OK,
        f"model version {version!r} loads and ships — " + "; ".join(lines),
    )


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
        return PreflightCheck("source_run", ERROR, f"source run {source_run_id!r} does not exist")
    problems: list[str] = []
    if run.get("mode") != "committee":
        problems.append(f"mode is {run.get('mode')!r}, need 'committee'")
    start_as_of = cast(date | None, run.get("start_as_of"))
    end_as_of = cast(date | None, run.get("end_as_of"))
    if start_as_of is None or start_as_of > cfg.start or end_as_of is None or end_as_of < cfg.end:
        problems.append(
            f"window [{start_as_of}, {end_as_of}] does not cover [{cfg.start}, {cfg.end}]"
        )
    # Universe compatibility: the committee run's verdicts only make sense
    # against the same universe tiers (checked via its stored config_json;
    # runs recorded before the config was stored are tolerated).
    source_config = run.get("config_json") or {}
    source_tiers = source_config.get("universe_tiers")
    if source_tiers is not None and tuple(source_tiers) != tuple(cfg.universe_tiers):
        problems.append(
            f"universe_tiers {tuple(source_tiers)} incompatible with "
            f"this config's {tuple(cfg.universe_tiers)}"
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


def run_preflight(
    engine_or_reads: Any,
    config: ReplayConfig,
    *,
    require_shipping_bundle: bool | None = None,
) -> PreflightReport:
    """All checks, always; raises :class:`PreflightError` when any errored.

    ``engine_or_reads``: a SQLAlchemy ``Engine`` (wrapped in
    :class:`EngineReads`) or any :class:`PreflightReads` implementation.
    ``require_shipping_bundle`` decides whether a pinned bundle that failed
    its own ship gate is an ERROR (default) or a loud WARNING; ``None``
    reads ``Settings.replay_require_shipping_bundle``.
    """
    if require_shipping_bundle is None:
        from dynamo.config.settings import Settings

        require_shipping_bundle = bool(Settings().replay_require_shipping_bundle)
    reads: PreflightReads = (
        EngineReads(engine_or_reads) if isinstance(engine_or_reads, Engine) else engine_or_reads
    )
    checks: list[PreflightCheck] = []

    # First, and cheapest: a database behind the code fails every downstream
    # check for the wrong reason, or worse, passes them all and dies hours in.
    checks.append(_check_schema(reads, config))

    universe_check, symbols = _check_universe(reads, config)
    checks.append(universe_check)
    # Discovery widens what the run trades, so it widens what has to be
    # checked: the adjustment audit below runs over the TRADED core plus the
    # surveillance universe when discovery is on. An unadjusted split reads
    # as a crash and the detectors hunt crashes — auditing only the 55-name
    # core while sweeping 11,500 would leave the flagged set contaminated in
    # exactly the population discovery draws its promotions from.
    audited = list(symbols)
    if config.discovery_mode != "off":
        surveillance_check, tier2 = _check_surveillance_universe(reads, config)
        checks.append(surveillance_check)
        if tier2:
            checks.append(_check_surveillance_coverage(reads, tier2, config))
            audited = sorted(set(audited) | set(tier2))
    if symbols:
        checks.append(_check_bar_coverage(reads, symbols, config.start, "start"))
        # The lookback check must be measured against the universe AS OF THE
        # LOOKBACK DATE, not as of the start. A wide universe carries listings:
        # measured on the real tier-2 universe, 2,592 of 10,429 members at
        # 2026-03-03 joined after the 750d-lookback date — IPOs cannot have
        # pre-IPO bars, and dividing by today's membership reported 75.8%
        # "coverage" against a 95% floor when coverage among symbols that
        # EXISTED was ~100%. Names without full history simply abstain via the
        # feature-coverage gate; their absence two years ago is not a data
        # failure.
        lookback_date = config.start - timedelta(days=RETURNS_LOOKBACK_DAYS)
        lookback_universe = [
            s
            for s in reads.universe_symbols(lookback_date, config.universe_tiers)
            if s in set(symbols)
        ]
        if lookback_universe:
            checks.append(_check_bar_coverage(reads, lookback_universe, lookback_date, "lookback"))
        else:
            checks.append(
                PreflightCheck(
                    "bar_coverage_lookback",
                    WARNING,
                    f"no current universe member was a member at {lookback_date} — "
                    "lookback coverage unmeasurable (very young universe)",
                )
            )
        checks.append(_check_bar_adjustment(reads, audited, config))
    checks.append(_check_macro(reads, config))
    checks.append(_check_quant_model(config, require_shipping_bundle))
    checks.append(_check_source_run(reads, config))

    report = PreflightReport(checks=tuple(checks))
    if not report.passed:
        raise PreflightError(report)
    return report
