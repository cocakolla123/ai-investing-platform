"""Replay discovery — the layer that lets a backtest find names it was not
already trading (wide-campaign design, §"How discovery participates").

The problem
-----------
Every replay so far ran over a static tier-1/3 list. That measures portfolio
construction and risk, and it measures NOTHING about the platform's actual
thesis, which is that a whole-market detector sweep surfaces opportunities a
55-name core never contains. The control run debated 20 distinct symbols in
74 sessions. Discovery is the missing prepend that changes what the universe
IS on each day, leaving every downstream stage untouched:

    1. panel      = SurveillancePanel.slice(as_of=D, lookback_days=420)
    2. flags      = run_detectors(...)          # dynamo.data.detectors, unforked
    3. promoted_D = promotion_gate(flags, cap)  # 'code' | 'haiku' | 'off'
    4. universe_D = tier1_at(D) | {promoted within the last N sessions}
    5-7. quant scores -> shortlist -> committee -> risk -> constructor

Steps 5-7 are untouched. ``discovery_mode='off'`` constructs none of this and
is byte-identical to the pre-discovery runner, which
``tests/unit/replay/test_discovery.py`` asserts directly — discovery has to
be a pure addition or the A/B it exists to measure is not an A/B.

Why the panel exists
--------------------
The naive shape is "read 420 days of bars for the tier-2 universe, per replay
day". Measured: 9 seconds and 3.09M rows per read, times 115 sessions — over
17 minutes of pure database round trips before a single detector runs, all of
it re-reading the same rows. :class:`SurveillancePanel` reads that window
ONCE for the whole run, holds per-symbol arrays sorted by date, and serves
each day by index.

Measured cost of what remains, on a synthetic 8,000-symbol x 300-session
panel: **5.3s per replay day** end to end (slice 1.3s + five detectors +
gate), plus a 1.8s one-off panel build. Over a 115-session window that is
about ten minutes of CPU for the entire discovery layer, against seventeen
minutes of database round trips for the naive shape alone.

Loading once means the loaded frame contains rows dated after the early
sessions' ``as_of``. That is deliberate and it is where point-in-time
correctness lives: the load happens through a gateway pinned at the run's
END date (so nothing after the run window can enter at all), and
:meth:`SurveillancePanel.slice` is the per-day bound, cutting each symbol's
array at ``as_of`` by ``searchsorted``. The slice cannot return a later bar
because the index it takes is computed from the bound itself — and
:meth:`SurveillancePanel.slice` re-asserts it anyway, raising
:class:`~dynamo.replay.gateway.ReplayLeakageError` rather than trusting the
arithmetic.

Why the scoping matters
-----------------------
Feature building costs ~4s + 0.135s per symbol. Scoped to ``universe_D``
(core plus promotions, ~300-600 names) that is ~60s/day. Run over all of
tier 2 (~11,500 names) it would be ~26 min/day, i.e. ~50 hours for a
115-session window. The scoping IS the architecture; discovery decides which
few hundred names are worth the feature build, and everything downstream sees
only those.

FIREWALL
--------
Discovery writes ONLY to ``replay_flags`` through
:meth:`~dynamo.persistence.repositories.replay.ReplayRepository.insert_flags`.
It must never write ``surveillance_flags`` or ``universe_membership``: those
are live tables, live triage promotes out of the first into the second, and
``HistoricalDataGateway.universe()`` reads the second. A single replay write
would contaminate production AND silently rewrite the point-in-time history
of every future replay. ``tests/unit/replay/test_firewall.py`` asserts the
live row counts are unchanged after a discovery run.

Determinism
-----------
No wall clock (every timestamp derives from the session date), no set
iteration order, ties broken ``(-score, symbol)``. The ``code`` gate is a
pure function of the flags, so the same config over the same data promotes
the same names — which is what makes ``discovery_mode`` meaningful inside
``config_hash``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from dynamo.data.detectors import DATA_QUALITY_DETECTOR, DetectorInputs, run_detectors
from dynamo.domain.models.market import SurveillanceFlag
from dynamo.domain.volume_source import (
    consolidated_equivalent_dollar_volume,
    dominant_source,
)
from dynamo.replay.gateway import ReplayLeakageError
from dynamo.research.flag_ranking import group_by_symbol, score_symbols, select_for_triage
from dynamo.research.shortlist import SessionHistory

#: The detector catalog's longest lookback, in calendar days. Derived from the
#: detectors themselves in ``dynamo.data.detector_windows`` and guarded by a
#: test there — a shorter window makes ``range_break`` report 3-month highs as
#: 52-week breakouts and stops ``range_bound`` firing at all, both silently.
DISCOVERY_LOOKBACK_DAYS = 420

#: Trailing sessions the liquidity filter measures over. 60 matches the
#: window ``price_zscore`` and ``volume_spike`` use for their own medians, so
#: the pre-filter and the detectors agree about what "typical volume" means.
LIQUIDITY_WINDOW_SESSIONS = 60

#: Valid ``ReplayConfig.discovery_mode`` values.
DISCOVERY_MODES: tuple[str, ...] = ("off", "code", "haiku")

#: The triage model the ``haiku`` gate mirrors. Named here for the knowledge
#: gate and the audit trail; the gate itself never resolves a client.
HAIKU_TIER = "screen"

#: Verdict that promotes, matching live ``_triage_flags``.
MATERIAL_VERDICT = "material"


class SurveillanceReads(Protocol):
    """The gateway slice the panel loads through."""

    as_of: date

    def surveillance_universe(self) -> list[str]: ...
    def surveillance_bars(
        self, symbols: Sequence[str], lookback_days: int
    ) -> pd.DataFrame: ...


class FlagSink(Protocol):
    """The ``ReplayRepository`` slice discovery writes through — the ONLY
    write path it has (see the module docstring's FIREWALL section)."""

    def insert_flags(self, rows: Sequence[Mapping[str, Any]]) -> int: ...


class TriageModel(Protocol):
    """The ``ResearchModel`` slice the ``haiku`` gate calls."""

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        model_tier: str,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSlice:
    """One replay day's view of the panel: per-symbol frames bounded at
    ``as_of``, already filtered for tradability."""

    as_of: date
    per_symbol: dict[str, pd.DataFrame]
    #: Symbols the liquidity/price filter removed, for the audit trail.
    filtered_out: int = 0

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.per_symbol))


class SurveillancePanel:
    """Every tier-2 bar the run needs, read once, sliced by index per day.

    Construct with :meth:`load`. The instance is immutable after that and
    :meth:`slice` is pure, so the same panel serves every session of a run
    (and every run over the same window, in a test) identically.
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], end: date) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._dates: dict[str, np.ndarray] = {}
        self.end = end
        for symbol in sorted(frames):
            frame = frames[symbol].sort_values("ts").reset_index(drop=True)
            self._frames[symbol] = frame
            # Session dates as naive datetime64[D], the axis ``slice``
            # searches. Naive because the bound is a calendar date and
            # comparing an aware column against one is a pandas error rather
            # than a comparison.
            ts = pd.to_datetime(frame["ts"])
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
            self._dates[symbol] = ts.to_numpy(dtype="datetime64[D]")

    @classmethod
    def load(
        cls,
        gw: SurveillanceReads,
        symbols: Sequence[str],
        *,
        lookback_days: int = DISCOVERY_LOOKBACK_DAYS,
        span_days: int = 0,
    ) -> SurveillancePanel:
        """Read ``[gw.as_of - span - lookback, gw.as_of]`` in ONE query.

        ``gw`` must be pinned at the run's END date and ``span_days`` set to
        the run's length, so the read covers the whole window plus the
        detectors' lookback. The per-day point-in-time bound is
        :meth:`slice`, not this read — see the module docstring.
        """
        frame = gw.surveillance_bars(list(symbols), lookback_days + max(span_days, 0))
        if frame.empty:
            return cls({}, gw.as_of)
        return cls(
            {str(sym): grp for sym, grp in frame.groupby("symbol")},
            gw.as_of,
        )

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._frames)

    def slice(
        self,
        as_of: date,
        *,
        lookback_days: int = DISCOVERY_LOOKBACK_DAYS,
        symbols: Sequence[str] | None = None,
        min_dollar_volume: float = 0.0,
        min_price: float = 0.0,
    ) -> PanelSlice:
        """The per-symbol frames for ``as_of``, bounded and pre-filtered.

        THIS is the point-in-time boundary. Each symbol's array is cut at
        ``as_of`` by ``searchsorted``, so a later bar cannot be returned; the
        assertion below re-checks it and raises rather than trusting index
        arithmetic to stay correct through a future edit.

        ``min_dollar_volume`` is measured in CONSOLIDATED-EQUIVALENT dollars
        (``dynamo.domain.volume_source``) over the trailing
        :data:`LIQUIDITY_WINDOW_SESSIONS`, because the free feed is IEX-only
        for ~11,200 of the ~11,500 names and its raw volume is ~2-3% of the
        tape. Filtering on the observed number would keep essentially
        nothing. ``min_price`` is on the last close: sub-dollar names produce
        the loudest z-scores and the least tradable ones.
        """
        wanted = self._frames.keys() if symbols is None else symbols
        lo_bound = np.datetime64(as_of - timedelta(days=lookback_days), "D")
        # Exclusive upper bound at as_of + 1 day == inclusive at as_of.
        hi_bound = np.datetime64(as_of + timedelta(days=1), "D")

        out: dict[str, pd.DataFrame] = {}
        filtered = 0
        for symbol in wanted:
            dates = self._dates.get(symbol)
            if dates is None:
                continue
            lo = int(np.searchsorted(dates, lo_bound, side="left"))
            hi = int(np.searchsorted(dates, hi_bound, side="left"))
            if hi <= lo:
                continue
            frame = self._frames[symbol].iloc[lo:hi].reset_index(drop=True)
            if not _tradable(frame, min_dollar_volume, min_price):
                filtered += 1
                continue
            out[symbol] = frame

        self._assert_bounded(out, as_of)
        return PanelSlice(as_of=as_of, per_symbol=out, filtered_out=filtered)

    @staticmethod
    def _assert_bounded(frames: Mapping[str, pd.DataFrame], as_of: date) -> None:
        """The panel's own leakage tripwire — the gateway's cannot see this
        slice, because the read that fed it happened at the run's end date."""
        for symbol, frame in frames.items():
            if frame.empty:
                continue
            last = pd.Timestamp(frame["ts"].iloc[-1])
            if last is pd.NaT:
                continue
            end = cast(date, last.date())
            if end > as_of:
                raise ReplayLeakageError(
                    f"leakage tripwire: panel slice for {symbol} ends "
                    f"{end}, after the as_of {as_of} cutoff"
                )


def _tradable(frame: pd.DataFrame, min_dollar_volume: float, min_price: float) -> bool:
    """Liquidity + price pre-filter, applied before the detectors run.

    Rejecting here rather than inside the detectors is what keeps flags/day
    in a readable range: the first whole-market sweep produced 1,155 flags in
    one night, 39 of the 61 ``price_zscore`` flags on names whose MEDIAN
    daily dollar volume was exactly zero.
    """
    if frame.empty:
        return False
    close = float(frame["close"].iloc[-1])
    if not np.isfinite(close) or close < min_price:
        return False
    if min_dollar_volume <= 0.0:
        return True
    tail = frame.tail(LIQUIDITY_WINDOW_SESSIONS)
    observed = float((tail["close"] * tail["volume"]).median())
    if not np.isfinite(observed):
        return False
    source = dominant_source(tail["volume_source"]) if "volume_source" in tail else None
    return consolidated_equivalent_dollar_volume(observed, source) >= min_dollar_volume


# --------------------------------------------------------------------------
# promotion gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Promotion:
    """One symbol admitted into ``universe_D`` by the day's gate."""

    symbol: str
    rank: int  # 1-based, in gate order
    score: float  # flag_ranking composite (0 when the gate does not rank)
    severity_total: float  # raw severity sum, for the audit trail
    detectors: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GateResult:
    """What a gate decided for one day."""

    promoted: tuple[Promotion, ...]
    considered: tuple[str, ...]  # symbols the gate looked at (cooldown input)
    llm_calls: int = 0

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(p.symbol for p in self.promoted)


def tradable_flags(flags: Sequence[SurveillanceFlag]) -> list[SurveillanceFlag]:
    """Flags that could become a position.

    ``data_quality`` flags are dropped, exactly as live triage drops them
    before spending a call: they do not describe a price move, they describe
    a bar the detectors could not trust (an unadjusted corporate action).
    Promoting on one would put a fabricated return into the universe.
    """
    return [f for f in flags if f.detector != DATA_QUALITY_DETECTOR]


def _candidates(
    flags: Sequence[SurveillanceFlag],
    *,
    cap: int,
    sectors: Mapping[str, str],
    history: SessionHistory,
    cooldown_sessions: int,
) -> tuple[list[str], dict[str, list[SurveillanceFlag]], dict[str, float]]:
    """The day's ranked, capped, diversified candidate symbols.

    REUSED, not reimplemented. ``select_for_triage`` is the live ranking
    stage (percentile-within-detector, weighted composite, corroboration
    bonus) wrapped around ``select_shortlist``'s cooldown and sector
    round-robin. Ranking flagged symbols by raw severity — the obvious
    implementation — is exactly the bug ``flag_ranking`` exists to fix: the
    detectors do not share a severity scale, so ``volume_spike`` (which
    saturates past 6x median volume) outranked every 52-week breakout, every
    night, structurally.
    """
    ranked = select_for_triage(
        flags,
        cap,
        sectors=sectors,
        history=history,
        cooldown_sessions=cooldown_sessions,
    )
    return ranked, group_by_symbol(flags, ranked), score_symbols(flags)


def _promotion(
    symbol: str,
    rank: int,
    scores: Mapping[str, float],
    sym_flags: Sequence[SurveillanceFlag],
    reason: str,
) -> Promotion:
    return Promotion(
        symbol=symbol,
        rank=rank,
        score=round(float(scores.get(symbol, 0.0)), 6),
        severity_total=round(sum(float(f.severity) for f in sym_flags), 6),
        detectors=tuple(sorted({f.detector for f in sym_flags})),
        reason=reason,
    )


def off_gate(flags: Sequence[SurveillanceFlag], **_: Any) -> GateResult:
    """Promote nothing. The control arm — and the reason ``off`` mode never
    constructs a panel at all."""
    return GateResult(promoted=(), considered=())


def code_gate(
    flags: Sequence[SurveillanceFlag],
    *,
    cap: int,
    sectors: Mapping[str, str] | None = None,
    history: SessionHistory = (),
    cooldown_sessions: int = 3,
) -> GateResult:
    """Deterministic promotion: rank, cap, promote. $0, no model, hashable.

    Every symbol the ranking selects is promoted — the gate spends no
    judgement, which is the point. It is the denominator the ``haiku`` gate
    is measured against: the difference between the two arms is what the
    triage LLM adds over a ranking anyone can compute for free.

    Pure and total: no wall clock, no randomness, no set-iteration order.
    Ties inside the ranking break ``(-score, symbol)``.
    """
    usable = tradable_flags(flags)
    if cap <= 0 or not usable:
        return GateResult(promoted=(), considered=())
    ranked, by_symbol, scores = _candidates(
        usable,
        cap=cap,
        sectors=sectors or {},
        history=history,
        cooldown_sessions=cooldown_sessions,
    )
    promoted = tuple(
        _promotion(
            symbol,
            rank,
            scores,
            by_symbol.get(symbol, ()),
            "code gate: ranked "
            f"{rank}/{len(ranked)} on "
            + ", ".join(sorted({f.detector for f in by_symbol.get(symbol, ())})),
        )
        for rank, symbol in enumerate(ranked, start=1)
    )
    return GateResult(promoted=promoted, considered=tuple(ranked))


def haiku_gate(
    flags: Sequence[SurveillanceFlag],
    *,
    cap: int,
    as_of: date,
    model: TriageModel,
    system_prompt: str,
    schema: dict[str, Any],
    sectors: Mapping[str, str] | None = None,
    history: SessionHistory = (),
    cooldown_sessions: int = 3,
    max_tokens: int = 512,
) -> GateResult:
    """The live ``_triage_flags`` decision, replayed.

    Same ranking, same cap, same per-symbol prompt shape, same
    ``model_tier='screen'`` and ``max_tokens``, same promote-iff-``material``
    rule as ``orchestration/assets/research.py::_triage_flags``. What differs
    is only where the answer goes: a :class:`Promotion` for the day's
    universe instead of a ``universe_membership`` row (see the FIREWALL
    section of the module docstring).

    ``model`` is injected, never resolved here — this function makes real API
    calls when handed a real client, and the unit tests hand it a fake. A
    symbol whose response is unparsable or missing a verdict is NOT promoted:
    a triage call that did not happen is not evidence of materiality.
    """
    usable = tradable_flags(flags)
    if cap <= 0 or not usable:
        return GateResult(promoted=(), considered=())
    ranked, by_symbol, scores = _candidates(
        usable,
        cap=cap,
        sectors=sectors or {},
        history=history,
        cooldown_sessions=cooldown_sessions,
    )
    promoted: list[Promotion] = []
    calls = 0
    for symbol in ranked:
        sym_flags = by_symbol.get(symbol, [])
        parsed = model.complete_json(
            role="triage",
            system=system_prompt,
            context=f"## Surveillance flags for {symbol}\n{_flag_text(sym_flags)}",
            instruction=(
                f"Classify the materiality of these flags for {symbol} as of {as_of}."
            ),
            schema=schema,
            model_tier=HAIKU_TIER,
            max_tokens=max_tokens,
        )
        calls += 1
        verdict = str((parsed or {}).get("verdict", "noise"))
        if verdict != MATERIAL_VERDICT:
            continue
        why = str((parsed or {}).get("reason", "")).strip()
        promoted.append(
            _promotion(
                symbol,
                len(promoted) + 1,
                scores,
                sym_flags,
                f"triage material: {why}" if why else "triage material",
            )
        )
    return GateResult(
        promoted=tuple(promoted), considered=tuple(ranked), llm_calls=calls
    )


def _flag_text(flags: Sequence[SurveillanceFlag]) -> str:
    """The live triage prompt's flag block, verbatim (``_triage_flags``)."""
    return "\n".join(
        f"- {f.detector} severity={f.severity:.2f} "
        f"details={json.dumps(f.details, sort_keys=True, default=str)}"
        for f in flags
    )


# --------------------------------------------------------------------------
# the per-run driver
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryDay:
    """One replay day's discovery result (the audit record)."""

    as_of: date
    scanned: int  # symbols that reached the detectors
    filtered_out: int  # symbols the liquidity/price filter removed
    flags: tuple[SurveillanceFlag, ...]
    promoted: tuple[Promotion, ...]
    universe: tuple[str, ...]  # universe_D handed downstream
    llm_calls: int = 0


@dataclass
class PromotionWindow:
    """Symbols promoted within the last ``sessions`` REPLAY SESSIONS.

    Sessions, not calendar days: a promotion made on the Friday before a long
    weekend should get the same number of trading days in the universe as one
    made on a Tuesday. Deterministic — insertion order never reaches the
    output, which is always sorted.
    """

    sessions: int
    _log: list[tuple[date, tuple[str, ...]]] = field(default_factory=list)

    def record(self, as_of: date, symbols: Iterable[str]) -> None:
        self._log.append((as_of, tuple(sorted(set(symbols)))))

    def active(self) -> tuple[str, ...]:
        if self.sessions <= 0:
            return ()
        window = self._log[-self.sessions :]
        return tuple(sorted({sym for _, syms in window for sym in syms}))

    @property
    def history(self) -> SessionHistory:
        """This run's sessions and what each considered — the cooldown input
        ``select_for_triage`` expects (ascending, one entry per session
        INCLUDING sessions that promoted nothing, so quiet days still age a
        symbol out of its cooldown)."""
        return list(self._log)


class ReplayDiscovery:
    """Per-day discovery for one replay run: panel -> detectors -> gate.

    Stateful across the run (the promotion window and the cooldown history)
    and deterministic given the same config, data and gate. Constructed ONLY
    when ``discovery_mode != 'off'`` — the off path builds none of this, so
    "off reproduces today's behavior" is a structural property, not a
    promise.
    """

    def __init__(
        self,
        panel: SurveillancePanel,
        *,
        run_id: str,
        mode: str,
        promotions_per_day: int,
        promotion_days: int,
        min_dollar_volume: float = 0.0,
        min_price: float = 0.0,
        cooldown_sessions: int = 3,
        sectors: Mapping[str, str] | None = None,
        sector_etf_map: Mapping[str, str] | None = None,
        model: TriageModel | None = None,
        system_prompt: str = "",
        schema: dict[str, Any] | None = None,
        flag_sink: FlagSink | None = None,
    ) -> None:
        if mode not in DISCOVERY_MODES:
            raise ValueError(
                f"discovery mode {mode!r} is not one of {DISCOVERY_MODES}"
            )
        if mode == "haiku" and model is None:
            raise ValueError(
                "discovery_mode='haiku' needs a triage model — the composition "
                "layer injects one; nothing here resolves a client"
            )
        self.panel = panel
        self.run_id = run_id
        self.mode = mode
        self.promotions_per_day = promotions_per_day
        self.min_dollar_volume = min_dollar_volume
        self.min_price = min_price
        self.cooldown_sessions = cooldown_sessions
        self._sectors = dict(sectors or {})
        self._sector_etf_map = dict(sector_etf_map or {})
        self._model = model
        self._system_prompt = system_prompt
        self._schema = dict(schema or {})
        self._flags = flag_sink
        self.window = PromotionWindow(sessions=promotion_days)
        self.days: list[DiscoveryDay] = []

    # -- the per-day pipeline ---------------------------------------------

    def universe_for(
        self, gw: SurveillanceReads, day: date, base_universe: Sequence[str]
    ) -> list[str]:
        """``universe_D`` = the traded core at D, plus everything promoted in
        the last ``promotion_days`` sessions. The one method the runner and
        the committee driver call."""
        return list(self.run_day(gw, day, base_universe).universe)

    def run_day(
        self, gw: SurveillanceReads, day: date, base_universe: Sequence[str]
    ) -> DiscoveryDay:
        # Point-in-time membership, per day: a name whose tier-2 membership
        # starts after D must not be scanned on D even though the panel
        # (loaded over the whole window) holds its bars.
        eligible = set(gw.surveillance_universe())
        # Sector ETFs are kept regardless of membership — rel_strength needs
        # the benchmark's bars present to compute an excess return at all.
        eligible |= set(self._sector_etf_map.values())
        panel_slice = self.panel.slice(
            day,
            symbols=[s for s in self.panel.symbols if s in eligible],
            min_dollar_volume=self.min_dollar_volume,
            min_price=self.min_price,
        )
        flags = self._detect(panel_slice, day)
        result = self._gate(flags, day)

        self.window.record(day, result.considered)
        universe = tuple(sorted(set(base_universe) | set(self.window.active())))
        record = DiscoveryDay(
            as_of=day,
            scanned=len(panel_slice.per_symbol),
            filtered_out=panel_slice.filtered_out,
            flags=tuple(flags),
            promoted=result.promoted,
            universe=universe,
            llm_calls=result.llm_calls,
        )
        self.days.append(record)
        self._persist(record)
        return record

    def _detect(self, panel_slice: PanelSlice, day: date) -> list[SurveillanceFlag]:
        """The five BAR detectors over the day's slice.

        The four event detectors (``filing_event``, ``insider_cluster``,
        ``earnings_surprise``, ``news_burst``) are deliberately not run here,
        and that is a limitation to state rather than hide: ``filings``,
        ``insider_tx`` and ``earnings_events`` hold zero rows for the tier-2
        universe, and ``news`` covers 50 core symbols. Discovery is therefore
        tested at roughly 5/9 strength, and the missing four are the
        fundamental/event detectors — a weak result falsifies the
        price/volume half of the design, not the design.
        """
        if not panel_slice.per_symbol:
            return []
        return run_detectors(
            DetectorInputs(
                # Empty by design: the grouping below is what the detectors
                # read, and materializing a combined frame for ~7,600 symbols
                # would undo the panel's whole reason for existing.
                bars=pd.DataFrame(),
                # Midnight UTC of the session, never a wall clock — the flag
                # timestamp has to be a function of the replay day alone or
                # two runs of the same config produce different rows.
                as_of=datetime(day.year, day.month, day.day, tzinfo=UTC),
                per_symbol=panel_slice.per_symbol,
                sector_etf_map={
                    sym: etf
                    for sym, etf in self._sector_etf_map.items()
                    if sym in panel_slice.per_symbol and etf in panel_slice.per_symbol
                },
            )
        )

    def _gate(self, flags: Sequence[SurveillanceFlag], day: date) -> GateResult:
        if self.mode == "off":
            return off_gate(flags)
        common: dict[str, Any] = {
            "cap": self.promotions_per_day,
            "sectors": self._sectors,
            "history": self.window.history,
            "cooldown_sessions": self.cooldown_sessions,
        }
        if self.mode == "code":
            return code_gate(flags, **common)
        assert self._model is not None  # enforced in __init__
        return haiku_gate(
            flags,
            as_of=day,
            model=self._model,
            system_prompt=self._system_prompt,
            schema=self._schema,
            **common,
        )

    # -- persistence (the ONLY write path) --------------------------------

    def _persist(self, record: DiscoveryDay) -> None:
        if self._flags is None or not record.flags:
            return
        self._flags.insert_flags(flag_rows(self.run_id, record))

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Run-level discovery counters for ``replay_runs.config_json``."""
        promoted = {p.symbol for d in self.days for p in d.promoted}
        flags = sum(len(d.flags) for d in self.days)
        sessions = len(self.days) or 1
        return {
            "mode": self.mode,
            "sessions": len(self.days),
            "flags_total": flags,
            "flags_per_session": round(flags / sessions, 2),
            "promotions_total": sum(len(d.promoted) for d in self.days),
            "distinct_promoted": len(promoted),
            "llm_calls": sum(d.llm_calls for d in self.days),
            "mean_universe": round(
                sum(len(d.universe) for d in self.days) / sessions, 1
            ),
        }


def flag_rows(run_id: str, record: DiscoveryDay) -> list[dict[str, Any]]:
    """``replay_flags`` rows for one discovery day, deterministically ided.

    ``{run_id}:{as_of}:{symbol}:{detector}:{n}`` — ``n`` disambiguates the
    detectors that can emit more than one flag per symbol per day, in the
    order the detectors produced them, which is itself deterministic.
    """
    promoted = {p.symbol for p in record.promoted}
    seen: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    for flag in record.flags:
        key = (flag.symbol, flag.detector)
        n = seen.get(key, 0)
        seen[key] = n + 1
        rows.append(
            {
                "id": (
                    f"{run_id}:{record.as_of.isoformat()}:"
                    f"{flag.symbol}:{flag.detector}:{n}"
                ),
                "run_id": run_id,
                "as_of": record.as_of,
                "symbol": flag.symbol,
                "detector": flag.detector,
                "severity": float(flag.severity),
                "details": dict(flag.details),
                "promoted": flag.symbol in promoted,
            }
        )
    return rows
