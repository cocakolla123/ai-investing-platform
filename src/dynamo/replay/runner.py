"""Quant + verdict-reuse portfolio replay runner (plan Phase 4; campaign Phase 3).

Two modes share one loop:

- ``mode="quant"``: verdicts are synthesized deterministically from quant
  scores (:func:`synthesize_verdicts`, NO LLM);
- ``mode="portfolio"`` (backtest-campaign Phase 3): verdicts are REUSED from
  a completed committee run's persisted ``replay_verdicts``
  (``source_run_id``), keyed ``(as_of, symbol)`` — ZERO LLM calls, so three
  bankrolls can replay one paid committee run for free. Days without
  verdicts open no new risk; existing holdings are still managed (horizon
  exits, ladder/risk flow unchanged). Verdicts for symbols outside the
  day's universe (and not held) are ignored gracefully.

Per session day D in [start, end] (NYSE calendar):

1. fill yesterday's target at TODAY's open with the backtest cost model
   (``cost_per_side_fraction`` — reused, never forked; no same-bar fills);
2. mark to close; update equity / high-water mark / drawdown ladder;
3. read point-in-time inputs through the day's ``HistoricalDataGateway``
   (the ONLY read path);
4. quant scores: persisted historical scores where fresh, else the same
   heuristic "baseline-v0" path production uses (``score_universe`` with an
   empty ``ModelBundle`` — all-None predictions ⇒ abstention);
5. synthesize quant-only verdicts (deterministic mapping documented on
   :func:`synthesize_verdicts`; NO LLM anywhere);
6. REAL ``RiskEngine.compute_verdict`` with inputs assembled the way the
   nightly ``risk_verdict`` asset assembles them (returns panel, regime
   history from SPY + macro, ADV, ladder) — not the backtest engine's thin
   ``_session_verdict``;
7. production ``RuleBasedConstructor`` (which delegates sizing to the risk
   cascade's ``size_positions``) -> ``TargetPortfolio`` queued for the next
   open;
8. at full exit (horizon reached, avoid/sell synthesis, or window end) the
   round trip resolves into a ``DecisionExperience`` with ``source='replay'``
   and ``replay_run_id`` set, via the Phase 1 builders + counterfactuals.

Determinism: no wall clock, no randomness — every timestamp derives from the
run configuration; the same config produces byte-identical decisions and
experiences (the run id itself is a hash of the config).

Crypto (ADR-0013), v1 convention: crypto pairs flow through the same loop
when present in the universe + bars, but the calendar stays NYSE — crypto
trades AND marks on equity sessions only. Weekend bars sit in the bars frame
harmlessly (weekend rows never match a session date in ``_day_prices``); the
first session after a weekend marks positions at that session's close, so
weekend price moves surface on Monday. Sizing/caps come from the same
production constructor + cascade (crypto sleeve caps included).

FIREWALL: the runner writes ONLY through ``ReplayRepository`` (replay_runs /
replay_decisions) and ``ExperienceRepository.insert`` (source='replay').
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dynamo.backtest.costs import (
    CostParams,
    order_cost,
    order_skipped,
    whole_share_qty,
)
from dynamo.domain.models.learning import DecisionExperience
from dynamo.domain.models.portfolio import Lot, PortfolioState, Position, TargetPortfolio
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    QuantAlignment,
    QuantScore,
    Sleeve,
    WatchlistRow,
)
from dynamo.domain.models.risk import RegimeState, RiskConfig
from dynamo.learning.experience import PathInputs, TradeFacts, build_experience
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor
from dynamo.research.quant.features import build_features
from dynamo.research.quant.scoring import ModelBundle, score_universe
from dynamo.risk.engine import RiskEngine, RiskEngineInputs
from dynamo.risk.ladder import LadderState, step
from dynamo.risk.regime import RegimeHistory
from dynamo.risk.types import ReturnsPanel

BENCHMARK_SYMBOLS = ("SPY", "TLT")
RETURNS_LOOKBACK_DAYS = 750
REGIME_MACRO_LOOKBACK_DAYS = 900
SIGMA_SESSIONS = 120
_ANNUALIZE = math.sqrt(252.0)

# --- verdict synthesis thresholds (documented mapping, see synthesize_verdicts)
BUY_PERCENTILE = 0.8
AVOID_PERCENTILE = 0.2
DOWNSIDE_AVOID_PROB = 0.5
HOLD_CONVICTION = 0.5
# doc 04's downside label threshold: the quant model has no per-name downside
# magnitude, so the synthesized expectation uses the -10% label boundary.
DEFAULT_EXPECTED_DOWNSIDE_PCT = 10.0

PROMPT_VERSION = "replay-quant-synth-v1"


class ReplayConfig(BaseModel):
    """Frozen, hashable replay run configuration (determinism anchor)."""

    model_config = ConfigDict(frozen=True)

    start: date
    end: date
    mode: str = "quant"
    label: str = ""
    universe_tiers: tuple[int, ...] = (1, 3)
    initial_equity: float = Field(default=100_000.0, gt=0.0)
    horizon_sessions: int = Field(default=20, gt=0)
    quant_score_max_age_days: int = Field(default=5, ge=0)
    # Trained quant bundle (backtest-campaign Phase 2): empty = the baseline-v0
    # heuristic (all-None scores => honest abstention). A registered bundle
    # version participates in config_hash automatically (frozen pydantic model:
    # model_dump_json covers every field), so runs with and without a model can
    # never share a run id.
    quant_model_version: str = ""
    # Committee mode only (plan Phase 5): per-day deep-dive shortlist cap.
    committee_deep_dives_per_day: int = Field(default=4, gt=0)
    # Portfolio mode (backtest-campaign Phase 3): the completed committee run
    # whose replay_verdicts this run replays. Required (non-empty) when
    # mode="portfolio". In config_hash automatically, like every field.
    source_run_id: str = ""
    # Bankroll realism (backtest-campaign Phase 3) — defaults preserve prior
    # behavior bit-for-bit; all in config_hash, so three bankroll configs on
    # the same source run get three deterministic, distinct run ids.
    whole_shares: bool = False
    min_order_notional: float = Field(default=0.0, ge=0.0)
    commission_per_order: float = Field(default=0.0, ge=0.0)
    # Committee-mode context completeness (backtest-campaign Phase 4; both in
    # config_hash like every field — they change behavior AND spend):
    # generate_macro_digest: when the gateway has no macro digest for a day,
    # generate one from vintage-bounded macro series via the analyst tier
    # (costed through ReplayBudget) and inject it into that day's dossiers.
    # within_run_learning: prior-case retrieval additionally sees THIS run's
    # own resolved replay experiences (as_of-bounded, this run only) so later
    # replay months learn from earlier ones — live-parity, still firewalled.
    generate_macro_digest: bool = True
    within_run_learning: bool = True
    code_version: str = ""

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        """Deterministic: the same config always yields the same run id, so
        re-running a config is idempotent (same rows, same experience ids)."""
        return f"replay-{self.config_hash[:12]}"


# --------------------------------------------------------------------------
# ports (structural) — the gateway and repos the runner drives
# --------------------------------------------------------------------------


class ReplayGateway(Protocol):
    """The slice of ``HistoricalDataGateway`` the quant runner reads."""

    as_of: date

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]: ...
    def bars(self, symbols: Sequence[str] | None, lookback_days: int) -> pd.DataFrame: ...
    def feature_inputs(self, symbols: Sequence[str]) -> Any: ...
    def quant_scores(self, symbols: Sequence[str] | None = None) -> pd.DataFrame: ...
    def macro_series(self, lookback_days: int) -> pd.DataFrame: ...
    def sectors(self, symbols: Sequence[str] | None = None) -> dict[str, str]: ...
    def regime_state(self) -> RegimeState | None: ...


class ExperienceSink(Protocol):
    def insert(self, exp: DecisionExperience) -> bool: ...


class RunSink(Protocol):
    def create_run(
        self,
        *,
        run_id: str,
        label: str,
        mode: str,
        start_as_of: date,
        end_as_of: date,
        config: Mapping[str, Any],
        code_version: str = "",
        budget_usd: float | None = None,
        status: str = "created",
        created_at: datetime | None = None,
    ) -> None: ...
    def update_status(self, run_id: str, status: str) -> None: ...
    def insert_decisions(self, rows: Sequence[Mapping[str, Any]]) -> int: ...


# --------------------------------------------------------------------------
# pure synthesis (no LLM, unit-tested directly)
# --------------------------------------------------------------------------


def synthesize_verdicts(
    scores: Mapping[str, QuantScore], as_of: date, horizon_sessions: int
) -> dict[str, CommitteeVerdict]:
    """Deterministic quant-only verdict synthesis (NO LLM).

    Mapping (documented, deliberately simple):

    - only symbols with a non-None ``expected_excess_return`` participate;
      baseline-v0's all-None scores therefore synthesize NOTHING — honest
      abstention, exactly what a quant-only night without models produces;
    - percentile = cross-sectional rank of expected excess return among
      today's scored symbols, in (0, 1] (1 = best; ties break by symbol);
    - action: AVOID when ``downside_prob >= 0.5``; else BUY when percentile
      >= 0.8 AND expected excess return > 0; else AVOID when percentile
      <= 0.2; else HOLD;
    - conviction: BUY -> percentile; AVOID -> max(1 - percentile,
      downside_prob or 0); HOLD -> 0.5;
    - expectations: upside = expected excess return x 100 (pct), downside =
      the doc-04 label boundary (10%), horizon = the run's horizon.
    """
    scored = [
        (sym, s)
        for sym, s in sorted(scores.items())
        if s.expected_excess_return is not None
    ]
    if not scored:
        return {}
    order = sorted(scored, key=lambda t: (-(t[1].expected_excess_return or 0.0), t[0]))
    n = len(order)
    out: dict[str, CommitteeVerdict] = {}
    for idx, (sym, s) in enumerate(order):
        eer = float(s.expected_excess_return or 0.0)
        percentile = (n - idx) / n
        downside = s.downside_prob
        if downside is not None and downside >= DOWNSIDE_AVOID_PROB:
            action = Action.AVOID
            conviction = max(1.0 - percentile, float(downside))
        elif percentile >= BUY_PERCENTILE and eer > 0.0:
            action = Action.BUY
            conviction = percentile
        elif percentile <= AVOID_PERCENTILE:
            action = Action.AVOID
            conviction = 1.0 - percentile
        else:
            action = Action.HOLD
            conviction = HOLD_CONVICTION
        out[sym] = CommitteeVerdict(
            id=f"rv-{as_of.isoformat()}-{sym}",
            symbol=sym,
            as_of=as_of,
            action=action,
            conviction=round(min(max(conviction, 0.0), 1.0), 6),
            thesis=(
                f"quant-only replay synthesis: percentile {percentile:.3f}, "
                f"expected excess {eer:+.4f}"
            ),
            bull_case="",
            bear_case="",
            expected_upside_pct=max(eer, 0.0) * 100.0,
            expected_downside_pct=DEFAULT_EXPECTED_DOWNSIDE_PCT,
            horizon_days=horizon_sessions,
            quant_alignment=QuantAlignment.AGREE,
            prompt_version=PROMPT_VERSION,
            model_id="none",
        )
    return out


def synthesize_watchlist(
    verdicts: Mapping[str, CommitteeVerdict], as_of: date
) -> tuple[WatchlistRow, ...]:
    """Long-term watchlist rows for the BUY verdicts, ranked by conviction
    (ties break by symbol). ``quality_component`` mirrors the conviction
    percentile — replay has no quality scores, so the constructor runs with
    ``quality_floor=0`` and this field is informational only."""
    buys = sorted(
        (v for v in verdicts.values() if v.action is Action.BUY),
        key=lambda v: (-v.conviction, v.symbol),
    )
    return tuple(
        WatchlistRow(
            as_of=as_of,
            sleeve=Sleeve.LONG_TERM,
            symbol=v.symbol,
            rank=rank,
            blend_score=round(100.0 * v.conviction, 6),
            committee_component=v.conviction,
            quant_component=v.conviction,
            quality_component=round(100.0 * v.conviction, 6),
            badges=("replay-quant",),
            verdict_ref=v.id,
        )
        for rank, v in enumerate(buys, start=1)
    )


def scores_from_frame(
    frame: pd.DataFrame, as_of: date, max_age_days: int
) -> dict[str, QuantScore]:
    """Latest fresh persisted QuantScore per symbol from a gateway frame.

    Rows older than ``max_age_days`` before ``as_of`` are dropped (a stale
    score is worse than honest baseline abstention). Deterministic: latest
    ``as_of`` wins, ties break by ``model_version``."""
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["as_of"] = frame["as_of"].map(lambda v: pd.Timestamp(v).date())
    frame = cast(
        pd.DataFrame, frame[frame["as_of"] >= as_of - timedelta(days=max_age_days)]
    )
    if frame.empty:
        return {}
    frame = (
        frame.sort_values(["symbol", "as_of", "model_version"])
        .drop_duplicates(subset=["symbol"], keep="last")
        .reset_index(drop=True)
    )
    out: dict[str, QuantScore] = {}
    fields = set(QuantScore.model_fields)
    for row in frame.to_dict(orient="records"):
        payload = {k: v for k, v in row.items() if k in fields and not _is_na(v)}
        score = QuantScore.model_validate(payload)
        out[score.symbol] = score
    return out


def _is_na(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class QuantScoreReads(Protocol):
    """The two gateway reads :func:`quant_scores_for` needs — satisfied by
    both the quant runner's gateway slice and committee replay's."""

    def quant_scores(self, symbols: Sequence[str] | None = None) -> pd.DataFrame: ...
    def feature_inputs(self, symbols: Sequence[str]) -> Any: ...


def quant_scores_for(
    gw: QuantScoreReads,
    universe: Sequence[str],
    day: date,
    max_age_days: int,
    bundle: ModelBundle | None = None,
) -> dict[str, QuantScore]:
    """Persisted historical scores where fresh; the model bundle for the rest.

    Shared by the quant runner and committee replay (Phase 5) so the two
    modes can never drift on what "the quant view at D" means. Persisted
    rows win because they are what-was-actually-known-then; symbols without
    a fresh persisted score are scored through ``bundle`` — a TRAINED bundle
    when ``ReplayConfig.quant_model_version`` is set (Phase 2), else the
    production-parity baseline: ``score_universe`` with an empty
    ``ModelBundle`` — all-None predictions, so unscored symbols synthesize
    nothing (abstention)."""
    universe = list(universe)
    persisted = scores_from_frame(gw.quant_scores(universe), day, max_age_days)
    missing = sorted(set(universe) - set(persisted))
    if missing:
        inputs = gw.feature_inputs(universe)
        features = build_features(inputs, day, missing)
        if not features.empty:
            active = bundle if bundle is not None else ModelBundle(model_version="baseline-v0")
            for s in score_universe(active, features, day):
                persisted[s.symbol] = s
    return persisted


# --------------------------------------------------------------------------
# result records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySessionRecord:
    session: date
    equity: float
    cash: float
    regime: str
    ladder: str
    new_risk_allowed: bool
    degraded: bool
    vetoes: tuple[tuple[str, str], ...]  # (symbol, rule)


@dataclass(frozen=True)
class ReplayFill:
    session: date
    symbol: str
    side: str
    qty: float
    price: float
    cost: float


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    sessions: tuple[date, ...]
    session_records: tuple[ReplaySessionRecord, ...]
    decisions: tuple[dict[str, Any], ...]
    experiences: tuple[DecisionExperience, ...]
    fills: tuple[ReplayFill, ...]
    equity_curve: tuple[tuple[date, float], ...]
    status: str


@dataclass
class _OpenLot:
    """Entry bookkeeping for one open replay position (in-memory only)."""

    symbol: str
    entry_date: date
    entry_session_index: int
    entry_price: float
    qty: float
    verdict: CommitteeVerdict | None
    quant: QuantScore | None
    regime: RegimeState | None
    sector: str | None
    # Bankroll context (campaign Phase 3): the portfolio equity at the
    # DECISION that opened the lot (kept at first entry across adds).
    equity_at_decision: float | None = None


@dataclass
class _StagedDecisions:
    """Decisions computed on day D, persisted once D+1's fills are known."""

    as_of: date
    rows: list[dict[str, Any]] = field(default_factory=list)


def _cents(x: float) -> float:
    return round(x, 2)


def _sessions_between(start: date, end: date) -> list[date]:
    """NYSE trading sessions in [start, end] via exchange-calendars."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    return [
        cast(date, pd.Timestamp(s).date())
        for s in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    ]


def _aligned_macro(
    macro: pd.DataFrame, series_id: str, index: pd.DatetimeIndex, default: float
) -> np.ndarray:
    """Forward-filled macro series aligned to a session index (asset parity)."""
    n = len(index)
    if macro.empty:
        return np.full(n, default)
    sub = cast(pd.DataFrame, macro[macro["series_id"] == series_id])
    if len(sub) == 0:
        return np.full(n, default)
    sub = sub.sort_values(["ts", "vintage_ts"]).drop_duplicates(subset=["ts"], keep="last")
    s = pd.Series(sub["value"].to_numpy(dtype=float), index=pd.to_datetime(sub["ts"]))
    aligned = s.reindex(index, method="ffill").fillna(default)
    return np.asarray(aligned, dtype=float)


class ReplayRunner:
    """The quant-only replay loop. See the module docstring for the sequence."""

    def __init__(
        self,
        gateway_factory: Callable[[date], ReplayGateway],
        experience_repo: ExperienceSink,
        run_repo: RunSink,
        config: ReplayConfig,
        *,
        risk_config: RiskConfig | None = None,
        construction_config: ConstructionConfig | None = None,
        run_config_extras: Mapping[str, Any] | None = None,
        bundle_factory: Callable[[str], ModelBundle] | None = None,
        verdict_source: Callable[[str], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        if config.mode not in {"quant", "portfolio"}:
            raise ValueError(
                f"ReplayRunner runs quant/portfolio modes; mode {config.mode!r} "
                "runs through dynamo.replay.committee.CommitteeReplay"
            )
        if config.mode == "portfolio":
            if not config.source_run_id:
                raise ValueError(
                    "portfolio mode requires a non-empty source_run_id "
                    "(a completed committee run whose verdicts to replay)"
                )
            if verdict_source is None:
                raise ValueError(
                    "portfolio mode requires a verdict_source (the composition "
                    "layer passes ReplayRepository.verdicts_for)"
                )
        self._gateway_factory = gateway_factory
        self._experiences = experience_repo
        self._runs = run_repo
        self.config = config
        # Run-metadata extras (e.g. the preflight report) recorded into
        # replay_runs.config_json alongside the config — deliberately NOT part
        # of ReplayConfig, so they never perturb the deterministic run id.
        self._run_config_extras = dict(run_config_extras or {})
        self._risk_config = risk_config or RiskConfig()
        # Replay construction defaults: quality_floor=0 (replay has no quality
        # scores; the synthesized watchlist carries an informational
        # quality_component instead) and long_term_top_fraction=1.0 (the
        # synthesized watchlist already contains ONLY the BUY verdicts — the
        # top-fraction cut is redundant and would let a vetoed name consume
        # the candidate slots of eligible ones).
        self._construction_config = construction_config or ConstructionConfig(
            quality_floor=0.0, long_term_top_fraction=1.0
        )
        self._engine = RiskEngine()
        self._constructor = RuleBasedConstructor()
        # Trained-bundle seam (backtest-campaign Phase 2): the composition
        # layer (cli/replay.py, tests) may inject a factory; the default is
        # the artifact loader. Resolved ONCE at run start, never per session.
        self._bundle_factory = bundle_factory
        self._bundle: ModelBundle | None = None
        # Verdict-reuse seam (backtest-campaign Phase 3): portfolio mode
        # preloads the source committee run's persisted verdicts ONCE at run
        # start, keyed (as_of, symbol). Injected like the bundle factory so
        # tests drive it with fixtures and the CLI passes the repository read.
        self._verdict_source = verdict_source
        self._source_verdicts: dict[date, dict[str, CommitteeVerdict]] = {}
        # Realism cost knobs, shared helper semantics with the backtest engine
        # (see backtest.costs) — defaults reproduce prior behavior exactly.
        self._cost_params = CostParams(
            commission_per_order=config.commission_per_order,
            min_order_notional=config.min_order_notional,
        )
        # Entry context for fills: the synthesized verdict/quant recorded when
        # the target was queued (fills happen the NEXT session, before the
        # next synthesis overwrites these).
        self._entry_verdicts: dict[str, CommitteeVerdict] = {}
        self._entry_quant: dict[str, QuantScore] = {}

    # ------------------------------------------------------------------ run
    def _resolve_bundle(self) -> ModelBundle | None:
        """Load the trained bundle ONCE at run start (loud on any failure:
        unknown version, sha mismatch, feature drift). Empty version ⇒ None
        ⇒ :func:`quant_scores_for` falls back to baseline-v0 abstention."""
        version = self.config.quant_model_version
        if not version:
            return None
        factory = self._bundle_factory
        if factory is None:
            from dynamo.research.quant.bundle import bundle_from_version

            factory = bundle_from_version
        return factory(version)

    def _resolve_source_verdicts(self) -> dict[date, dict[str, CommitteeVerdict]]:
        """Portfolio mode: preload the source run's replay_verdicts ONCE,
        keyed as_of -> symbol -> CommitteeVerdict (parsed from the persisted
        payload). Loud on unparsable payloads — a silently dropped verdict
        would make the portfolio replay dishonest."""
        if self.config.mode != "portfolio":
            return {}
        source = self._verdict_source
        assert source is not None  # enforced in __init__
        out: dict[date, dict[str, CommitteeVerdict]] = {}
        fields = set(CommitteeVerdict.model_fields)
        for row in source(self.config.source_run_id):
            # The persisted payload carries the raw per-stage payloads under
            # "stages" (Phase 4 audit trail) beside the verdict fields —
            # filter to the model's own fields (extra="forbid").
            payload = {k: v for k, v in dict(row["payload"]).items() if k in fields}
            verdict = CommitteeVerdict.model_validate(payload)
            as_of = row["as_of"]
            if isinstance(as_of, str):
                as_of = date.fromisoformat(as_of)
            out.setdefault(as_of, {})[str(row["symbol"])] = verdict
        return out

    def run(self, progress: Callable[[str], None] | None = None) -> ReplayResult:
        cfg = self.config
        run_id = cfg.run_id
        # Before create_run: fail loud, write nothing.
        self._bundle = self._resolve_bundle()
        self._source_verdicts = self._resolve_source_verdicts()
        sessions = _sessions_between(cfg.start, cfg.end)
        self._runs.create_run(
            run_id=run_id,
            label=cfg.label,
            mode=cfg.mode,
            start_as_of=cfg.start,
            end_as_of=cfg.end,
            config={**cfg.model_dump(mode="json"), **self._run_config_extras},
            code_version=cfg.code_version,
            budget_usd=None,
            status="running",
        )
        try:
            result = self._loop(run_id, sessions, progress)
        except Exception:
            self._runs.update_status(run_id, "failed")
            raise
        self._runs.update_status(run_id, "completed")
        return result

    # ----------------------------------------------------------------- loop
    def _loop(
        self,
        run_id: str,
        sessions: list[date],
        progress: Callable[[str], None] | None,
    ) -> ReplayResult:
        cfg = self.config
        cash = _cents(cfg.initial_equity)
        shares: dict[str, float] = {}
        avg_cost: dict[str, float] = {}
        open_lots: dict[str, _OpenLot] = {}
        last_close: dict[str, float] = {}
        ladder = LadderState()
        hwm = cfg.initial_equity
        turnover_mtd = 0.0
        current_month: tuple[int, int] | None = None

        pending_target: TargetPortfolio | None = None
        pending_equity = cfg.initial_equity
        pending_closes: dict[str, float] = {}
        pending_sigma: dict[str, float] = {}
        pending_adv: dict[str, float] = {}
        staged: _StagedDecisions | None = None

        all_decisions: list[dict[str, Any]] = []
        experiences: list[DecisionExperience] = []
        fills: list[ReplayFill] = []
        session_records: list[ReplaySessionRecord] = []
        equity_curve: list[tuple[date, float]] = []

        for i, day in enumerate(sessions):
            if current_month != (day.year, day.month):
                current_month = (day.year, day.month)
                turnover_mtd = 0.0

            gw = self._gateway_factory(day)
            universe = gw.universe(cfg.universe_tiers)
            panel_symbols = sorted(
                {*universe, *shares, *BENCHMARK_SYMBOLS}
            )
            bars = gw.bars(panel_symbols, RETURNS_LOOKBACK_DAYS)
            opens_today, closes_today = _day_prices(bars, day)
            for sym, px in closes_today.items():
                last_close[sym] = px

            # 1. Fill yesterday's target at TODAY's open (never same-bar).
            if pending_target is not None:
                cash, day_fills, traded = self._execute(
                    pending_target,
                    pending_equity,
                    pending_closes,
                    opens_today,
                    day,
                    cash,
                    shares,
                    avg_cost,
                    pending_sigma,
                    pending_adv,
                )
                fills.extend(day_fills)
                turnover_mtd += traded / pending_equity / 2.0 if pending_equity > 0 else 0.0
                # Entry bookkeeping + resolution of full exits.
                # pending_equity is still YESTERDAY's decision equity here
                # (overwritten in step 8) — the bankroll context for the lot.
                self._track_lots(
                    day_fills, day, i, shares, avg_cost, open_lots, gw, bars,
                    experiences, run_id, pending_equity,
                )
                if staged is not None:
                    fill_price = {f.symbol: f.price for f in day_fills}
                    for row in staged.rows:
                        row["fill_price"] = fill_price.get(row["symbol"])
                    self._runs.insert_decisions(staged.rows)
                    all_decisions.extend(staged.rows)
                    staged = None
                pending_target = None

            # 2. Mark to close.
            position_value = 0.0
            for sym in sorted(shares):
                px = last_close.get(sym)
                if px is not None and math.isfinite(px) and shares[sym] != 0.0:
                    position_value += _cents(shares[sym] * px)
            equity = _cents(cash + position_value)
            hwm = max(hwm, equity)
            equity_curve.append((day, equity))

            # 3. Real RiskEngine verdict on point-in-time inputs.
            sigma_annual, adv = _sigma_and_adv(bars)
            sectors = gw.sectors(panel_symbols)
            state = _portfolio_state(day, equity, cash, shares, avg_cost, hwm, open_lots)
            prices = {s: last_close[s] for s in sorted(shares) if s in last_close}
            panel = _returns_panel(bars)
            regime_history = _regime_history(bars, gw.macro_series(REGIME_MACRO_LOOKBACK_DAYS))
            verdict = self._engine.compute_verdict(
                RiskEngineInputs(
                    portfolio=state,
                    returns=panel,
                    prices=prices,
                    sector_by_symbol=sectors,
                    adv_dollar_20d=adv,
                    regime_history=regime_history,
                    prev_ladder=ladder,
                    config=self._risk_config,
                    as_of=datetime(day.year, day.month, day.day, tzinfo=UTC),
                    session_index=i,
                )
            )
            ladder = step(ladder, equity, hwm, i)

            # 4./5. Verdicts: synthesized from quant scores (quant mode) or
            # reused from the source committee run (portfolio mode). Either
            # way, quant scores flow to the constructor (production parity).
            quant_scores = self._quant_scores(gw, universe, day)
            if cfg.mode == "portfolio":
                # Days without persisted verdicts open no new risk; verdicts
                # for symbols outside today's universe (and not held) are
                # ignored gracefully.
                tradable = {*universe, *(s for s in shares if shares[s])}
                synth = {
                    sym: v
                    for sym, v in self._source_verdicts.get(day, {}).items()
                    if sym in tradable
                }
            else:
                synth = synthesize_verdicts(quant_scores, day, cfg.horizon_sessions)
            synth = self._with_horizon_exits(synth, open_lots, day, i)
            watchlist = synthesize_watchlist(synth, day)

            # 6./7. Production construction (sizing delegated to the cascade).
            holdings = {
                s: _cents(shares[s] * last_close.get(s, 0.0)) / equity
                for s in sorted(shares)
                if shares[s] != 0.0 and equity > 0
            }
            target = self._constructor.construct(
                AdvisoryInputs(
                    as_of=datetime(day.year, day.month, day.day, tzinfo=UTC),
                    watchlist=watchlist,
                    verdicts=synth,
                    quant_scores=quant_scores,
                    sigma_annual=sigma_annual,
                    adv_dollar_20d=adv,
                    sector_by_symbol=sectors,
                ),
                verdict,
                state,
                self._construction_config,
                risk_config=self._risk_config,
                current_weights=holdings,
                turnover_used_mtd=turnover_mtd,
            )

            staged = _StagedDecisions(
                as_of=day,
                rows=_decision_rows(run_id, day, synth, target),
            )
            session_records.append(
                ReplaySessionRecord(
                    session=day,
                    equity=equity,
                    cash=cash,
                    regime=verdict.regime.regime.value,
                    ladder=verdict.ladder.value,
                    new_risk_allowed=verdict.new_risk_allowed,
                    degraded=verdict.degraded,
                    vetoes=tuple((v.symbol, v.rule) for v in verdict.vetoes),
                )
            )
            if progress is not None:
                progress(
                    f"{day.isoformat()} equity={equity:,.2f} "
                    f"verdicts={len(synth)} targets={len(target.weights)} "
                    f"regime={verdict.regime.regime.value}"
                )

            # 8. Queue for the next open (entry context recorded for lots).
            if i < len(sessions) - 1:
                pending_target = target
                pending_equity = equity
                pending_closes = dict(last_close)
                pending_sigma = sigma_annual
                pending_adv = adv
                self._record_entry_context(synth, quant_scores)

        # Window end: persist the final (unfilled) decisions and resolve the
        # remaining open lots at the last close (synthetic exit, no costs).
        if staged is not None:
            self._runs.insert_decisions(staged.rows)
            all_decisions.extend(staged.rows)
        if sessions:
            final_day = sessions[-1]
            final_gw = self._gateway_factory(final_day)
            final_bars = final_gw.bars(
                sorted({*open_lots, *BENCHMARK_SYMBOLS}), RETURNS_LOOKBACK_DAYS
            )
            for sym in sorted(open_lots):
                lot = open_lots[sym]
                exit_price = last_close.get(sym)
                if exit_price is None or exit_price <= 0:
                    continue
                experiences.append(
                    self._resolve(lot, final_day, exit_price, final_bars, run_id)
                )
        return ReplayResult(
            run_id=run_id,
            sessions=tuple(sessions),
            session_records=tuple(session_records),
            decisions=tuple(all_decisions),
            experiences=tuple(experiences),
            fills=tuple(fills),
            equity_curve=tuple(equity_curve),
            status="completed",
        )

    # ----------------------------------------------------------- scores/exits
    def _quant_scores(
        self, gw: ReplayGateway, universe: list[str], day: date
    ) -> dict[str, QuantScore]:
        """See :func:`quant_scores_for` (shared with committee replay)."""
        return quant_scores_for(
            gw, universe, day, self.config.quant_score_max_age_days, bundle=self._bundle
        )

    def _with_horizon_exits(
        self,
        synth: dict[str, CommitteeVerdict],
        open_lots: Mapping[str, _OpenLot],
        day: date,
        session_index: int,
    ) -> dict[str, CommitteeVerdict]:
        """Quant-only replay holds to the verdict horizon then exits: lots at
        or past ``horizon_sessions`` get a synthesized SELL verdict, which the
        production constructor treats as a mandatory exit."""
        out = dict(synth)
        for sym in sorted(open_lots):
            lot = open_lots[sym]
            if session_index - lot.entry_session_index >= self.config.horizon_sessions:
                out[sym] = CommitteeVerdict(
                    id=f"rv-exit-{day.isoformat()}-{sym}",
                    symbol=sym,
                    as_of=day,
                    action=Action.SELL,
                    conviction=1.0,
                    thesis="replay horizon reached: mandatory exit",
                    bull_case="",
                    bear_case="",
                    horizon_days=self.config.horizon_sessions,
                    quant_alignment=QuantAlignment.NEUTRAL,
                    prompt_version=PROMPT_VERSION,
                    model_id="none",
                )
        return out

    # ---------------------------------------------------------------- fills
    def _execute(
        self,
        target: TargetPortfolio,
        decision_equity: float,
        decision_closes: Mapping[str, float],
        opens_today: Mapping[str, float],
        day: date,
        cash: float,
        shares: dict[str, float],
        avg_cost: dict[str, float],
        sigma_annual: Mapping[str, float],
        adv: Mapping[str, float],
    ) -> tuple[float, list[ReplayFill], float]:
        """Backtest-parity fills: sells first, next-open prices, cost model
        reused from ``backtest.costs`` (never forked)."""
        target_by_symbol: dict[str, float] = {}
        for w in target.weights:
            target_by_symbol[w.symbol] = target_by_symbol.get(w.symbol, 0.0) + w.weight
        desired: dict[str, float] = {}
        for sym in sorted(set(target_by_symbol) | {s for s in shares if shares[s]}):
            weight = target_by_symbol.get(sym, 0.0)
            close = decision_closes.get(sym, float("nan"))
            if weight > 0.0 and math.isfinite(close) and close > 0.0:
                desired[sym] = whole_share_qty(
                    sym,
                    _cents(weight * decision_equity) / close,
                    self.config.whole_shares,
                )
            else:
                desired[sym] = 0.0

        day_fills: list[ReplayFill] = []
        traded_value = 0.0
        deltas = sorted(
            ((s, q - shares.get(s, 0.0)) for s, q in desired.items()),
            key=lambda t: (t[1] >= 0.0, t[0]),  # sells first, then symbol asc
        )
        for sym, delta in deltas:
            if abs(delta) < 1e-12:
                continue
            open_px = opens_today.get(sym, float("nan"))
            if not (math.isfinite(open_px) and open_px > 0.0):
                continue  # no fill possible; position unchanged
            qty = abs(delta)
            value = _cents(qty * open_px)
            if value < 0.01:
                continue
            if order_skipped(value, self._cost_params):
                continue  # below the minimum order notional — never filled
            sigma = sigma_annual.get(sym, float("nan"))
            sigma_daily = sigma / _ANNUALIZE if math.isfinite(sigma) else float("nan")
            cost = _cents(
                order_cost(
                    value, sigma_daily, adv.get(sym, float("nan")), self._cost_params
                )
            )
            if delta > 0.0:
                cash = _cents(cash - value - cost)
                prev_qty = shares.get(sym, 0.0)
                prev_cost = avg_cost.get(sym, 0.0)
                new_qty = prev_qty + qty
                avg_cost[sym] = (
                    (prev_qty * prev_cost + value + cost) / new_qty if new_qty else 0.0
                )
                shares[sym] = new_qty
                side = "buy"
            else:
                cash = _cents(cash + value - cost)
                shares[sym] = shares.get(sym, 0.0) - qty
                if abs(shares[sym]) < 1e-9:
                    shares.pop(sym, None)
                    avg_cost.pop(sym, None)
                side = "sell"
            traded_value += value
            day_fills.append(
                ReplayFill(
                    session=day, symbol=sym, side=side, qty=qty, price=open_px, cost=cost
                )
            )
        return cash, day_fills, traded_value

    def _track_lots(
        self,
        day_fills: Sequence[ReplayFill],
        day: date,
        session_index: int,
        shares: Mapping[str, float],
        avg_cost: Mapping[str, float],
        open_lots: dict[str, _OpenLot],
        gw: ReplayGateway,
        bars: pd.DataFrame,
        experiences: list[DecisionExperience],
        run_id: str,
        decision_equity: float | None = None,
    ) -> None:
        """Open lots on buys; resolve experiences on full exits."""
        for f in day_fills:
            if f.side == "buy":
                if f.symbol not in open_lots:
                    open_lots[f.symbol] = _OpenLot(
                        symbol=f.symbol,
                        entry_date=day,
                        entry_session_index=session_index,
                        entry_price=f.price,
                        qty=f.qty,
                        verdict=self._entry_verdict_for(f.symbol),
                        quant=self._entry_quant_for(f.symbol),
                        regime=gw.regime_state(),
                        sector=gw.sectors([f.symbol]).get(f.symbol),
                        equity_at_decision=decision_equity,
                    )
                else:
                    lot = open_lots[f.symbol]
                    lot.qty += f.qty
                    lot.entry_price = avg_cost.get(f.symbol, lot.entry_price)
            elif f.side == "sell" and shares.get(f.symbol, 0.0) == 0.0:
                lot = open_lots.pop(f.symbol, None)
                if lot is not None:
                    experiences.append(self._resolve(lot, day, f.price, bars, run_id))

    def _entry_verdict_for(self, symbol: str) -> CommitteeVerdict | None:
        return self._entry_verdicts.get(symbol)

    def _entry_quant_for(self, symbol: str) -> QuantScore | None:
        return self._entry_quant.get(symbol)

    def _resolve(
        self,
        lot: _OpenLot,
        exit_date: date,
        exit_price: float,
        bars: pd.DataFrame,
        run_id: str,
    ) -> DecisionExperience:
        """One round trip -> a resolved DecisionExperience (source='replay')."""
        path = _path_inputs(bars, lot.symbol, lot.entry_date, exit_date)
        pnl_pct = (
            (exit_price / lot.entry_price - 1.0) * 100.0 if lot.entry_price > 0 else 0.0
        )
        trade = TradeFacts(
            trade_id=f"{lot.symbol}-{lot.entry_date.isoformat()}",
            symbol=lot.symbol,
            qty=lot.qty,
            opened_at=datetime(
                lot.entry_date.year, lot.entry_date.month, lot.entry_date.day, tzinfo=UTC
            ),
            closed_at=datetime(exit_date.year, exit_date.month, exit_date.day, tzinfo=UTC),
            entry_price=lot.entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            holding_days=(exit_date - lot.entry_date).days,
        )
        exp = build_experience(
            trade=trade,
            # Deterministic: derived from the run window, never wall clock.
            now=datetime(exit_date.year, exit_date.month, exit_date.day, tzinfo=UTC),
            verdict=lot.verdict,
            quant=lot.quant,
            regime=lot.regime,
            path=path,
            sector=lot.sector,
            source="replay",
            replay_run_id=run_id,
            # Bankroll context (campaign Phase 3): equity at the opening
            # decision + the realized position notional (avg entry x qty).
            equity_at_decision=lot.equity_at_decision,
            position_notional=(
                _cents(lot.qty * lot.entry_price) if lot.entry_price > 0 else None
            ),
        )
        self._experiences.insert(exp)
        return exp

    # ------------------------------------------------------------ decisions
    def _record_entry_context(
        self,
        synth: Mapping[str, CommitteeVerdict],
        quant_scores: Mapping[str, QuantScore],
    ) -> None:
        self._entry_verdicts = dict(synth)
        self._entry_quant = dict(quant_scores)


def _decision_rows(
    run_id: str,
    day: date,
    synth: Mapping[str, CommitteeVerdict],
    target: TargetPortfolio,
) -> list[dict[str, Any]]:
    """One replay_decisions row per (day, symbol): synthesized verdicts plus
    verdict-less retained holdings that still carry a target weight."""
    target_by_symbol: dict[str, float] = {}
    for w in target.weights:
        target_by_symbol[w.symbol] = target_by_symbol.get(w.symbol, 0.0) + w.weight
    rows: list[dict[str, Any]] = []
    for sym in sorted(set(synth) | set(target_by_symbol)):
        v = synth.get(sym)
        rows.append(
            {
                "id": f"{run_id}:{day.isoformat()}:{sym}",
                "run_id": run_id,
                "as_of": day,
                "symbol": sym,
                "action": v.action.value if v is not None else "hold",
                "conviction": v.conviction if v is not None else None,
                "target_weight": target_by_symbol.get(sym, 0.0),
                "fill_price": None,
                "payload": {
                    "verdict": v.model_dump(mode="json") if v is not None else None,
                },
            }
        )
    return rows


# --------------------------------------------------------------------------
# pure frame helpers
# --------------------------------------------------------------------------


def _day_prices(bars: pd.DataFrame, day: date) -> tuple[dict[str, float], dict[str, float]]:
    """(opens, closes) for one session from a canonical bars frame."""
    if bars.empty:
        return {}, {}
    ts = pd.to_datetime(bars["ts"])
    mask = ts.dt.date == day
    sub = cast(pd.DataFrame, bars[mask])
    opens = {
        str(r["symbol"]): float(r["open"]) for r in sub.to_dict(orient="records")
    }
    closes = {
        str(r["symbol"]): float(r["close"]) for r in sub.to_dict(orient="records")
    }
    return opens, closes


def _sigma_and_adv(bars: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Annualized close-to-close sigma (last ~120 sessions) + 20d median
    dollar volume per symbol (asset parity with portfolio_exec)."""
    sigma: dict[str, float] = {}
    adv: dict[str, float] = {}
    if bars.empty:
        return sigma, adv
    for sym, grp in bars.groupby("symbol"):
        grp = grp.sort_values("ts")
        closes = grp["close"].astype(float).tail(SIGMA_SESSIONS + 1)
        if len(closes) > 21:
            sigma[str(sym)] = float(closes.pct_change().std() * _ANNUALIZE)
        adv[str(sym)] = float(
            (grp["close"].astype(float) * grp["volume"].astype(float)).tail(20).median()
        )
    return sigma, adv


def _returns_panel(bars: pd.DataFrame) -> ReturnsPanel:
    """Daily simple-return panel from canonical bars (asset parity). Columns
    with fewer than 2 finite returns are dropped so sparse universe names
    cannot degrade the engine's correlation/VaR math."""
    if bars.empty:
        return ReturnsPanel(dates=(), symbols=(), values=np.zeros((0, 0)))
    wide = bars.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    returns = wide.pct_change().dropna(how="all").fillna(0.0)
    return ReturnsPanel(
        dates=tuple(cast(date, pd.Timestamp(t).date()) for t in returns.index),
        symbols=tuple(str(c) for c in returns.columns),
        values=returns.to_numpy(dtype=np.float64),
    )


def _regime_history(bars: pd.DataFrame, macro: pd.DataFrame) -> RegimeHistory:
    """SPY + macro regime inputs, mirroring the nightly ``regime_state`` /
    ``risk_verdict`` assets (breadth proxy from the 200d MA)."""
    if bars.empty:
        return RegimeHistory(
            dates=(),
            spy_close=np.zeros(0),
            hy_oas=np.zeros(0),
            curve_slope=np.zeros(0),
            breadth=np.zeros(0),
        )
    spy = (
        cast(pd.DataFrame, bars[bars["symbol"] == "SPY"])
        .sort_values("ts")
        .set_index("ts")["close"]
        .dropna()
    )
    if spy.empty:
        return RegimeHistory(
            dates=(),
            spy_close=np.zeros(0),
            hy_oas=np.zeros(0),
            curve_slope=np.zeros(0),
            breadth=np.zeros(0),
        )
    idx = pd.to_datetime(spy.index)
    close = spy.to_numpy(dtype=float)
    ma200 = np.asarray(pd.Series(close).rolling(200, min_periods=1).mean(), dtype=float)
    breadth = np.clip((close / ma200 - 1.0) * 5.0 + 0.5, 0.0, 1.0)
    return RegimeHistory(
        dates=tuple(cast(date, pd.Timestamp(t).date()) for t in idx),
        spy_close=close,
        hy_oas=_aligned_macro(macro, "BAMLH0A0HYM2", idx, 4.0),
        curve_slope=_aligned_macro(macro, "T10Y3M", idx, 1.0),
        breadth=breadth,
    )


def _portfolio_state(
    day: date,
    equity: float,
    cash: float,
    shares: Mapping[str, float],
    avg_cost: Mapping[str, float],
    hwm: float,
    open_lots: Mapping[str, _OpenLot],
) -> PortfolioState:
    positions = tuple(
        Position(
            symbol=s,
            qty=shares[s],
            avg_entry_price=avg_cost.get(s, 0.0),
            lots=(
                Lot(
                    symbol=s,
                    sleeve=Sleeve.LONG_TERM,
                    qty=shares[s],
                    entry_price=avg_cost.get(s, 0.0),
                    entry_date=(
                        open_lots[s].entry_date if s in open_lots else day
                    ),
                ),
            ),
        )
        for s in sorted(shares)
        if shares[s] != 0.0
    )
    return PortfolioState(
        as_of=datetime(day.year, day.month, day.day, tzinfo=UTC),
        equity=equity,
        cash=cash,
        positions=positions,
        high_water_mark=hwm,
    )


def _path_inputs(
    bars: pd.DataFrame, symbol: str, entry_date: date, exit_date: date
) -> PathInputs | None:
    """Session-ordered OHLC path from entry to exit (index 0 = entry) plus
    the SPY benchmark closes over the same window."""
    if bars.empty:
        return None
    ts = pd.to_datetime(bars["ts"]).dt.date
    window = cast(pd.DataFrame, bars[(ts >= entry_date) & (ts <= exit_date)])
    sym_bars = cast(pd.DataFrame, window[window["symbol"] == symbol]).sort_values("ts")
    if sym_bars.empty:
        return None
    spy = cast(pd.DataFrame, window[window["symbol"] == "SPY"]).sort_values("ts")
    return PathInputs(
        opens=tuple(float(x) for x in sym_bars["open"]),
        highs=tuple(float(x) for x in sym_bars["high"]),
        lows=tuple(float(x) for x in sym_bars["low"]),
        closes=tuple(float(x) for x in sym_bars["close"]),
        exit_session_index=len(sym_bars) - 1,
        benchmark_closes=tuple(float(x) for x in spy["close"]),
    )
