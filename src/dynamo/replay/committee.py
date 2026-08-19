"""Committee replay — the ONLY token-spending replay mode (plan Phase 5).

Per session day D in [start, end]:

1. shortlist: point-in-time quant scores through the day's gateway ->
   ``shortlist_candidates`` (the ranked candidate band) ->
   ``select_shortlist`` (sector-bucketed round-robin, crypto slot cap,
   cooldown on recently debated names, one rotation slot) ->
   ``committee_deep_dives_per_day`` symbols. Days without candidates run
   zero deep dives (honest abstention, $0);
2. historical dossier context assembled STRICTLY via the gateway: instrument
   profile, quant score, price stats, as-of news (if present), as-of macro
   digest, as-of journal hits + prior-case evidence — all bounded at D.
   Missing sources render as absent sections exactly like live thin dossiers
   (no filings/transcripts in replay — accepted plan caveat);
3. the REAL 4-stage committee (dossier -> bull & bear -> arbiter) through the
   injected ``ResearchModel``, with the production ``CommitteeEngine``
   (prompts, schemas, guardrails — nothing forked);
4. verdicts persist to ``replay_verdicts`` ONLY — never committee_verdicts
   (firewall extension); entry verdicts resolve into ``DecisionExperience``
   rows (source='replay') at next-open entry / horizon-or-window-end close,
   via the Phase-1 builders, when the outcome lands inside the run window.

PRIMARY CALL PATH (documented decision): the Message Batches API, submitted
and collected WITHIN ONE PROCESS per stage-batch (a sleep/backoff poll loop
on ``batch_status`` — no Dagster sensor). This keeps the 50% batch discount
and 1h-TTL prompt caching AND sidesteps the client's process-local
``_batch_schemas`` restart hazard: results are always collected by the same
process (and the same client instance) that submitted them. The sync
``complete_json`` path is deliberately NOT used — one path, no drift.

BUDGET: before every stage submission the projected cost (conservative
overestimate, see ``replay.budget``) is checked against the per-run cap;
``ReplayBudgetExhausted`` marks the run ``budget_exhausted`` and keeps every
already-persisted verdict/experience (partial results are results). Actual
spend is recorded from the client's own metered usage into
``replay_runs.spent_usd`` as it happens, so a restarted run resumes with
accurate accounting. The live calendar-month CostMeter (REAL wall clock)
stays wired inside the client as the outer bound.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pandas as pd
import structlog

from dynamo.domain.models.learning import DecisionExperience
from dynamo.domain.models.market import Instrument, NewsItem
from dynamo.domain.models.research import (
    Action,
    AdvocateBrief,
    CommitteeVerdict,
    QuantScore,
    Side,
)
from dynamo.domain.models.risk import RegimeState
from dynamo.domain.ports import ResearchModel
from dynamo.learning.experience import TradeFacts, build_experience
from dynamo.replay.budget import ReplayBudget, estimate_batch_cost
from dynamo.replay.runner import (
    ExperienceSink,
    ReplayConfig,
    _day_prices,
    _path_inputs,
    _sessions_between,
    quant_scores_for,
    shortlist_candidates,
)
from dynamo.replay.shortlist import SessionHistory, select_shortlist
from dynamo.research.committee import (
    TEMPORAL_DISCIPLINE,
    CommitteeEngine,
    advocate_context_for,
    arbiter_context_for,
    custom_id_for,
    split_custom_id,
)
from dynamo.research.dossier import assemble_dossier_context, transcript_excerpt
from dynamo.research.fundamentals import build_fundamentals
from dynamo.research.market_stats import peer_context, peer_symbols, price_stats
from dynamo.research.news_digest import partition_news
from dynamo.research.quant.scoring import ModelBundle

log = structlog.get_logger(__name__)


ENTRY_ACTIONS = frozenset({Action.BUY, Action.ADD})
# 250-session returns, the 52-week range and a 120-session beta need more than
# a 120-day window (live build_deep_dive_contexts uses the same constant).
PRICE_STATS_LOOKBACK_DAYS = 420
NEWS_LOOKBACK_DAYS = 7  # live parity
NEWS_MAX_ITEMS = 10  # live parity
JOURNAL_TOP_K = 3  # live experience_retrieval_top_k default
RESOLUTION_BARS_LOOKBACK_DAYS = 750

# Poll cadence for in-process batch collection: start fast, back off, never
# exceed the ceiling; give up after the API's own 24h batch expiry window.
POLL_INITIAL_S = 15.0
POLL_BACKOFF = 1.5
POLL_MAX_S = 300.0
POLL_TIMEOUT_S = 24.0 * 3600.0

# Evidence-line format for prior similar cases — byte-parity with the live
# pipeline's line (orchestration/assets/research.py) so replay dossiers read
# exactly like thin live dossiers.
PRIOR_CASE_PREFIX = "Prior similar case (evidence, not rule): "

# Model knowledge gate (backtest-campaign Phase 4): default safety margin
# added on top of each model's documented training-data cutoff. The CLI
# overrides it from Settings.replay_cutoff_margin_days.
DEFAULT_CUTOFF_MARGIN_DAYS = 30

# Replay macro digest generation (live build_macro_digest / _macro_context
# parity): same series lookback, same max_tokens, same instruction template.
MACRO_DIGEST_STAGE = "macro_digest"
MACRO_DIGEST_OUTPUT_STAGE = "macro_digest_output"
MACRO_DIGEST_SYMBOL = "MACRO"
MACRO_SERIES_LOOKBACK_DAYS = 120
MACRO_DIGEST_MAX_TOKENS = 3072


class ModelKnowledgeError(RuntimeError):
    """The replay window starts inside (or too close to) a committee model's
    documented training-data window — refuse to spend unless the caller
    explicitly acknowledges (``--acknowledge-model-knowledge``)."""


def model_knowledge_gate(
    start: date,
    tier_models: Mapping[str, str],
    margin_days: int = DEFAULT_CUTOFF_MARGIN_DAYS,
) -> dict[str, Any]:
    """Resolve every tier model's documented training-data cutoff and decide
    whether the window is tainted: BLOCKED when
    ``start <= cutoff + margin_days`` for any model. Unknown models raise
    :class:`dynamo.llm.pricing.UnknownModelError` loudly — a model without a
    documented cutoff must never gate silently."""
    from dynamo.llm.pricing import spec_for

    cutoffs: dict[str, str] = {}
    blocking: list[str] = []
    for _tier, model_id in sorted(tier_models.items()):
        spec = spec_for(model_id)
        cutoffs[model_id] = spec.knowledge_cutoff.isoformat()
        if start <= spec.knowledge_cutoff + timedelta(days=margin_days):
            blocking.append(model_id)
    return {
        "window_start": start.isoformat(),
        "margin_days": int(margin_days),
        "cutoffs": cutoffs,
        "blocked": bool(blocking),
        "blocking_models": sorted(set(blocking)),
    }


class ReplayCommitteeGateway(Protocol):
    """The slice of ``HistoricalDataGateway`` committee replay reads."""

    as_of: date

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]: ...
    def bars(self, symbols: Sequence[str] | None, lookback_days: int) -> pd.DataFrame: ...
    def feature_inputs(self, symbols: Sequence[str]) -> Any: ...
    # Optional (duck-typed): the dossier's longer-window fundamentals read.
    # ``_fundamentals_frame`` falls back to ``feature_inputs`` without it.
    def quant_scores(self, symbols: Sequence[str] | None = None) -> pd.DataFrame: ...
    def macro_digest(self) -> dict[str, Any] | None: ...
    def macro_series(self, lookback_days: int) -> pd.DataFrame: ...
    def transcripts(self, symbols: Sequence[str]) -> pd.DataFrame: ...
    def instruments(self, symbols: Sequence[str]) -> dict[str, Instrument]: ...
    def news(
        self, symbols: Sequence[str], lookback_days: int = 7
    ) -> dict[str, list[NewsItem]]: ...
    def journal_search(
        self,
        *,
        embedding: Sequence[float] | None = None,
        text_query: str | None = None,
        kinds: Sequence[Any] = (),
        symbols: Sequence[str] = (),
        top_k: int = 5,
    ) -> list[tuple[Any, float]]: ...
    def similar_experiences(
        self,
        *,
        symbol: str,
        sector: str | None = None,
        regime: str | None = None,
        sleeve: str | None = None,
        limit: int = 3,
        include_replay_run_id: str | None = None,
    ) -> list[DecisionExperience]: ...
    def sectors(self, symbols: Sequence[str] | None = None) -> dict[str, str]: ...
    def regime_state(self) -> RegimeState | None: ...


class CommitteeRunSink(Protocol):
    """The slice of ``ReplayRepository`` the committee driver writes."""

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
    def insert_verdicts(self, rows: Sequence[Mapping[str, Any]]) -> int: ...
    def insert_contexts(self, rows: Sequence[Mapping[str, Any]]) -> int: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------
# custom-id namespacing (run-scoped, never collides with live ids)
# --------------------------------------------------------------------------


def namespace_custom_id(run_id: str, custom_id: str) -> str:
    """``<run_id>:<live custom id>`` — live ids contain no ``:`` (they are
    ``{as_of}-{symbol}-{stage}``), so the namespace strips unambiguously."""
    return f"{run_id}:{custom_id}"


def strip_namespace(run_id: str, custom_id: str) -> str:
    prefix = f"{run_id}:"
    if not custom_id.startswith(prefix):
        raise ValueError(f"custom id {custom_id!r} is not namespaced by run {run_id!r}")
    return custom_id[len(prefix) :]


# --------------------------------------------------------------------------
# shortlist + context assembly (gateway-only reads)
# --------------------------------------------------------------------------


def committee_shortlist(
    gw: ReplayCommitteeGateway,
    config: ReplayConfig,
    day: date,
    bundle: ModelBundle | None = None,
    history: SessionHistory = (),
) -> tuple[dict[str, QuantScore], list[str]]:
    """(all quant scores, deep-dive shortlist) for day D.

    Scoring is the quant runner's own: ``bundle`` is the trained quant bundle
    when the run pins a ``quant_model_version`` (Phase 2), None keeps
    baseline-v0 abstention, and names below ``config.min_feature_coverage``
    abstain rather than compete for a deep-dive slot (campaign findings
    Phase 1).

    SELECTION is no longer ``argmax_k(expected_excess_return)`` (campaign
    findings item 19: that debated WMT on 52 of 74 days and never once
    debated AAPL/GOOGL/AMZN/MSFT). The ranked candidate BAND
    (``shortlist_candidate_percentile`` — wider than the BUY threshold,
    which is unchanged) goes through
    :func:`dynamo.replay.shortlist.select_shortlist`: sector-bucketed
    round-robin, a hard crypto slot cap, a cooldown on recently debated
    names, and a slot reserved for the longest-unseen candidate.

    ``history`` is this run's prior sessions in ascending order, each with
    the symbols debated that day — the driver carries it across the run AND
    seeds it from persisted ``replay_verdicts``, so a resumed run's cooldown
    is continuous. Deterministic throughout: same config + same data => same
    slate.
    """
    universe = gw.universe(config.universe_tiers)
    scores = quant_scores_for(
        gw, universe, day, config.quant_score_max_age_days, bundle=bundle
    )
    candidates = shortlist_candidates(
        scores,
        min_feature_coverage=config.min_feature_coverage,
        candidate_percentile=config.shortlist_candidate_percentile,
    )
    if not candidates:
        return scores, []
    shortlist = select_shortlist(
        candidates,
        config.committee_deep_dives_per_day,
        sectors=gw.sectors(candidates),
        history=history,
        crypto_slots=config.shortlist_crypto_slots,
        cooldown_sessions=config.shortlist_cooldown_sessions,
        rotation_slots=config.shortlist_rotation_slots,
        rotation_lookback_sessions=config.shortlist_rotation_lookback_sessions,
    )
    return scores, shortlist


def _prior_case_line(exp: DecisionExperience) -> str:
    """Live-parity evidence line for one prior experience."""
    conviction = "n/a" if exp.conviction is None else f"{float(exp.conviction):.2f}"
    pnl = "n/a" if exp.pnl_pct is None else f"{float(exp.pnl_pct):+.1f}%"
    regime = exp.regime or "unknown-regime"
    sector = exp.sector or "unknown-sector"
    action = exp.action or "unknown-action"
    return (
        f"{PRIOR_CASE_PREFIX}{exp.symbol} {action} conviction {conviction} "
        f"in {regime}/{sector} -> {exp.outcome_quality.value}, "
        f"pnl {pnl} ({exp.as_of.isoformat()})"
    )


def replay_journal_hits(
    gw: ReplayCommitteeGateway,
    symbols: Sequence[str],
    *,
    embedder: Embedder | None,
    top_k: int = JOURNAL_TOP_K,
    include_replay_run_id: str | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """(entry/experience id, evidence text) per symbol, ALL through the
    gateway: as-of journal vector hits (skipped without an embedder) plus
    prior-similar-case lines — the live pipeline's shape, point-in-time.
    ``include_replay_run_id`` (within-run learning) additionally surfaces
    THIS run's own resolved replay experiences."""
    hits: dict[str, list[tuple[str, str]]] = {sym: [] for sym in symbols}
    if embedder is not None and symbols:
        from dynamo.learning.journal import JournalFilters, rank_similar

        queries = [
            f"{sym} investment thesis, lessons and post-trade reviews" for sym in symbols
        ]
        vectors = embedder.embed(queries)
        for sym, vector in zip(symbols, vectors, strict=True):
            # Filter to the symbol in SQL, not just via rank_similar's boost:
            # the journal is symbol-scoped evidence, and without the filter a
            # weak embedding backend returns other tickers' entries as this
            # ticker's "similar situations".
            candidates = gw.journal_search(
                embedding=vector, symbols=[sym], top_k=max(top_k * 4, top_k)
            )
            ranked = rank_similar(
                candidates, JournalFilters(symbols=frozenset({sym})), top_k=top_k
            )
            hits[sym] = [(r.entry.id, r.entry.text) for r in ranked]
    sectors = gw.sectors(list(symbols))
    regime = gw.regime_state()
    regime_name = regime.regime.value if regime is not None else None
    for sym in symbols:
        cases = gw.similar_experiences(
            symbol=sym,
            sector=sectors.get(sym),
            regime=regime_name,
            limit=top_k,
            include_replay_run_id=include_replay_run_id,
        )
        hits[sym].extend((c.id, _prior_case_line(c)) for c in cases)
    return hits


def _macro_context_text(gw: ReplayCommitteeGateway) -> str:
    """Live ``_macro_context`` (orchestration/assets/research.py) parity over
    the gateway's vintage-bounded series: latest value + change over the
    120-day window, one sorted line per series."""
    frame = gw.macro_series(MACRO_SERIES_LOOKBACK_DAYS)
    if frame.empty:
        return "no macro series ingested"
    lines: list[str] = []
    for series_id, grp in frame.sort_values("ts").groupby("series_id"):
        latest = grp.iloc[-1]
        first = grp.iloc[0]
        lines.append(
            f"{series_id}: {float(latest['value']):.3f}"
            f" (from {float(first['value']):.3f} over window)"
        )
    return "\n".join(sorted(lines))


def _transcript_excerpts(
    gw: ReplayCommitteeGateway, symbols: Sequence[str]
) -> dict[str, str]:
    """Latest-call excerpt per symbol from the gateway's 30-day transcript
    window — mirrors the live dossier builder (assets/research.py) exactly:
    latest event per symbol, same head/tail truncation."""
    frame = gw.transcripts(symbols)
    if frame.empty:
        return {}
    latest = frame.sort_values(["symbol", "event_date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    out: dict[str, str] = {}
    for r in latest.to_dict(orient="records"):
        event_date = r["event_date"]
        if isinstance(event_date, datetime):
            event_date = event_date.date()
        out[str(r["symbol"])] = transcript_excerpt(
            str(r["content"]),
            symbol=str(r["symbol"]),
            event_date=event_date.isoformat(),
            source=str(r["source"]),
        )
    return out


def build_replay_contexts(
    gw: ReplayCommitteeGateway,
    symbols: Sequence[str],
    quant_scores: Mapping[str, QuantScore],
    *,
    embedder: Embedder | None = None,
    macro_digest_text: str | None = None,
    include_replay_run_id: str | None = None,
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """(symbol -> dossier context, symbol -> journal hit ids/texts).

    EVERY read goes through the gateway — the only data the committee sees is
    what was knowable at ``gw.as_of``. Sections without historical data
    ("none") render exactly like a thin live dossier's absent sections.
    ``macro_digest_text`` overrides the gateway's persisted digest (used when
    the replay generated one for a day with none); ``include_replay_run_id``
    turns on within-run learning for the prior-case section.
    """
    symbols = list(symbols)
    instruments = gw.instruments(symbols)
    news_by_symbol = gw.news(symbols, NEWS_LOOKBACK_DAYS)
    if macro_digest_text is None:
        digest = gw.macro_digest()
        macro_text = str((digest or {}).get("narrative", ""))
    else:
        macro_text = macro_digest_text
    transcripts_by_symbol = _transcript_excerpts(gw, symbols)
    sectors = gw.sectors()
    # Peers/sector ETFs/benchmark share ONE bars read with the subject names,
    # over the window the 250-session and 52-week statistics need.
    reference_symbols = peer_symbols(sectors, symbols)
    bars = gw.bars(sorted(set(symbols) | set(reference_symbols)), PRICE_STATS_LOOKBACK_DAYS)
    facts = _fundamentals_frame(gw, sorted(set(symbols) | set(reference_symbols)))
    closes = _latest_closes(bars)
    journal_hits = replay_journal_hits(
        gw, symbols, embedder=embedder, include_replay_run_id=include_replay_run_id
    )

    contexts: dict[str, str] = {}
    for sym in symbols:
        instrument = instruments.get(sym) or Instrument(symbol=sym, name=sym)
        sector = instrument.sector or sectors.get(sym)
        fundamentals = build_fundamentals(
            facts,
            symbol=sym,
            as_of=gw.as_of,
            close=closes.get(sym),
            sector=sector,
            peer_closes=closes,
            peer_sectors=sectors,
        )
        primary_news, market_news = partition_news(
            news_by_symbol.get(sym, []),
            symbol=sym,
            name=instrument.name,
            max_items=NEWS_MAX_ITEMS,
        )
        contexts[sym] = assemble_dossier_context(
            instrument=instrument,
            quant_score=quant_scores.get(sym),
            price_stats=price_stats(bars, sym),
            fundamentals=fundamentals,
            peers=peer_context(bars, sym, sector=sector),
            news_items=primary_news,
            market_news_items=market_news,
            macro_digest_text=macro_text,
            journal_hits=[text for _, text in journal_hits.get(sym, [])],
            earnings_call_excerpt=transcripts_by_symbol.get(sym, ""),
        )
    return contexts, journal_hits


def _fundamentals_frame(gw: ReplayCommitteeGateway, symbols: Sequence[str]) -> pd.DataFrame:
    """Point-in-time XBRL facts for the dossier's fundamentals section.

    The gateway's dedicated (longer-window, symbol-filtered) read when it has
    one; otherwise the fundamentals frame inside ``feature_inputs`` — both are
    ``filed_at <= as_of`` bounded and swept by the leakage tripwire."""
    reader = getattr(gw, "fundamentals", None)
    if callable(reader):
        frame = reader(list(symbols))
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    inputs = gw.feature_inputs(list(symbols))
    frame = getattr(inputs, "fundamentals", None)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _latest_closes(bars: pd.DataFrame) -> dict[str, float]:
    """symbol -> last close in the (already as-of bounded) bars frame."""
    if bars.empty:
        return {}
    ordered = bars.sort_values("ts")
    last = ordered.drop_duplicates(subset=["symbol"], keep="last")
    return {
        str(r["symbol"]): float(r["close"])
        for r in last.to_dict(orient="records")
        if pd.notna(r["close"])
    }


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayDeepDive:
    """One completed deep dive: the verdict plus its decision-time context.

    ``stages`` carries the raw per-stage payloads (dossier/bull/bear JSON and
    the raw arbiter output) persisted under ``replay_verdicts.payload["stages"]``
    — the audit trail the schema always promised."""

    as_of: date
    symbol: str
    verdict: CommitteeVerdict
    quant: QuantScore | None
    regime: RegimeState | None
    sector: str | None
    stages: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitteeReplayResult:
    run_id: str
    sessions: tuple[date, ...]
    deep_dives: tuple[ReplayDeepDive, ...]
    experiences: tuple[DecisionExperience, ...]
    spent_usd: float
    cap_usd: float
    status: str  # completed | budget_exhausted


def _verdict_rows(run_id: str, dives: Sequence[ReplayDeepDive]) -> list[dict[str, Any]]:
    """replay_verdicts rows — id namespaced by run for cross-run idempotency.

    ``payload`` = full CommitteeVerdict + the raw per-stage payloads under
    "stages" (the schema comment's promise, made true in Phase 4). Portfolio
    mode filters "stages" back out before ``CommitteeVerdict.model_validate``.
    """
    return [
        {
            "id": f"{run_id}:{d.verdict.id}",
            "run_id": run_id,
            "symbol": d.symbol,
            "as_of": d.as_of,
            "action": d.verdict.action.value,
            "conviction": d.verdict.conviction,
            "disagreement_score": d.verdict.disagreement_score,
            "payload": {**d.verdict.model_dump(mode="json"), "stages": dict(d.stages)},
        }
        for d in dives
    ]


class CommitteeReplay:
    """The committee-replay driver. See the module docstring for the flow."""

    def __init__(
        self,
        *,
        gateway_factory: Callable[[date], ReplayCommitteeGateway],
        research_model: ResearchModel,
        budget: ReplayBudget,
        run_repo: CommitteeRunSink,
        experience_repo: ExperienceSink,
        config: ReplayConfig,
        tier_models: Mapping[str, str],
        prompts: Mapping[str, Any] | None = None,
        schemas: Mapping[str, dict[str, Any]] | None = None,
        embedder: Embedder | None = None,
        sleep_fn: Callable[[float], None] = _time.sleep,
        poll_initial_s: float = POLL_INITIAL_S,
        poll_timeout_s: float = POLL_TIMEOUT_S,
        run_config_extras: Mapping[str, Any] | None = None,
        bundle_factory: Callable[[str], ModelBundle] | None = None,
        acknowledge_model_knowledge: bool = False,
        cutoff_margin_days: int = DEFAULT_CUTOFF_MARGIN_DAYS,
        resume_skip_days: Collection[date] = (),
        prior_dives: Mapping[date, Sequence[str]] | None = None,
    ) -> None:
        if config.mode != "committee":
            raise ValueError(f"CommitteeReplay requires mode='committee', got {config.mode!r}")
        # Model knowledge gate (Phase 4): resolve every tier model's
        # documented training-data cutoff BEFORE anything can spend. A
        # blocked window refuses unless explicitly acknowledged; either way
        # the resolved cutoffs, gate decision and acknowledge flag are
        # recorded into replay_runs.config_json via run_config_extras.
        gate = model_knowledge_gate(config.start, tier_models, cutoff_margin_days)
        if gate["blocked"] and not acknowledge_model_knowledge:
            raise ModelKnowledgeError(
                f"replay window start {config.start} is within "
                f"{cutoff_margin_days}d of the documented training-data "
                f"cutoff for {', '.join(gate['blocking_models'])} "
                f"(cutoffs: {gate['cutoffs']}). The committee could 'know the "
                "future'. Shift the window or pass "
                "acknowledge_model_knowledge=True (--acknowledge-model-knowledge)."
            )
        if prompts is None:
            from dynamo.research.prompt_loader import load_prompt

            prompts = {
                name: load_prompt(name)
                for name in ("dossier_analyst", "bull_advocate", "bear_advocate", "arbiter")
            }
        if schemas is None:
            from dynamo.llm.schemas import SCHEMAS

            schemas = SCHEMAS
        self._gateway_factory = gateway_factory
        self._model = research_model
        self._budget = budget
        self._runs = run_repo
        self._experiences = experience_repo
        self.config = config
        self._tier_models = dict(tier_models)
        self._prompts = dict(prompts)
        self._schemas = dict(schemas)
        self._engine = CommitteeEngine(research_model, self._prompts, self._schemas)
        self._embedder = embedder
        self._sleep = sleep_fn
        self._poll_initial_s = poll_initial_s
        self._poll_timeout_s = poll_timeout_s
        # Run-metadata extras (e.g. the preflight report) merged into
        # replay_runs.config_json — NOT part of ReplayConfig (run-id stability).
        self._run_config_extras = dict(run_config_extras or {})
        self._run_config_extras["model_knowledge"] = {
            **gate,
            "acknowledged": bool(acknowledge_model_knowledge),
        }
        # Trained-bundle seam (Phase 2), mirroring ReplayRunner: injected
        # factory or the artifact loader; resolved ONCE at run start.
        self._bundle_factory = bundle_factory
        self._bundle: ModelBundle | None = None
        # Incremental resume (ADR-0014 claude_code plan windows, but backend-
        # agnostic): days whose verdicts are ALREADY persisted for this run id
        # are never re-submitted — no re-spend, no plan-usage burn. Their
        # experiences were persisted by the prior attempt's resolution pass,
        # and today's within-run-learning retrieval still sees them through
        # the gateway's DB-backed reads.
        self._resume_skip_days = frozenset(resume_skip_days)
        # Shortlist diversity history (campaign findings item 19): which
        # symbols were debated on which session of THIS run. Seeded from the
        # persisted replay_verdicts the composition layer passes in, so a
        # resumed run's cooldown/rotation windows are continuous rather than
        # restarting empty (which would re-debate the names just covered).
        self._debated: dict[date, tuple[str, ...]] = {
            day: tuple(sorted(symbols)) for day, symbols in (prior_dives or {}).items()
        }

    def _resolve_bundle(self) -> ModelBundle | None:
        version = self.config.quant_model_version
        if not version:
            return None
        factory = self._bundle_factory
        if factory is None:
            from dynamo.research.quant.bundle import bundle_from_version

            factory = bundle_from_version
        return factory(version)

    # ------------------------------------------------------------------ run
    def run(self, progress: Callable[[str], None] | None = None) -> CommitteeReplayResult:
        from dynamo.replay.budget import ReplayBudgetExhausted

        cfg = self.config
        run_id = cfg.run_id
        self._bundle = self._resolve_bundle()  # loud + $0 before any run row
        sessions = _sessions_between(cfg.start, cfg.end)
        self._runs.create_run(
            run_id=run_id,
            label=cfg.label,
            mode=cfg.mode,
            start_as_of=cfg.start,
            end_as_of=cfg.end,
            config={**cfg.model_dump(mode="json"), **self._run_config_extras},
            code_version=cfg.code_version,
            budget_usd=self._budget.cap_usd,
            status="running",
        )
        dives: list[ReplayDeepDive] = []
        experiences: list[DecisionExperience] = []
        resolved: set[tuple[date, str]] = set()
        status = "completed"
        try:
            for day_index, day in enumerate(sessions):
                if day in self._resume_skip_days:
                    if progress is not None:
                        progress(
                            f"{day.isoformat()} verdicts already persisted — "
                            "skipped (resume)"
                        )
                    continue
                gw = self._gateway_factory(day)
                # Within-run learning: resolve any dive whose horizon exit
                # already CLOSED strictly before today, so today's prior-case
                # retrieval (through the gateway, as_of-bounded) can see it —
                # exactly like the live system's accumulating experiences.
                if cfg.within_run_learning:
                    experiences.extend(
                        self._resolve_pending(gw, sessions, day_index, dives, resolved, run_id)
                    )
                history = tuple(
                    (session, self._debated.get(session, ()))
                    for session in sessions[:day_index]
                )
                scores, shortlist = committee_shortlist(
                    gw, cfg, day, bundle=self._bundle, history=history
                )
                if not shortlist:
                    if progress is not None:
                        progress(f"{day.isoformat()} shortlist empty — no deep dives ($0)")
                    continue
                # Selection, not completion, drives the cooldown: a name that
                # consumed a slot today must not consume tomorrow's too, even
                # if its dive came back unusable.
                self._debated[day] = tuple(shortlist)
                day_dives = self._run_day(gw, day, shortlist, scores)
                if day_dives:
                    self._runs.insert_verdicts(_verdict_rows(run_id, day_dives))
                    dives.extend(day_dives)
                if progress is not None:
                    progress(
                        f"{day.isoformat()} deep_dives={len(day_dives)} "
                        f"spent=${self._budget.spent_usd:.2f} of ${self._budget.cap_usd:.2f}"
                    )
        except ReplayBudgetExhausted as exc:
            # Partial results kept: verdicts already persisted stay, and the
            # resolution pass below still builds their experiences.
            status = "budget_exhausted"
            self._runs.update_status(run_id, status)
            if progress is not None:
                progress(f"budget exhausted: {exc}")
        except Exception:
            self._runs.update_status(run_id, "failed")
            raise

        experiences.extend(self._resolve_experiences(sessions, dives, run_id, resolved))
        if status == "completed":
            self._runs.update_status(run_id, status)
        return CommitteeReplayResult(
            run_id=run_id,
            sessions=tuple(sessions),
            deep_dives=tuple(dives),
            experiences=tuple(experiences),
            spent_usd=self._budget.spent_usd,
            cap_usd=self._budget.cap_usd,
            status=status,
        )

    # ----------------------------------------------------------------- day
    def _run_day(
        self,
        gw: ReplayCommitteeGateway,
        day: date,
        shortlist: Sequence[str],
        scores: Mapping[str, QuantScore],
    ) -> list[ReplayDeepDive]:
        macro_text: str | None = None
        if self.config.generate_macro_digest and gw.macro_digest() is None:
            macro_text = self._generate_macro_digest(gw, day)
        contexts, journal_hits = build_replay_contexts(
            gw,
            shortlist,
            scores,
            embedder=self._embedder,
            macro_digest_text=macro_text,
            include_replay_run_id=(
                self.config.run_id if self.config.within_run_learning else None
            ),
        )

        # Stage 1: dossier.
        dossier_requests = self._engine.build_batch_requests("dossier", contexts, day)
        dossiers, _ = CommitteeEngine.parse_stage_results(
            self._submit_and_collect(dossier_requests), "dossier"
        )
        if not dossiers:
            return []

        # Stage 2+3: bull & bear in ONE batch (live pipeline parity).
        adv_items = {
            sym: advocate_context_for(contexts[sym], dossiers[sym]) for sym in sorted(dossiers)
        }
        adv_requests = self._engine.build_batch_requests("bull", adv_items, day)
        adv_requests += self._engine.build_batch_requests("bear", adv_items, day)
        adv_results = self._submit_and_collect(adv_requests)
        bulls, _ = CommitteeEngine.parse_stage_results(adv_results, "bull")
        bears, _ = CommitteeEngine.parse_stage_results(adv_results, "bear")
        ready = sorted(set(bulls) & set(bears) & set(dossiers))
        if not ready:
            return []

        # Stage 4: arbiter.
        arb_items = {
            sym: arbiter_context_for(
                contexts[sym], dossiers[sym], bulls[sym], bears[sym], scores.get(sym)
            )
            for sym in ready
        }
        arb_requests = self._engine.build_batch_requests("arbiter", arb_items, day)
        arbiters, _ = CommitteeEngine.parse_stage_results(
            self._submit_and_collect(arb_requests), "arbiter"
        )

        sectors = gw.sectors(list(shortlist))
        regime = gw.regime_state()
        dives: list[ReplayDeepDive] = []
        for sym in sorted(arbiters):
            try:
                verdict = CommitteeEngine.assemble_verdict(
                    symbol=sym,
                    as_of=day,
                    arbiter_payload=arbiters[sym],
                    bull=_brief(bulls[sym], sym, Side.BULL),
                    bear=_brief(bears[sym], sym, Side.BEAR),
                    quant_score=scores.get(sym),
                    prompt_version=self._prompts["arbiter"].version,
                    model_id=self._tier_models.get("arbiter", ""),
                    journal_context_used=tuple(eid for eid, _ in journal_hits.get(sym, [])),
                )
            except Exception:
                # Fault isolation (live-path quarantine parity): one dive with
                # unsalvageable model output must never void the day or the run.
                log.exception(
                    "replay_dive_quarantined", symbol=sym, as_of=day.isoformat()
                )
                continue
            dives.append(
                ReplayDeepDive(
                    as_of=day,
                    symbol=sym,
                    verdict=verdict,
                    quant=scores.get(sym),
                    regime=regime,
                    sector=sectors.get(sym),
                    stages={
                        "dossier": dict(dossiers[sym]),
                        "bull": dict(bulls[sym]),
                        "bear": dict(bears[sym]),
                        "arbiter": dict(arbiters[sym]),
                    },
                )
            )
        return dives

    # ------------------------------------------------- replay macro digest
    def _macro_prompt(self) -> Any:
        prompt = self._prompts.get(MACRO_DIGEST_STAGE)
        if prompt is None:
            from dynamo.research.prompt_loader import load_prompt

            prompt = load_prompt(MACRO_DIGEST_STAGE)
            self._prompts[MACRO_DIGEST_STAGE] = prompt
        return prompt

    def _macro_schema(self) -> dict[str, Any]:
        schema = self._schemas.get(MACRO_DIGEST_STAGE)
        if schema is None:
            from dynamo.llm.schemas import SCHEMAS

            schema = SCHEMAS[MACRO_DIGEST_STAGE]
            self._schemas[MACRO_DIGEST_STAGE] = schema
        return schema

    def _generate_macro_digest(self, gw: ReplayCommitteeGateway, day: date) -> str | None:
        """One replay macro digest for a day with none persisted, from the
        gateway's vintage-bounded macro series through the analyst tier —
        live ``build_macro_digest`` parity (same prompt, same instruction,
        same max_tokens), submitted through the ONE batch funnel so the cost
        flows through ReplayBudget and the context row persists like every
        other request. Prediction-market context is honestly "none" (no
        vintaged PM history in replay — accepted plan caveat).

        The digest is stored replay-namespaced ONLY: its request context row
        (stage='macro_digest') plus the generated JSON payload
        (stage='macro_digest_output') in ``replay_contexts`` — NEVER the live
        ``macro_digests`` table. Returns the narrative, or None on failure
        (the day's dossiers then render the absent-macro section honestly).
        """
        from dynamo.research.macro import _MACRO_INSTRUCTION

        prompt = self._macro_prompt()
        macro_context = (
            f"## Macro series and market context\n{_macro_context_text(gw)}\n\n"
            f"## Prediction-market digest\nnone"
        )
        instruction = _MACRO_INSTRUCTION.format(as_of=day) + TEMPORAL_DISCIPLINE.format(
            as_of=day
        )
        request = {
            "custom_id": custom_id_for(day, MACRO_DIGEST_SYMBOL, MACRO_DIGEST_STAGE),
            "model_tier": "analyst",
            "system": prompt.system,
            "context": macro_context,
            "instruction": instruction,
            "schema": self._macro_schema(),
            "max_tokens": MACRO_DIGEST_MAX_TOKENS,
        }
        results = self._submit_and_collect([request])
        payload = next((p for _cid, p in results if p is not None), None)
        if payload is None:
            return None
        text = json.dumps(dict(payload), sort_keys=True, default=str)
        self._runs.insert_contexts(
            [
                {
                    "id": namespace_custom_id(
                        self.config.run_id,
                        custom_id_for(day, MACRO_DIGEST_SYMBOL, MACRO_DIGEST_OUTPUT_STAGE),
                    ),
                    "run_id": self.config.run_id,
                    "as_of": day,
                    "symbol": MACRO_DIGEST_SYMBOL,
                    "stage": MACRO_DIGEST_OUTPUT_STAGE,
                    "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "context_text": text,
                    "model_id": self._tier_models.get("analyst", ""),
                    "created_at": datetime.now(UTC),
                }
            ]
        )
        return str(payload.get("narrative", ""))

    # --------------------------------------------------- batch submit/collect
    def _submit_and_collect(
        self, requests: list[dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """One stage-batch: project -> cap-check -> persist contexts ->
        submit -> poll -> collect.

        Runs entirely in-process: the same client that stored the batch's
        schemas parses its results (the ``_batch_schemas`` process-local
        contract). The budget records actual metered cost as results stream.

        EVERY request's decision-time text (context + instruction; system
        prompts excluded — versioned in the repo) is persisted to
        ``replay_contexts`` BEFORE submission, sha256-stamped, so the
        post-run leakage audit sees exactly what the model saw even if the
        batch later fails.
        """
        if not requests:
            return []
        run_id = self.config.run_id
        projected = estimate_batch_cost(requests, self._tier_models)
        self._budget.check_projection(projected)
        self._runs.insert_contexts([self._context_row(req) for req in requests])
        namespaced = [
            {**req, "custom_id": namespace_custom_id(run_id, str(req["custom_id"]))}
            for req in requests
        ]
        batch_id = self._model.submit_batch(namespaced)
        self._wait_for_batch(batch_id)
        return [
            (strip_namespace(run_id, custom_id), payload)
            for custom_id, payload in self._model.batch_results(batch_id)
        ]

    def _context_row(self, req: Mapping[str, Any]) -> dict[str, Any]:
        """One ``replay_contexts`` row for one batch request (context +
        instruction — the exact volatile text; system prompt excluded)."""
        custom_id = str(req["custom_id"])
        as_of, symbol, stage = split_custom_id(custom_id)
        text = f"{req['context']}\n\n{req['instruction']}"
        return {
            "id": namespace_custom_id(self.config.run_id, custom_id),
            "run_id": self.config.run_id,
            "as_of": as_of,
            "symbol": symbol,
            "stage": stage,
            "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "context_text": text,
            "model_id": self._tier_models.get(str(req["model_tier"]), ""),
            "created_at": datetime.now(UTC),
        }

    def _wait_for_batch(self, batch_id: str) -> None:
        waited = 0.0
        interval = self._poll_initial_s
        while self._model.batch_status(batch_id) != "ended":
            if waited >= self._poll_timeout_s:
                raise TimeoutError(
                    f"batch {batch_id} not finished after {self._poll_timeout_s:.0f}s"
                )
            self._sleep(interval)
            waited += interval
            interval = min(interval * POLL_BACKOFF, POLL_MAX_S)

    # --------------------------------------------------------- resolution
    def _resolve_one(
        self,
        dive: ReplayDeepDive,
        sessions: Sequence[date],
        session_index: Mapping[date, int],
        bars: pd.DataFrame,
        run_id: str,
        *,
        clip_to_window: bool,
    ) -> DecisionExperience | None:
        """One dive -> one resolved experience (or None when unresolvable).

        Entry at the NEXT session's open (never same-bar), exit at the close
        of entry + horizon sessions. ``clip_to_window`` (the final pass)
        clips the exit to the window end — the quant runner's window-end
        synthetic-exit parity; the incremental (within-run) pass never clips:
        it only resolves fully-elapsed horizons.
        """
        i = session_index.get(dive.as_of)
        if i is None or i + 1 >= len(sessions):
            return None  # no next open inside the window
        entry_i = i + 1
        exit_i = entry_i + self.config.horizon_sessions
        if clip_to_window:
            exit_i = min(exit_i, len(sessions) - 1)
            if exit_i <= entry_i:
                return None  # entry IS the last session — outcome outside window
        elif exit_i >= len(sessions):
            return None
        entry_date = sessions[entry_i]
        exit_date = sessions[exit_i]
        entry_price = _day_prices(bars, entry_date)[0].get(dive.symbol)
        exit_price = _day_prices(bars, exit_date)[1].get(dive.symbol)
        if not entry_price or not exit_price or entry_price <= 0 or exit_price <= 0:
            return None
        pnl_pct = (exit_price / entry_price - 1.0) * 100.0
        trade = TradeFacts(
            trade_id=f"{dive.symbol}-{entry_date.isoformat()}",
            symbol=dive.symbol,
            qty=1.0,
            opened_at=_midnight(entry_date),
            closed_at=_midnight(exit_date),
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            holding_days=(exit_date - entry_date).days,
        )
        return build_experience(
            trade=trade,
            now=_midnight(exit_date),  # deterministic, never wall clock
            verdict=dive.verdict,
            quant=dive.quant,
            regime=dive.regime,
            path=_path_inputs(bars, dive.symbol, entry_date, exit_date),
            sector=dive.sector,
            source="replay",
            replay_run_id=run_id,
        )

    def _resolve_pending(
        self,
        gw: ReplayCommitteeGateway,
        sessions: Sequence[date],
        day_index: int,
        dives: Sequence[ReplayDeepDive],
        resolved: set[tuple[date, str]],
        run_id: str,
    ) -> list[DecisionExperience]:
        """Within-run learning (Phase 4): resolve entry dives whose horizon
        exit closed STRICTLY before ``sessions[day_index]`` — the data was
        knowable yesterday, so today's gateway reads (as_of-bounded) may see
        the resulting experience. Bars come from TODAY's gateway (<= today),
        never the final one: no future prices touch an in-flight day."""
        session_index = {d: i for i, d in enumerate(sessions)}
        ready = [
            d
            for d in dives
            if d.verdict.action in ENTRY_ACTIONS
            and (d.as_of, d.symbol) not in resolved
            and session_index[d.as_of] + 1 + self.config.horizon_sessions < day_index
        ]
        if not ready:
            return []
        symbols = sorted({d.symbol for d in ready})
        bars = gw.bars([*symbols, "SPY"], RESOLUTION_BARS_LOOKBACK_DAYS)
        out: list[DecisionExperience] = []
        for dive in ready:
            exp = self._resolve_one(
                dive, sessions, session_index, bars, run_id, clip_to_window=False
            )
            if exp is None:
                continue
            self._experiences.insert(exp)
            resolved.add((dive.as_of, dive.symbol))
            out.append(exp)
        return out

    def _resolve_experiences(
        self,
        sessions: Sequence[date],
        dives: Sequence[ReplayDeepDive],
        run_id: str,
        resolved: set[tuple[date, str]],
    ) -> list[DecisionExperience]:
        """Final pass: entry verdicts not already resolved incrementally ->
        experiences, with exits clipped to the window end (the quant runner's
        window-end synthetic-exit parity). Entries whose next open falls
        outside the window never resolve — no experience row."""
        if not sessions or not dives:
            return []
        session_index = {d: i for i, d in enumerate(sessions)}
        pending = [
            d
            for d in dives
            if d.verdict.action in ENTRY_ACTIONS and (d.as_of, d.symbol) not in resolved
        ]
        if not pending:
            return []
        final_gw = self._gateway_factory(sessions[-1])
        symbols = sorted({d.symbol for d in pending})
        bars = final_gw.bars([*symbols, "SPY"], RESOLUTION_BARS_LOOKBACK_DAYS)
        out: list[DecisionExperience] = []
        for dive in pending:
            exp = self._resolve_one(
                dive, sessions, session_index, bars, run_id, clip_to_window=True
            )
            if exp is None:
                continue
            self._experiences.insert(exp)
            resolved.add((dive.as_of, dive.symbol))
            out.append(exp)
        return out


def _brief(payload: Mapping[str, Any], symbol: str, side: Side) -> AdvocateBrief:
    # Tolerant boundary parse: nested LLM payloads may carry extra/padded keys
    # (same real-run lesson as research.committee._validated — reuse it).
    from dynamo.research.committee import _validated

    fields = set(AdvocateBrief.model_fields)
    filtered = {k: v for k, v in payload.items() if k in fields}
    filtered.update({"symbol": symbol, "side": side})
    return _validated(AdvocateBrief, filtered)


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)
