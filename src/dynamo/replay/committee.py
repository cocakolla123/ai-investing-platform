"""Committee replay — the ONLY token-spending replay mode (plan Phase 5).

Per session day D in [start, end]:

1. shortlist: reuse the quant runner's selection logic — point-in-time quant
   scores through the day's gateway -> ``synthesize_verdicts`` ->
   ``synthesize_watchlist`` (BUY candidates ranked by conviction) -> top
   ``committee_deep_dives_per_day`` symbols. Days without BUY candidates run
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

import time as _time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

import pandas as pd

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
    synthesize_verdicts,
    synthesize_watchlist,
)
from dynamo.research.committee import (
    CommitteeEngine,
    advocate_context_for,
    arbiter_context_for,
)
from dynamo.research.dossier import assemble_dossier_context
from dynamo.research.quant.scoring import ModelBundle

ENTRY_ACTIONS = frozenset({Action.BUY, Action.ADD})
PRICE_STATS_LOOKBACK_DAYS = 120  # live build_deep_dive_contexts parity
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


class ReplayCommitteeGateway(Protocol):
    """The slice of ``HistoricalDataGateway`` committee replay reads."""

    as_of: date

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]: ...
    def bars(self, symbols: Sequence[str] | None, lookback_days: int) -> pd.DataFrame: ...
    def feature_inputs(self, symbols: Sequence[str]) -> Any: ...
    def quant_scores(self, symbols: Sequence[str] | None = None) -> pd.DataFrame: ...
    def macro_digest(self) -> dict[str, Any] | None: ...
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
) -> tuple[dict[str, QuantScore], list[str]]:
    """(all quant scores, deep-dive shortlist) for day D — the quant runner's
    own selection logic, capped at ``committee_deep_dives_per_day``.
    ``bundle`` is the trained quant bundle when the run pins a
    ``quant_model_version`` (Phase 2); None keeps baseline-v0 abstention."""
    universe = gw.universe(config.universe_tiers)
    scores = quant_scores_for(
        gw, universe, day, config.quant_score_max_age_days, bundle=bundle
    )
    synth = synthesize_verdicts(scores, day, config.horizon_sessions)
    watchlist = synthesize_watchlist(synth, day)
    shortlist = [w.symbol for w in watchlist[: config.committee_deep_dives_per_day]]
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
) -> dict[str, list[tuple[str, str]]]:
    """(entry/experience id, evidence text) per symbol, ALL through the
    gateway: as-of journal vector hits (skipped without an embedder) plus
    prior-similar-case lines — the live pipeline's shape, point-in-time."""
    hits: dict[str, list[tuple[str, str]]] = {sym: [] for sym in symbols}
    if embedder is not None and symbols:
        from dynamo.learning.journal import JournalFilters, rank_similar

        queries = [
            f"{sym} investment thesis, lessons and post-trade reviews" for sym in symbols
        ]
        vectors = embedder.embed(queries)
        for sym, vector in zip(symbols, vectors, strict=True):
            candidates = gw.journal_search(embedding=vector, top_k=max(top_k * 4, top_k))
            ranked = rank_similar(
                candidates, JournalFilters(symbols=frozenset({sym})), top_k=top_k
            )
            hits[sym] = [(r.entry.id, r.entry.text) for r in ranked]
    sectors = gw.sectors(list(symbols))
    regime = gw.regime_state()
    regime_name = regime.regime.value if regime is not None else None
    for sym in symbols:
        cases = gw.similar_experiences(
            symbol=sym, sector=sectors.get(sym), regime=regime_name, limit=top_k
        )
        hits[sym].extend((c.id, _prior_case_line(c)) for c in cases)
    return hits


def build_replay_contexts(
    gw: ReplayCommitteeGateway,
    symbols: Sequence[str],
    quant_scores: Mapping[str, QuantScore],
    *,
    embedder: Embedder | None = None,
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """(symbol -> dossier context, symbol -> journal hit ids/texts).

    EVERY read goes through the gateway — the only data the committee sees is
    what was knowable at ``gw.as_of``. Sections without historical data
    ("none") render exactly like a thin live dossier's absent sections.
    """
    symbols = list(symbols)
    instruments = gw.instruments(symbols)
    news_by_symbol = gw.news(symbols, NEWS_LOOKBACK_DAYS)
    digest = gw.macro_digest()
    macro_text = str((digest or {}).get("narrative", ""))
    bars = gw.bars(symbols, PRICE_STATS_LOOKBACK_DAYS)
    journal_hits = replay_journal_hits(gw, symbols, embedder=embedder)

    contexts: dict[str, str] = {}
    for sym in symbols:
        instrument = instruments.get(sym) or Instrument(symbol=sym, name=sym)
        price_stats: dict[str, float] = {}
        if not bars.empty:
            sym_bars = cast(pd.DataFrame, bars[bars["symbol"] == sym]).sort_values("ts")
            if not sym_bars.empty:
                closes = sym_bars["close"].astype(float)
                price_stats["close"] = float(closes.iloc[-1])
                if len(closes) > 21:
                    price_stats["ret_20d_pct"] = float(
                        (closes.iloc[-1] / closes.iloc[-21] - 1.0) * 100.0
                    )
                    price_stats["vol_20d_ann_pct"] = float(
                        closes.pct_change().tail(20).std() * (252**0.5) * 100.0
                    )
                dollar = sym_bars["close"].astype(float) * sym_bars["volume"].astype(float)
                price_stats["dollar_volume"] = float(dollar.tail(20).median())
        contexts[sym] = assemble_dossier_context(
            instrument=instrument,
            quant_score=quant_scores.get(sym),
            price_stats=price_stats,
            news_items=news_by_symbol.get(sym, [])[:NEWS_MAX_ITEMS],
            macro_digest_text=macro_text,
            journal_hits=[text for _, text in journal_hits.get(sym, [])],
        )
    return contexts, journal_hits


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayDeepDive:
    """One completed deep dive: the verdict plus its decision-time context."""

    as_of: date
    symbol: str
    verdict: CommitteeVerdict
    quant: QuantScore | None
    regime: RegimeState | None
    sector: str | None


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
    """replay_verdicts rows — id namespaced by run for cross-run idempotency."""
    return [
        {
            "id": f"{run_id}:{d.verdict.id}",
            "run_id": run_id,
            "symbol": d.symbol,
            "as_of": d.as_of,
            "action": d.verdict.action.value,
            "conviction": d.verdict.conviction,
            "disagreement_score": d.verdict.disagreement_score,
            "payload": d.verdict.model_dump(mode="json"),
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
    ) -> None:
        if config.mode != "committee":
            raise ValueError(f"CommitteeReplay requires mode='committee', got {config.mode!r}")
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
        self._engine = CommitteeEngine(research_model, self._prompts, dict(schemas))
        self._embedder = embedder
        self._sleep = sleep_fn
        self._poll_initial_s = poll_initial_s
        self._poll_timeout_s = poll_timeout_s
        # Run-metadata extras (e.g. the preflight report) merged into
        # replay_runs.config_json — NOT part of ReplayConfig (run-id stability).
        self._run_config_extras = dict(run_config_extras or {})
        # Trained-bundle seam (Phase 2), mirroring ReplayRunner: injected
        # factory or the artifact loader; resolved ONCE at run start.
        self._bundle_factory = bundle_factory
        self._bundle: ModelBundle | None = None

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
        status = "completed"
        try:
            for day in sessions:
                gw = self._gateway_factory(day)
                scores, shortlist = committee_shortlist(gw, cfg, day, bundle=self._bundle)
                if not shortlist:
                    if progress is not None:
                        progress(f"{day.isoformat()} shortlist empty — no deep dives ($0)")
                    continue
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

        experiences = self._resolve_experiences(sessions, dives, run_id)
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
        contexts, journal_hits = build_replay_contexts(
            gw, shortlist, scores, embedder=self._embedder
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
            dives.append(
                ReplayDeepDive(
                    as_of=day,
                    symbol=sym,
                    verdict=verdict,
                    quant=scores.get(sym),
                    regime=regime,
                    sector=sectors.get(sym),
                )
            )
        return dives

    # --------------------------------------------------- batch submit/collect
    def _submit_and_collect(
        self, requests: list[dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any] | None]]:
        """One stage-batch: project -> cap-check -> submit -> poll -> collect.

        Runs entirely in-process: the same client that stored the batch's
        schemas parses its results (the ``_batch_schemas`` process-local
        contract). The budget records actual metered cost as results stream.
        """
        if not requests:
            return []
        run_id = self.config.run_id
        projected = estimate_batch_cost(requests, self._tier_models)
        self._budget.check_projection(projected)
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
    def _resolve_experiences(
        self,
        sessions: Sequence[date],
        dives: Sequence[ReplayDeepDive],
        run_id: str,
    ) -> list[DecisionExperience]:
        """Entry verdicts -> resolved experiences within the run window.

        Deterministic per-decision resolution (no portfolio loop): entry at
        the NEXT session's open (never same-bar), exit at the close of
        entry + horizon sessions, clipped to the window end (the quant
        runner's window-end synthetic exit parity). Entries whose next open
        falls outside the window never resolve — no experience row.
        """
        if not sessions or not dives:
            return []
        session_index = {d: i for i, d in enumerate(sessions)}
        final_gw = self._gateway_factory(sessions[-1])
        symbols = sorted({d.symbol for d in dives if d.verdict.action in ENTRY_ACTIONS})
        if not symbols:
            return []
        bars = final_gw.bars([*symbols, "SPY"], RESOLUTION_BARS_LOOKBACK_DAYS)
        prices: dict[date, tuple[dict[str, float], dict[str, float]]] = {
            day: _day_prices(bars, day) for day in sessions
        }
        out: list[DecisionExperience] = []
        for dive in dives:
            if dive.verdict.action not in ENTRY_ACTIONS:
                continue
            i = session_index.get(dive.as_of)
            if i is None or i + 1 >= len(sessions):
                continue  # no next open inside the window
            entry_date = sessions[i + 1]
            exit_i = min(i + 1 + self.config.horizon_sessions, len(sessions) - 1)
            if exit_i <= i + 1:
                continue  # entry IS the last session — outcome outside window
            exit_date = sessions[exit_i]
            entry_price = prices[entry_date][0].get(dive.symbol)
            exit_price = prices[exit_date][1].get(dive.symbol)
            if not entry_price or not exit_price or entry_price <= 0 or exit_price <= 0:
                continue
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
            exp = build_experience(
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
            self._experiences.insert(exp)
            out.append(exp)
        return out


def _brief(payload: Mapping[str, Any], symbol: str, side: Side) -> AdvocateBrief:
    fields = set(AdvocateBrief.model_fields)
    filtered = {k: v for k, v in payload.items() if k in fields}
    filtered.update({"symbol": symbol, "side": side})
    return AdvocateBrief.model_validate(filtered)


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)
