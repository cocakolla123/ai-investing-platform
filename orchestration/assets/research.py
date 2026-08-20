"""Research assets + committee batch jobs — docs 02 §4, 03 §1/§3.

Thin wiring: read frames via repositories, call core functions
(detectors, quality checks, quant scoring, selection, committee engine),
persist results, return metadata. Guardrails and all business logic live in
``src/dynamo``; missing configuration degrades per-asset, never all-or-nothing.
"""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

import pandas as pd
from dagster import (
    AssetExecutionContext,
    Config,
    MaterializeResult,
    OpExecutionContext,
    job,
    op,
)
from dagster import asset as dagster_asset
from sqlalchemy import select, update

from dynamo.persistence import schema
from orchestration.resources import DynamoResource, utcnow

_log = logging.getLogger(__name__)

RESEARCH_GROUP = "research"

# Sync fallback threshold: <= this many deep dives run inline (no batch round-trip).
SYNC_DEEP_DIVE_MAX = 3
BATCH_TERMINAL_STATUSES = frozenset({"ended", "completed"})
COMMITTEE_STAGE_AFTER = {"dossier": "advocates", "advocates": "arbiter", "arbiter": None}


# --------------------------------------------------------------------------
# shared thin helpers
# --------------------------------------------------------------------------


def _read_df(dynamo: DynamoResource, stmt: Any) -> pd.DataFrame:
    with dynamo.engine().connect() as conn:
        return pd.read_sql(stmt, conn)


def _subset(model_cls: type, data: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Filter a mapping to a pydantic model's fields, then apply overrides."""
    fields: Mapping[str, Any] = model_cls.model_fields  # type: ignore[attr-defined]
    payload = {k: v for k, v in data.items() if k in fields and v is not None}
    payload.update(overrides)
    return payload


def _since(as_of: date, days: int) -> datetime:
    return datetime.combine(as_of - timedelta(days=days), time.min, tzinfo=UTC)


def _pydate(value: Any) -> date:
    return cast(date, pd.Timestamp(value).date())


def _pydatetime(value: Any) -> datetime:
    return cast(datetime, pd.Timestamp(value).to_pydatetime())


def _rows_to_models(model_cls: type, frame: pd.DataFrame) -> list[Any]:
    out: list[Any] = []
    for row in frame.to_dict(orient="records"):
        out.append(model_cls.model_validate(_subset(model_cls, row)))  # type: ignore[attr-defined]
    return out


def _quant_score_for(dynamo: DynamoResource, symbol: str) -> Any | None:
    from dynamo.domain.models.research import QuantScore

    frame = dynamo.market_repo().latest_quant_scores([symbol])
    if frame.empty:
        return None
    row = dict(frame.iloc[-1])
    row["as_of"] = _pydate(row["as_of"])
    return QuantScore.model_validate(_subset(QuantScore, row))


# --------------------------------------------------------------------------
# data_quality — doc 02 §6, between ingestion and features
# --------------------------------------------------------------------------


@dagster_asset(group_name=RESEARCH_GROUP)
def data_quality(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Run the doc 02 §6 checks; failures quarantine symbols, not the pipeline."""
    from dynamo.data.quality import (
        corp_action_sanity,
        dual_source_close_check,
        gap_check,
        run_quality_checks,
        staleness_check,
    )

    as_of = utcnow().date()
    start = as_of - timedelta(days=8)
    repo = dynamo.market_repo()
    tier1 = dynamo.universe_symbols(as_of, tiers=(1, 3))

    t = schema.bars_1d
    raw = _read_df(
        dynamo,
        select(t.c.symbol, t.c.ts, t.c.close, t.c.source).where(t.c.ts >= _since(as_of, 8)),
    )
    alpaca_df = cast(pd.DataFrame, raw[raw["source"] == "alpaca"][["symbol", "ts", "close"]])
    tiingo_df = cast(pd.DataFrame, raw[raw["source"] == "tiingo"][["symbol", "ts", "close"]])

    canonical = dynamo.bars_window(tier1 or None, as_of, 8)
    ca = schema.corporate_actions
    actions = _read_df(dynamo, select(ca.c.symbol, ca.c.ex_date))

    latest_ts: dict[str, datetime] = {}
    if not canonical.empty:
        for sym, grp in canonical.groupby("symbol"):
            latest_ts[str(sym)] = _pydatetime(grp["ts"].max())

    issues = (
        dual_source_close_check(alpaca_df, tiingo_df)
        + gap_check(canonical, start, as_of)
        + corp_action_sanity(canonical, actions)
        + staleness_check(latest_ts, tier1, as_of=utcnow(), max_age_days=2)
    )
    quarantined = run_quality_checks(repo, issues)
    return MaterializeResult(metadata={"issues": len(issues), "symbols_quarantined": quarantined})


# --------------------------------------------------------------------------
# tier2_scan — code-only detectors + Haiku triage (doc 02 §2)
# --------------------------------------------------------------------------


def _instrument_sectors(dynamo: DynamoResource) -> dict[str, str]:
    """symbol -> GICS sector from ``instruments`` (blank sectors dropped)."""
    inst = schema.instruments
    frame = _read_df(dynamo, select(inst.c.symbol, inst.c.sector))
    return {
        str(r["symbol"]): str(r["sector"]) for r in frame.to_dict(orient="records") if r["sector"]
    }


def _triage_history(dynamo: DynamoResource, as_of: date, sessions: int) -> list[Any]:
    """Prior sessions and the symbols triaged on each, ascending.

    One entry per calendar day in the window, INCLUDING days that triaged
    nothing: ``select_shortlist``'s cooldown counts entries, so a run of
    quiet days must still age a symbol out of its cooldown.
    """
    a = schema.agent_opinions
    frame = _read_df(
        dynamo,
        select(a.c.symbol, a.c.as_of).where(
            a.c.role == "triage", a.c.as_of >= as_of - timedelta(days=sessions)
        ),
    )
    by_day: dict[date, list[str]] = {}
    for row in frame.to_dict(orient="records"):
        by_day.setdefault(_pydate(row["as_of"]), []).append(str(row["symbol"]))
    return [
        (day, sorted(set(by_day.get(day, []))))
        for day in (as_of - timedelta(days=n) for n in range(sessions, 0, -1))
    ]


# How far back the triage queue reaches. Flags persist in the table, so
# triage drains a BACKLOG, not just tonight's batch — but not an unbounded
# one: a flag older than this describes a price move that today's triage
# cannot usefully judge, and should expire unread rather than be answered
# against stale context. Two weeks covers a long holiday weekend plus a
# multi-day outage without reaching back into a different market regime.
TRIAGE_QUEUE_LOOKBACK_DAYS = 14

# Hard ceiling on rows pulled into memory per run. The first whole-market
# sweep wrote 1,155 flags in one night; the cap on LLM calls is applied after
# ranking, so this only bounds the frame, not the spend.
TRIAGE_QUEUE_MAX_ROWS = 20_000

# `data_quality` flags are not trading candidates — they mark a bar the
# detectors could not trust (an unadjusted corporate action; see
# dynamo.data.detectors). They must never reach the LLM or the promotion
# path, and they must not sit in the queue forever either, so they are
# stamped with their own verdict without an API call.
DATA_QUALITY_VERDICT = "data_quality"


def _flags_from_rows(frame: pd.DataFrame) -> tuple[list[Any], dict[int, int]]:
    """Rebuild ``SurveillanceFlag`` objects from queue rows.

    Returns the flags and a map from each flag's *object identity* to its
    database id. Identity is the right key here: the ranking layer hands the
    same objects back, and two genuinely distinct rows can otherwise be equal
    by value (same symbol, detector and severity on two dates).
    """
    from dynamo.domain.models.market import SurveillanceFlag

    flags: list[Any] = []
    ids: dict[int, int] = {}
    for row in frame.to_dict(orient="records"):
        details = row.get("details") or {}
        flag = SurveillanceFlag(
            symbol=str(row["symbol"]),
            detector=str(row["detector"]),
            severity=float(row["severity"]),
            details=dict(details) if isinstance(details, dict) else {},
            ts=_pydatetime(row["ts"]),
        )
        flags.append(flag)
        ids[id(flag)] = int(row["id"])
    return flags, ids


def _triage_flags(
    context: AssetExecutionContext | OpExecutionContext,
    dynamo: DynamoResource,
    as_of: date,
    *,
    lookback_days: int = TRIAGE_QUEUE_LOOKBACK_DAYS,
) -> int:
    """Haiku triage over the best-ranked UN-TRIAGED flags; material -> Tier 3.

    Reads the queue from ``surveillance_flags`` rather than from whatever this
    process happened to compute. That is the difference between a pipeline and
    a side effect: flags written by the ``surveillance-sweep`` CLI, or by a
    run that died before triage, or on a night the model was unavailable, are
    now picked up by the next run instead of being written and forgotten. The
    first whole-market sweep left 1,155 such rows.
    """
    from dynamo.data.universe import UniverseManager
    from dynamo.llm.schemas import TRIAGE_SCHEMA
    from dynamo.research.flag_ranking import group_by_symbol, select_for_triage
    from dynamo.research.prompt_loader import load_prompt

    repo_for_queue = dynamo.market_repo()
    since = _since(as_of, lookback_days)

    # Retire data-quality flags first, without an API call, so they neither
    # reach the model nor accumulate in the queue forever.
    dq = repo_for_queue.untriaged_flags(since, limit=TRIAGE_QUEUE_MAX_ROWS)
    dq_ids = [int(r["id"]) for r in dq.to_dict(orient="records") if r["detector"] == "data_quality"]
    if dq_ids:
        repo_for_queue.set_triage_verdicts(dq_ids, DATA_QUALITY_VERDICT, triaged_at=utcnow())
        context.log.info("tier2 triage: retired %d data_quality flags (no LLM)", len(dq_ids))

    queue = repo_for_queue.untriaged_flags(
        since, exclude_detectors=("data_quality",), limit=TRIAGE_QUEUE_MAX_ROWS
    )
    flags, flag_ids = _flags_from_rows(queue)

    model = dynamo.research_model()
    if model is None or not flags:
        if flags:
            context.log.warning(
                "tier2 triage skipped: ANTHROPIC_API_KEY unset (%d flags left un-triaged)",
                len(flags),
            )
        return 0

    settings = dynamo.settings()
    repo = dynamo.market_repo()
    prompt = load_prompt("triage")
    manager = UniverseManager(
        repo,
        default_promotion_days=settings.tier3_promotion_window_days,
        include_crypto=bool(settings.crypto_enabled),
    )

    # Ranking stage (dynamo.research.flag_ranking). This used to be
    # `sorted(flags, key=-severity)[:cap]`, which ranked by a number the
    # detectors compute on incompatible scales: volume_spike saturates at
    # 1.0 past 6x median volume while range_break emits a constant 0.6, so
    # a routine volume blip outranked every 52-week breakout, every night.
    # Percentile-within-detector makes them comparable; the shared
    # shortlist layer then applies cooldown and sector round-robin so one
    # macro move cannot consume the whole cap.
    cap = settings.tier2_triage_cap_per_night
    selected = select_for_triage(
        flags,
        cap,
        sectors=_instrument_sectors(dynamo),
        history=_triage_history(dynamo, as_of, settings.tier2_triage_cooldown_sessions * 4),
        cooldown_sessions=settings.tier2_triage_cooldown_sessions,
    )
    by_symbol = group_by_symbol(flags, selected)
    context.log.info(
        "tier2 triage: %d flags on %d symbols -> %d selected (cap %d)",
        len(flags),
        len({f.symbol for f in flags}),
        len(selected),
        cap,
    )
    triaged = 0
    for symbol, sym_flags in by_symbol.items():
        flag_text = "\n".join(
            f"- {f.detector} severity={f.severity:.2f} "
            f"details={json.dumps(f.details, sort_keys=True, default=str)}"
            for f in sym_flags
        )
        parsed = model.complete_json(
            role="triage",
            system=prompt.system,
            context=f"## Surveillance flags for {symbol}\n{flag_text}",
            instruction=f"Classify the materiality of these flags for {symbol} as of {as_of}.",
            schema=TRIAGE_SCHEMA,
            model_tier="screen",
            max_tokens=512,
        )
        verdict = str(parsed.get("verdict", "noise"))
        # Stamp EXACTLY the rows this verdict was formed from. The previous
        # write matched `symbol AND triage_verdict IS NULL` with no time
        # bound, so one night's verdict for a symbol also overwrote every
        # historical un-triaged row that symbol had — a verdict about
        # yesterday's flags that nobody had read.
        repo.set_triage_verdicts(
            [flag_ids[id(f)] for f in sym_flags if id(f) in flag_ids],
            verdict,
            triaged_at=utcnow(),
        )
        repo.insert_rows(
            schema.agent_opinions,
            [
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "role": "triage",
                    "payload": parsed,
                    "prompt_version": prompt.version,
                    "model_id": settings.model_screen,
                }
            ],
        )
        if verdict == "material":
            reason = f"triage material: {parsed.get('reason', '')}"
            manager.promote(symbol, as_of=as_of, reason=reason)
        triaged += 1
    return triaged


@dagster_asset(group_name=RESEARCH_GROUP, deps=["data_quality"])
def tier2_scan(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Tier-2 detectors -> flags -> Haiku triage -> Tier-3 promotions."""
    from dynamo.data.detector_windows import TIER2_SCAN_LOOKBACK_DAYS
    from dynamo.data.detectors import DetectorInputs, run_detectors
    from dynamo.domain.models.market import EarningsEvent, Filing, InsiderTx, NewsItem

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    tier2 = dynamo.universe_symbols(as_of, tiers=(2,))
    # 420 calendar days, NOT the 100 this used to pass. The detectors share
    # one frame and the longest lookback in the catalog is range_break's 252
    # sessions; at 100 days (~167 sessions) range_break reported 3-month
    # highs as 52-week breakouts (61% false on the live universe) and
    # range_bound — which gates at 131 sessions — could not fire at all.
    # The number is derived from the detectors themselves in
    # dynamo.data.detector_windows and guarded by a test there, so a
    # longer-lookback detector fails loudly instead of silently no-op'ing.
    bars = dynamo.bars_window(tier2 or None, as_of, TIER2_SCAN_LOOKBACK_DAYS)
    if bars.empty and not tier2:
        context.log.warning("tier2_scan: no bars and no tier-2 members — skipping")
        return MaterializeResult(metadata={"flags": 0, "skipped": "no data"})

    lookback = _since(as_of, 14)
    f, i, n, e = schema.filings, schema.insider_tx, schema.news, schema.earnings_events
    filings = _rows_to_models(Filing, _read_df(dynamo, select(f).where(f.c.filed_at >= lookback)))
    insiders = _rows_to_models(
        InsiderTx, _read_df(dynamo, select(i).where(i.c.filed_at >= lookback))
    )
    news_items = _rows_to_models(
        NewsItem, _read_df(dynamo, select(n).where(n.c.published_at >= lookback))
    )
    earnings = _rows_to_models(
        EarningsEvent,
        _read_df(dynamo, select(e).where(e.c.event_date >= as_of - timedelta(days=7))),
    )

    flags = run_detectors(
        DetectorInputs(
            bars=bars,
            as_of=utcnow(),
            earnings=earnings,
            filings=filings,
            insider_txs=insiders,
            news_items=news_items,
        )
    )
    # Persist FIRST, then triage the table. Triage no longer takes the flags
    # it should look at as an argument — it reads the un-triaged queue, which
    # now contains tonight's batch plus anything earlier runs left behind.
    inserted = repo.insert_surveillance_flags(flags)
    triaged = _triage_flags(context, dynamo, as_of)
    return MaterializeResult(metadata={"flags": inserted, "triaged_symbols": triaged})


# --------------------------------------------------------------------------
# quant features/scores, quality, strategies, regime — docs 04, 05, 06
# --------------------------------------------------------------------------

# Feature-frame lookback windows. These MIRROR the replay gateway's constants
# of the same names (``dynamo.replay.gateway``) — the two builders feed the
# SAME catalog, so a window that differs between them is a train/serve skew,
# not a tuning knob. ``tests/unit/orchestration/test_feature_inputs_parity.py``
# asserts each one equals the gateway's; they are duplicated rather than
# imported so the nightly pipeline carries no dependency on the replay
# composition root.
FEATURE_BARS_LOOKBACK_DAYS = 420
FEATURE_FUNDAMENTALS_LOOKBACK_DAYS = 500
FEATURE_INSIDER_LOOKBACK_DAYS = 200
FEATURE_MACRO_LOOKBACK_DAYS = 400
FEATURE_NEWS_LOOKBACK_DAYS = 365
FEATURE_SHORT_INTEREST_LOOKBACK_DAYS = 120
FEATURE_EARNINGS_LOOKAHEAD_DAYS = 30


def _feature_inputs(dynamo: DynamoResource, symbols: list[str], as_of: date) -> Any:
    """The nightly ``FeatureInputs``, field-for-field with the replay gateway.

    Every frame the gateway populates is populated here, over the same window
    — anything the gateway reads and this does not is a feature that trains
    and backtests fine and then returns NaN in production. Four features were
    exactly that until this builder learned to read ``news`` and
    ``short_interest`` (``news_count_60d``, ``unusual_volume_no_news``,
    ``short_interest_pct_float``, ``days_to_cover``, ``short_interest_change``).

    The bounds here are LOWER only: ``as_of`` is tonight, so "not after
    as_of" is free. The gateway adds the upper bound because it runs at a
    historical date. One bound is NOT free even here: short interest is
    filtered on ``publication_date``, never ``settlement_date``. FINRA
    disseminates a cycle about eight days after the settlement date it
    describes, so bounding on the settlement date would hand the features
    ~8 days of look-ahead in replay and silently disagree with training.
    """
    from dynamo.data.universe import default_sector_etf_map
    from dynamo.research.quant.catalog import FeatureInputs

    bars = dynamo.bars_window(symbols, as_of, FEATURE_BARS_LOOKBACK_DAYS)
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector, inst.c.cik))
    sectors = {
        str(r["symbol"]): str(r["sector"]) for r in inst_df.to_dict(orient="records") if r["sector"]
    }
    cik_to_symbol = {
        str(r["cik"]): str(r["symbol"]) for r in inst_df.to_dict(orient="records") if r["cik"]
    }
    x = schema.xbrl_facts
    fundamentals = _read_df(
        dynamo, select(x).where(x.c.filed_at >= _since(as_of, FEATURE_FUNDAMENTALS_LOOKBACK_DAYS))
    )
    if not fundamentals.empty:
        fundamentals["symbol"] = fundamentals["cik"].astype(str).map(cik_to_symbol.get)
        fundamentals = fundamentals.dropna(subset=["symbol"])
    i = schema.insider_tx
    insider = _read_df(
        dynamo, select(i).where(i.c.filed_at >= _since(as_of, FEATURE_INSIDER_LOOKBACK_DAYS))
    )
    m = schema.macro_series
    macro = _read_df(
        dynamo,
        select(m).where(m.c.ts >= as_of - timedelta(days=FEATURE_MACRO_LOOKBACK_DAYS)),
    )
    e = schema.earnings_events
    earnings = _read_df(
        dynamo,
        select(e.c.symbol, e.c.event_date).where(
            e.c.event_date >= as_of - timedelta(days=FEATURE_EARNINGS_LOOKAHEAD_DAYS)
        ),
    )
    ev = schema.earnings_estimate_vintages
    # Append-only vintages: the whole (small) table, exactly as the gateway
    # reads it. Estimate-revision features stay NaN until the first vintage
    # is collected — history that was never captured is not fabricated.
    estimates = _read_df(
        dynamo, select(ev.c.symbol, ev.c.event_date, ev.c.fetched_at, ev.c.eps_est)
    )
    nw = schema.news
    # ``news.symbols`` is an ARRAY column; the feature path wants one row per
    # (symbol, story), so explode here exactly as the gateway does. The full
    # year is needed even though ``news_count_60d`` reads 60 days:
    # ``unusual_volume_no_news`` uses the year to decide whether the feed
    # covers a symbol at all (catalog group 12).
    news = _read_df(
        dynamo,
        select(nw.c.symbols, nw.c.published_at).where(
            nw.c.published_at >= _since(as_of, FEATURE_NEWS_LOOKBACK_DAYS)
        ),
    )
    if not news.empty:
        news = news.explode("symbols").rename(columns={"symbols": "symbol"})
        news = news.dropna(subset=["symbol"])
    si = schema.short_interest
    # publication_date, NOT settlement_date — see the docstring. The 120-day
    # floor always contains the most recently published semi-monthly cycle
    # and keeps this off the full 2.4M-row table.
    short_interest = _read_df(
        dynamo,
        select(
            si.c.symbol,
            si.c.settlement_date,
            si.c.publication_date,
            si.c.shares_short,
            si.c.previous_shares_short,
            si.c.avg_daily_volume,
            si.c.days_to_cover,
        ).where(
            si.c.publication_date >= as_of - timedelta(days=FEATURE_SHORT_INTEREST_LOOKBACK_DAYS),
            si.c.publication_date <= as_of,
        ),
    )
    return FeatureInputs(
        bars=bars,
        fundamentals=fundamentals if not fundamentals.empty else None,
        insider=insider if not insider.empty else None,
        estimates=estimates if not estimates.empty else None,
        macro=macro if not macro.empty else None,
        earnings_calendar=earnings if not earnings.empty else None,
        news=news if not news.empty else None,
        short_interest=short_interest if not short_interest.empty else None,
        sector_map=default_sector_etf_map(sectors),
    )


@dagster_asset(group_name=RESEARCH_GROUP, deps=["data_quality"])
def quant_features_scores(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Features -> quant scores (Tier 1 + promoted) -> QuantScore rows."""
    from dynamo.research.quant.features import build_features
    from dynamo.research.quant.scoring import ModelBundle, score_universe

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    universe = dynamo.universe_symbols(as_of, tiers=(1, 3))
    if not universe:
        context.log.warning("quant_features_scores: empty active universe — skipping")
        return MaterializeResult(metadata={"scores": 0, "skipped": "empty universe"})
    inputs = _feature_inputs(dynamo, universe, as_of)
    features = build_features(inputs, as_of, universe)
    if features.empty:
        return MaterializeResult(metadata={"scores": 0, "skipped": "no features"})
    # Candidate/production split (backtest-campaign Phase 2): PRODUCTION
    # scoring loads ONLY a registry row promoted to status=production
    # (model_type='quant_bundle') via the promotion tool (doc 08 §7).
    # CANDIDATE bundles are replay-only, opted into explicitly with
    # `replay run --quant-model-version <id>` — they never score the nightly
    # pipeline. No production row -> the honest baseline-v0 abstention. A
    # promoted bundle that fails to load (missing/corrupt artifact, feature
    # drift) raises and fails the asset LOUDLY — an incident, never a silent
    # fallback to the heuristic.
    from pathlib import Path

    from dynamo.persistence.repositories.learning import ModelRegistryRepository
    from dynamo.research.quant.bundle import bundle_from_version

    production = ModelRegistryRepository(dynamo.engine()).latest_production("quant_bundle")
    if production is not None:
        artifact_dir = Path(production.artifact_path).parent if production.artifact_path else None
        bundle = bundle_from_version(production.id, artifact_dir)
        context.log.info("quant_features_scores: production bundle %s", production.id)
    else:
        bundle = ModelBundle(model_version="baseline-v0")
    scores = score_universe(bundle, features, as_of)
    rows = [s.model_dump(mode="json") for s in scores]
    written = repo.upsert_rows(schema.quant_scores, rows) if rows else 0
    return MaterializeResult(
        metadata={
            "scores": written,
            "features": len(features),
            "model_version": bundle.model_version,
        }
    )


@dagster_asset(group_name=RESEARCH_GROUP, deps=["data_quality"])
def quality_scores(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Durability/quality composites (doc 05) for the active universe."""
    from dynamo.research.quality import compute_quality_scores

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    universe = dynamo.universe_symbols(as_of, tiers=(1, 3))
    if not universe:
        return MaterializeResult(metadata={"scores": 0, "skipped": "empty universe"})
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector, inst.c.cik))
    sector_map = {
        str(r["symbol"]): str(r["sector"])
        for r in inst_df.to_dict(orient="records")
        if r["sector"] and str(r["symbol"]) in set(universe)
    }
    cik_to_symbol = {
        str(r["cik"]): str(r["symbol"]) for r in inst_df.to_dict(orient="records") if r["cik"]
    }
    x = schema.xbrl_facts
    fundamentals = _read_df(dynamo, select(x).where(x.c.filed_at >= _since(as_of, 800)))
    if not fundamentals.empty:
        fundamentals["symbol"] = fundamentals["cik"].astype(str).map(cik_to_symbol.get)
        fundamentals = fundamentals.dropna(subset=["symbol"])
    i = schema.insider_tx
    insider = _read_df(dynamo, select(i).where(i.c.filed_at >= _since(as_of, 200)))
    scores = compute_quality_scores(
        fundamentals=fundamentals if not fundamentals.empty else None,
        sector_map=sector_map,
        as_of=as_of,
        bars=dynamo.bars_window(universe, as_of, 400),
        insider=insider if not insider.empty else None,
    )
    rows = [s.model_dump(mode="json") for s in scores]
    written = repo.upsert_rows(schema.quality_scores, rows) if rows else 0
    return MaterializeResult(metadata={"scores": written})


@dagster_asset(group_name=RESEARCH_GROUP, deps=["quant_features_scores"])
def strategy_signals(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Evaluate the doc 05 strategy set over recent feature history -> signals."""
    from dynamo.research.quant.features import build_raw_features
    from dynamo.research.strategies import DEFAULT_STRATEGIES, evaluate_strategies

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    universe = dynamo.universe_symbols(as_of, tiers=(1, 3))
    if not universe:
        return MaterializeResult(metadata={"signals": 0, "skipped": "empty universe"})
    inputs = _feature_inputs(dynamo, universe, as_of)
    bars = inputs.bars
    if bars is None or bars.empty:
        return MaterializeResult(metadata={"signals": 0, "skipped": "no bars"})

    sessions = sorted(pd.to_datetime(bars["ts"]).dt.normalize().unique())[-10:]
    frames: list[pd.DataFrame] = []
    for ts in sessions:
        frame = build_raw_features(inputs, cast(pd.Timestamp, pd.Timestamp(ts)), universe)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ts"] = pd.Timestamp(ts)
        frame["symbol"] = frame.index if "symbol" not in frame.columns else frame["symbol"]
        frames.append(frame.reset_index(drop=True))
    if not frames:
        return MaterializeResult(metadata={"signals": 0, "skipped": "no features"})
    history = pd.concat(frames, ignore_index=True)

    # Only run strategies whose referenced features exist in tonight's frame —
    # skipped strategies are reported, never silently evaluated on NaNs.
    available = set(history.columns)
    runnable, skipped = [], []
    for strat in DEFAULT_STRATEGIES.values():
        referenced = {
            r.feature
            for r in (*strat.entry_rules, *strat.exit_rules, *strat.universe_filter.extra_rules)
        }
        (runnable if referenced <= available else skipped).append(strat)
    if skipped:
        context.log.warning(
            "strategy_signals: skipping %s (missing feature columns)",
            [s.strategy_id for s in skipped],
        )
    signals = evaluate_strategies(runnable, history, as_of) if runnable else []
    rows = [
        {k: v for k, v in s.model_dump(mode="json").items() if k in schema.signals.c}
        for s in signals
    ]
    written = repo.insert_rows(schema.signals, rows) if rows else 0
    return MaterializeResult(
        metadata={
            "signals": written,
            "strategies_run": len(runnable),
            "strategies_skipped": len(skipped),
        }
    )


@dagster_asset(group_name=RESEARCH_GROUP, deps=["data_quality"])
def regime_state(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Rule-based regime classification (doc 06 §8) from SPY + macro inputs."""
    import numpy as np

    from dynamo.domain.models.risk import RiskConfig
    from dynamo.risk.regime import RegimeHistory, latest_regime

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    spy = dynamo.bars_window(["SPY"], as_of, 500)
    if spy.empty:
        context.log.warning("regime_state: no SPY bars — skipping")
        return MaterializeResult(metadata={"skipped": "no SPY bars"})
    spy = spy.sort_values("ts")
    dates = tuple(_pydate(t) for t in spy["ts"])
    n = len(dates)

    m = schema.macro_series
    macro = _read_df(
        dynamo,
        select(m.c.series_id, m.c.ts, m.c.value).where(m.c.ts >= as_of - timedelta(days=900)),
    )

    def _aligned(series_id: str, default: float) -> Any:
        if macro.empty:
            return np.full(n, default)
        sub = cast(pd.DataFrame, macro[macro["series_id"] == series_id]).sort_values("ts")
        if sub.empty:
            return np.full(n, default)
        s = pd.Series(
            sub["value"].to_numpy(),
            index=pd.to_datetime(sub["ts"]),
        )
        idx = pd.to_datetime([d.isoformat() for d in dates])
        aligned = s.reindex(idx, method="ffill").fillna(default)
        return cast(Any, aligned).to_numpy(dtype=float)

    close = spy["close"].to_numpy(dtype=float)
    ma200 = np.asarray(pd.Series(close).rolling(200, min_periods=1).mean(), dtype=float)
    # breadth proxy until real Tier-1 breadth lands (doc 06 open item)
    breadth = np.clip((close / ma200 - 1.0) * 5.0 + 0.5, 0.0, 1.0)
    history = RegimeHistory(
        dates=dates,
        spy_close=close,
        hy_oas=_aligned("BAMLH0A0HYM2", 4.0),
        curve_slope=_aligned("T10Y3M", 1.0),
        breadth=breadth,
    )
    state = latest_regime(history, RiskConfig())
    if state is None:
        return MaterializeResult(metadata={"skipped": "insufficient history"})
    repo.upsert_rows(
        schema.regime_states,
        [
            {
                "as_of": state.as_of,
                "regime": state.regime.value,
                "composite_score": state.composite_score,
                "inputs": state.inputs,
                "sessions_in_state": state.sessions_in_state,
            }
        ],
    )
    return MaterializeResult(
        metadata={"regime": state.regime.value, "composite_score": state.composite_score}
    )


# --------------------------------------------------------------------------
# macro digest + prediction-market views (doc 03 §6)
# --------------------------------------------------------------------------


def _macro_context(dynamo: DynamoResource, as_of: date) -> str:
    m = schema.macro_series
    macro = _read_df(
        dynamo,
        select(m.c.series_id, m.c.ts, m.c.value).where(m.c.ts >= as_of - timedelta(days=120)),
    )
    if macro.empty:
        return "no macro series ingested"
    lines = []
    for series_id, grp in macro.sort_values("ts").groupby("series_id"):
        latest = grp.iloc[-1]
        first = grp.iloc[0]
        lines.append(
            f"{series_id}: {float(latest['value']):.3f}"
            f" (from {float(first['value']):.3f} over window)"
        )
    return "\n".join(sorted(lines))


def _cross_venue_digest(dynamo: DynamoResource, as_of: date) -> str:
    """Top-5 OddsPipe cross-venue divergences as context lines ('' when none)."""
    cv = schema.pm_cross_venue_spreads
    frame = _read_df(dynamo, select(cv).where(cv.c.ts >= _since(as_of, 3)))
    if frame.empty:
        return ""
    frame = frame.sort_values("ts").drop_duplicates(subset=["kalshi_market_id"], keep="last")
    frame = frame.assign(_abs_diff=frame["yes_diff"].abs()).sort_values(
        "_abs_diff", ascending=False
    )
    lines = ["Cross-venue divergences (OddsPipe; same question on both venues):"]
    for row in frame.head(5).to_dict(orient="records"):
        lines.append(
            f"- {row['question']}: kalshi yes {float(row['kalshi_yes']):.0%} vs "
            f"polymarket yes {float(row['polymarket_yes']):.0%} "
            f"(diff {float(row['yes_diff']):+.2f})"
        )
    return "\n".join(lines)


def _pm_digest(dynamo: DynamoResource, as_of: date) -> tuple[str, list[tuple[Any, float]]]:
    from dynamo.domain.models.market import PredictionMarket

    pm, pp = schema.prediction_markets, schema.prediction_market_prices
    cross_venue = _cross_venue_digest(dynamo, as_of)
    prices = _read_df(
        dynamo,
        select(pp).where(pp.c.ts >= _since(as_of, 3), pp.c.trusted.is_(True)),
    )
    if prices.empty:
        return cross_venue, []
    prices = (
        prices.sort_values("ts")
        .drop_duplicates(subset=["venue", "market_id"], keep="last")
        .reset_index(drop=True)
    )
    markets = _read_df(dynamo, select(pm))
    merged = prices.merge(markets, on=["venue", "market_id"])
    lines: list[str] = []
    tracked: list[tuple[Any, float]] = []
    for row in merged.sort_values("volume", ascending=False).head(8).to_dict(orient="records"):
        market = PredictionMarket.model_validate(_subset(PredictionMarket, row))
        implied = float(row["implied_prob"])
        lines.append(f"- {market.question}: implied {implied:.0%} ({market.venue})")
        tracked.append((market, implied))
    text = "\n".join(lines)
    if cross_venue:
        text = f"{text}\n\n{cross_venue}" if text else cross_venue
    return text, tracked


@dagster_asset(group_name=RESEARCH_GROUP)
def macro_digest(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Once-daily whole-market digest + prediction-market views (Sonnet)."""
    from dynamo.llm.schemas import SCHEMAS
    from dynamo.research.macro import build_macro_digest, build_prediction_market_views
    from dynamo.research.prompt_loader import load_prompt

    model = dynamo.research_model()
    if model is None:
        context.log.warning("macro_digest: ANTHROPIC_API_KEY unset — skipping")
        return MaterializeResult(metadata={"skipped": "no research model"})
    as_of = utcnow().date()
    repo = dynamo.market_repo()
    pm_text, tracked = _pm_digest(dynamo, as_of)
    prompt = load_prompt("macro_digest")
    digest = build_macro_digest(
        model,
        _macro_context(dynamo, as_of),
        pm_text,
        as_of,
        prompt=prompt,
        schema=SCHEMAS["macro_digest"],
    )
    repo.upsert_rows(
        schema.macro_digests,
        [{"as_of": as_of, "payload": digest.model_dump(mode="json")}],
    )
    repo.insert_rows(
        schema.agent_opinions,
        [
            {
                "symbol": "*",
                "as_of": as_of,
                "role": "macro",
                "payload": digest.model_dump(mode="json"),
                "prompt_version": prompt.version,
                "model_id": dynamo.settings().model_analyst,
            }
        ],
    )
    views = build_prediction_market_views(
        model,
        tracked,
        as_of,
        prompt=load_prompt("prediction_market"),
        schema=SCHEMAS["prediction_market"],
    )
    if views:
        repo.upsert_rows(
            schema.prediction_market_views,
            [
                {
                    "venue": v.venue,
                    "market_id": v.market_id,
                    "as_of": as_of,
                    "payload": v.model_dump(mode="json"),
                }
                for v in views
            ],
        )
    return MaterializeResult(metadata={"pm_views": len(views), "tracked_markets": len(tracked)})


# --------------------------------------------------------------------------
# deep-dive selection (doc 03 §1) — shared by the asset and the committee job
# --------------------------------------------------------------------------


def select_deep_dive_symbols(dynamo: DynamoResource, as_of: date) -> list[str]:
    from dynamo.research.selection import DeepDiveCandidate, select_deep_dives

    repo = dynamo.market_repo()
    active = dynamo.universe_symbols(as_of, tiers=(1, 3))
    if not active:
        return []
    active_set = set(active)

    promoted_today = {
        m.symbol for m in repo.universe_at(as_of) if int(m.tier) == 3 and m.valid_from == as_of
    }

    e = schema.earnings_events
    earnings = _read_df(
        dynamo,
        select(e.c.symbol, e.c.event_date).where(
            e.c.event_date >= as_of, e.c.event_date <= as_of + timedelta(days=4)
        ),
    )
    earnings_sessions = {
        str(r["symbol"]): int((_pydate(r["event_date"]) - as_of).days)
        for r in earnings.to_dict(orient="records")
    }

    f = schema.filings
    fresh_filings = {
        str(s)
        for s in _read_df(dynamo, select(f.c.symbol).where(f.c.filed_at >= _since(as_of, 1)))[
            "symbol"
        ].dropna()
    }

    sf = schema.surveillance_flags
    flags = _read_df(
        dynamo,
        select(sf.c.symbol, sf.c.detector, sf.c.details).where(sf.c.ts >= _since(as_of, 1)),
    )
    move_sigma: dict[str, float] = {}
    for r in flags.to_dict(orient="records"):
        if r["detector"] == "price_zscore":
            details = r["details"] or {}
            z = details.get("zscore", details.get("z", 0.0))
            try:
                move_sigma[str(r["symbol"])] = abs(float(z))
            except (TypeError, ValueError):
                continue

    verdicts = repo.latest_verdicts(active)
    verdict_age: dict[str, int] = {}
    if not verdicts.empty:
        for r in verdicts.to_dict(orient="records"):
            verdict_age[str(r["symbol"])] = (as_of - _pydate(r["as_of"])).days

    candidates = [
        DeepDiveCandidate(
            symbol=sym,
            newly_promoted=sym in promoted_today,
            earnings_in_sessions=earnings_sessions.get(sym),
            has_new_filing=sym in fresh_filings,
            move_sigma=move_sigma.get(sym, 0.0),
            last_verdict_age_days=verdict_age.get(sym),
        )
        for sym in sorted(active_set)
    ]
    settings = dynamo.settings()
    selected = select_deep_dives(
        candidates,
        settings.deep_dive_cap_per_night,
        # Without this reserve, ~18 promotions a night claimed all 8 slots
        # and the core universe was never re-examined (doc 03 §1 /
        # dynamo.research.funnel).
        promotion_slots=settings.deep_dive_promotion_slots,
        sectors=_instrument_sectors(dynamo),
        history=_deep_dive_history(dynamo, as_of, settings.replay_shortlist_cooldown_sessions * 4),
        cooldown_sessions=settings.replay_shortlist_cooldown_sessions,
        rotation_slots=settings.replay_shortlist_rotation_slots,
        rotation_lookback_sessions=settings.replay_shortlist_rotation_lookback_sessions,
        crypto_slots=settings.replay_shortlist_crypto_slots,
    )
    return [c.symbol for c in selected]


def _deep_dive_history(dynamo: DynamoResource, as_of: date, sessions: int) -> list[Any]:
    """Prior sessions and the symbols the committee debated on each."""
    v = schema.committee_verdicts
    frame = _read_df(
        dynamo,
        select(v.c.symbol, v.c.as_of).where(v.c.as_of >= as_of - timedelta(days=sessions)),
    )
    by_day: dict[date, list[str]] = {}
    for row in frame.to_dict(orient="records"):
        by_day.setdefault(_pydate(row["as_of"]), []).append(str(row["symbol"]))
    return [
        (day, sorted(set(by_day.get(day, []))))
        for day in (as_of - timedelta(days=n) for n in range(sessions, 0, -1))
    ]


@dagster_asset(group_name=RESEARCH_GROUP, deps=["tier2_scan", "quant_features_scores"])
def deep_dive_selection(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Tonight's deep-dive set: promotions > events > staleness, hard-capped."""
    as_of = utcnow().date()
    symbols = select_deep_dive_symbols(dynamo, as_of)
    context.log.info("deep dive selection for %s: %s", as_of, symbols)
    return MaterializeResult(metadata={"selected": len(symbols), "symbols": symbols})


# --------------------------------------------------------------------------
# universe lifecycle (doc 02 §2 exit paths) — nightly, code-only
# --------------------------------------------------------------------------


@dagster_asset(group_name=RESEARCH_GROUP, deps=["deep_dive_selection"])
def universe_lifecycle(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Close lapsed promotions, and file Tier-1 add/demote recommendations.

    Three things that existed in :class:`UniverseManager` but had no
    production caller, so none of them ever happened:

    - ``expire_promotions`` — Tier-3 windows lapsed by interval semantics,
      which is correct but invisible. Calling it records the expiry and
      defensively closes any promotion written without a ``valid_to``.
    - ``recommend_addition`` / ``recommend_demotion`` — the human-approval
      queue behind ``dashboard/pages/7_Ops.py`` and
      ``api/routers/admin.py`` is fed by these alerts and nothing else, so
      it was permanently empty. Tier 1 is the one thing the platform may
      never change on its own (doc 02 §2), so the ONLY way a discovered
      name can become permanent is for a recommendation to reach a human.

    Promotion criterion: a name whose Tier-3 window is closing and whose
    committee conviction held up across repeat dives has earned a *proposal*
    for core membership — sustained interest, not one good night. The
    demotion criterion is deliberately absent for now (see below).
    """
    from dynamo.data.universe import UniverseManager

    as_of = utcnow().date()
    settings = dynamo.settings()
    repo = dynamo.market_repo()
    manager = UniverseManager(
        repo,
        default_promotion_days=settings.tier3_promotion_window_days,
        include_crypto=bool(settings.crypto_enabled),
    )

    lapsed = manager.expire_promotions(as_of)

    # Sustained interest: promoted names the committee debated at least
    # twice inside the promotion window and never rated a SELL. Repeat
    # conviction is the signal; a single dive is not.
    window_start = as_of - timedelta(days=settings.tier3_promotion_window_days)
    promoted = {m.symbol for m in repo.universe_at(as_of) if int(m.tier) == 3}
    recommended: list[str] = []
    if promoted:
        v = schema.committee_verdicts
        verdicts = _read_df(
            dynamo,
            select(v.c.symbol, v.c.as_of, v.c.action).where(v.c.as_of >= window_start),
        )
        by_symbol: dict[str, list[str]] = {}
        for row in verdicts.to_dict(orient="records"):
            symbol = str(row["symbol"])
            if symbol in promoted:
                by_symbol.setdefault(symbol, []).append(str(row["action"]).upper())
        now = utcnow()
        for symbol, actions in sorted(by_symbol.items()):
            if len(actions) >= 2 and not any(a.startswith("SELL") for a in actions):
                manager.recommend_addition(
                    symbol,
                    f"{len(actions)} committee verdicts in the last "
                    f"{settings.tier3_promotion_window_days}d, none SELL",
                    now=now,
                )
                recommended.append(symbol)

    context.log.info(
        "universe lifecycle %s: %d promotions expired, %d core additions recommended",
        as_of,
        len(lapsed),
        len(recommended),
    )
    return MaterializeResult(
        metadata={
            "expired_promotions": len(lapsed),
            "recommended_additions": len(recommended),
            "symbols_recommended": recommended,
            "funnel": _funnel_budget(settings).diagnosis(),
        }
    )


def _funnel_budget(settings: Any) -> Any:
    from dynamo.research.funnel import FunnelBudget

    return FunnelBudget(
        triage_cap_per_night=settings.tier2_triage_cap_per_night,
        triage_material_rate=settings.triage_material_rate,
        deep_dive_cap_per_night=settings.deep_dive_cap_per_night,
        deep_dive_promotion_slots=settings.deep_dive_promotion_slots,
        promotion_window_days=settings.tier3_promotion_window_days,
    )


# --------------------------------------------------------------------------
# committee: stage-by-stage batches + sync fallback (doc 03 §3)
# --------------------------------------------------------------------------

# Journal embedder cached per process: LocalEmbedder() probes the optional
# sentence-transformers extra on construction (hash fallback otherwise).
_EMBEDDER: Any = None


def _journal_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from dynamo.llm.embeddings import LocalEmbedder

        _EMBEDDER = LocalEmbedder()
    return _EMBEDDER


def _write_journal_entries(dynamo: DynamoResource, entries: Sequence[Any]) -> int:
    """Embed + insert journal entries (idempotent on deterministic ids)."""
    from dynamo.persistence.repositories.learning import JournalRepository

    entries = list(entries)
    if not entries:
        return 0
    embedder = _journal_embedder()
    vectors = embedder.embed([e.text for e in entries])
    repo = JournalRepository(dynamo.engine())
    for entry, vector in zip(entries, vectors, strict=True):
        repo.insert(entry, vector, embedding_version=embedder.backend)
    return len(entries)


PRIOR_CASE_PREFIX = "Prior similar case (evidence, not rule): "


def _prior_case_line(exp: Any) -> str:
    """Compact evidence string for one prior experience. The 'evidence, not
    rule' guidance is embedded in the text itself so the committee prompt
    needs no change (plan Phase 2)."""
    conviction = "n/a" if exp.conviction is None else f"{float(exp.conviction):.2f}"
    pnl = "n/a" if exp.pnl_pct is None else f"{float(exp.pnl_pct):+.1f}%"
    regime = exp.regime or "unknown-regime"
    sector = exp.sector or "unknown-sector"
    action = exp.action or "unknown-action"
    outcome = exp.outcome_quality.value
    return (
        f"{PRIOR_CASE_PREFIX}{exp.symbol} {action} conviction {conviction} "
        f"in {regime}/{sector} -> {outcome}, pnl {pnl} ({exp.as_of.isoformat()})"
    )


def _prior_case_hits(
    exp_repo: Any,
    symbols: Sequence[str],
    as_of: date,
    *,
    top_k: int,
    sector_for: Mapping[str, str | None],
    regime: str | None,
) -> dict[str, list[tuple[str, str]]]:
    """Top-k prior similar experiences per symbol as (experience_id, evidence
    line). Pure over the injected repository; ``similar()`` hard-filters
    ``source='live'``, resolved, and strictly ``as_of <`` the query date."""
    hits: dict[str, list[tuple[str, str]]] = {}
    for sym in symbols:
        cases = exp_repo.similar(
            symbol=sym,
            as_of=as_of,
            sector=sector_for.get(sym),
            regime=regime,
            limit=top_k,
        )
        hits[sym] = [(c.id, _prior_case_line(c)) for c in cases]
    return hits


def _sectors_for(dynamo: DynamoResource, symbols: Sequence[str]) -> dict[str, str | None]:
    inst = schema.instruments
    frame = _read_df(
        dynamo, select(inst.c.symbol, inst.c.sector).where(inst.c.symbol.in_(list(symbols)))
    )
    return {
        str(r["symbol"]): (str(r["sector"]) if r["sector"] else None)
        for r in frame.to_dict(orient="records")
    }


# Dossier price history: 52-week range, 250-session returns and a 120-session
# beta need more than the 120 days the four original stats used (replay's
# PRICE_STATS_LOOKBACK_DAYS carries the same value — the paths stay identical).
DOSSIER_BARS_LOOKBACK_DAYS = 420
# Filed-fundamentals window for the dossier: a 3-year revenue CAGR spans
# sixteen filed quarters, which the 500-day feature window does not reach.
DOSSIER_FUNDAMENTALS_LOOKBACK_DAYS = 1500


def _all_sectors(dynamo: DynamoResource) -> dict[str, str]:
    """symbol -> GICS sector for every instrument that has one."""
    inst = schema.instruments
    frame = _read_df(dynamo, select(inst.c.symbol, inst.c.sector))
    return {
        str(r["symbol"]): str(r["sector"]) for r in frame.to_dict(orient="records") if r["sector"]
    }


def _dossier_fundamentals(dynamo: DynamoResource, symbols: Sequence[str], as_of: date) -> Any:
    """``xbrl_facts`` as ``[symbol, tag, period_end, value, filed_at]`` with
    ``filed_at <= as_of`` — the live twin of
    ``HistoricalDataGateway.fundamentals`` (same columns, same bounds, same
    lookback), so both dossier paths see the same fundamentals."""
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.cik))
    cik_to_symbol = {
        str(r["cik"]): str(r["symbol"]) for r in inst_df.to_dict(orient="records") if r["cik"]
    }
    wanted = set(symbols)
    ciks = [cik for cik, symbol in cik_to_symbol.items() if symbol in wanted]
    cols = ["symbol", "tag", "period_end", "value", "filed_at"]
    if not ciks:
        return pd.DataFrame(columns=pd.Index(cols))
    x = schema.xbrl_facts
    frame = _read_df(
        dynamo,
        select(x.c.cik, x.c.tag, x.c.period_end, x.c.value, x.c.filed_at).where(
            x.c.cik.in_(ciks),
            x.c.filed_at >= _since(as_of, DOSSIER_FUNDAMENTALS_LOOKBACK_DAYS),
            x.c.filed_at <= datetime.combine(as_of, time.max, tzinfo=UTC),
        ),
    )
    if frame.empty:
        return pd.DataFrame(columns=pd.Index(cols))
    frame["symbol"] = frame["cik"].astype(str).map(cik_to_symbol.get)
    return frame.dropna(subset=["symbol"]).loc[:, cols]


def _latest_closes(bars: Any) -> dict[str, float]:
    """symbol -> most recent close in an as-of bounded bars frame."""
    if bars is None or bars.empty:
        return {}
    last = bars.sort_values("ts").drop_duplicates(subset=["symbol"], keep="last")
    return {
        str(r["symbol"]): float(r["close"])
        for r in last.to_dict(orient="records")
        if pd.notna(r["close"])
    }


def _regime_at(dynamo: DynamoResource, as_of: date) -> str | None:
    rs = schema.regime_states
    frame = _read_df(
        dynamo,
        select(rs.c.regime).where(rs.c.as_of <= as_of).order_by(rs.c.as_of.desc()).limit(1),
    )
    return str(frame.iloc[0]["regime"]) if not frame.empty else None


def _journal_hits_for(
    dynamo: DynamoResource, symbols: Sequence[str], as_of: date
) -> dict[str, list[tuple[str, str]]]:
    """Similar-situation journal hits per symbol: (entry_id, text), best first,
    followed by prior-similar-case evidence lines from ``decision_experience``
    (plan Phase 2) under the same gate and top-k.

    Vector search filtered to entries written on/before ``as_of`` — the run's
    point-in-time, mandatory for replay correctness later. Deterministic for a
    fixed journal, so the batch stages that re-derive it agree.

    Vector search ABSTAINS on hash-fallback embeddings (see
    ``dynamo.learning.journal``): those vectors are random, and their hits
    would enter the committee prompt labelled "similar past situations". The
    prior-similar-case lines below are structured retrieval over
    ``decision_experience`` and use no embedding at all, so they keep flowing
    either way — the abstention removes the noise, not the evidence.
    """
    settings = dynamo.settings()
    if not settings.journal_retrieval_enabled or not symbols:
        return {sym: [] for sym in symbols}
    from dynamo.learning.journal import (
        HASH_FALLBACK_REMEDIATION,
        JournalFilters,
        drop_hash_fallback_entries,
        journal_retrieval_gate,
        rank_similar,
    )
    from dynamo.persistence.repositories.learning import JournalRepository

    repo = JournalRepository(dynamo.engine())
    top_k = settings.experience_retrieval_top_k
    cutoff = datetime.combine(as_of, time.max, tzinfo=UTC)
    allow_fallback = settings.journal_allow_hash_fallback
    embedder = _journal_embedder()
    gate = journal_retrieval_gate(
        query_backend=embedder.backend, allow_hash_fallback=allow_fallback
    )
    if gate.reason:
        _log.warning("journal_retrieval: %s", gate.reason)
    hits: dict[str, list[tuple[str, str]]] = {sym: [] for sym in symbols}
    if gate.allowed:
        queries = [f"{sym} investment thesis, lessons and post-trade reviews" for sym in symbols]
        vectors = embedder.embed(queries)
        dropped = 0
        for sym, vector in zip(symbols, vectors, strict=True):
            # Symbol filtered in SQL (not only boosted in rank_similar):
            # journal entries are symbol-scoped evidence, and a weak embedding
            # backend would otherwise return other tickers' entries as this
            # one's "similar situations".
            candidates = repo.search(
                embedding=vector,
                symbols=[sym],
                top_k=max(top_k * 4, top_k),
                as_of=cutoff,
                allow_hash_fallback=allow_fallback,
            )
            # The other half of the guard. The repository already excludes
            # fallback-stamped rows in SQL — this is the same rule applied
            # again in the one place that can COUNT what it removed, so a
            # regression there surfaces as a log line rather than as silence.
            candidates, n = drop_hash_fallback_entries(
                candidates,
                repo.embedding_versions([e.id for e, _ in candidates]),
                allow_hash_fallback=allow_fallback,
            )
            dropped += n
            ranked = rank_similar(candidates, JournalFilters(symbols=frozenset({sym})), top_k=top_k)
            hits[sym] = [(r.entry.id, r.entry.text) for r in ranked]
        if dropped:
            _log.warning(
                "journal_retrieval: dropped %d candidate(s) still carrying a "
                "hash-fallback vector — they are random, not similar. Fix: %s",
                dropped,
                HASH_FALLBACK_REMEDIATION,
            )
    # Prior similar cases (plan Phase 2): structured retrieval from resolved
    # live experiences, appended through the SAME journal_hits channel so the
    # dossier needs no new parameter.
    prior = _prior_case_hits(
        dynamo.experience_repo(),
        symbols,
        as_of,
        top_k=top_k,
        sector_for=_sectors_for(dynamo, symbols),
        regime=_regime_at(dynamo, as_of),
    )
    for sym in symbols:
        hits[sym].extend(prior.get(sym, []))
    return hits


def build_deep_dive_contexts(
    dynamo: DynamoResource, symbols: Sequence[str], as_of: date
) -> dict[str, str]:
    """Deterministic per-ticker dossier context (doc 03 §3, stable -> volatile)."""
    from dynamo.domain.models.market import Instrument, NewsItem
    from dynamo.research.dossier import assemble_dossier_context
    from dynamo.research.fundamentals import build_fundamentals
    from dynamo.research.market_stats import peer_context, peer_symbols, price_stats
    from dynamo.research.news_digest import partition_news

    inst_t = schema.instruments
    inst_df = _read_df(dynamo, select(inst_t).where(inst_t.c.symbol.in_(list(symbols))))
    instruments = {
        str(r["symbol"]): Instrument.model_validate(_subset(Instrument, r))
        for r in inst_df.to_dict(orient="records")
    }
    digest_payload = dynamo.market_repo().latest_macro_digest()
    macro_text = str((digest_payload or {}).get("narrative", ""))

    # Peers, sector ETFs and the benchmark ride along in the same bars read,
    # over the window the 52-week range / 250-session returns need.
    sectors = _all_sectors(dynamo)
    reference_symbols = peer_symbols(sectors, list(symbols))
    all_symbols = sorted(set(symbols) | set(reference_symbols))
    bars = dynamo.bars_window(all_symbols, as_of, DOSSIER_BARS_LOOKBACK_DAYS)
    facts = _dossier_fundamentals(dynamo, all_symbols, as_of)
    closes = _latest_closes(bars)
    n = schema.news
    news_df = _read_df(dynamo, select(n).where(n.c.published_at >= _since(as_of, 7)))

    # Latest earnings-call transcript per symbol within 30 days -> labeled,
    # truncated excerpt in the dossier context (the earnings-call reader
    # prompt guidance already covers transcripts; no prompt change needed).
    from dynamo.research.dossier import transcript_excerpt

    tr = schema.transcripts
    tr_df = _read_df(
        dynamo,
        select(tr.c.symbol, tr.c.event_date, tr.c.source, tr.c.content).where(
            tr.c.symbol.in_(list(symbols)),
            tr.c.event_date >= as_of - timedelta(days=30),
            tr.c.event_date <= as_of,
        ),
    )
    transcript_by_symbol: dict[str, str] = {}
    if not tr_df.empty:
        latest = tr_df.sort_values(["symbol", "event_date"]).drop_duplicates(
            subset=["symbol"], keep="last"
        )
        for r in latest.to_dict(orient="records"):
            transcript_by_symbol[str(r["symbol"])] = transcript_excerpt(
                str(r["content"]),
                symbol=str(r["symbol"]),
                event_date=_pydate(r["event_date"]).isoformat(),
                source=str(r["source"]),
            )

    journal_hits = _journal_hits_for(dynamo, symbols, as_of)

    contexts: dict[str, str] = {}
    for sym in symbols:
        instrument = instruments.get(sym) or Instrument(symbol=sym, name=sym)
        sector = instrument.sector or sectors.get(sym)
        sym_news: list[NewsItem] = []
        if not news_df.empty:
            for r in news_df.to_dict(orient="records"):
                if sym in (r.get("symbols") or []):
                    sym_news.append(NewsItem.model_validate(_subset(NewsItem, r)))
        primary_news, market_news = partition_news(
            sym_news, symbol=sym, name=instrument.name, max_items=10
        )
        contexts[sym] = assemble_dossier_context(
            instrument=instrument,
            quant_score=_quant_score_for(dynamo, sym),
            price_stats=price_stats(bars, sym),
            fundamentals=build_fundamentals(
                facts,
                symbol=sym,
                as_of=as_of,
                close=closes.get(sym),
                sector=sector,
                peer_closes=closes,
                peer_sectors=sectors,
            ),
            peers=peer_context(bars, sym, sector=sector),
            news_items=primary_news,
            market_news_items=market_news,
            macro_digest_text=macro_text,
            journal_hits=[text for _, text in journal_hits.get(sym, [])],
            earnings_call_excerpt=transcript_by_symbol.get(sym, ""),
        )
    return contexts


def _record_batch(dynamo: DynamoResource, batch_id: str, stage: str, request_count: int) -> None:
    dynamo.market_repo().upsert_rows(
        schema.batch_jobs,
        [
            {
                "batch_id": batch_id,
                "stage": stage,
                "submitted_at": utcnow(),
                "status": "submitted",
                "request_count": request_count,
            }
        ],
    )


def _resolve_batch(dynamo: DynamoResource, batch_id: str, status: str = "ended") -> None:
    with dynamo.engine().begin() as conn:
        conn.execute(
            update(schema.batch_jobs)
            .where(schema.batch_jobs.c.batch_id == batch_id)
            .values(status=status, resolved_at=utcnow())
        )


def _quarantine_failures(dynamo: DynamoResource, stage: str, failed: Iterable[str]) -> None:
    repo = dynamo.market_repo()
    for custom_id in failed:
        repo.add_quarantine(
            scope="llm_call",
            key=custom_id,
            reason=f"committee {stage} batch call failed/off-schema",
        )


def _persist_dossiers(
    dynamo: DynamoResource, parsed: Mapping[str, Mapping[str, Any]], as_of: date
) -> None:
    from dynamo.research.prompt_loader import load_prompt

    version = load_prompt("dossier_analyst").version
    model_id = dynamo.settings().model_analyst
    repo = dynamo.market_repo()
    repo.upsert_rows(
        schema.dossiers,
        [
            {
                "symbol": sym,
                "as_of": as_of,
                "payload": dict(payload),
                "prompt_version": version,
                "model_id": model_id,
            }
            for sym, payload in sorted(parsed.items())
        ],
    )
    _persist_opinions(dynamo, "dossier", parsed, as_of, version, model_id)


def _persist_opinions(
    dynamo: DynamoResource,
    role: str,
    parsed: Mapping[str, Mapping[str, Any]],
    as_of: date,
    prompt_version: str,
    model_id: str,
) -> None:
    dynamo.market_repo().insert_rows(
        schema.agent_opinions,
        [
            {
                "symbol": sym,
                "as_of": as_of,
                "role": role,
                "payload": dict(payload),
                "prompt_version": prompt_version,
                "model_id": model_id,
            }
            for sym, payload in sorted(parsed.items())
        ],
    )


def _persist_verdict(dynamo: DynamoResource, verdict: Any) -> None:
    from dynamo.learning.journal import thesis_entry

    dynamo.market_repo().upsert_rows(
        schema.committee_verdicts,
        [
            {
                "id": verdict.id,
                "symbol": verdict.symbol,
                "as_of": verdict.as_of,
                "action": verdict.action.value,
                "conviction": verdict.conviction,
                "disagreement_score": verdict.disagreement_score,
                "payload": verdict.model_dump(mode="json"),
            }
        ],
    )
    # Journal write path (doc 10 §6): every persisted verdict leaves a thesis
    # entry; the review/lesson entries follow in post_trade_reviews.
    _write_journal_entries(dynamo, [thesis_entry(verdict, utcnow())])


def _opinion_payloads(
    dynamo: DynamoResource, role: str, symbols: Sequence[str], as_of: date
) -> dict[str, dict[str, Any]]:
    ao = schema.agent_opinions
    frame = _read_df(
        dynamo,
        select(ao).where(ao.c.role == role, ao.c.as_of == as_of, ao.c.symbol.in_(list(symbols))),
    )
    out: dict[str, dict[str, Any]] = {}
    for r in frame.to_dict(orient="records"):
        out[str(r["symbol"])] = dict(r["payload"] or {})
    return out


def run_deep_dives_sync_impl(
    context: OpExecutionContext | AssetExecutionContext,
    dynamo: DynamoResource,
    symbols: Sequence[str],
    as_of: date,
) -> int:
    """Sequential 4-call deep dives — the small-count fallback path."""
    from dynamo.research.prompt_loader import load_prompt

    engine = dynamo.committee_engine()
    if engine is None:
        context.log.warning("run_deep_dives_sync: no research model — skipping")
        return 0
    contexts = build_deep_dive_contexts(dynamo, symbols, as_of)
    journal_ids = _journal_hits_for(dynamo, symbols, as_of)
    settings = dynamo.settings()
    completed = 0
    for sym in symbols:
        dossier, bull, bear, verdict = engine.run_deep_dive(
            sym,
            as_of,
            contexts[sym],
            quant_score=_quant_score_for(dynamo, sym),
            journal_context_used=tuple(eid for eid, _ in journal_ids.get(sym, [])),
        )
        _persist_dossiers(dynamo, {sym: dossier.model_dump(mode="json")}, as_of)
        _persist_opinions(
            dynamo,
            "bull",
            {sym: bull.model_dump(mode="json")},
            as_of,
            load_prompt("bull_advocate").version,
            settings.model_analyst,
        )
        _persist_opinions(
            dynamo,
            "bear",
            {sym: bear.model_dump(mode="json")},
            as_of,
            load_prompt("bear_advocate").version,
            settings.model_analyst,
        )
        _persist_opinions(
            dynamo,
            "arbiter",
            {sym: verdict.model_dump(mode="json")},
            as_of,
            load_prompt("arbiter").version,
            settings.model_arbiter,
        )
        _persist_verdict(dynamo, verdict)
        completed += 1
    return completed


@op
def submit_committee_deep_dives(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Nightly 23:30 ET: select the deep-dive set and submit the dossier batch.

    Small sets (<= SYNC_DEEP_DIVE_MAX) run synchronously — no batch round-trip.
    """
    as_of = utcnow().date()
    symbols = select_deep_dive_symbols(dynamo, as_of)
    if not symbols:
        context.log.info("committee: nothing selected for %s", as_of)
        return
    engine = dynamo.committee_engine()
    model = dynamo.research_model()
    if engine is None or model is None:
        context.log.warning("committee: ANTHROPIC_API_KEY unset — degraded quant-only night")
        return
    if len(symbols) <= SYNC_DEEP_DIVE_MAX:
        completed = run_deep_dives_sync_impl(context, dynamo, symbols, as_of)
        context.log.info("committee: %d deep dives completed synchronously", completed)
        return
    contexts = build_deep_dive_contexts(dynamo, symbols, as_of)
    requests = engine.build_batch_requests("dossier", contexts, as_of)
    batch_id = model.submit_batch(requests)
    _record_batch(dynamo, batch_id, "dossier", len(requests))
    context.log.info("committee: dossier batch %s submitted (%d requests)", batch_id, len(requests))


class CommitteeAdvanceConfig(Config):
    """Run config injected by batch_completion_sensor."""

    batch_id: str
    stage: str  # dossier | advocates | arbiter


@op
def advance_committee_stage(
    context: OpExecutionContext, config: CommitteeAdvanceConfig, dynamo: DynamoResource
) -> None:
    """Collect one finished batch stage, persist it, submit the next stage."""
    from dynamo.research.committee import CommitteeEngine, split_custom_id
    from dynamo.research.prompt_loader import load_prompt

    engine = dynamo.committee_engine()
    model = dynamo.research_model()
    if engine is None or model is None:
        context.log.warning("advance_committee_stage: no research model")
        return
    settings = dynamo.settings()
    results = list(model.batch_results(config.batch_id))
    if not results:
        context.log.warning("batch %s returned no results", config.batch_id)
        _resolve_batch(dynamo, config.batch_id, status="empty")
        return
    as_of = split_custom_id(results[0][0])[0]

    if config.stage == "dossier":
        parsed, failed = CommitteeEngine.parse_stage_results(results, "dossier")
        _quarantine_failures(dynamo, "dossier", failed)
        _persist_dossiers(dynamo, parsed, as_of)
        if parsed:
            from dynamo.research.committee import advocate_context_for

            contexts = build_deep_dive_contexts(dynamo, sorted(parsed), as_of)
            items = {
                sym: advocate_context_for(contexts[sym], payload) for sym, payload in parsed.items()
            }
            requests = engine.build_batch_requests("bull", items, as_of)
            requests += engine.build_batch_requests("bear", items, as_of)
            next_batch = model.submit_batch(requests)
            _record_batch(dynamo, next_batch, "advocates", len(requests))
            context.log.info("committee: advocates batch %s submitted", next_batch)

    elif config.stage == "advocates":
        bulls, failed_bull = CommitteeEngine.parse_stage_results(results, "bull")
        bears, failed_bear = CommitteeEngine.parse_stage_results(results, "bear")
        _quarantine_failures(dynamo, "advocates", [*failed_bull, *failed_bear])
        version = load_prompt("bull_advocate").version
        _persist_opinions(dynamo, "bull", bulls, as_of, version, settings.model_analyst)
        _persist_opinions(
            dynamo,
            "bear",
            bears,
            as_of,
            load_prompt("bear_advocate").version,
            settings.model_analyst,
        )
        ready = sorted(set(bulls) & set(bears))
        if ready:
            from dynamo.research.committee import arbiter_context_for

            dossier_payloads = _opinion_payloads(dynamo, "dossier", ready, as_of)
            contexts = build_deep_dive_contexts(dynamo, ready, as_of)
            items = {
                sym: arbiter_context_for(
                    contexts[sym],
                    dossier_payloads.get(sym, {}),
                    bulls[sym],
                    bears[sym],
                    _quant_score_for(dynamo, sym),
                )
                for sym in ready
            }
            requests = engine.build_batch_requests("arbiter", items, as_of)
            next_batch = model.submit_batch(requests)
            _record_batch(dynamo, next_batch, "arbiter", len(requests))
            context.log.info("committee: arbiter batch %s submitted", next_batch)

    elif config.stage == "arbiter":
        from dynamo.domain.models.research import AdvocateBrief, Side

        parsed, failed = CommitteeEngine.parse_stage_results(results, "arbiter")
        _quarantine_failures(dynamo, "arbiter", failed)
        symbols = sorted(parsed)
        bull_payloads = _opinion_payloads(dynamo, "bull", symbols, as_of)
        bear_payloads = _opinion_payloads(dynamo, "bear", symbols, as_of)
        # Re-derived deterministically — the same hits the dossier stage put
        # into the context, so the verdict records what was actually shown.
        journal_ids = _journal_hits_for(dynamo, symbols, as_of)
        arbiter_version = load_prompt("arbiter").version
        verdicts = 0
        for sym in symbols:
            if sym not in bull_payloads or sym not in bear_payloads:
                context.log.warning("arbiter result for %s lacks advocate briefs", sym)
                continue
            bull = AdvocateBrief.model_validate(
                _subset(AdvocateBrief, bull_payloads[sym], symbol=sym, side=Side.BULL)
            )
            bear = AdvocateBrief.model_validate(
                _subset(AdvocateBrief, bear_payloads[sym], symbol=sym, side=Side.BEAR)
            )
            verdict = CommitteeEngine.assemble_verdict(
                symbol=sym,
                as_of=as_of,
                arbiter_payload=parsed[sym],
                bull=bull,
                bear=bear,
                quant_score=_quant_score_for(dynamo, sym),
                prompt_version=arbiter_version,
                model_id=settings.model_arbiter,
                journal_context_used=tuple(eid for eid, _ in journal_ids.get(sym, [])),
            )
            _persist_opinions(
                dynamo,
                "arbiter",
                {sym: verdict.model_dump(mode="json")},
                as_of,
                arbiter_version,
                settings.model_arbiter,
            )
            _persist_verdict(dynamo, verdict)
            verdicts += 1
        context.log.info("committee: %d verdicts persisted for %s", verdicts, as_of)
    else:
        raise ValueError(f"unknown committee stage {config.stage!r}")

    _resolve_batch(dynamo, config.batch_id)


class SyncDeepDiveConfig(Config):
    symbols: list[str] = []


@op
def run_deep_dives_sync(
    context: OpExecutionContext, config: SyncDeepDiveConfig, dynamo: DynamoResource
) -> None:
    """Manual/sync path: run deep dives inline for an explicit symbol list."""
    as_of = utcnow().date()
    symbols = config.symbols or select_deep_dive_symbols(dynamo, as_of)[:SYNC_DEEP_DIVE_MAX]
    completed = run_deep_dives_sync_impl(context, dynamo, symbols, as_of)
    context.log.info("run_deep_dives_sync: %d deep dives for %s", completed, symbols)


@job
def committee_job() -> None:
    """23:30 ET: submit the dossier stage (or run small sets synchronously)."""
    submit_committee_deep_dives()


@job
def committee_advance_job() -> None:
    """Sensor-triggered: collect a finished stage and submit the next."""
    advance_committee_stage()


@job
def committee_sync_job() -> None:
    """Manual sync fallback for small deep-dive sets."""
    run_deep_dives_sync()


# --------------------------------------------------------------------------
# watchlist blend (doc 05 §4)
# --------------------------------------------------------------------------


@dagster_asset(
    name="watchlist",
    group_name=RESEARCH_GROUP,
    deps=["quant_features_scores", "quality_scores", "strategy_signals"],
)
def watchlist_asset(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Blend verdicts + quant + quality + signals into the ranked watchlist."""
    from dynamo.domain.models.research import (
        CommitteeVerdict,
        QualityScore,
        QuantScore,
        Signal,
    )
    from dynamo.research.watchlist import build_watchlist

    as_of = utcnow().date()
    repo = dynamo.market_repo()

    verdicts: list[CommitteeVerdict] = []
    vf = repo.latest_verdicts()
    if not vf.empty:
        verdicts = [CommitteeVerdict.model_validate(p) for p in vf["payload"]]

    quant: list[QuantScore] = []
    qf = repo.latest_quant_scores()
    if not qf.empty:
        qf = qf.copy()
        qf["as_of"] = qf["as_of"].map(lambda v: pd.Timestamp(v).date())
        quant = _rows_to_models(QuantScore, qf)

    qs = schema.quality_scores
    quality_df = _read_df(dynamo, select(qs).where(qs.c.as_of >= as_of - timedelta(days=7)))
    quality: list[QualityScore] = []
    if not quality_df.empty:
        quality_df["as_of"] = quality_df["as_of"].map(lambda v: pd.Timestamp(v).date())
        quality = _rows_to_models(QualityScore, quality_df)

    sg = schema.signals
    signals_df = _read_df(
        dynamo,
        select(sg).where(sg.c.as_of == as_of, sg.c.status == "active"),
    )
    signals: list[Signal] = []
    if not signals_df.empty:
        signals_df["as_of"] = signals_df["as_of"].map(lambda v: pd.Timestamp(v).date())
        signals = _rows_to_models(Signal, signals_df)

    rows = build_watchlist(
        as_of=as_of,
        verdicts=verdicts,
        quant_scores=quant,
        quality_scores=quality,
        signals=signals,
    )
    written = 0
    if rows:
        written = repo.upsert_rows(
            schema.watchlist,
            [
                {k: v for k, v in r.model_dump(mode="json").items() if k in schema.watchlist.c}
                for r in rows
            ],
        )
    return MaterializeResult(metadata={"rows": written, "verdicts": len(verdicts)})


RESEARCH_ASSETS = [
    data_quality,
    tier2_scan,
    quant_features_scores,
    quality_scores,
    strategy_signals,
    regime_state,
    macro_digest,
    deep_dive_selection,
    universe_lifecycle,
    watchlist_asset,
]
COMMITTEE_JOBS = [committee_job, committee_advance_job, committee_sync_job]
