"""Research assets + committee batch jobs — docs 02 §4, 03 §1/§3.

Thin wiring: read frames via repositories, call core functions
(detectors, quality checks, quant scoring, selection, committee engine),
persist results, return metadata. Guardrails and all business logic live in
``src/dynamo``; missing configuration degrades per-asset, never all-or-nothing.
"""

import json
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


def _triage_flags(
    context: AssetExecutionContext | OpExecutionContext,
    dynamo: DynamoResource,
    flags: Sequence[Any],
    as_of: date,
) -> int:
    """Haiku triage over flagged symbols; material -> Tier-3 promotion."""
    model = dynamo.research_model()
    if model is None or not flags:
        if flags:
            context.log.warning("tier2 triage skipped: ANTHROPIC_API_KEY unset")
        return 0
    from dynamo.data.universe import UniverseManager
    from dynamo.llm.schemas import TRIAGE_SCHEMA
    from dynamo.research.prompt_loader import load_prompt

    settings = dynamo.settings()
    repo = dynamo.market_repo()
    prompt = load_prompt("triage")
    manager = UniverseManager(repo)

    by_symbol: dict[str, list[Any]] = {}
    for f in sorted(flags, key=lambda f: -f.severity):
        by_symbol.setdefault(f.symbol, []).append(f)
    triaged = 0
    for symbol, sym_flags in list(by_symbol.items())[: settings.tier2_triage_cap_per_night]:
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
        now = utcnow()
        with dynamo.engine().begin() as conn:
            conn.execute(
                update(schema.surveillance_flags)
                .where(
                    schema.surveillance_flags.c.symbol == symbol,
                    schema.surveillance_flags.c.triage_verdict.is_(None),
                )
                .values(triage_verdict=verdict, triaged_at=now)
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
    from dynamo.data.detectors import DetectorInputs, run_detectors
    from dynamo.domain.models.market import EarningsEvent, Filing, InsiderTx, NewsItem

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    tier2 = dynamo.universe_symbols(as_of, tiers=(2,))
    bars = dynamo.bars_window(tier2 or None, as_of, 100)
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
    inserted = repo.insert_surveillance_flags(flags)
    triaged = _triage_flags(context, dynamo, flags, as_of)
    return MaterializeResult(metadata={"flags": inserted, "triaged_symbols": triaged})


# --------------------------------------------------------------------------
# quant features/scores, quality, strategies, regime — docs 04, 05, 06
# --------------------------------------------------------------------------


def _feature_inputs(dynamo: DynamoResource, symbols: list[str], as_of: date) -> Any:
    from dynamo.data.universe import default_sector_etf_map
    from dynamo.research.quant.catalog import FeatureInputs

    bars = dynamo.bars_window(symbols, as_of, 420)
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector, inst.c.cik))
    sectors = {
        str(r["symbol"]): str(r["sector"]) for r in inst_df.to_dict(orient="records") if r["sector"]
    }
    cik_to_symbol = {
        str(r["cik"]): str(r["symbol"]) for r in inst_df.to_dict(orient="records") if r["cik"]
    }
    x = schema.xbrl_facts
    fundamentals = _read_df(dynamo, select(x).where(x.c.filed_at >= _since(as_of, 500)))
    if not fundamentals.empty:
        fundamentals["symbol"] = fundamentals["cik"].astype(str).map(cik_to_symbol.get)
        fundamentals = fundamentals.dropna(subset=["symbol"])
    i = schema.insider_tx
    insider = _read_df(dynamo, select(i).where(i.c.filed_at >= _since(as_of, 200)))
    m = schema.macro_series
    macro = _read_df(dynamo, select(m).where(m.c.ts >= as_of - timedelta(days=400)))
    e = schema.earnings_events
    earnings = _read_df(
        dynamo,
        select(e.c.symbol, e.c.event_date).where(e.c.event_date >= as_of - timedelta(days=30)),
    )
    return FeatureInputs(
        bars=bars,
        fundamentals=fundamentals if not fundamentals.empty else None,
        insider=insider if not insider.empty else None,
        macro=macro if not macro.empty else None,
        earnings_calendar=earnings if not earnings.empty else None,
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
        artifact_dir = (
            Path(production.artifact_path).parent if production.artifact_path else None
        )
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
    selected = select_deep_dives(candidates, dynamo.settings().deep_dive_cap_per_night)
    return [c.symbol for c in selected]


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
        str(r["symbol"]): str(r["sector"])
        for r in frame.to_dict(orient="records")
        if r["sector"]
    }


def _dossier_fundamentals(
    dynamo: DynamoResource, symbols: Sequence[str], as_of: date
) -> Any:
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

    Vector search (hash-fallback embedder keeps this functional without the
    embeddings extra) filtered to entries written on/before ``as_of`` — the
    run's point-in-time, mandatory for replay correctness later. Deterministic
    for a fixed journal, so the batch stages that re-derive it agree.
    """
    settings = dynamo.settings()
    if not settings.journal_retrieval_enabled or not symbols:
        return {sym: [] for sym in symbols}
    from dynamo.learning.journal import JournalFilters, rank_similar
    from dynamo.persistence.repositories.learning import JournalRepository

    repo = JournalRepository(dynamo.engine())
    top_k = settings.experience_retrieval_top_k
    cutoff = datetime.combine(as_of, time.max, tzinfo=UTC)
    queries = [f"{sym} investment thesis, lessons and post-trade reviews" for sym in symbols]
    vectors = _journal_embedder().embed(queries)
    hits: dict[str, list[tuple[str, str]]] = {}
    for sym, vector in zip(symbols, vectors, strict=True):
        candidates = repo.search(embedding=vector, top_k=max(top_k * 4, top_k), as_of=cutoff)
        ranked = rank_similar(
            candidates, JournalFilters(symbols=frozenset({sym})), top_k=top_k
        )
        hits[sym] = [(r.entry.id, r.entry.text) for r in ranked]
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
    watchlist_asset,
]
COMMITTEE_JOBS = [committee_job, committee_advance_job, committee_sync_job]
