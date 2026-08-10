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
    # No promoted ML models yet -> heuristic bundle; the registry swaps this in
    # via the promotion tool (doc 08 §7), not code changes here.
    bundle = ModelBundle(model_version="baseline-v0")
    scores = score_universe(bundle, features, as_of)
    rows = [s.model_dump(mode="json") for s in scores]
    written = repo.upsert_rows(schema.quant_scores, rows) if rows else 0
    return MaterializeResult(metadata={"scores": written, "features": len(features)})


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


def build_deep_dive_contexts(
    dynamo: DynamoResource, symbols: Sequence[str], as_of: date
) -> dict[str, str]:
    """Deterministic per-ticker dossier context (doc 03 §3, stable -> volatile)."""
    from dynamo.domain.models.market import Instrument, NewsItem
    from dynamo.research.dossier import assemble_dossier_context

    inst_t = schema.instruments
    inst_df = _read_df(dynamo, select(inst_t).where(inst_t.c.symbol.in_(list(symbols))))
    instruments = {
        str(r["symbol"]): Instrument.model_validate(_subset(Instrument, r))
        for r in inst_df.to_dict(orient="records")
    }
    md = schema.macro_digests
    digest_df = _read_df(dynamo, select(md).order_by(md.c.as_of.desc()).limit(1))
    macro_text = ""
    if not digest_df.empty:
        macro_text = str((digest_df.iloc[0]["payload"] or {}).get("narrative", ""))

    bars = dynamo.bars_window(list(symbols), as_of, 120)
    n = schema.news
    news_df = _read_df(dynamo, select(n).where(n.c.published_at >= _since(as_of, 7)))

    contexts: dict[str, str] = {}
    for sym in symbols:
        instrument = instruments.get(sym) or Instrument(symbol=sym, name=sym)
        price_stats: dict[str, float] = {}
        sym_bars = bars[bars["symbol"] == sym].sort_values("ts") if not bars.empty else bars
        if sym_bars is not None and not sym_bars.empty:
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
        sym_news: list[NewsItem] = []
        if not news_df.empty:
            for r in news_df.to_dict(orient="records"):
                if sym in (r.get("symbols") or []):
                    sym_news.append(NewsItem.model_validate(_subset(NewsItem, r)))
        contexts[sym] = assemble_dossier_context(
            instrument=instrument,
            quant_score=_quant_score_for(dynamo, sym),
            price_stats=price_stats,
            news_items=sym_news[:10],
            macro_digest_text=macro_text,
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
    settings = dynamo.settings()
    completed = 0
    for sym in symbols:
        dossier, bull, bear, verdict = engine.run_deep_dive(
            sym, as_of, contexts[sym], quant_score=_quant_score_for(dynamo, sym)
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
