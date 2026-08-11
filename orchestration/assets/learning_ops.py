"""Learning-loop, metrics, export, and dead-man-switch wiring — docs 10, 12.

Reviews respect the Claude budget (the cost meter refuses submissions at cap);
everything else here is LLM-free and runs in degraded mode too.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    OpExecutionContext,
    job,
    op,
)
from dagster import asset as dagster_asset
from sqlalchemy import func, select

from dynamo.persistence import schema
from orchestration.assets.portfolio_exec import _latest_prices, _portfolio_state
from orchestration.assets.research import _read_df, _since, _write_journal_entries
from orchestration.resources import DynamoResource, utcnow

LEARNING_GROUP = "learning"

HEALTHCHECK_URL_ENV = "HEALTHCHECK_URL"
MODELED_SLIPPAGE_BPS = 10.0  # doc 08 cost model baseline; drift monitor compares realized

AGENT_ROLES = ("bull", "bear", "arbiter")


def _trade_id(closed: Any) -> str:
    """Deterministic closed-trade id shared by reviews + completeness check."""
    return f"{closed.symbol}-{closed.closed_at.date().isoformat()}"


def _decision_context_for(closed: Any, trades_repo: Any, orders_repo: Any, dc_repo: Any) -> Any:
    """The exit fill's decision context via its order (the repaired dc-id join)."""
    fills = [f for f in trades_repo.fills(closed.symbol) if f.filled_at <= closed.closed_at]
    for fill in reversed(fills):
        order = orders_repo.get(fill.client_order_id)
        if order is not None and order.decision_context_id:
            dc = dc_repo.get(order.decision_context_id)
            if dc is not None:
                return dc
    return None


@dagster_asset(group_name=LEARNING_GROUP)
def post_trade_reviews(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Closed trades without a review -> Opus review -> post_trade_reviews rows."""
    from dynamo.learning.journal import lesson_entries, review_entry
    from dynamo.learning.reviews import TradeOutcome, build_review_request, parse_review
    from dynamo.llm.cost_meter import BudgetExceededError
    from dynamo.llm.schemas import strict_schema
    from dynamo.persistence.repositories.execution import (
        OrderRepository,
        TradeRepository,
    )
    from dynamo.persistence.repositories.learning import (
        DecisionContextRepository,
        ReviewRepository,
    )
    from dynamo.research.prompt_loader import load_prompt

    model = dynamo.research_model()
    if model is None:
        context.log.warning("post_trade_reviews: ANTHROPIC_API_KEY unset — skipping")
        return MaterializeResult(metadata={"skipped": "no research model", "reviews": 0})
    from dynamo.domain.models.learning import PostTradeReview

    engine = dynamo.engine()
    trades_repo = TradeRepository(engine)
    orders_repo = OrderRepository(engine)
    reviews_repo = ReviewRepository(engine)
    dc_repo = DecisionContextRepository(engine)
    reviewed = reviews_repo.reviewed_trade_ids()
    prompt = load_prompt("post_trade_review")
    review_schema = strict_schema(PostTradeReview)
    settings = dynamo.settings()

    written = 0
    skipped = 0
    journal_written = 0
    for closed in trades_repo.closed_trades():
        trade_id = _trade_id(closed)
        if trade_id in reviewed:
            continue
        # Find the exit fill's decision context via its order.
        dc = _decision_context_for(closed, trades_repo, orders_repo, dc_repo)
        if dc is None:
            context.log.warning("post_trade_reviews: no decision context for %s", trade_id)
            skipped += 1
            continue
        outcome = TradeOutcome(
            trade_id=trade_id,
            symbol=closed.symbol,
            pnl_pct=closed.pnl_pct,
            holding_days=closed.holding_days,
            entry_price=closed.entry_avg_price,
            exit_price=closed.exit_avg_price,
            closed_at=closed.closed_at,
        )
        request = build_review_request(dc, outcome)
        import json as _json

        try:
            parsed = model.complete_json(
                role="post_trade_review",
                system=prompt.system,
                context=_json.dumps(request, sort_keys=True, default=str),
                instruction=f"Produce the post-trade review JSON for trade {trade_id}.",
                schema=review_schema,
                model_tier="arbiter",
                max_tokens=3072,
            )
        except BudgetExceededError:
            context.log.warning("post_trade_reviews: Claude budget reached — stopping")
            break
        payload = {
            **parsed,
            "trade_id": trade_id,
            "decision_context_id": dc.id,
            "symbol": closed.symbol,
            "as_of": utcnow().date(),
            "prompt_version": prompt.version,
            "model_id": settings.model_arbiter,
            "outcome_pnl_pct": closed.pnl_pct,
            "holding_days": closed.holding_days,
        }
        review = parse_review(payload, outcome)
        reviews_repo.insert(review)
        written += 1
        # Journal write path (doc 10 §6): each review leaves a review entry
        # plus one entry per extracted lesson, embedded for retrieval.
        now = utcnow()
        journal_written += _write_journal_entries(
            dynamo,
            [
                review_entry(review, now, strategy=dc.sleeve.value),
                *lesson_entries(review, now, strategy=dc.sleeve.value),
            ],
        )
    return MaterializeResult(
        metadata={"reviews": written, "skipped": skipped, "journal_entries": journal_written}
    )


def _date_key(value: Any) -> str:
    """Normalize date-ish values (date, Timestamp, str) to an ISO day string."""
    return str(pd.Timestamp(value).date())


def _cum_returns_pct(
    dynamo: DynamoResource, symbol: str, start: datetime, horizon_days: int
) -> list[float]:
    """Daily cumulative returns (%) from the first close on/after ``start``,
    over the next ``horizon_days`` sessions — the path trade_outcome_achieved
    walks. Empty when bars are missing."""
    b = schema.bars_1d
    end = start + timedelta(days=horizon_days * 2 + 10)  # calendar buffer over sessions
    frame = _read_df(
        dynamo,
        select(b.c.ts, b.c.close)
        .where(b.c.symbol == symbol, b.c.ts >= start, b.c.ts <= end)
        .order_by(b.c.ts),
    )
    if len(frame) < 2:
        return []
    closes = [float(x) for x in frame["close"]]
    base = closes[0]
    if base <= 0:
        return []
    return [(c / base - 1.0) * 100.0 for c in closes[1 : horizon_days + 1]]


def _outcome_for(
    row: Mapping[str, Any],
    cum_returns: Callable[[str, datetime, int], Sequence[float]],
) -> bool:
    """Path-based outcome where the decision context carries real committee
    expectations (post-denormalization rows); otherwise the naive sign of
    realized pnl — pre-repair rows hardcoded (0, 0) expectations, for which
    the doc 10 §5 path definition is undefined."""
    from dynamo.learning.calibration import trade_outcome_achieved

    upside = float(row.get("expected_upside_pct") or 0.0)
    downside = float(row.get("expected_downside_pct") or 0.0)
    horizon = int(row.get("horizon_days") or 0)
    if upside > 0.0 and downside != 0.0 and horizon > 0:
        path = list(cum_returns(str(row["symbol"]), row["created_at"], horizon))
        if path:
            return trade_outcome_achieved(path, upside, downside, horizon)
    payload = row.get("payload") or {}
    return float(payload.get("outcome_pnl_pct", 0.0)) > 0.0


def _calibration_observations(
    merged: Sequence[Mapping[str, Any]],
    opinions: Sequence[Mapping[str, Any]],
    outcome_for: Callable[[Mapping[str, Any]], bool],
) -> list[Any]:
    """Observations per reviewed trade: committee/arbiter (as before), plus
    strategy/<sleeve> from the decision context and agent/<role> for each
    bull/bear/arbiter opinion of the underlying verdict (same outcome)."""
    from dynamo.learning.calibration import CalibrationObservation

    opinions_by_key: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for op_row in opinions:
        payload = op_row.get("payload") or {}
        conviction = payload.get("conviction")
        if conviction is None:
            continue
        key = (str(op_row["symbol"]), _date_key(op_row["as_of"]))
        opinions_by_key.setdefault(key, []).append(
            (str(op_row["role"]), max(0.0, min(1.0, float(conviction))))
        )

    observations: list[Any] = []
    for row in merged:
        conviction = row.get("conviction")
        if conviction is None or pd.isna(conviction):
            continue
        outcome = outcome_for(row)
        conviction = float(conviction)
        observations.append(
            CalibrationObservation(
                scope_type="committee", scope_id="arbiter", conviction=conviction, outcome=outcome
            )
        )
        sleeve = row.get("sleeve")
        if sleeve and not pd.isna(sleeve):
            observations.append(
                CalibrationObservation(
                    scope_type="strategy",
                    scope_id=str(sleeve),
                    conviction=conviction,
                    outcome=outcome,
                )
            )
        verdict_key = (str(row.get("symbol_v")), _date_key(row.get("as_of")))
        for role, agent_conviction in opinions_by_key.get(verdict_key, []):
            observations.append(
                CalibrationObservation(
                    scope_type="agent",
                    scope_id=role,
                    conviction=agent_conviction,
                    outcome=outcome,
                )
            )
    return observations


@dagster_asset(group_name=LEARNING_GROUP, deps=["post_trade_reviews"])
def calibration_update(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Conviction vs realized trade outcomes -> calibration_points, per scope
    (committee/arbiter + strategy/<sleeve> + agent/<role>)."""
    from dynamo.learning.calibration import compute_calibration
    from dynamo.persistence.repositories.learning import CalibrationRepository

    as_of = utcnow().date()
    r, d, v = schema.post_trade_reviews, schema.decision_context, schema.committee_verdicts
    reviews = _read_df(dynamo, select(r.c.trade_id, r.c.decision_context_id, r.c.payload))
    if reviews.empty:
        return MaterializeResult(metadata={"observations": 0, "skipped": "no reviews"})
    contexts = _read_df(
        dynamo,
        select(
            d.c.id,
            d.c.verdict_id,
            d.c.symbol,
            d.c.sleeve,
            d.c.created_at,
            d.c.expected_upside_pct,
            d.c.expected_downside_pct,
            d.c.horizon_days,
        ),
    )
    verdicts = _read_df(dynamo, select(v.c.id, v.c.conviction, v.c.symbol, v.c.as_of))
    merged = reviews.merge(
        contexts, left_on="decision_context_id", right_on="id", how="left"
    ).merge(verdicts, left_on="verdict_id", right_on="id", how="left", suffixes=("", "_v"))
    ao = schema.agent_opinions
    opinions = _read_df(
        dynamo,
        select(ao.c.symbol, ao.c.as_of, ao.c.role, ao.c.payload).where(
            ao.c.role.in_(AGENT_ROLES)
        ),
    )
    observations = _calibration_observations(
        merged.to_dict(orient="records"),
        opinions.to_dict(orient="records"),
        lambda row: _outcome_for(
            row, lambda sym, start, horizon: _cum_returns_pct(dynamo, sym, start, horizon)
        ),
    )
    if not observations:
        return MaterializeResult(metadata={"observations": 0, "skipped": "no linked verdicts"})
    points = compute_calibration(observations, as_of)
    CalibrationRepository(dynamo.engine()).upsert(points)
    return MaterializeResult(metadata={"observations": len(observations), "points": len(points)})


@dagster_asset(group_name=LEARNING_GROUP)
def drift_monitors(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Slippage / incident-trend / IC-decay monitors -> improvement_proposals."""
    from dynamo.learning.drift import (
        ic_decay_monitor,
        incident_trend_monitor,
        slippage_monitor,
    )
    from dynamo.persistence.repositories.learning import ProposalRepository

    as_of = utcnow().date()
    now = utcnow()
    proposals = []

    t = schema.trades
    fills = _read_df(
        dynamo,
        select(t.c.slippage_bps)
        .where(t.c.slippage_bps.isnot(None))
        .order_by(t.c.filled_at.desc())
        .limit(200),
    )
    if len(fills) >= 20:
        realized = [float(x) for x in fills["slippage_bps"]]
        modeled = [MODELED_SLIPPAGE_BPS] * len(realized)
        p = slippage_monitor(realized, modeled, as_of, now)
        if p is not None:
            proposals.append(p)

    a = schema.alerts
    provider_alerts = _read_df(
        dynamo,
        select(a.c.created_at).where(
            a.c.name.like("provider%"), a.c.created_at >= _since(as_of, 30)
        ),
    )
    if not provider_alerts.empty:
        counts = pd.to_datetime(provider_alerts["created_at"]).dt.date.value_counts().sort_index()
        daily = [float(counts.get(as_of - timedelta(days=i), 0.0)) for i in range(29, -1, -1)]
        p = incident_trend_monitor(daily, "all-providers", as_of, now)
        if p is not None:
            proposals.append(p)

    m = schema.metrics_daily
    ic_rows = _read_df(
        dynamo,
        select(m.c.scope_id, m.c.as_of, m.c.value)
        .where(m.c.metric == "quant_ic", m.c.scope_type == "model")
        .order_by(m.c.as_of),
    )
    mv = schema.model_versions
    promoted = _read_df(dynamo, select(mv.c.id, mv.c.metrics).where(mv.c.status == "production"))
    if not ic_rows.empty and not promoted.empty:
        for row in promoted.to_dict(orient="records"):
            model_id = str(row["id"])
            promo_ic = (row["metrics"] or {}).get("ic")
            series = ic_rows[ic_rows["scope_id"] == model_id]["value"].tolist()
            if promo_ic is None or not series:
                continue
            p = ic_decay_monitor([float(x) for x in series], float(promo_ic), model_id, as_of, now)
            if p is not None:
                proposals.append(p)

    repo = ProposalRepository(dynamo.engine())
    for proposal in proposals:
        repo.insert(proposal)
    return MaterializeResult(metadata={"proposals": len(proposals)})


@dagster_asset(name="metrics_daily", group_name=LEARNING_GROUP)
def metrics_daily_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Nightly platform metrics rollup (doc 12 §6 catalog, DB-backed subset)."""
    from dynamo.persistence.repositories.learning import MetricsRepository

    as_of = utcnow().date()
    repo = MetricsRepository(dynamo.engine())

    def _count(stmt: object) -> float:
        with dynamo.engine().connect() as conn:
            return float(conn.execute(stmt).scalar() or 0)  # type: ignore[arg-type]

    v, sf, q, w = (
        schema.committee_verdicts,
        schema.surveillance_flags,
        schema.quarantine,
        schema.watchlist,
    )
    metrics: dict[str, float] = {
        "verdicts_today": _count(select(func.count()).select_from(v).where(v.c.as_of == as_of)),
        "flags_24h": _count(
            select(func.count()).select_from(sf).where(sf.c.ts >= _since(as_of, 1))
        ),
        "quarantine_open": _count(
            select(func.count()).select_from(q).where(q.c.resolved_at.is_(None))
        ),
        "watchlist_rows": _count(select(func.count()).select_from(w).where(w.c.as_of == as_of)),
        "claude_mtd_usd": dynamo.cost_meter().month_to_date_usd(),
        "claude_budget_utilization": dynamo.cost_meter().budget_utilization(),
    }
    verdicts_30d = _read_df(
        dynamo, select(v.c.action).where(v.c.as_of >= as_of - timedelta(days=30))
    )
    if not verdicts_30d.empty:
        metrics["abstention_rate_30d"] = float((verdicts_30d["action"] == "no_action").mean())
    for name, value in metrics.items():
        repo.upsert_metric(name, "platform", "all", as_of, value)
    return MaterializeResult(metadata={k: v for k, v in metrics.items()})


@dagster_asset(name="equity_curve", group_name=LEARNING_GROUP)
def equity_curve_asset(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Mark-to-market equity snapshot -> equity_curve row."""
    from dynamo.persistence.repositories.execution import EquityCurveRepository

    as_of = utcnow().date()
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("equity_curve: no portfolio state — skipping")
        return MaterializeResult(metadata={"skipped": "no portfolio state"})
    prices = _latest_prices(dynamo, sorted({p.symbol for p in state.positions}), as_of)
    market_value = sum(p.qty * prices.get(p.symbol, p.avg_entry_price) for p in state.positions)
    equity = state.cash + market_value
    gross = market_value / equity if equity > 0 else 0.0
    EquityCurveRepository(dynamo.engine()).upsert(
        as_of, equity=equity, cash=state.cash, gross_exposure=gross
    )
    return MaterializeResult(metadata={"equity": equity, "cash": state.cash, "gross": gross})


@dagster_asset(name="parquet_export", group_name=LEARNING_GROUP, deps=["metrics_daily"])
def parquet_export_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """DuckDB/Parquet research sidecar export (doc 02 §3; 02:00 ET)."""
    from dynamo.data.parquet_export import export_all

    as_of = utcnow().date()
    parquet_dir = dynamo.settings().parquet_dir
    export_all(
        dynamo.market_repo(),
        dynamo.engine(),
        parquet_dir,
        start=as_of - timedelta(days=750),
        end=as_of,
    )
    return MaterializeResult(metadata={"parquet_dir": parquet_dir})


# --- learning-loop safety checks (doc 10 §9; nightly 02:20/02:25 ET) ---


@dagster_asset(name="chain_verify_check", group_name=LEARNING_GROUP)
def chain_verify_check(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Nightly decision_context hash-chain verification; page on any fault."""
    from dynamo.persistence.repositories.learning import DecisionContextRepository

    result = DecisionContextRepository(dynamo.engine()).verify()
    if not result.ok:
        message = (
            f"decision_context hash chain FAILED: {result.fault.value if result.fault else '?'} "
            f"at index {result.index} — {result.detail}"
        )
        context.log.error("chain_verify_check: %s", message)
        dynamo.market_repo().add_alert(severity="page", name="chain_verify_fault", message=message)
    return MaterializeResult(
        metadata={
            "ok": result.ok,
            "fault": result.fault.value if result.fault else "",
            "fault_index": -1 if result.index is None else result.index,
        }
    )


@dagster_asset(name="experience_completeness_check", group_name=LEARNING_GROUP)
def experience_completeness_check(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Closed trades must be reviewed within 24h (doc 00 §4) — alert on overdue."""
    from dynamo.learning.reviews import completeness_check
    from dynamo.persistence.repositories.execution import TradeRepository
    from dynamo.persistence.repositories.learning import ReviewRepository

    engine = dynamo.engine()
    closed = [(_trade_id(c), c.closed_at) for c in TradeRepository(engine).closed_trades()]
    reviewed = ReviewRepository(engine).reviewed_trade_ids()
    overdue = completeness_check(closed, reviewed, utcnow())
    if overdue:
        sample = ", ".join(sorted(overdue)[:10])
        message = f"{len(overdue)} closed trades overdue for post-trade review: {sample}"
        context.log.warning("experience_completeness_check: %s", message)
        dynamo.market_repo().add_alert(
            severity="warn", name="experience_completeness_overdue", message=message
        )
    return MaterializeResult(
        metadata={"closed_trades": len(closed), "reviewed": len(reviewed), "overdue": len(overdue)}
    )


# --- dead-man switch: final step of the nightly chain (doc 12 §6) ---


@op
def ping_healthcheck(context: OpExecutionContext) -> None:
    url = os.environ.get(HEALTHCHECK_URL_ENV, "")
    if not url:
        context.log.info("healthcheck: %s unset — skipping ping", HEALTHCHECK_URL_ENV)
        return
    import httpx

    response = httpx.get(url, timeout=10.0)
    context.log.info("healthcheck ping -> %d", response.status_code)


@job
def healthcheck_job() -> None:
    """Dead-man switch ping after the nightly chain completes."""
    ping_healthcheck()


LEARNING_ASSETS = [
    post_trade_reviews,
    calibration_update,
    drift_monitors,
    metrics_daily_asset,
    equity_curve_asset,
    parquet_export_asset,
    chain_verify_check,
    experience_completeness_check,
]
