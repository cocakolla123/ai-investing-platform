"""Learning-loop, metrics, export, and dead-man-switch wiring — docs 10, 12.

Reviews respect the Claude budget (the cost meter refuses submissions at cap);
everything else here is LLM-free and runs in degraded mode too.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetSelection,
    MaterializeResult,
    OpExecutionContext,
    define_asset_job,
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


# --- decision-experience resolution (plan Phase 1) --------------------------

BENCHMARK_SYMBOL = "SPY"
DEFAULT_HORIZON_DAYS = 20


def build_experiences(
    *,
    closed_trades: Sequence[Any],
    review_for: Callable[[str], Any],
    decision_context_for: Callable[[Any], Any],
    verdict_for: Callable[[str | None], Any],
    quant_for: Callable[[str, date], Any],
    regime_for: Callable[[date], Any],
    vix_for: Callable[[date], float | None],
    sector_for: Callable[[str], str | None],
    path_for: Callable[[Any, int | None], Any],
    existing_dc_ids: frozenset[str],
    insert: Callable[[Any], bool],
    now: datetime,
    log: Any,
) -> dict[str, int]:
    """For each reviewed closed trade lacking an experience row: join the
    decision spine, build the ``DecisionExperience``, insert (idempotent).

    Broken joins (no review / no decision context) skip gracefully; a single
    bad trade NEVER fails the run — logged and counted. Pure over the injected
    callables so the asset test drives it with in-memory fakes.
    """
    from dynamo.learning.experience import TradeFacts, build_experience

    built = skipped = existing = failed = 0
    for closed in closed_trades:
        trade_id = _trade_id(closed)
        try:
            review = review_for(trade_id)
            if review is None:
                skipped += 1
                continue
            dc = decision_context_for(closed)
            if dc is None:
                log.warning("experience_resolution: no decision context for %s", trade_id)
                skipped += 1
                continue
            if dc.id in existing_dc_ids:
                existing += 1
                continue
            verdict = verdict_for(dc.verdict_id)
            horizon = _horizon_for(dc, verdict)
            decided = dc.created_at.date()
            exp = build_experience(
                trade=TradeFacts(
                    trade_id=trade_id,
                    symbol=closed.symbol,
                    qty=closed.qty,
                    opened_at=closed.opened_at,
                    closed_at=closed.closed_at,
                    entry_price=closed.entry_avg_price,
                    exit_price=closed.exit_avg_price,
                    pnl_pct=closed.pnl_pct,
                    holding_days=closed.holding_days,
                ),
                now=now,
                decision_context=dc,
                verdict=verdict,
                quant=quant_for(closed.symbol, decided),
                regime=regime_for(decided),
                review=review,
                path=path_for(closed, horizon),
                sector=sector_for(closed.symbol),
                vix=vix_for(decided),
                source="live",
            )
            if insert(exp):
                built += 1
        except Exception as exc:  # one bad trade must never fail the run
            log.error("experience_resolution: %s failed (%s) — continuing", trade_id, exc)
            failed += 1
    return {"built": built, "skipped": skipped, "existing": existing, "failed": failed}


def _horizon_for(dc: Any, verdict: Any) -> int:
    """Bar-window horizon: the frozen context first, verdict fallback, then
    the platform default (pre-repair rows hardcoded zeros)."""
    for source_obj in (dc, verdict):
        horizon = int(getattr(source_obj, "horizon_days", 0) or 0)
        if horizon > 0:
            return horizon
    return DEFAULT_HORIZON_DAYS


def resolve_experiences(dynamo: DynamoResource, log: Any) -> dict[str, int]:
    """Live wiring for :func:`build_experiences` over the repaired decision
    spine (decision_context → orders → trades → FIFO closed trades + verdicts
    + quant scores + regime states + reviews + bar paths). Shared by the
    ``experience_resolution`` asset and the backfill CLI."""
    from sqlalchemy import select as sa_select

    from dynamo.domain.models.research import CommitteeVerdict, QuantScore
    from dynamo.domain.models.risk import Regime, RegimeState
    from dynamo.learning.experience import PathInputs
    from dynamo.persistence.repositories.execution import (
        OrderRepository,
        TradeRepository,
    )
    from dynamo.persistence.repositories.learning import (
        DecisionContextRepository,
        ReviewRepository,
    )

    engine = dynamo.engine()
    trades_repo = TradeRepository(engine)
    orders_repo = OrderRepository(engine)
    dc_repo = DecisionContextRepository(engine)
    reviews_repo = ReviewRepository(engine)
    exp_repo = dynamo.experience_repo()

    def verdict_for(verdict_id: str | None) -> CommitteeVerdict | None:
        if not verdict_id:
            return None
        v = schema.committee_verdicts
        with engine.connect() as cx:
            row = cx.execute(sa_select(v.c.payload).where(v.c.id == verdict_id)).first()
        if row is None:
            return None
        try:
            return CommitteeVerdict.model_validate(row.payload)
        except Exception:  # malformed historical payload — degrade per-field
            log.warning("experience_resolution: unparsable verdict payload %s", verdict_id)
            return None

    def quant_for(symbol: str, as_of: date) -> QuantScore | None:
        q = schema.quant_scores
        with engine.connect() as cx:
            row = cx.execute(
                sa_select(q)
                .where(q.c.symbol == symbol, q.c.as_of <= as_of)
                .order_by(q.c.as_of.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        return QuantScore(
            symbol=row.symbol,
            as_of=row.as_of,
            model_version=row.model_version,
            expected_excess_return=row.expected_excess_return,
            downside_prob=row.downside_prob,
            vol_forecast=row.vol_forecast,
            rank=row.rank,
            confidence=row.confidence,
            feature_coverage=row.feature_coverage,
        )

    def regime_for(as_of: date) -> RegimeState | None:
        rs = schema.regime_states
        with engine.connect() as cx:
            row = cx.execute(
                sa_select(rs).where(rs.c.as_of <= as_of).order_by(rs.c.as_of.desc()).limit(1)
            ).first()
        if row is None:
            return None
        return RegimeState(
            as_of=row.as_of,
            regime=Regime(row.regime),
            composite_score=row.composite_score,
            inputs=row.inputs or {},
            sessions_in_state=row.sessions_in_state,
        )

    def vix_for(as_of: date) -> float | None:
        m = schema.macro_series
        with engine.connect() as cx:
            row = cx.execute(
                sa_select(m.c.value)
                .where(m.c.series_id == "VIXCLS", m.c.ts <= as_of)
                .order_by(m.c.ts.desc(), m.c.vintage_ts.desc())
                .limit(1)
            ).first()
        return float(row.value) if row is not None else None

    def sector_for(symbol: str) -> str | None:
        inst = schema.instruments
        with engine.connect() as cx:
            row = cx.execute(sa_select(inst.c.sector).where(inst.c.symbol == symbol)).first()
        return row.sector if row is not None else None

    def path_for(closed: Any, horizon: int | None) -> PathInputs | None:
        """Session bars entry → exit + horizon (calendar buffer over sessions),
        plus benchmark closes over the holding window."""
        days = horizon or DEFAULT_HORIZON_DAYS
        start = closed.opened_at.date()
        end = closed.closed_at.date() + timedelta(days=days * 2 + 10)
        frame = dynamo.market_repo().bars_frame([closed.symbol, BENCHMARK_SYMBOL], start, end)
        if frame.empty:
            return None
        from typing import cast as _cast

        exit_ts = pd.Timestamp(closed.closed_at)
        sym = _cast(pd.DataFrame, frame[frame["symbol"] == closed.symbol]).sort_values("ts")
        if sym.empty:
            return None
        exit_index = int((pd.to_datetime(sym["ts"], utc=True) <= exit_ts).sum()) - 1
        if exit_index < 0:
            return None
        bench = _cast(pd.DataFrame, frame[frame["symbol"] == BENCHMARK_SYMBOL]).sort_values("ts")
        bench = _cast(pd.DataFrame, bench[pd.to_datetime(bench["ts"], utc=True) <= exit_ts])
        return PathInputs(
            opens=tuple(float(x) for x in sym["open"]),
            highs=tuple(float(x) for x in sym["high"]),
            lows=tuple(float(x) for x in sym["low"]),
            closes=tuple(float(x) for x in sym["close"]),
            exit_session_index=exit_index,
            benchmark_closes=tuple(float(x) for x in bench["close"]),
        )

    return build_experiences(
        closed_trades=trades_repo.closed_trades(),
        review_for=reviews_repo.get,
        decision_context_for=lambda closed: _decision_context_for(
            closed, trades_repo, orders_repo, dc_repo
        ),
        verdict_for=verdict_for,
        quant_for=quant_for,
        regime_for=regime_for,
        vix_for=vix_for,
        sector_for=sector_for,
        path_for=path_for,
        existing_dc_ids=exp_repo.live_decision_context_ids(),
        insert=exp_repo.insert,
        now=utcnow(),
        log=log,
    )


@dagster_asset(name="experience_resolution", group_name=LEARNING_GROUP, deps=["post_trade_reviews"])
def experience_resolution(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Reviewed closed trades lacking an experience row → decision-spine join
    → ``decision_experience`` rows with ``source='live'`` (plan Phase 1).

    LLM-free; incomplete joins skip gracefully and one bad trade never fails
    the run (see :func:`build_experiences`).
    """
    counts = resolve_experiences(dynamo, context.log)
    return MaterializeResult(metadata={k: v for k, v in counts.items()})


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


def hier_calibration_inputs(exp_repo: Any, enabled: bool) -> tuple[list[Any], list[Any]]:
    """Hierarchical-scope feedstock (plan Phase 2): resolved LIVE experiences
    and one ``hier``-scope observation per chain prefix each. Gated by
    ``hierarchical_calibration_enabled`` — off means empty AND unqueried."""
    if not enabled:
        return [], []
    from dynamo.learning.hierarchy import hier_observations

    experiences = list(exp_repo.resolved_live())
    return experiences, hier_observations(experiences)


def log_calibrated_probabilities(
    experiences: Sequence[Any],
    hier_obs: Sequence[Any],
    update: Callable[[str, float, str], None],
) -> int:
    """Advisory calibrated-p logging (plan Phase 2): for each resolved live
    experience not yet calibrated, run its conviction through the
    :class:`~dynamo.learning.hierarchy.HierarchicalCalibrator` and persist
    ``(value, scope_used)`` via ``update``. ADVISORY ONLY — nothing consumes
    the logged probability for sizing.

    v1 limitation (documented in the plan): the calibrator is built from
    ALL current observations — including each experience's own outcome —
    rather than strictly ``created_at < as_of`` prior ones. Acceptable for
    advisory logging; the strictly-prior build lands with replay (Phase 4+),
    which needs point-in-time reads anyway.
    """
    from dynamo.learning.hierarchy import HierarchicalCalibrator, experience_scope_key

    calibrator = HierarchicalCalibrator.from_observations(list(hier_obs))
    written = 0
    for exp in experiences:
        if exp.conviction is None or exp.calibrated_probability is not None:
            continue
        result = calibrator.calibrate(exp.conviction, experience_scope_key(exp))
        update(exp.id, result.value, result.scope_used)
        written += 1
    return written


@dagster_asset(group_name=LEARNING_GROUP, deps=["post_trade_reviews", "experience_resolution"])
def calibration_update(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Conviction vs realized trade outcomes -> calibration_points, per scope
    (committee/arbiter + strategy/<sleeve> + agent/<role>), plus — plan
    Phase 2, gated by ``hierarchical_calibration_enabled`` — one ``hier``
    scope per chain prefix of every resolved live ``decision_experience``
    (path-string scope ids; ``calibration_points.scope_id`` is Text, so
    arbitrary-length paths fit). After upserting, the hierarchical calibrator
    logs an ADVISORY calibrated probability onto each new experience row —
    nothing consumes it for sizing."""
    from dynamo.learning.calibration import compute_calibration
    from dynamo.persistence.repositories.learning import CalibrationRepository

    as_of = utcnow().date()
    observations: list[Any] = []
    r, d, v = schema.post_trade_reviews, schema.decision_context, schema.committee_verdicts
    reviews = _read_df(dynamo, select(r.c.trade_id, r.c.decision_context_id, r.c.payload))
    if not reviews.empty:
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
    exp_repo = dynamo.experience_repo()
    experiences, hier_obs = hier_calibration_inputs(
        exp_repo, dynamo.settings().hierarchical_calibration_enabled
    )
    all_observations = observations + hier_obs
    if not all_observations:
        return MaterializeResult(metadata={"observations": 0, "skipped": "no resolved outcomes"})
    points = compute_calibration(all_observations, as_of)
    CalibrationRepository(dynamo.engine()).upsert(points)
    calibrated = 0
    if hier_obs:
        calibrated = log_calibrated_probabilities(
            experiences, hier_obs, exp_repo.update_calibration
        )
    return MaterializeResult(
        metadata={
            "observations": len(all_observations),
            "points": len(points),
            "hier_observations": len(hier_obs),
            "calibrated_experiences": calibrated,
        }
    )


# --- advisory meta-model scoring (plan Phase 3; 23:45 ET) -------------------

META_NEIGHBORS = 3  # k nearest prior experiences per scored decision


def build_meta_score_rows(
    model: Any,
    model_version: str,
    decisions: Sequence[tuple[str, str | None, Any, Any, Sequence[Any]]],
    now: datetime,
    as_of: date,
) -> list[dict[str, Any]]:
    """meta_decision_scores rows for tonight's decisions (pure over inputs).

    ``decisions``: ``(symbol, verdict_id, snapshot, calib, neighbors)`` per
    verdict. ``meta_score`` and ``effective_probability`` are both the
    model's calibrated P(win) — ``MetaModel.predict_proba`` already includes
    the calibration head, and ``effective_probability`` is DEFINED as that
    output (plan Phase 3). Deterministic ids make re-runs idempotent.
    """
    import hashlib as _hashlib
    import json as _json

    from dynamo.learning.meta_model import build_features

    rows: list[dict[str, Any]] = []
    if not decisions:
        return rows
    features = [
        build_features(snapshot, calib, neighbors)
        for (_, _, snapshot, calib, neighbors) in decisions
    ]
    probs = model.predict_proba(features)
    for (symbol, verdict_id, _, _, _), feature_row, p in zip(
        decisions, features, probs, strict=True
    ):
        canonical = _json.dumps(feature_row, sort_keys=True)
        rows.append(
            {
                "id": f"mds:{as_of.isoformat()}:{symbol}:{model_version}",
                "created_at": now,
                "as_of": as_of,
                "symbol": symbol,
                "verdict_id": verdict_id,
                "model_version": model_version,
                "meta_score": float(p),
                "effective_probability": float(p),
                "features_hash": _hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
                "source": "live",
            }
        )
    return rows


def run_meta_scoring(
    *,
    production: Any,
    load: Callable[[Any], Any],
    decisions: Callable[[], Sequence[tuple[str, str | None, Any, Any, Sequence[Any]]]],
    insert: Callable[[dict[str, Any]], bool],
    now: datetime,
    as_of: date,
    log: Any,
) -> dict[str, Any]:
    """Asset core, pure over injected callables (test-driven with fakes).

    No production-stage meta-model — the NORMAL state until one survives the
    full promotion pipeline — is a silent, successful skip, never an error.
    """
    if production is None:
        return {"skipped": "no production meta-model", "scored": 0}
    from dynamo.learning.artifacts import ArtifactIntegrityError

    try:
        model = load(production)
    except (ArtifactIntegrityError, FileNotFoundError) as exc:
        log.error("meta_scoring: cannot load %s: %s", production.id, exc)
        return {"skipped": f"artifact unusable: {exc}", "scored": 0}
    rows = build_meta_score_rows(model, production.id, list(decisions()), now, as_of)
    written = sum(1 for row in rows if insert(row))
    return {"scored": len(rows), "written": written, "model_version": production.id}


def _tonights_decisions(
    dynamo: DynamoResource, as_of: date, log: Any
) -> list[tuple[str, str | None, Any, Any, Sequence[Any]]]:
    """(symbol, verdict_id, snapshot, calib, neighbors) for today's verdicts.

    The experience row does not exist yet at decision time, so snapshots are
    built straight from the verdict + quant + regime (shared builder:
    ``dynamo.learning.meta_model.snapshot_from_decision``).
    """
    from sqlalchemy import select as sa_select

    from dynamo.domain.models.research import CommitteeVerdict, QuantScore
    from dynamo.domain.models.risk import Regime, RegimeState
    from dynamo.learning.hierarchy import HierarchicalCalibrator, ScopeKey
    from dynamo.learning.meta_model import snapshot_from_decision
    from orchestration.assets.research import _pydate

    engine = dynamo.engine()
    v = schema.committee_verdicts
    verdicts: list[CommitteeVerdict] = []
    with engine.connect() as cx:
        for row in cx.execute(sa_select(v.c.payload).where(v.c.as_of == as_of)).all():
            try:
                verdicts.append(CommitteeVerdict.model_validate(row.payload))
            except Exception:
                log.warning("meta_scoring: unparsable verdict payload — skipping one row")
    if not verdicts:
        return []

    symbols = sorted({vd.symbol for vd in verdicts})
    quant_by_symbol: dict[str, QuantScore] = {}
    qf = dynamo.market_repo().latest_quant_scores(symbols)
    if not qf.empty:
        for r in qf.to_dict(orient="records"):
            quant_by_symbol[str(r["symbol"])] = QuantScore(
                symbol=str(r["symbol"]),
                as_of=_pydate(r["as_of"]),
                model_version=str(r["model_version"]),
                expected_excess_return=r["expected_excess_return"],
                downside_prob=r["downside_prob"],
                vol_forecast=r["vol_forecast"],
                rank=r["rank"],
                confidence=float(r["confidence"] or 0.0),
                feature_coverage=float(r["feature_coverage"] or 1.0),
            )

    rs = schema.regime_states
    regime: RegimeState | None = None
    with engine.connect() as cx:
        r = cx.execute(
            sa_select(rs).where(rs.c.as_of <= as_of).order_by(rs.c.as_of.desc()).limit(1)
        ).first()
    if r is not None:
        regime = RegimeState(
            as_of=r.as_of,
            regime=Regime(r.regime),
            composite_score=r.composite_score,
            inputs=r.inputs or {},
            sessions_in_state=r.sessions_in_state,
        )

    m = schema.macro_series
    with engine.connect() as cx:
        vix_row = cx.execute(
            sa_select(m.c.value)
            .where(m.c.series_id == "VIXCLS", m.c.ts <= as_of)
            .order_by(m.c.ts.desc(), m.c.vintage_ts.desc())
            .limit(1)
        ).first()
    vix = float(vix_row.value) if vix_row is not None else None

    spy = dynamo.bars_window(["SPY"], as_of, 60)
    spy_ret_20d: float | None = None
    if not spy.empty:
        closes = [float(x) for x in spy.sort_values("ts")["close"]]
        if len(closes) >= 21 and closes[-21] > 0:
            spy_ret_20d = closes[-1] / closes[-21] - 1.0

    inst = schema.instruments
    with engine.connect() as cx:
        sector_by_symbol = {
            str(r.symbol): r.sector
            for r in cx.execute(
                sa_select(inst.c.symbol, inst.c.sector).where(inst.c.symbol.in_(symbols))
            ).all()
        }
    wl = schema.watchlist
    with engine.connect() as cx:
        sleeve_by_symbol = {
            str(r.symbol): str(r.sleeve)
            for r in cx.execute(
                sa_select(wl.c.symbol, wl.c.sleeve).where(
                    wl.c.as_of == as_of, wl.c.symbol.in_(symbols)
                )
            ).all()
        }

    exp_repo = dynamo.experience_repo()
    calibrator: HierarchicalCalibrator | None = None
    _, hier_obs = hier_calibration_inputs(
        exp_repo, dynamo.settings().hierarchical_calibration_enabled
    )
    if hier_obs:
        calibrator = HierarchicalCalibrator.from_observations(hier_obs)

    decisions: list[tuple[str, str | None, Any, Any, Sequence[Any]]] = []
    for vd in verdicts:
        sector = sector_by_symbol.get(vd.symbol)
        sleeve = sleeve_by_symbol.get(vd.symbol)
        calib = None
        if calibrator is not None:
            calib = calibrator.calibrate(
                vd.conviction,
                ScopeKey(
                    sleeve=sleeve,
                    regime=regime.regime.value if regime is not None else None,
                    sector=sector,
                    quant_alignment=vd.quant_alignment.value,
                ),
            )
        neighbors = exp_repo.similar(
            symbol=vd.symbol,
            as_of=as_of,
            sector=sector,
            regime=regime.regime.value if regime is not None else None,
            sleeve=sleeve,
            limit=META_NEIGHBORS,
        )
        snapshot = snapshot_from_decision(
            vd,
            quant=quant_by_symbol.get(vd.symbol),
            regime=regime,
            sector=sector,
            sleeve=sleeve,
            vix=vix,
            spy_ret_20d=spy_ret_20d,
        )
        decisions.append((vd.symbol, vd.id, snapshot, calib, neighbors))
    return decisions


@dagster_asset(name="meta_scoring", group_name=LEARNING_GROUP)
def meta_scoring(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Score tonight's committee verdicts with the PRODUCTION-stage advisory
    meta-model → ``meta_decision_scores`` rows (plan Phase 3; 23:45 ET).

    ADVISORY-FIRST: with no production meta-model in the registry (the normal
    state for months) this skips silently. Written scores are consumed for
    sizing ONLY behind ``meta_sizing_enabled`` (default False). LLM-free.
    """
    from pathlib import Path

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from dynamo.learning.artifacts import load_model
    from dynamo.persistence.repositories.learning import ModelRegistryRepository

    as_of = utcnow().date()
    production = ModelRegistryRepository(dynamo.engine()).latest_production("meta")

    def _load(mv: Any) -> Any:
        artifact = Path(mv.artifact_path)
        expected = (mv.metrics or {}).get("artifact_sha256")
        return load_model(
            artifact.parent, mv.id, expected_sha=str(expected) if expected else None
        )

    def _insert(row: dict[str, Any]) -> bool:
        t = schema.meta_decision_scores
        stmt = pg_insert(t).values(**row).on_conflict_do_nothing(index_elements=[t.c.id])
        with dynamo.engine().begin() as cx:
            inserted = cx.execute(stmt.returning(t.c.id)).first()
        return inserted is not None

    result = run_meta_scoring(
        production=production,
        load=_load,
        decisions=lambda: _tonights_decisions(dynamo, as_of, context.log),
        insert=_insert,
        now=utcnow(),
        as_of=as_of,
        log=context.log,
    )
    return MaterializeResult(metadata={k: v for k, v in result.items()})


@dagster_asset(group_name=LEARNING_GROUP)
def drift_monitors(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Slippage / incident-trend / IC-decay / PSI / strategy-expectation
    monitors -> improvement_proposals.

    The PSI and strategy-expectation monitors (dormant since doc 10 §7) are
    wired with HONEST inputs only (plan Phase 6): they read the
    ``metrics_daily`` rows a trainer/backtest is expected to record
    (``feature_psi`` per model:feature, ``strategy_perf`` per strategy) and
    the expectation metadata on registry rows. TODAY nothing records those
    inputs — the quant model is baseline-v0 with no training distribution and
    no strategy writes a rolling performance metric — so both monitors see
    empty inputs and emit NOTHING. That is deliberate: the plumbing is live,
    and the first trainer that records real inputs arms it without changes
    here. No inputs are ever fabricated.
    """
    from dynamo.learning.drift import (
        feature_psi_monitor,
        ic_decay_monitor,
        incident_trend_monitor,
        slippage_monitor,
        strategy_expectation_monitor,
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

    # PSI per top-importance feature (honest inputs; see docstring). Expected
    # rows: metric='feature_psi', scope_type='model',
    # scope_id='<model_id>:<feature>'; latest value per (model, feature) wins.
    psi_rows = _read_df(
        dynamo,
        select(m.c.scope_id, m.c.as_of, m.c.value)
        .where(m.c.metric == "feature_psi", m.c.scope_type == "model")
        .order_by(m.c.as_of),
    )
    if not psi_rows.empty and not promoted.empty:
        latest_psi: dict[tuple[str, str], float] = {}
        for psi_row in psi_rows.to_dict(orient="records"):
            model_id, _, feature = str(psi_row["scope_id"]).partition(":")
            if feature:
                latest_psi[(model_id, feature)] = float(psi_row["value"])
        for row in promoted.to_dict(orient="records"):
            model_id = str(row["id"])
            psi_by_feature = {f: v for (mid, f), v in latest_psi.items() if mid == model_id}
            if not psi_by_feature:
                continue
            # top_features from training metadata; when absent, every recorded
            # feature counts as top-importance (conservative, still honest).
            top = list((row["metrics"] or {}).get("top_features") or sorted(psi_by_feature))
            p = feature_psi_monitor(psi_by_feature, top, model_id, as_of, now)
            if p is not None:
                proposals.append(p)

    # Strategy performance vs its backtest expectation band (honest inputs).
    # Expected rows: metric='strategy_perf', scope_type='strategy'; the band
    # comes from the strategy's registry metrics (expectation_low/high).
    strat_rows = _read_df(
        dynamo,
        select(m.c.scope_id, m.c.as_of, m.c.value)
        .where(m.c.metric == "strategy_perf", m.c.scope_type == "strategy")
        .order_by(m.c.as_of),
    )
    strat_models = _read_df(
        dynamo,
        select(mv.c.id, mv.c.metrics).where(
            mv.c.model_type == "strategy", mv.c.status == "production"
        ),
    )
    if not strat_rows.empty and not strat_models.empty:
        for row in strat_models.to_dict(orient="records"):
            strategy_id = str(row["id"])
            meta = row["metrics"] or {}
            low, high = meta.get("expectation_low"), meta.get("expectation_high")
            series = strat_rows[strat_rows["scope_id"] == strategy_id]["value"].tolist()
            if low is None or high is None or not series:
                continue
            p = strategy_expectation_monitor(
                [float(x) for x in series], (float(low), float(high)), strategy_id, as_of, now
            )
            if p is not None:
                proposals.append(p)

    repo = ProposalRepository(dynamo.engine())
    for proposal in proposals:
        repo.insert(proposal)
    return MaterializeResult(metadata={"proposals": len(proposals)})


# --- learning-metrics suite (plan Phase 6; weekly after calibration_update) --


def upsert_learning_metrics(
    metric_rows: Sequence[tuple[str, str, str, float]],
    upsert: Callable[[str, str, str, date, float], None],
    as_of: date,
) -> int:
    """Write the ``learning.*`` rows through a MetricsRepository-shaped
    callable (pure over the injection so the asset test drives fakes)."""
    for metric, scope_type, scope_id, value in metric_rows:
        upsert(metric, scope_type, scope_id, as_of, value)
    return len(metric_rows)


@dagster_asset(
    name="learning_metrics", group_name=LEARNING_GROUP, deps=["calibration_update"]
)
def learning_metrics_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Weekly learning-metrics suite over resolved LIVE experiences →
    ``metrics_daily`` rows namespaced ``learning.*`` (plan Phase 6).

    All computation happens in the pure ``dynamo.learning.metrics`` module;
    this asset only joins the inputs: resolved live experiences (the replay
    firewall applies — ``resolved_live()`` filters ``source='live'``),
    all-time committee_verdicts action counts (the decisions-considered
    denominator experiences alone cannot supply), and generation_results
    rows for the value-of-learning passthrough. LLM-free.
    """
    from dynamo.learning.metrics import learning_metric_rows
    from dynamo.persistence.repositories.learning import MetricsRepository

    as_of = utcnow().date()
    experiences = dynamo.experience_repo().resolved_live()

    v = schema.committee_verdicts
    actions = _read_df(
        dynamo, select(v.c.action, func.count().label("n")).group_by(v.c.action)
    )
    action_counts = (
        {str(r["action"]): int(r["n"]) for r in actions.to_dict(orient="records")}
        if not actions.empty
        else {}
    )

    gr = schema.generation_results
    generations = _read_df(
        dynamo, select(gr.c.generation, gr.c.value_of_learning, gr.c.n_test)
    )
    generation_rows = generations.to_dict(orient="records") if not generations.empty else []

    rows = learning_metric_rows(experiences, action_counts, generation_rows)
    written = upsert_learning_metrics(
        rows, MetricsRepository(dynamo.engine()).upsert_metric, as_of
    )
    headline = {
        metric: value for metric, scope_type, _, value in rows if scope_type == "platform"
    }
    return MaterializeResult(
        metadata={
            "rows_upserted": written,
            "n_resolved": int(headline.get("learning.n_resolved", 0.0)),
            **{k: round(val, 6) for k, val in headline.items()},
        }
    )


# --- failure clustering (plan Phase 6; sensor-triggered) ---------------------

CLUSTER_WATERMARK_METRIC = "learning.cluster_watermark"


def run_failure_clustering(
    *,
    failures: Sequence[Any],
    review_text_for: Callable[[str | None], str],
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    insert_proposal: Callable[[Any], bool],
    replace_clusters: Callable[[Sequence[Any], Sequence[Any]], int],
    upsert_metric: Callable[[str, str, str, date, float], None],
    as_of: date,
    now: datetime,
) -> dict[str, int]:
    """Asset core, pure over injected callables (test-driven with fakes).

    Builds one :class:`~dynamo.learning.clustering.FailureCase` per failed
    experience (review text via ``review_text_for``; embeddings via the
    injected embedder — the pure clustering module never embeds), clusters,
    inserts OPEN proposals for clusters >= min size (idempotent via
    ``insert_proposal`` returning False on an existing id), persists cluster
    rows for the dashboard, and advances the durable watermark: the
    ``metrics_daily`` row ``learning.cluster_watermark`` records how many
    failure experiences had been seen at this run — the sensor triggers on
    (current count - max recorded watermark) >= threshold.
    """
    from dynamo.learning.clustering import (
        FailureCase,
        cluster_failures,
        proposals_from_clusters,
    )

    texts = [review_text_for(getattr(exp, "trade_id", None)) or "" for exp in failures]
    vectors = list(embed(texts)) if texts else []
    cases = [
        FailureCase(
            case_id=exp.id,
            symbol=exp.symbol,
            as_of=exp.as_of,
            failure_class=exp.failure_primary,
            text=text,
            embedding=tuple(float(x) for x in vector) if vector is not None else None,
        )
        for exp, text, vector in zip(failures, texts, vectors, strict=True)
    ]
    clusters = cluster_failures(cases)
    proposals = proposals_from_clusters(clusters, as_of, now)
    inserted = sum(1 for p in proposals if insert_proposal(p))
    stored = replace_clusters(clusters, proposals)
    upsert_metric(CLUSTER_WATERMARK_METRIC, "platform", "all", as_of, float(len(failures)))
    return {
        "failures": len(failures),
        "clusters": len(clusters),
        "proposals": len(proposals),
        "proposals_inserted": inserted,
        "cluster_rows": stored,
    }


@dagster_asset(name="failure_clustering", group_name=LEARNING_GROUP)
def failure_clustering_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Cluster failed LIVE experiences into shared failure modes →
    ``failure_clusters`` rows + OPEN ``improvement_proposals`` (plan Phase 6).

    Triggered by ``failure_clustering_sensor`` every ~150 new resolved
    failures (manual materialization works too). LLM-free: review texts are
    embedded locally (bge-small or the deterministic hash fallback). Proposals
    are ONLY ever opened for a human to review — nothing auto-applies.
    """
    from sqlalchemy.exc import IntegrityError

    from dynamo.domain.models.learning import FailureClass
    from dynamo.llm.embeddings import LocalEmbedder
    from dynamo.persistence.repositories.learning import (
        MetricsRepository,
        ProposalRepository,
    )

    as_of = utcnow().date()
    now = utcnow()
    failures = [
        e
        for e in dynamo.experience_repo().resolved_live()
        if e.failure_primary is not None
        and e.failure_primary != FailureClass.NORMAL_UNCERTAINTY
    ]

    r = schema.post_trade_reviews
    reviews = _read_df(dynamo, select(r.c.trade_id, r.c.payload))
    text_by_trade: dict[str, str] = {}
    if not reviews.empty:
        for row in reviews.to_dict(orient="records"):
            payload = row.get("payload") or {}
            parts = [*(payload.get("what_failed") or ()), *(payload.get("lessons") or ())]
            text_by_trade[str(row["trade_id"])] = " ".join(str(p) for p in parts)

    proposal_repo = ProposalRepository(dynamo.engine())

    def insert_proposal(proposal: Any) -> bool:
        try:
            proposal_repo.insert(proposal)
        except IntegrityError:
            return False  # deterministic id already inserted by an earlier run
        return True

    def replace_clusters(clusters: Sequence[Any], proposals: Sequence[Any]) -> int:
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        proposal_by_target = {p.target: p.id for p in proposals}
        t = schema.failure_clusters
        written = 0
        with dynamo.engine().begin() as cx:
            cx.execute(t.delete().where(t.c.run_as_of == as_of))
            for cluster in clusters:
                stmt = pg_insert(t).values(
                    id=f"fc:{as_of.isoformat()}:{cluster.label}",
                    run_as_of=as_of,
                    created_at=now,
                    label=cluster.label,
                    failure_class=cluster.failure_class.value,
                    size=cluster.size,
                    example_refs=list(cluster.example_refs),
                    suggested_kind=cluster.suggested_kind.value,
                    proposal_id=proposal_by_target.get(cluster.label),
                )
                cx.execute(stmt.on_conflict_do_nothing(index_elements=[t.c.id]))
                written += 1
        return written

    embedder = LocalEmbedder()
    counts = run_failure_clustering(
        failures=failures,
        review_text_for=lambda trade_id: text_by_trade.get(trade_id or "", ""),
        embed=embedder.embed,
        insert_proposal=insert_proposal,
        replace_clusters=replace_clusters,
        upsert_metric=MetricsRepository(dynamo.engine()).upsert_metric,
        as_of=as_of,
        now=now,
    )
    context.log.info("failure_clustering: %s (embedder=%s)", counts, embedder.backend)
    return MaterializeResult(metadata={**counts, "embedder": embedder.backend})


failure_clustering_job = define_asset_job(
    "failure_clustering_job",
    # Sensor target (plan Phase 6): requested by failure_clustering_sensor
    # when >= 150 new resolved failures accumulate; never scheduled.
    selection=AssetSelection.assets("failure_clustering"),
)


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
    experience_resolution,
    calibration_update,
    meta_scoring,
    drift_monitors,
    learning_metrics_asset,
    failure_clustering_asset,
    metrics_daily_asset,
    equity_curve_asset,
    parquet_export_asset,
    chain_verify_check,
    experience_completeness_check,
]
