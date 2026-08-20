"""Risk, portfolio construction, and paper-execution wiring — docs 06, 07, 09.

The risk engine is independent of the committee and never degrades (doc 06);
execution is gated: paper orders only with Alpaca keys AND DYNAMO_ENV=PAPER,
and never with the kill switch engaged.
"""

from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

import pandas as pd
from dagster import (
    AssetExecutionContext,
    In,
    MaterializeResult,
    Nothing,
    OpExecutionContext,
    Out,
    job,
    op,
)
from dagster import asset as dagster_asset
from sqlalchemy import select

from dynamo.config.settings import DynamoEnv, Settings
from dynamo.persistence import schema
from orchestration.assets.research import _pydate, _read_df, _rows_to_models
from orchestration.resources import DynamoResource, utcnow

PORTFOLIO_GROUP = "portfolio"

# Paper bootstrap equity when neither equity_curve nor a broker account exists.
RETURNS_LOOKBACK_DAYS = 750
BENCHMARK_SYMBOLS = ("SPY", "TLT")


class UtcClock:
    """Injectable wall clock for execution components (domain Clock port)."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def today(self) -> date:
        return self.now().date()


def _broker(dynamo: DynamoResource) -> Any | None:
    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        return None
    from dynamo.execution.alpaca_broker import AlpacaBroker

    return AlpacaBroker(s)


def _portfolio_state(dynamo: DynamoResource) -> Any | None:
    """Internal portfolio state: positions table + equity curve (broker fallback)."""
    from dynamo.domain.models.portfolio import PortfolioState
    from dynamo.persistence.repositories.execution import PositionRepository

    positions = tuple(PositionRepository(dynamo.engine()).all_positions())
    eq = schema.equity_curve
    frame = _read_df(dynamo, select(eq).order_by(eq.c.as_of.desc()).limit(1))
    if not frame.empty:
        row = frame.iloc[0]
        equity, cash = float(row["equity"]), float(row["cash"])
    else:
        broker = _broker(dynamo)
        if broker is None:
            return None
        account = broker.account()
        equity = float(account.get("equity", 0.0))
        cash = float(account.get("cash", equity))
    if equity <= 0.0:
        return None
    return PortfolioState(as_of=utcnow(), equity=equity, cash=cash, positions=positions)


def _latest_prices(dynamo: DynamoResource, symbols: list[str], as_of: date) -> dict[str, float]:
    bars = dynamo.bars_window(symbols or None, as_of, 7)
    if bars.empty:
        return {}
    latest = bars.sort_values("ts").drop_duplicates(subset=["symbol"], keep="last")
    return {str(r["symbol"]): float(r["close"]) for r in latest.to_dict(orient="records")}


def _meta_effective_probability(dynamo: DynamoResource, as_of: date) -> dict[str, float]:
    """symbol -> meta-model effective_probability for tonight's decisions.

    DARK PATH (plan Phase 3): returns {} — leaving sizing bit-identical to
    the pre-meta platform — unless ``meta_sizing_enabled`` is True, which is
    legitimate ONLY after a meta-model survives the full promotion pipeline
    with signed human approval. When enabled, reads the most recent
    ``meta_decision_scores`` night within 3 days (23:45 ET scoring vs the
    00:30 ET construction run crosses a calendar day).
    """
    if not dynamo.settings().meta_sizing_enabled:
        return {}
    mds = schema.meta_decision_scores
    frame = _read_df(
        dynamo,
        select(mds.c.symbol, mds.c.effective_probability, mds.c.as_of).where(
            mds.c.as_of >= as_of - timedelta(days=3),
            mds.c.as_of <= as_of,
            mds.c.source == "live",
        ),
    )
    if frame.empty:
        return {}
    latest = cast(pd.DataFrame, frame[frame["as_of"] == frame["as_of"].max()])
    return {
        str(r["symbol"]): float(r["effective_probability"])
        for r in latest.to_dict(orient="records")
    }


def _ballast_symbol(dynamo: DynamoResource) -> str:
    """The active ballast symbol ("" when disabled). Owner default: SPY ON."""
    s = dynamo.settings()
    return s.ballast_symbol if s.ballast_enabled else ""


def _with_ballast_sector(sector_by_symbol: dict[str, str], ballast: str) -> dict[str, str]:
    """Give the ballast symbol a dedicated sector tag.

    The risk engine treats held weight as sector exposure when computing
    per-symbol caps (sector_cap minus same-sector usage). A large held
    ballast position sharing a sector label (e.g. "unknown") with active
    candidates would zero their caps and block active picks — the one thing
    ballast must never do. A dedicated tag keeps the engine's math intact
    while isolating ballast in its own bucket.
    """
    if ballast:
        sector_by_symbol[ballast] = "ballast"
    return sector_by_symbol


def _latest_risk_verdict(dynamo: DynamoResource) -> tuple[str, Any] | None:
    from dynamo.domain.models.risk import RiskVerdict

    rv = schema.risk_verdicts
    frame = _read_df(dynamo, select(rv).order_by(rv.c.as_of.desc()).limit(1))
    if frame.empty:
        return None
    row = frame.iloc[0]
    return str(row["id"]), RiskVerdict.model_validate(row["payload"])


# --------------------------------------------------------------------------
# risk_verdict — doc 06; independent of the committee, uses no LLM
# --------------------------------------------------------------------------


@dagster_asset(name="risk_verdict", group_name=PORTFOLIO_GROUP, deps=["regime_state"])
def risk_verdict_asset(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    import numpy as np

    from dynamo.domain.models.risk import LadderRung, RiskConfig
    from dynamo.risk.engine import RiskEngine, RiskEngineInputs
    from dynamo.risk.ladder import LadderState
    from dynamo.risk.regime import RegimeHistory

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("risk_verdict: no portfolio state (no equity curve, no broker)")
        return MaterializeResult(metadata={"skipped": "no portfolio state"})

    held = sorted({p.symbol for p in state.positions})
    panel_symbols: list[str] = sorted({*held, *BENCHMARK_SYMBOLS})
    bars = dynamo.bars_window(panel_symbols, as_of, RETURNS_LOOKBACK_DAYS)
    if bars.empty:
        context.log.warning("risk_verdict: no bars for %s", panel_symbols)
        return MaterializeResult(metadata={"skipped": "no bars"})
    wide = bars.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    returns = wide.pct_change().dropna(how="all").fillna(0.0)
    from dynamo.risk.types import ReturnsPanel

    panel = ReturnsPanel(
        dates=tuple(_pydate(t) for t in returns.index),
        symbols=tuple(str(c) for c in returns.columns),
        values=returns.to_numpy(dtype=np.float64),
    )

    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector))
    sector_by_symbol = _with_ballast_sector(
        {
            str(r["symbol"]): str(r["sector"] or "unknown")
            for r in inst_df.to_dict(orient="records")
        },
        _ballast_symbol(dynamo),
    )
    adv: dict[str, float] = {}
    for sym, grp in bars.groupby("symbol"):
        adv[str(sym)] = float(
            (grp["close"].astype(float) * grp["volume"].astype(float)).tail(20).median()
        )

    spy = wide.get("SPY")
    if spy is None or spy.dropna().empty:
        context.log.warning("risk_verdict: no SPY history — cannot build regime inputs")
        return MaterializeResult(metadata={"skipped": "no SPY history"})
    spy = spy.dropna()
    n = len(spy)
    m = schema.macro_series
    macro = _read_df(
        dynamo,
        select(m.c.series_id, m.c.ts, m.c.value).where(
            m.c.ts >= as_of - timedelta(days=RETURNS_LOOKBACK_DAYS + 200)
        ),
    )

    def _aligned(series_id: str, default: float) -> Any:
        if macro.empty:
            return np.full(n, default)
        sub = cast(pd.DataFrame, macro[macro["series_id"] == series_id]).sort_values("ts")
        if len(sub) == 0:
            return np.full(n, default)
        s = pd.Series(sub["value"].to_numpy(), index=pd.to_datetime(sub["ts"]))
        aligned = s.reindex(pd.to_datetime(spy.index), method="ffill").fillna(default)
        return np.asarray(aligned, dtype=float)

    close = spy.to_numpy(dtype=float)
    ma200 = np.asarray(pd.Series(close).rolling(200, min_periods=1).mean(), dtype=float)
    breadth = np.clip((close / ma200 - 1.0) * 5.0 + 0.5, 0.0, 1.0)
    regime_history = RegimeHistory(
        dates=tuple(_pydate(t) for t in spy.index),
        spy_close=close,
        hy_oas=_aligned("BAMLH0A0HYM2", 4.0),
        curve_slope=_aligned("T10Y3M", 1.0),
        breadth=breadth,
    )

    ls = schema.ladder_states
    ladder_df = _read_df(dynamo, select(ls).order_by(ls.c.as_of.desc()).limit(1))
    prev_ladder = LadderState()
    if not ladder_df.empty:
        prev_ladder = LadderState(rung=LadderRung(str(ladder_df.iloc[0]["rung"])))

    q = schema.quarantine
    quarantined = frozenset(
        str(s)
        for s in _read_df(
            dynamo,
            select(q.c.key).where(q.c.scope == "symbol", q.c.resolved_at.is_(None)),
        )["key"]
    )

    engine = RiskEngine()
    verdict = engine.compute_verdict(
        RiskEngineInputs(
            portfolio=state,
            returns=panel,
            prices=_latest_prices(dynamo, panel_symbols, as_of),
            sector_by_symbol=sector_by_symbol,
            adv_dollar_20d=adv,
            regime_history=regime_history,
            prev_ladder=prev_ladder,
            config=RiskConfig(),
            as_of=utcnow(),
            session_index=n - 1,
            quarantined=quarantined,
        )
    )
    verdict_id = f"risk-{as_of.isoformat()}"
    repo.upsert_rows(
        schema.risk_verdicts,
        [{"id": verdict_id, "as_of": verdict.as_of, "payload": verdict.model_dump(mode="json")}],
    )
    hwm = state.high_water_mark or state.equity
    repo.upsert_rows(
        schema.ladder_states,
        [
            {
                "as_of": as_of,
                "rung": verdict.ladder.value,
                "drawdown": verdict.metrics.drawdown_from_hwm,
                "hwm": hwm,
            }
        ],
    )
    return MaterializeResult(
        metadata={
            "id": verdict_id,
            "ladder": verdict.ladder.value,
            "new_risk_allowed": verdict.new_risk_allowed,
            "degraded": verdict.degraded,
            "vetoes": len(verdict.vetoes),
        }
    )


# --------------------------------------------------------------------------
# target_portfolio — doc 07; advisory in, constraint-clamped target out
# --------------------------------------------------------------------------


@dagster_asset(
    name="target_portfolio",
    group_name=PORTFOLIO_GROUP,
    # meta_scoring is a lineage dep only: it materializes successfully even
    # when it skips (no production meta-model — the normal state), so it can
    # never block construction; the 23:45-before-00:30 schedule ordering is
    # the operational guarantee that scores exist when the flag is on.
    deps=["risk_verdict", "watchlist", "meta_scoring"],
)
def target_portfolio_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    from dynamo.domain.models.research import (
        CommitteeVerdict,
        QualityScore,
        QuantScore,
        Signal,
        WatchlistRow,
    )
    from dynamo.portfolio.config import ConstructionConfig
    from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor

    as_of = utcnow().date()
    repo = dynamo.market_repo()
    latest = _latest_risk_verdict(dynamo)
    if latest is None:
        context.log.warning("target_portfolio: no risk verdict — skipping")
        return MaterializeResult(metadata={"skipped": "no risk verdict"})
    verdict_id, risk = latest
    state = _portfolio_state(dynamo)
    if state is None:
        return MaterializeResult(metadata={"skipped": "no portfolio state"})

    wl = schema.watchlist
    wl_df = _read_df(dynamo, select(wl).where(wl.c.as_of == as_of))
    watchlist_rows: tuple[Any, ...] = ()
    if not wl_df.empty:
        wl_df["as_of"] = wl_df["as_of"].map(lambda v: pd.Timestamp(v).date())
        watchlist_rows = tuple(_rows_to_models(WatchlistRow, wl_df))

    verdicts: dict[str, CommitteeVerdict] = {}
    vf = repo.latest_verdicts()
    if not vf.empty:
        for payload in vf["payload"]:
            v = CommitteeVerdict.model_validate(payload)
            verdicts[v.symbol] = v

    quant: dict[str, QuantScore] = {}
    qf = repo.latest_quant_scores()
    if not qf.empty:
        qf = qf.copy()
        qf["as_of"] = qf["as_of"].map(lambda v: pd.Timestamp(v).date())
        for s in _rows_to_models(QuantScore, qf):
            quant[s.symbol] = s

    qs = schema.quality_scores
    quality_df = _read_df(dynamo, select(qs).where(qs.c.as_of >= as_of - timedelta(days=7)))
    quality: dict[str, QualityScore] = {}
    if not quality_df.empty:
        quality_df["as_of"] = quality_df["as_of"].map(lambda v: pd.Timestamp(v).date())
        for s in _rows_to_models(QualityScore, quality_df):
            quality[s.symbol] = s

    sg = schema.signals
    signals_df = _read_df(dynamo, select(sg).where(sg.c.as_of == as_of, sg.c.status == "active"))
    signals: tuple[Signal, ...] = ()
    if not signals_df.empty:
        signals_df["as_of"] = signals_df["as_of"].map(lambda v: pd.Timestamp(v).date())
        signals = tuple(_rows_to_models(Signal, signals_df))

    candidate_symbols = sorted(
        {r.symbol for r in watchlist_rows} | set(verdicts) | {p.symbol for p in state.positions}
    )
    bars = dynamo.bars_window(candidate_symbols or None, as_of, 120)
    sigma_annual: dict[str, float] = {}
    adv: dict[str, float] = {}
    if not bars.empty:
        for sym, grp in bars.groupby("symbol"):
            closes = grp.sort_values("ts")["close"].astype(float)
            if len(closes) > 21:
                sigma_annual[str(sym)] = float(closes.pct_change().std() * (252**0.5))
            adv[str(sym)] = float(
                (grp["close"].astype(float) * grp["volume"].astype(float)).tail(20).median()
            )
    ballast = _ballast_symbol(dynamo)
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector))
    sector_by_symbol = _with_ballast_sector(
        {
            str(r["symbol"]): str(r["sector"] or "unknown")
            for r in inst_df.to_dict(orient="records")
        },
        ballast,
    )

    target = RuleBasedConstructor().construct(
        AdvisoryInputs(
            as_of=utcnow(),
            watchlist=watchlist_rows,
            verdicts=verdicts,
            quant_scores=quant,
            quality_scores=quality,
            signals=signals,
            sigma_annual=sigma_annual,
            adv_dollar_20d=adv,
            sector_by_symbol=sector_by_symbol,
            # {} unless meta_sizing_enabled (dark path, plan Phase 3).
            effective_probability=_meta_effective_probability(dynamo, as_of),
        ),
        risk,
        state,
        # Idle-cash ballast (settings.ballast_enabled, default ON): the
        # residual above the cash floor rides SPY instead of dollars.
        ConstructionConfig(ballast_symbol=ballast),
        risk_verdict_ref=verdict_id,
    )
    repo.upsert_rows(
        schema.target_portfolios,
        [{"id": target.id, "as_of": target.as_of, "payload": target.model_dump(mode="json")}],
    )
    return MaterializeResult(
        metadata={
            "id": target.id,
            "weights": len(target.weights),
            "cash": target.cash,
            "changes": len(target.changes),
        }
    )


# --------------------------------------------------------------------------
# paper execution / reconciliation / stops — op jobs (doc 09)
# --------------------------------------------------------------------------


def _execution_ready(context: OpExecutionContext, dynamo: DynamoResource) -> bool:
    from dynamo.execution.gate import kill_switch_engaged

    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        context.log.warning("execution: alpaca keys unset — skipping")
        return False
    if s.dynamo_env not in (DynamoEnv.PAPER, DynamoEnv.LIVE):
        context.log.warning("execution: DYNAMO_ENV=%s is not PAPER — skipping", s.dynamo_env)
        return False
    if kill_switch_engaged(s.kill_switch_file):
        context.log.error("execution: kill switch engaged — refusing all orders")
        return False
    return True


def _verdict_expectations(
    dynamo: DynamoResource, verdict_ids: Any
) -> dict[str, tuple[float, float, int]]:
    """verdict_id -> (expected_upside_pct, expected_downside_pct, horizon_days).

    Denormalized onto new DecisionContext rows so path-based outcome scoring
    (trade_outcome_achieved) has real expectations to walk against.
    """
    ids = sorted({str(v) for v in verdict_ids if v})
    if not ids:
        return {}
    v = schema.committee_verdicts
    frame = _read_df(dynamo, select(v.c.id, v.c.payload).where(v.c.id.in_(ids)))
    out: dict[str, tuple[float, float, int]] = {}
    for r in frame.to_dict(orient="records"):
        payload = r["payload"] or {}
        out[str(r["id"])] = (
            float(payload.get("expected_upside_pct") or 0.0),
            float(payload.get("expected_downside_pct") or 0.0),
            int(payload.get("horizon_days") or 20),
        )
    return out


def _equity_gate(settings: Settings, readiness: Any = None) -> Callable[[Any], Any]:
    """The doc 09 §7 gate as a callable over equity/crypto Orders.

    The SAME function the options and PM sleeves use. Non-LIVE short-circuits
    to permitted *unless* the kill-switch is engaged; LIVE additionally
    requires the signed approval artifact covering the order's symbol (or
    ``all``) plus met readiness — fail-closed when either is absent.

    Crucially the kill-switch file is re-read inside the closure, so it is
    consulted on EVERY submission: a switch thrown mid-job blocks the very
    next order instead of being noticed at the start of the next job.
    """
    from pathlib import Path

    from dynamo.execution.gate import (
        Readiness,
        kill_switch_engaged,
        live_orders_permitted,
        load_approval,
    )

    approval = None
    if settings.dynamo_env == DynamoEnv.LIVE:
        path = Path(settings.approvals_dir) / f"live-{utcnow().date().isoformat()}.yaml"
        if path.exists():
            approval = load_approval(path)

    def gate(order: Any) -> Any:
        return live_orders_permitted(
            settings.dynamo_env,
            approval,
            kill_switch_engaged(settings.kill_switch_file),
            readiness if readiness is not None else Readiness(),
            scope_target=order.symbol,
        )

    return gate


# pm/options reconciliation runs are scope-prefixed; anything else in the table
# belongs to the equity/crypto sleeve — including rows written before the
# equity reconciler started labelling its runs, which is the fail-closed read.
_FOREIGN_HALT_PREFIXES = ("pm", "options")


def _equities_sticky_halt_active(dynamo: DynamoResource) -> bool:
    """Unacknowledged divergent equity reconciliation runs block new orders.

    The halt is sticky because it lives in the database, not in the
    ``HaltRegistry`` of whichever Reconciler instance happened to find it: the
    15-minute op builds a fresh Reconciler every cycle, so an in-memory halt
    would evaporate before the next order was ever proposed.
    """
    from dynamo.persistence.repositories.execution import ReconciliationRepository

    runs = ReconciliationRepository(dynamo.engine()).unacked_divergent_runs()
    return any(
        not str(r.get("scope", "")).startswith(_FOREIGN_HALT_PREFIXES) for r in runs
    )


def _order_manager(dynamo: DynamoResource, broker: Any) -> Any:
    """The equity/crypto OrderManager, gated and halt-aware per submission."""
    from dynamo.execution.order_manager import OrderManager
    from dynamo.persistence.repositories.execution import OrderRepository, TradeRepository

    s = dynamo.settings()
    engine = dynamo.engine()
    return OrderManager(
        broker,
        OrderRepository(engine),
        TradeRepository(engine),
        UtcClock(),
        auto_approve=s.auto_approve,
        approval_ttl_minutes=s.approval_ttl_minutes,
        gate=_equity_gate(s, dynamo.readiness()),
        halt_check=lambda _order: _equities_sticky_halt_active(dynamo),
    )


@op
def execute_paper_orders(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Propose -> risk-check -> approve -> submit the latest target (gated)."""
    from dynamo.domain.models.portfolio import DecisionContext, OrderState, TargetPortfolio
    from dynamo.domain.models.research import Sleeve
    from dynamo.execution.errors import ExecutionError
    from dynamo.persistence.repositories.learning import DecisionContextRepository

    if not _execution_ready(context, dynamo):
        return
    if _equities_sticky_halt_active(dynamo):
        context.log.error(
            "execute_paper_orders: sticky reconciliation halt active — refusing all orders"
        )
        return
    tp = schema.target_portfolios
    frame = _read_df(dynamo, select(tp).order_by(tp.c.as_of.desc()).limit(1))
    if frame.empty:
        context.log.warning("execute_paper_orders: no target portfolio — skipping")
        return
    target = TargetPortfolio.model_validate(frame.iloc[0]["payload"])
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("execute_paper_orders: no portfolio state — skipping")
        return
    as_of = utcnow().date()
    symbols = sorted({w.symbol for w in target.weights} | {p.symbol for p in state.positions})
    prices = _latest_prices(dynamo, symbols, as_of)
    broker = _broker(dynamo)
    if broker is None:
        return
    manager = _order_manager(dynamo, broker)
    manager.startup_reconciliation()  # doc 09 §4: broker truth before submission
    dc_repo = DecisionContextRepository(dynamo.engine())
    settings = dynamo.settings()

    verdict_by_symbol = {w.symbol: w.verdict_id for w in target.weights}
    expectations = _verdict_expectations(dynamo, verdict_by_symbol.values())
    orders = manager.propose_orders(target, state, prices)
    submitted = 0
    for order in orders:
        if order.state is not OrderState.PROPOSED:
            # Deterministic ids mean a re-run re-derives the same intents; an
            # order already past ``proposed`` was handled by the earlier run.
            context.log.info(
                "execute_paper_orders: %s already %s — skipping (idempotent)",
                order.client_order_id,
                order.state.value,
            )
            continue
        verdict_id = verdict_by_symbol.get(order.symbol)
        # Committee expectations denormalized from the verdict where one
        # exists; (0, 0, 20) only for verdict-less orders (e.g. pure exits).
        upside, downside, horizon = expectations.get(verdict_id or "", (0.0, 0.0, 20))
        dc = DecisionContext(
            id=f"dc-{order.client_order_id}",
            created_at=utcnow(),
            env=settings.dynamo_env.value,
            symbol=order.symbol,
            sleeve=order.sleeve or Sleeve.LONG_TERM,
            verdict_id=verdict_id,
            target_portfolio_id=target.id,
            payload={
                "order": order.model_dump(mode="json"),
                "target_portfolio_id": target.id,
                "risk_verdict_ref": target.risk_verdict_ref,
                "decision_price": prices.get(order.symbol),
            },
            expected_upside_pct=upside,
            expected_downside_pct=downside,
            horizon_days=horizon,
        )
        dc_id = dc_repo.append(dc)
        # proposed -> risk_checked -> pending_approval: the constructor already
        # clamped this order to the risk verdict's constraints, and skipping
        # the hop is an IllegalTransition, not a shortcut (doc 09 §2).
        manager.transition(
            order.client_order_id, OrderState.RISK_CHECKED, "target diff within risk verdict"
        )
        manager.attach_decision_context(order.client_order_id, dc_id)
        manager.request_approval(order.client_order_id)
        if settings.auto_approve:
            try:
                manager.submit(order.client_order_id)
                submitted += 1
            except ExecutionError as exc:
                # A refusal (gate, kill-switch, halt) stops THIS order; the
                # rest of the batch is still evaluated and refused in turn.
                context.log.error(
                    "execute_paper_orders: %s not submitted: %s", order.client_order_id, exc
                )
    context.log.info(
        "execute_paper_orders: %d proposed, %d submitted for target %s",
        len(orders),
        submitted,
        target.id,
    )


@job
def paper_execution_job() -> None:
    """Gated paper-trading execution of the latest approved target."""
    execute_paper_orders()


class _InternalReader:
    """InternalStateReader over our repositories (doc 09 reconciliation)."""

    def __init__(self, dynamo: DynamoResource) -> None:
        self._dynamo = dynamo

    def positions(self) -> Any:
        from dynamo.persistence.repositories.execution import PositionRepository

        return PositionRepository(self._dynamo.engine()).all_positions()

    def open_order_ids(self) -> Any:
        from dynamo.persistence.repositories.execution import OrderRepository

        return [
            (o.client_order_id, o.state, o.symbol)
            for o in OrderRepository(self._dynamo.engine()).open_orders()
        ]

    def _equity_row(self) -> tuple[float, float]:
        eq = schema.equity_curve
        frame = _read_df(self._dynamo, select(eq).order_by(eq.c.as_of.desc()).limit(1))
        if frame.empty:
            return 0.0, 0.0
        return float(frame.iloc[0]["cash"]), float(frame.iloc[0]["equity"])

    def cash(self) -> float:
        return self._equity_row()[0]

    def equity(self) -> float:
        return self._equity_row()[1]


@op(out=Out(Nothing))
def poll_equity_fills(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Poll live equity/crypto orders for fills; write trades AND positions.

    The missing half of the equity loop: submission wrote an order and nothing
    ever asked the broker what became of it, so a real fill left ``trades`` and
    ``positions`` empty and the reconciler below — correctly — halted the
    sleeve for divergence. Mirrors the options/PM sync_orders + apply_fill
    pattern; partial fills top up on the next poll and re-polling an unchanged
    order records nothing.
    """
    from dynamo.persistence.repositories.execution import PositionRepository

    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        context.log.info("equity_fills: no broker configured — skipping")
        return
    broker = _broker(dynamo)
    if broker is None:
        return
    manager = _order_manager(dynamo, broker)
    trades = manager.sync_orders()
    positions = PositionRepository(dynamo.engine())
    now = utcnow()
    for trade in trades:
        positions.apply_fill(trade, now)
    context.log.info("equity_fills: %d new fill(s) recorded", len(trades))


@job
def equity_fills_job() -> None:
    """Manual/ad-hoc equity fill poll (also runs inside reconciliation_job)."""
    poll_equity_fills()


@op(ins={"fills_polled": In(Nothing)})
def reconcile_broker_state(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """15-minute positions/orders/cash reconciliation during market hours.

    Runs *after* the fill poller in the same job: reconciling before recording
    known fills would flag every one of them as a divergence.
    """
    from dynamo.execution.reconciliation import Reconciler
    from dynamo.persistence.repositories.execution import ReconciliationRepository

    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        context.log.info("reconciliation: no broker configured — skipping")
        return
    def alert(name: str, message: str) -> None:
        dynamo.alert("page", name, message)

    broker = _broker(dynamo)
    if broker is None:
        return
    reconciler = Reconciler(
        broker,
        _InternalReader(dynamo),
        ReconciliationRepository(dynamo.engine()),
        alert,
        cash_tolerance=s.reconciliation_cash_tolerance,
    )
    result = reconciler.reconcile(utcnow())
    context.log.info(
        "reconciliation: %d divergences, halted scopes %s",
        len(result.divergences),
        sorted(result.halted_scopes),
    )


@job
def reconciliation_job() -> None:
    reconcile_broker_state(poll_equity_fills())


def _pdt_filtered_specs(
    context: OpExecutionContext, dynamo: DynamoResource, specs: list[Any], equity: float
) -> list[Any]:
    """PDT guard (ADR-0010): a same-day-entered name is never auto-exited when
    the exit would trip the pattern day-trader rule under $25k equity — log +
    alert instead, UNLESS it is a disaster stop."""
    from dynamo.domain.models.risk import StopKind
    from dynamo.risk.intraday import count_day_trades, pdt_exit_allowed, pdt_relevant_fills

    today = utcnow().date()
    t = schema.trades
    frame = _read_df(
        dynamo,
        select(t.c.symbol, t.c.side, t.c.filled_at).where(
            t.c.filled_at >= datetime.combine(today - timedelta(days=7), time.min, tzinfo=UTC)
        ),
    )
    # PDT is a securities-account rule: crypto fills are excluded up front
    # (ADR-0013), so crypto never counts a day trade and a crypto stop exit
    # is never deferred — its symbol cannot appear in ``entered_today``.
    fills: list[tuple[str, str, date]] = pdt_relevant_fills(
        (str(r["symbol"]), str(r["side"]), _pydate(r["filled_at"]))
        for r in frame.to_dict(orient="records")
    )
    entered_today = {sym for sym, side, day in fills if side == "buy" and day == today}
    day_trades = count_day_trades(fills)
    allowed: list[Any] = []
    for spec in specs:
        decision = pdt_exit_allowed(
            equity=equity,
            day_trades_in_window=day_trades,
            entered_today=spec.symbol in entered_today,
            is_disaster_stop=spec.kind == StopKind.DISASTER,
        )
        if decision.allowed:
            allowed.append(spec)
        else:
            message = (
                f"stop exit for {spec.symbol} deferred: {decision.reason} "
                f"(same-day entry, equity ${equity:,.0f}, "
                f"{day_trades} day trades in window)"
            )
            context.log.warning("stops: %s", message)
            dynamo.alert("warn", "pdt_stop_deferred", message)
    return allowed


@op
def maintain_stops(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Apply the risk engine's stop updates through the order state machine."""
    from dynamo.domain.models.portfolio import DecisionContext as DC
    from dynamo.domain.models.research import Sleeve
    from dynamo.domain.symbols import is_crypto_symbol
    from dynamo.execution.stops_maintenance import StopsMaintainer
    from dynamo.persistence.repositories.execution import PositionRepository
    from dynamo.persistence.repositories.learning import DecisionContextRepository

    if not _execution_ready(context, dynamo):
        return
    if _equities_sticky_halt_active(dynamo):
        context.log.error("stops: sticky reconciliation halt active — refusing all orders")
        return
    latest = _latest_risk_verdict(dynamo)
    if latest is None:
        context.log.info("stops: no risk verdict yet — skipping")
        return
    verdict_id, risk = latest
    specs = list(risk.stop_updates)
    if not specs:
        context.log.info("stops: no stop updates in %s", verdict_id)
        return
    state = _portfolio_state(dynamo)
    specs = _pdt_filtered_specs(context, dynamo, specs, state.equity if state else 0.0)
    if not specs:
        context.log.info("stops: all stop updates deferred by the PDT guard")
        return
    positions = {p.symbol: p for p in PositionRepository(dynamo.engine()).all_positions()}
    broker = _broker(dynamo)
    if broker is None:
        return
    manager = _order_manager(dynamo, broker)
    manager.startup_reconciliation()  # doc 09 §4: broker truth before submission
    dc_repo = DecisionContextRepository(dynamo.engine())
    settings = dynamo.settings()
    dc_ids: dict[str, str] = {}
    for spec in specs:
        if spec.symbol not in positions:
            continue
        dc_ids[spec.symbol] = dc_repo.append(
            DC(
                id=f"dc-stop-{spec.symbol}-{utcnow().date().isoformat()}",
                created_at=utcnow(),
                env=settings.dynamo_env.value,
                symbol=spec.symbol,
                # Crypto stop maintenance is attributed to its sleeve (ADR-0013).
                sleeve=Sleeve.CRYPTO if is_crypto_symbol(spec.symbol) else Sleeve.LONG_TERM,
                verdict_id=None,
                target_portfolio_id="stops",
                payload={"stop_spec": spec.model_dump(mode="json"), "risk_verdict_ref": verdict_id},
                # Genuinely no committee expectations here: stop maintenance is
                # a risk-engine action with no verdict. Zeros mean "no forecast"
                # and calibration falls back to sign-of-pnl for such rows.
                expected_upside_pct=0.0,
                expected_downside_pct=0.0,
                horizon_days=0,
            )
        )
    maintainer = StopsMaintainer(broker=broker, manager=manager, clock=UtcClock())
    touched = maintainer.apply_stops(specs, positions, dc_ids)
    context.log.info("stops: %d stop orders touched", len(touched))


@job
def stops_job() -> None:
    maintain_stops()


PORTFOLIO_ASSETS = [risk_verdict_asset, target_portfolio_asset]
EXECUTION_JOBS = [paper_execution_job, reconciliation_job, equity_fills_job, stops_job]
