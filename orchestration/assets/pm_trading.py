"""Prediction-market trading sleeve wiring — ADR-0009 (Kalshi), ADR-0011
(Polymarket shadow sleeve), docs 06/09 discipline.

Kalshi is demo-first: the broker talks to the Kalshi demo exchange unless
DYNAMO_ENV=LIVE, and every submission passes the SAME live gate as equities
(kill-switch checked in every environment; LIVE additionally needs a signed,
in-scope, unexpired approval and met readiness). The risk engine
(risk/pm_sizing) has final authority over what trades; every order carries a
hash-chained DecisionContext.

The Polymarket sleeve is SIMULATED, always (ADR-0011): the same nightly
pipeline sizes it against the sim bankroll, but the "broker" is the in-house
PolymarketSimBroker — fills happen internally at Gamma prices + slippage and
NO real order is ever sent. Polymarket has no real execution wiring here or
anywhere.
"""

import math
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
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
from sqlalchemy import select, update

from dynamo.config.settings import DynamoEnv, Settings
from dynamo.persistence import schema
from orchestration.assets.portfolio_exec import UtcClock, _portfolio_state
from orchestration.assets.research import _pydatetime, _read_df
from orchestration.resources import DynamoResource, utcnow

PM_GROUP = "prediction_markets"
PM_VENUE = "kalshi"
PM_SIM_VENUE = "polymarket"  # SIMULATED sleeve — internal fills only (ADR-0011)


def _pm_broker(dynamo: DynamoResource) -> Any | None:
    """KalshiBroker when keys + signer are available; None → skip (demo by default)."""
    s = dynamo.settings()
    if not (s.kalshi_api_key_id and s.kalshi_private_key_path):
        return None
    from dynamo.execution.kalshi_broker import KalshiBroker

    broker = KalshiBroker(s)
    return broker if broker.configured else None


def _pm_gate(settings: Settings, scope_fn: Callable[[str], str] | None = None) -> Any:
    """The doc 09 §7 gate as a callable over PMOrders (same function as equities).

    Non-LIVE short-circuits to permitted (unless the kill-switch is engaged);
    LIVE requires the signed approval artifact covering the order's scope
    (``KALSHI:<market_id>`` — or ``POLYMARKET:<market_id>`` for the simulated
    sleeve — or ``all``) plus readiness — fail-closed when either is absent.
    """
    from dynamo.execution.gate import (
        Readiness,
        kill_switch_engaged,
        live_orders_permitted,
        load_approval,
    )
    from dynamo.execution.pm_order_manager import pm_scope

    scope = scope_fn if scope_fn is not None else pm_scope
    approval = None
    if settings.dynamo_env == DynamoEnv.LIVE:
        path = Path(settings.approvals_dir) / f"live-{utcnow().date().isoformat()}.yaml"
        if path.exists():
            approval = load_approval(path)

    def gate(order: Any) -> Any:
        return live_orders_permitted(
            settings.dynamo_env,
            approval,
            kill_switch_engaged(settings.kill_switch_path),
            Readiness(),  # LIVE readiness comes from stored metrics; unset → refuse
            scope_target=scope(order.market_id),
        )

    return gate


def _pm_manager(dynamo: DynamoResource, broker: Any) -> Any:
    from dynamo.execution.pm_order_manager import PMOrderManager
    from dynamo.persistence.repositories.pm import PMOrderRepository, PMTradeRepository

    engine = dynamo.engine()
    return PMOrderManager(
        broker,
        PMOrderRepository(engine),
        PMTradeRepository(engine),
        UtcClock(),
        gate=_pm_gate(dynamo.settings()),
    )


def _pm_sticky_halt_active(dynamo: DynamoResource) -> bool:
    """Unacknowledged divergent PM reconciliation runs block new orders (sticky)."""
    from dynamo.persistence.repositories.execution import ReconciliationRepository

    runs = ReconciliationRepository(dynamo.engine()).unacked_divergent_runs()
    return any(str(r.get("scope", "")).startswith("pm") for r in runs)


def _latest_pm_frames(
    dynamo: DynamoResource, venue: str = PM_VENUE
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Latest view / price / market per market_id for one venue, as domain models."""
    from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
    from dynamo.domain.models.research import PredictionMarketView

    v = schema.prediction_market_views
    views_df = _read_df(
        dynamo,
        select(v).where(v.c.venue == venue).order_by(v.c.as_of.desc()),
    )
    views: dict[str, Any] = {}
    if not views_df.empty:
        views_df = views_df.drop_duplicates(subset=["market_id"], keep="first")
        for payload in views_df["payload"]:
            view = PredictionMarketView.model_validate(payload)
            views[view.market_id] = view

    p = schema.prediction_market_prices
    prices_df = _read_df(
        dynamo,
        select(p).where(p.c.venue == venue).order_by(p.c.ts.desc()),
    )
    prices: dict[str, Any] = {}
    if not prices_df.empty:
        prices_df = prices_df.drop_duplicates(subset=["market_id"], keep="first")
        for row in prices_df.to_dict(orient="records"):
            prices[str(row["market_id"])] = PredictionMarketPrice(
                venue=row["venue"],
                market_id=row["market_id"],
                ts=_pydatetime(row["ts"]),
                implied_prob=row["implied_prob"],
                volume=row["volume"],
                liquidity=row["liquidity"],
                trusted=bool(row["trusted"]),
            )

    m = schema.prediction_markets
    markets_df = _read_df(dynamo, select(m).where(m.c.venue == venue))
    markets: dict[str, Any] = {}
    for row in markets_df.to_dict(orient="records"):
        close_ts = row.get("close_ts")
        markets[str(row["market_id"])] = PredictionMarket(
            venue=row["venue"],
            market_id=row["market_id"],
            question=row["question"],
            category=row.get("category"),
            close_ts=(
                _pydatetime(close_ts).astimezone(UTC)
                if close_ts is not None and not pd.isna(close_ts)
                else None
            ),
            resolved_outcome=(
                bool(row["resolved_outcome"])
                if row.get("resolved_outcome") is not None
                else None
            ),
        )
    return views, prices, markets


def _latest_cross_venue(dynamo: DynamoResource) -> dict[str, float]:
    """Latest OddsPipe Polymarket YES price per kalshi_market_id ({} when none).

    Corroboration input for select_and_size_pm_trades (ADR-0009 addendum);
    an empty table — no OddsPipe key, API down — simply yields {}.
    """
    s = schema.pm_cross_venue_spreads
    frame = _read_df(dynamo, select(s).order_by(s.c.ts.desc()))
    if frame.empty:
        return {}
    frame = frame.drop_duplicates(subset=["kalshi_market_id"], keep="first")
    mapping: dict[str, float] = {}
    for row in frame.to_dict(orient="records"):
        raw = row.get("polymarket_yes")
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            mapping[str(row["kalshi_market_id"])] = value
    return mapping


def _latest_kalshi_mirror(dynamo: DynamoResource) -> dict[str, float]:
    """Latest OddsPipe Kalshi YES price per polymarket_id ({} when none).

    The symmetric twin of :func:`_latest_cross_venue`: corroboration input for
    the SIMULATED Polymarket sleeve wherever Kalshi lists the mirror market.
    """
    s = schema.pm_cross_venue_spreads
    frame = _read_df(dynamo, select(s).order_by(s.c.ts.desc()))
    if frame.empty:
        return {}
    frame = frame.drop_duplicates(subset=["polymarket_id"], keep="first")
    mapping: dict[str, float] = {}
    for row in frame.to_dict(orient="records"):
        raw = row.get("kalshi_yes")
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            mapping[str(row["polymarket_id"])] = value
    return mapping


def _apply_new_fills(dynamo: DynamoResource, manager: Any, markets: dict[str, Any]) -> int:
    """Record fills discovered by sync_orders into pm_positions."""
    from dynamo.persistence.repositories.pm import PMPositionRepository

    positions = PMPositionRepository(dynamo.engine())
    trades = manager.sync_orders()
    now = utcnow()
    for trade in trades:
        market = markets.get(trade.market_id)
        positions.apply_fill(trade, now, question=market.question if market else "")
    return len(trades)


def _apply_new_sim_fills(
    dynamo: DynamoResource, manager: Any, markets: dict[str, Any], ledger: Any
) -> int:
    """Simulated fills → pm_positions rows + idempotent sim-bankroll debits."""
    from dynamo.execution.polymarket_sim import fill_debit_entry
    from dynamo.persistence.repositories.pm import PMPositionRepository

    positions = PMPositionRepository(dynamo.engine())
    trades = manager.sync_orders()
    now = utcnow()
    for trade in trades:
        market = markets.get(trade.market_id)
        positions.apply_fill(trade, now, question=market.question if market else "")
        ledger.append(fill_debit_entry(trade, now))
    return len(trades)


# --------------------------------------------------------------------------
# pm_trading_job — daily 20:30 ET, after the 20:00 prediction-market snapshot
# --------------------------------------------------------------------------


@op
def pm_execute_orders(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Views + prices → risk.pm_sizing → gate-checked Kalshi orders."""
    from dynamo.domain.models.portfolio import DecisionContext
    from dynamo.domain.models.research import Sleeve
    from dynamo.execution.errors import ExecutionError
    from dynamo.execution.pm_order_manager import pm_scope
    from dynamo.persistence.repositories.learning import DecisionContextRepository
    from dynamo.persistence.repositories.pm import PMPositionRepository
    from dynamo.risk.pm_sizing import PMSizingParams, select_and_size_pm_trades

    settings = dynamo.settings()
    broker = _pm_broker(dynamo)
    if broker is None:
        context.log.warning("pm_trading: kalshi keys unset or signer unavailable — skipping")
        return
    if _pm_sticky_halt_active(dynamo):
        context.log.error("pm_trading: sticky reconciliation halt active — refusing all orders")
        return

    views, prices, markets = _latest_pm_frames(dynamo)
    if not views:
        context.log.info("pm_trading: no kalshi views — nothing to size")
        return
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("pm_trading: no portfolio state — skipping")
        return
    open_positions = PMPositionRepository(dynamo.engine()).open_positions(venue=PM_VENUE)
    cross_venue = _latest_cross_venue(dynamo)
    if cross_venue:
        context.log.info("pm_trading: %d cross-venue matches (OddsPipe)", len(cross_venue))

    result = select_and_size_pm_trades(
        list(views.values()),
        prices,
        markets,
        state.equity,
        open_positions,
        utcnow(),
        PMSizingParams(
            sleeve_cap=settings.pm_sleeve_cap,
            per_market_cap=settings.pm_per_market_cap,
            min_edge=settings.pm_min_edge,
            min_confidence=settings.pm_min_confidence,
            max_days_to_resolution=settings.pm_max_days_to_resolution,
            kelly_fraction_cap=settings.pm_kelly_fraction_cap,
            max_open_markets=settings.pm_max_open_markets,
            cross_venue_veto_gap=settings.pm_cross_venue_veto_gap,
        ),
        cross_venue=cross_venue or None,
    )
    for market_id, reason in result.skipped:
        context.log.info("pm_trading: skipped %s (%s)", market_id, reason)
    if not result.decisions:
        context.log.info("pm_trading: no trades pass the sleeve rules")
        return

    manager = _pm_manager(dynamo, broker)
    dc_repo = DecisionContextRepository(dynamo.engine())
    as_of = utcnow().date()
    decisions_by_market = {d.market_id: d for d in result.decisions}
    orders = manager.propose(result.decisions, as_of)
    submitted = 0
    for order in orders:
        decision = decisions_by_market[order.market_id]
        dc = DecisionContext(
            id=f"dc-{order.client_order_id}",
            created_at=utcnow(),
            env=settings.dynamo_env.value,
            symbol=pm_scope(order.market_id),
            sleeve=Sleeve.TACTICAL,
            verdict_id=None,
            target_portfolio_id=f"pm-{as_of.isoformat()}",
            payload={
                "kind": "prediction_market",
                "venue": order.venue,
                "market_id": order.market_id,
                "decision": decision.model_dump(mode="json"),
                "view_ref": decision.view_ref,
                "order": order.model_dump(mode="json"),
            },
            expected_upside_pct=0.0,
            expected_downside_pct=0.0,
            horizon_days=settings.pm_max_days_to_resolution,
        )
        dc_id = dc_repo.append(dc)
        manager.attach_decision_context(order.client_order_id, dc_id)
        manager.request_approval(order.client_order_id)
        if settings.auto_approve:
            try:
                manager.submit(order.client_order_id)
                submitted += 1
            except ExecutionError as exc:
                context.log.error("pm_trading: %s not submitted: %s", order.client_order_id, exc)
    fills = _apply_new_fills(dynamo, manager, markets)
    context.log.info(
        "pm_trading: %d sized, %d submitted, %d immediate fills (env=%s, base=%s)",
        len(orders),
        submitted,
        fills,
        settings.dynamo_env.value,
        broker.base_url,
    )


@op
def pm_sim_execute_orders(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Polymarket shadow sleeve: same sizing gates, SIMULATED internal fills.

    NO real order is ever sent (ADR-0011). Decisions run through
    risk.pm_sizing against the sim bankroll; the "broker" is
    PolymarketSimBroker, which fills internally at a fresh Gamma price plus
    slippage and rejects anything stale, closed, or overdrawing the bankroll.
    """
    from dynamo.domain.models.portfolio import DecisionContext, OrderState
    from dynamo.domain.models.research import Sleeve
    from dynamo.execution.errors import ExecutionError
    from dynamo.execution.pm_order_manager import PMOrderManager
    from dynamo.execution.polymarket_sim import PolymarketSimBroker, SimQuote, pm_sim_scope
    from dynamo.persistence.repositories.learning import DecisionContextRepository
    from dynamo.persistence.repositories.pm import (
        PMOrderRepository,
        PMPositionRepository,
        PMSimLedgerRepository,
        PMTradeRepository,
    )
    from dynamo.risk.pm_sizing import PMSizingParams, select_and_size_pm_trades

    settings = dynamo.settings()
    if not settings.polymarket_sim_enabled:
        context.log.info("pm_sim_trading: sleeve disabled — skipping")
        return
    provider = dynamo.polymarket()
    if provider is None:
        context.log.warning("pm_sim_trading: polymarket provider unavailable — skipping")
        return

    views, prices, markets = _latest_pm_frames(dynamo, PM_SIM_VENUE)
    if not views:
        context.log.info("pm_sim_trading: no polymarket views — nothing to size")
        return

    engine = dynamo.engine()
    now = utcnow()
    ledger = PMSimLedgerRepository(engine)
    ledger.ensure_seeded(round(settings.polymarket_sim_bankroll_usd * 100.0, 2), now)
    cash_cents = ledger.cash_cents()
    open_positions = PMPositionRepository(engine).open_positions(venue=PM_SIM_VENUE)
    # SIM equity = bankroll cash + open cost basis: every cap (quarter-Kelly,
    # 1% per market, 5% sleeve) binds against the sim bankroll, never the
    # real portfolio.
    sim_equity = cash_cents / 100.0 + sum(p.exposure_dollars for p in open_positions)

    cross_venue = _latest_kalshi_mirror(dynamo)
    if cross_venue:
        context.log.info("pm_sim_trading: %d kalshi mirror matches (OddsPipe)", len(cross_venue))
    result = select_and_size_pm_trades(
        list(views.values()),
        prices,
        markets,
        sim_equity,
        open_positions,
        now,
        PMSizingParams(
            sleeve_cap=settings.pm_sleeve_cap,
            per_market_cap=settings.pm_per_market_cap,
            min_edge=settings.pm_min_edge,
            min_confidence=settings.pm_min_confidence,
            max_days_to_resolution=settings.pm_max_days_to_resolution,
            kelly_fraction_cap=settings.pm_kelly_fraction_cap,
            max_open_markets=settings.pm_max_open_markets,
            cross_venue_veto_gap=settings.pm_cross_venue_veto_gap,
            venue=PM_SIM_VENUE,
        ),
        cross_venue=cross_venue or None,
    )
    for market_id, reason in result.skipped:
        context.log.info("pm_sim_trading: skipped %s (%s)", market_id, reason)
    if not result.decisions:
        context.log.info("pm_sim_trading: no trades pass the sleeve rules")
        return

    # Fresh Gamma quotes for the decided markets only (cheap keyless reads);
    # the sim broker fails closed on anything missing, stale, or closed.
    quotes = {
        q.market_id: SimQuote(yes_price=q.yes_price, ts=q.ts, closed=q.closed)
        for q in provider.quotes([d.market_id for d in result.decisions])
        if q.yes_price is not None
    }
    broker = PolymarketSimBroker(
        quotes=quotes,
        cash_cents=cash_cents,
        now=now,
        slippage_cents=settings.polymarket_sim_slippage_cents,
    )
    manager = PMOrderManager(
        broker,
        PMOrderRepository(engine),
        PMTradeRepository(engine),
        UtcClock(),
        gate=_pm_gate(settings, scope_fn=pm_sim_scope),
    )
    dc_repo = DecisionContextRepository(engine)
    as_of = now.date()
    decisions_by_market = {d.market_id: d for d in result.decisions}
    orders = manager.propose(result.decisions, as_of)
    submitted = 0
    for order in orders:
        if order.state is not OrderState.PROPOSED:
            # Same decision already placed today — deterministic ids dedupe it.
            context.log.info(
                "pm_sim_trading: %s already %s — skipping (idempotent)",
                order.client_order_id,
                order.state.value,
            )
            continue
        decision = decisions_by_market[order.market_id]
        dc = DecisionContext(
            id=f"dc-{order.client_order_id}",
            created_at=now,
            env=settings.dynamo_env.value,
            symbol=pm_sim_scope(order.market_id),
            sleeve=Sleeve.TACTICAL,
            verdict_id=None,
            target_portfolio_id=f"pm-sim-{as_of.isoformat()}",
            payload={
                "kind": "prediction_market",
                "simulated": True,  # SIMULATED — no real order is ever sent
                "venue": order.venue,
                "market_id": order.market_id,
                "decision": decision.model_dump(mode="json"),
                "view_ref": decision.view_ref,
                "order": order.model_dump(mode="json"),
            },
            expected_upside_pct=0.0,
            expected_downside_pct=0.0,
            horizon_days=settings.pm_max_days_to_resolution,
        )
        dc_id = dc_repo.append(dc)
        manager.attach_decision_context(order.client_order_id, dc_id)
        manager.request_approval(order.client_order_id)
        if settings.auto_approve:
            try:
                manager.submit(order.client_order_id)
                submitted += 1
            except ExecutionError as exc:
                context.log.error(
                    "pm_sim_trading: %s not filled: %s", order.client_order_id, exc
                )
    fills = _apply_new_sim_fills(dynamo, manager, markets, ledger)
    context.log.info(
        "pm_sim_trading: %d sized, %d submitted, %d SIMULATED fills, "
        "sim bankroll $%.2f (no real orders sent)",
        len(orders),
        submitted,
        fills,
        ledger.cash_cents() / 100.0,
    )


@job
def pm_trading_job() -> None:
    """Gated Kalshi sleeve execution (demo unless LIVE; ADR-0009) plus the
    SIMULATED Polymarket shadow sleeve (ADR-0011)."""
    pm_execute_orders()
    pm_sim_execute_orders()


# --------------------------------------------------------------------------
# pm_reconciliation_job — 15-minute loop alongside the equity reconciler
# --------------------------------------------------------------------------


@op
def pm_reconcile_broker_state(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Order sync + positions/balance reconciliation vs Kalshi; sticky halt."""
    from dynamo.execution.pm_order_manager import PMReconciler
    from dynamo.persistence.repositories.execution import ReconciliationRepository
    from dynamo.persistence.repositories.pm import PMPositionRepository

    broker = _pm_broker(dynamo)
    if broker is None:
        context.log.info("pm_reconciliation: no kalshi broker configured — skipping")
        return
    _views, _prices, markets = _latest_pm_frames(dynamo)
    manager = _pm_manager(dynamo, broker)
    fills = _apply_new_fills(dynamo, manager, markets)

    repo = dynamo.market_repo()

    def alert(name: str, message: str) -> None:
        repo.add_alert(severity="page", name=name, message=message)

    reconciler = PMReconciler(
        broker, ReconciliationRepository(dynamo.engine()), alert
    )
    result = reconciler.reconcile(
        utcnow(), PMPositionRepository(dynamo.engine()).open_positions(venue=PM_VENUE)
    )
    context.log.info(
        "pm_reconciliation: %d new fills, %d divergences, halted scopes %s",
        fills,
        len(result.divergences),
        sorted(result.halted_scopes),
    )


@job
def pm_reconciliation_job() -> None:
    pm_reconcile_broker_state()


# --------------------------------------------------------------------------
# pm_resolution — settle resolved markets, realize P&L, feed calibration
# --------------------------------------------------------------------------


@dagster_asset(name="pm_resolution", group_name=PM_GROUP)
def pm_resolution_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Resolved Kalshi markets → pm_trades settlements + calibration points."""
    from dynamo.domain.models.pm import PMSide, PMTrade, settlement_pnl_cents
    from dynamo.learning.calibration import CalibrationObservation, compute_calibration
    from dynamo.persistence.repositories.learning import CalibrationRepository
    from dynamo.persistence.repositories.pm import PMPositionRepository, PMTradeRepository

    as_of: date = utcnow().date()
    now: datetime = utcnow()
    m = schema.prediction_markets
    resolved_df = _read_df(
        dynamo,
        select(m).where(m.c.venue == PM_VENUE, m.c.resolved_outcome.is_not(None)),
    )
    if resolved_df.empty:
        return MaterializeResult(metadata={"settlements": 0, "observations": 0})
    resolved = {
        str(r["market_id"]): bool(r["resolved_outcome"])
        for r in resolved_df.to_dict(orient="records")
    }

    positions_repo = PMPositionRepository(dynamo.engine())
    trades_repo = PMTradeRepository(dynamo.engine())
    settlements = 0
    realized_cents = 0.0
    for position in positions_repo.open_positions(venue=PM_VENUE):
        outcome = resolved.get(position.market_id)
        if outcome is None:
            continue
        pnl = settlement_pnl_cents(
            position.side, position.contracts, position.avg_price_cents, outcome
        )
        won = outcome if position.side is PMSide.YES else not outcome
        trades_repo.insert(
            PMTrade(
                client_order_id=None,
                venue=position.venue,
                market_id=position.market_id,
                side=position.side,
                contracts=position.contracts,
                price_cents=100.0 if won else 0.0,
                kind="settlement",
                realized_pnl_cents=pnl,
                filled_at=now,
            )
        )
        positions_repo.close(position.venue, position.market_id)
        settlements += 1
        realized_cents += pnl

    # Calibration: our estimated probabilities vs realized outcomes, so the
    # learning loop can score the committee's prediction-market estimates.
    v = schema.prediction_market_views
    views_df = _read_df(
        dynamo,
        select(v)
        .where(v.c.venue == PM_VENUE, v.c.market_id.in_(list(resolved)))
        .order_by(v.c.as_of.desc()),
    )
    observations: list[CalibrationObservation] = []
    if not views_df.empty:
        views_df = views_df.drop_duplicates(subset=["market_id"], keep="first")
        for row in views_df.to_dict(orient="records"):
            payload = row["payload"] if isinstance(row["payload"], dict) else {}
            estimated = payload.get("estimated_prob")
            if estimated is None:
                continue
            observations.append(
                CalibrationObservation(
                    scope_type="prediction_market",
                    scope_id=PM_VENUE,
                    conviction=float(estimated),
                    outcome=resolved[str(row["market_id"])],
                )
            )
    if observations:
        points = compute_calibration(observations, as_of)
        CalibrationRepository(dynamo.engine()).upsert(points)
    return MaterializeResult(
        metadata={
            "settlements": settlements,
            "realized_pnl_usd": realized_cents / 100.0,
            "observations": len(observations),
        }
    )


# --------------------------------------------------------------------------
# pm_sim_resolution — settle the SIMULATED Polymarket sleeve (ADR-0011)
# --------------------------------------------------------------------------


@dagster_asset(name="pm_sim_resolution", group_name=PM_GROUP)
def pm_sim_resolution_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Gamma-resolved markets → SIMULATED settlements on the sim bankroll.

    Win → contracts * $1 credited to the sim bankroll, loss → nothing; both
    are recorded as settlement trades and idempotent ledger credits (settling
    twice is a no-op). Unresolved markets are held to resolution, matching
    the Kalshi sleeve's exit semantics.
    """
    from dynamo.execution.polymarket_sim import settle_sim_positions
    from dynamo.persistence.repositories.pm import (
        PMPositionRepository,
        PMSimLedgerRepository,
        PMTradeRepository,
    )

    settings = dynamo.settings()
    if not settings.polymarket_sim_enabled:
        return MaterializeResult(metadata={"skipped": "polymarket sim disabled"})
    engine = dynamo.engine()
    positions_repo = PMPositionRepository(engine)
    positions = positions_repo.open_positions(venue=PM_SIM_VENUE)
    if not positions:
        return MaterializeResult(metadata={"simulated": True, "settlements": 0})
    provider = dynamo.polymarket()
    if provider is None:
        return MaterializeResult(metadata={"skipped": "polymarket provider unavailable"})

    resolved: dict[str, bool] = {}
    for quote in provider.quotes([p.market_id for p in positions]):
        if quote.resolved_yes is not None:
            resolved[quote.market_id] = quote.resolved_yes
    if resolved:  # keep prediction_markets in sync for calibration/dashboard
        m = schema.prediction_markets
        with engine.begin() as cx:
            for market_id, outcome in resolved.items():
                cx.execute(
                    update(m)
                    .where(m.c.venue == PM_SIM_VENUE, m.c.market_id == market_id)
                    .values(resolved_outcome=outcome)
                )

    ledger = PMSimLedgerRepository(engine)
    settlements, realized_cents = settle_sim_positions(
        positions, resolved, PMTradeRepository(engine), positions_repo, ledger, utcnow()
    )
    context.log.info(
        "pm_sim_resolution: %d SIMULATED settlements, realized $%.2f, sim bankroll $%.2f",
        settlements,
        realized_cents / 100.0,
        ledger.cash_cents() / 100.0,
    )
    return MaterializeResult(
        metadata={
            "simulated": True,
            "settlements": settlements,
            "realized_pnl_usd": realized_cents / 100.0,
            "sim_bankroll_usd": ledger.cash_cents() / 100.0,
        }
    )


PM_ASSETS = [pm_resolution_asset, pm_sim_resolution_asset]
PM_JOBS = [pm_trading_job, pm_reconciliation_job]
