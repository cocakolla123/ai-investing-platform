"""Options sleeve wiring — defined-risk only (ADR-0010, docs 06/09 discipline).

Paper-first: Alpaca supports options in PAPER accounts, and every submission
passes the SAME live gate as equities (kill-switch checked in every
environment; LIVE additionally needs a signed, in-scope, unexpired approval
and met readiness). The risk engine (risk/options_sizing) has final authority
over what trades; unbounded-loss structures are unrepresentable in the domain
models; every order carries a hash-chained DecisionContext
(``symbol=OPT:<occ>``); covered-call/CSP collateral is validated at proposal
(decision construction) AND at submission (manager lookups).
"""

from datetime import date, timedelta
from typing import Any

from dagster import OpExecutionContext, job, op
from sqlalchemy import select

from dynamo.config.settings import DynamoEnv, Settings
from dynamo.persistence import schema
from orchestration.assets.portfolio_exec import UtcClock, _latest_prices, _portfolio_state
from orchestration.assets.research import _read_df
from orchestration.resources import DynamoResource, utcnow

OPTIONS_GROUP = "options"


def _options_broker(dynamo: DynamoResource) -> Any | None:
    """AlpacaOptionsBroker when keys are set; None → skip (paper by default)."""
    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        return None
    from dynamo.execution.options_broker import AlpacaOptionsBroker

    return AlpacaOptionsBroker(s)


def _options_chain(dynamo: DynamoResource) -> Any | None:
    """Chain fetcher; None when keys or the SDK options surface are missing."""
    s = dynamo.settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        return None
    from dynamo.data.options_chain import AlpacaOptionsChain

    chain = AlpacaOptionsChain(s.alpaca_api_key, s.alpaca_api_secret)
    return chain if chain.available else None


def _options_gate(settings: Settings, readiness: Any = None) -> Any:
    """The doc 09 §7 gate as a callable over OptionOrders (same function as
    equities). Non-LIVE short-circuits to permitted (unless the kill-switch is
    engaged); LIVE requires the signed approval covering ``OPT:<occ>`` (or
    ``all``) plus readiness — fail-closed when either is absent."""
    from dynamo.execution.gate import (
        Readiness,
        kill_switch_engaged,
        live_orders_permitted,
        load_approval,
    )
    from dynamo.execution.options_order_manager import opt_scope

    approval = None
    if settings.dynamo_env == DynamoEnv.LIVE:
        # approvals_path anchors to $DYNAMO_HOME (like the kill switch) —
        # never the process CWD, which differs across daemon/API/shell.
        path = settings.approvals_path / f"live-{utcnow().date().isoformat()}.yaml"
        if path.exists():
            approval = load_approval(path)

    def gate(order: Any) -> Any:
        return live_orders_permitted(
            settings.dynamo_env,
            approval,
            kill_switch_engaged(settings.kill_switch_file),
            readiness if readiness is not None else Readiness(),
            scope_target=opt_scope(order.occ_symbol),
        )

    return gate


def _options_manager(dynamo: DynamoResource, broker: Any) -> Any:
    from dynamo.execution.options_order_manager import OptionsOrderManager
    from dynamo.persistence.repositories.execution import PositionRepository
    from dynamo.persistence.repositories.options import (
        OptionsOrderRepository,
        OptionsTradeRepository,
    )

    engine = dynamo.engine()
    position_repo = PositionRepository(engine)

    def shares_lookup(underlying: str) -> float:
        for p in position_repo.all_positions():
            if p.symbol == underlying:
                return p.qty
        return 0.0

    def cash_lookup() -> float:
        state = _portfolio_state(dynamo)
        return state.cash if state is not None else 0.0

    return OptionsOrderManager(
        broker,
        OptionsOrderRepository(engine),
        OptionsTradeRepository(engine),
        UtcClock(),
        gate=_options_gate(dynamo.settings(), dynamo.readiness()),
        shares_lookup=shares_lookup,
        cash_lookup=cash_lookup,
    )


def _options_sticky_halt_active(dynamo: DynamoResource) -> bool:
    """Unacked divergent options reconciliation runs, or an unacked ladder rung.

    The ladder halt is portfolio-wide on purpose (doc 06 §3): a -8% drawdown
    is a statement about the account, not about one sleeve, and options
    premium at risk is drawn from the same equity.
    """
    from dynamo.persistence.repositories.execution import ReconciliationRepository
    from dynamo.persistence.repositories.risk import LadderEventRepository

    if LadderEventRepository(dynamo.engine()).is_halted():
        return True
    runs = ReconciliationRepository(dynamo.engine()).unacked_divergent_runs()
    return any(str(r.get("scope", "")).startswith("options") for r in runs)


def _apply_new_fills(dynamo: DynamoResource, manager: Any) -> int:
    """Record fills discovered by sync_orders into options_positions."""
    from dynamo.persistence.repositories.options import (
        OptionsOrderRepository,
        OptionsPositionRepository,
    )

    engine = dynamo.engine()
    orders_repo = OptionsOrderRepository(engine)
    positions = OptionsPositionRepository(engine)
    trades = manager.sync_orders()
    now = utcnow()
    for trade in trades:
        order = orders_repo.get(trade.client_order_id)
        if order is not None:
            positions.apply_fill(order, trade, now)
    return len(trades)


def _close_expired_positions(dynamo: DynamoResource, today: date) -> int:
    """Retire positions past expiry as ``kind='expiry'`` close-outs.

    Assignment/exercise nuances are surfaced to the human via an alert; the
    broker-truth reconciliation would otherwise raise a spurious sticky halt
    for every expired contract.
    """
    from dynamo.domain.models.options import OptionTrade
    from dynamo.persistence.repositories.options import (
        OptionsPositionRepository,
        OptionsTradeRepository,
    )

    engine = dynamo.engine()
    positions = OptionsPositionRepository(engine)
    trades = OptionsTradeRepository(engine)
    closed = 0
    for position in positions.expired_positions(today):
        trades.insert(
            OptionTrade(
                client_order_id=None,
                occ_symbol=position.occ_symbol,
                underlying=position.underlying,
                side=position.side,
                right=position.right,
                contracts=position.contracts,
                price=0.0,
                kind="expiry",
                filled_at=utcnow(),
            )
        )
        positions.close(position.occ_symbol)
        dynamo.alert(
            "warn",
            "options_position_expired",
            message=(
                f"{position.occ_symbol} expired ({position.structure.value}, "
                f"{position.contracts} contracts) — review for assignment/exercise"
            ),
        )
        closed += 1
    return closed


# --------------------------------------------------------------------------
# options_trading_job — nightly 00:45 ET, after target_portfolio
# --------------------------------------------------------------------------


@op
def options_execute_orders(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Verdicts + chains → risk.options_sizing → gate-checked limit orders."""
    from dynamo.data.options_chain import atm_iv
    from dynamo.domain.models.portfolio import DecisionContext
    from dynamo.domain.models.research import CommitteeVerdict, Sleeve
    from dynamo.execution.errors import ExecutionError
    from dynamo.execution.options_order_manager import opt_scope
    from dynamo.persistence.repositories.learning import DecisionContextRepository
    from dynamo.persistence.repositories.options import (
        OptionsChainRepository,
        OptionsPositionRepository,
    )
    from dynamo.risk.options_sizing import (
        OptionsSizingInputs,
        OptionsSizingParams,
        iv_rank,
        select_and_size_option_trades,
    )

    settings = dynamo.settings()
    broker = _options_broker(dynamo)
    if broker is None:
        context.log.warning("options_trading: alpaca keys unset — skipping")
        return
    chain_provider = _options_chain(dynamo)
    if chain_provider is None:
        context.log.warning(
            "options_trading: chain provider unavailable "
            "(keys unset or alpaca-py lacks options) — skipping"
        )
        return
    if _options_sticky_halt_active(dynamo):
        context.log.error(
            "options_trading: sticky reconciliation halt active — refusing all orders"
        )
        return
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("options_trading: no portfolio state — skipping")
        return

    as_of = utcnow().date()
    _close_expired_positions(dynamo, as_of)

    verdicts: dict[str, CommitteeVerdict] = {}
    vf = dynamo.market_repo().latest_verdicts()
    if not vf.empty:
        for payload in vf["payload"]:
            v = CommitteeVerdict.model_validate(payload)
            verdicts[v.symbol] = v
    if not verdicts:
        context.log.info("options_trading: no committee verdicts — nothing to size")
        return

    held = {p.symbol: p.qty for p in state.positions}
    long_term_holdings = frozenset(
        p.symbol for p in state.positions if any(lot.sleeve == Sleeve.LONG_TERM for lot in p.lots)
    )
    qs = schema.quality_scores
    quality_df = _read_df(
        dynamo,
        select(qs.c.symbol, qs.c.composite).where(qs.c.as_of >= as_of - timedelta(days=7)),
    )
    long_term_eligible = frozenset(
        str(r["symbol"]) for r in quality_df.to_dict(orient="records") if r["composite"] is not None
    )

    candidates = sorted(set(verdicts) | set(held))
    spot_prices = _latest_prices(dynamo, candidates, as_of)

    chain_repo = OptionsChainRepository(dynamo.engine())
    now = utcnow()
    chains: dict[str, Any] = {}
    iv_ranks: dict[str, float] = {}
    expiry_start = as_of + timedelta(days=settings.options_min_dte)
    expiry_end = as_of + timedelta(days=settings.options_max_dte)
    for underlying in sorted(verdicts):
        spot = spot_prices.get(underlying)
        if spot is None:
            continue
        rows = chain_provider.chain_snapshot(underlying, spot, expiry_start, expiry_end, now)
        if not rows:
            continue
        chains[underlying] = rows
        chain_repo.insert_snapshot(rows)
        current_iv = atm_iv(rows, spot)
        if current_iv is not None:
            chain_repo.append_iv(underlying, as_of, current_iv)
            rank = iv_rank(current_iv, chain_repo.iv_history(underlying))
            if rank is not None:
                iv_ranks[underlying] = rank

    open_positions = OptionsPositionRepository(dynamo.engine()).open_positions()
    result = select_and_size_option_trades(
        OptionsSizingInputs(
            verdicts=verdicts,
            chains=chains,
            spot_prices=spot_prices,
            iv_ranks=iv_ranks,
            equity=state.equity,
            cash_available=state.cash,
            shares_held=held,
            long_term_holdings=long_term_holdings,
            long_term_eligible=long_term_eligible,
            open_positions=open_positions,
            now=now,
        ),
        OptionsSizingParams(
            premium_at_risk_cap=settings.options_premium_at_risk_cap,
            per_position_cap=settings.options_per_position_cap,
            min_dte=settings.options_min_dte,
            max_dte=settings.options_max_dte,
            min_open_interest=settings.options_min_open_interest,
            max_spread_pct=settings.options_max_spread_pct,
            min_conviction=settings.options_min_conviction,
            iv_rank_max=settings.options_iv_rank_max,
            max_open_positions=settings.options_max_open_positions,
        ),
    )
    for scope, reason in result.skipped:
        context.log.info("options_trading: skipped %s (%s)", scope, reason)
    if not result.decisions:
        context.log.info("options_trading: no trades pass the sleeve rules")
        return

    manager = _options_manager(dynamo, broker)
    dc_repo = DecisionContextRepository(dynamo.engine())
    decisions_by_occ = {d.occ_symbol: d for d in result.decisions}
    orders = manager.propose(result.decisions, as_of)
    submitted = 0
    for order in orders:
        decision = decisions_by_occ[order.occ_symbol]
        dc = DecisionContext(
            id=f"dc-{order.client_order_id}",
            created_at=utcnow(),
            env=settings.dynamo_env.value,
            symbol=opt_scope(order.occ_symbol),
            sleeve=Sleeve.TACTICAL,
            verdict_id=decision.verdict_ref,
            target_portfolio_id=f"options-{as_of.isoformat()}",
            payload={
                "kind": "options",
                "structure": decision.structure.value,
                "decision": decision.model_dump(mode="json"),
                "order": order.model_dump(mode="json"),
                "spot": spot_prices.get(order.underlying),
                "iv_rank": decision.iv_rank,
            },
            # The verdict's expected move is for the UNDERLYING — it does not
            # describe the option position's P&L path, so no expectations are
            # denormalized here; zeros mean "no forecast" (calibration falls
            # back to sign-of-pnl). Horizon is the position's real DTE.
            expected_upside_pct=0.0,
            expected_downside_pct=0.0,
            horizon_days=decision.dte,
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
                    "options_trading: %s not submitted: %s", order.client_order_id, exc
                )
    fills = _apply_new_fills(dynamo, manager)
    context.log.info(
        "options_trading: %d sized, %d submitted, %d immediate fills (env=%s)",
        len(orders),
        submitted,
        fills,
        settings.dynamo_env.value,
    )


@job
def options_trading_job() -> None:
    """Gated defined-risk options execution (paper unless LIVE; ADR-0010)."""
    options_execute_orders()


# --------------------------------------------------------------------------
# options_reconciliation_job — 15-minute loop alongside the equity reconciler
# --------------------------------------------------------------------------


@op
def options_reconcile_broker_state(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Order sync + positions reconciliation vs the broker; sticky halt."""
    from dynamo.execution.options_order_manager import OptionsReconciler
    from dynamo.persistence.repositories.execution import ReconciliationRepository
    from dynamo.persistence.repositories.options import OptionsPositionRepository

    broker = _options_broker(dynamo)
    if broker is None:
        context.log.info("options_reconciliation: no broker configured — skipping")
        return
    manager = _options_manager(dynamo, broker)
    fills = _apply_new_fills(dynamo, manager)

    def alert(name: str, message: str) -> None:
        dynamo.alert("page", name, message)

    reconciler = OptionsReconciler(broker, ReconciliationRepository(dynamo.engine()), alert)
    result = reconciler.reconcile(
        utcnow(), OptionsPositionRepository(dynamo.engine()).open_positions()
    )
    context.log.info(
        "options_reconciliation: %d new fills, %d divergences, halted scopes %s",
        fills,
        len(result.divergences),
        sorted(result.halted_scopes),
    )


@job
def options_reconciliation_job() -> None:
    options_reconcile_broker_state()


OPTIONS_JOBS = [options_trading_job, options_reconciliation_job]
