"""Same-day tactical entries — the 15-minute intraday loop (ADR-0010).

During market hours the loop evaluates (a) newly promoted Tier-3 names with a
material triage verdict today and (b) breakout-strategy triggers vs live
quotes for watchlist tacticals. Triggered names run the SAME risk sizing
cascade as the nightly path and propose equity orders through the standard
OrderManager (marketable limit, decision context, gate) — no shortcut exists.

Hard limits wired here: ``intraday_entry_cap_per_day`` entries per day, and
the PDT guard — a name entered same-day is never sold same-day automatically
(the stops op applies :func:`dynamo.risk.intraday.pdt_exit_allowed`).
"""

from datetime import UTC, date, datetime, time, timedelta

from dagster import OpExecutionContext, job, op
from sqlalchemy import select

from dynamo.persistence import schema
from orchestration.assets.portfolio_exec import (
    UtcClock,
    _broker,
    _execution_ready,
    _latest_risk_verdict,
    _meta_effective_probability,
    _order_manager,
    _portfolio_state,
)
from orchestration.assets.research import _pydate, _read_df, _rows_to_models
from orchestration.resources import DynamoResource, utcnow

BREAKOUT_STRATEGY_ID = "breakout-v1"
SIGNAL_LOOKBACK_DAYS = 3


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def _intraday_target_id(day: date) -> str:
    return f"intraday-{day.isoformat()}"


def _entries_today(dynamo: DynamoResource, day: date) -> list[str]:
    """Symbols already proposed by the intraday loop today (cap accounting)."""
    dc = schema.decision_context
    frame = _read_df(
        dynamo,
        select(dc.c.symbol).where(dc.c.target_portfolio_id == _intraday_target_id(day)),
    )
    return sorted({str(s) for s in frame["symbol"]}) if not frame.empty else []


def _fills_window(dynamo: DynamoResource, day: date) -> list[tuple[str, str, date]]:
    """(symbol, side, fill_date) fills over the trailing week (PDT window)."""
    t = schema.trades
    frame = _read_df(
        dynamo,
        select(t.c.symbol, t.c.side, t.c.filled_at).where(
            t.c.filled_at >= _day_start(day - timedelta(days=7))
        ),
    )
    out: list[tuple[str, str, date]] = []
    for row in frame.to_dict(orient="records"):
        out.append((str(row["symbol"]), str(row["side"]), _pydate(row["filled_at"])))
    return out


@op
def intraday_evaluate_entries(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    """Promotions + breakout triggers → sizing cascade → standard order path."""
    from dynamo.config.risk_config import load_risk_config
    from dynamo.domain.models.portfolio import (
        DecisionContext,
        Order,
        OrderSide,
        OrderState,
        OrderType,
    )
    from dynamo.domain.models.research import CommitteeVerdict, Signal, Sleeve
    from dynamo.execution.client_order_id import client_order_id
    from dynamo.execution.errors import ExecutionError
    from dynamo.execution.order_manager import marketable_limit_price
    from dynamo.execution.stops_maintenance import intraday_ts_bucket
    from dynamo.persistence.repositories.learning import DecisionContextRepository
    from dynamo.risk.intraday import IntradayEntryParams, select_intraday_entries
    from dynamo.risk.sizing import SizingCandidate, SizingContext

    settings = dynamo.settings()
    if not _execution_ready(context, dynamo):
        return
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("intraday_entries: no portfolio state — skipping")
        return
    latest = _latest_risk_verdict(dynamo)
    if latest is None:
        context.log.info("intraday_entries: no risk verdict — skipping")
        return
    verdict_id, risk = latest
    if not risk.new_risk_allowed:
        context.log.info("intraday_entries: risk verdict forbids new risk — skipping")
        return

    as_of = utcnow().date()
    repo = dynamo.market_repo()

    # (a) Newly promoted Tier-3 names with a MATERIAL triage verdict today.
    promoted_today = {
        m.symbol for m in repo.universe_at(as_of) if int(m.tier) == 3 and m.valid_from == as_of
    }
    sf = schema.surveillance_flags
    material_df = _read_df(
        dynamo,
        select(sf.c.symbol).where(sf.c.triage_verdict == "material", sf.c.ts >= _day_start(as_of)),
    )
    material_today = {str(s) for s in material_df["symbol"]} if not material_df.empty else set()
    promoted_material = sorted(promoted_today & material_today)

    # (b) Breakout signals for watchlist tacticals, checked vs live quotes.
    wl = schema.watchlist
    wl_df = _read_df(
        dynamo,
        select(wl.c.symbol)
        .where(wl.c.sleeve == Sleeve.TACTICAL.value)
        .order_by(wl.c.as_of.desc())
        .limit(200),
    )
    tactical_watchlist = {str(s) for s in wl_df["symbol"]} if not wl_df.empty else set()
    sg = schema.signals
    signals_df = _read_df(
        dynamo,
        select(sg).where(
            sg.c.strategy_id == BREAKOUT_STRATEGY_ID,
            sg.c.status == "active",
            sg.c.as_of >= as_of - timedelta(days=SIGNAL_LOOKBACK_DAYS),
        ),
    )
    breakout_signals: list[Signal] = []
    if not signals_df.empty:
        signals_df["as_of"] = signals_df["as_of"].map(_pydate)
        breakout_signals = [
            s for s in _rows_to_models(Signal, signals_df) if s.symbol in tactical_watchlist
        ]

    symbols = sorted(set(promoted_material) | {s.symbol for s in breakout_signals})
    if not symbols:
        context.log.info("intraday_entries: no triggers this cycle")
        return

    alpaca = dynamo.alpaca()
    if alpaca is None:
        context.log.warning("intraday_entries: no market data provider — skipping")
        return
    live_quotes = alpaca.latest_quotes(symbols)

    verdicts: dict[str, CommitteeVerdict] = {}
    vf = repo.latest_verdicts(symbols)
    if not vf.empty:
        for payload in vf["payload"]:
            v = CommitteeVerdict.model_validate(payload)
            verdicts[v.symbol] = v

    bars = dynamo.bars_window(symbols, as_of, 120)
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
    inst = schema.instruments
    inst_df = _read_df(dynamo, select(inst.c.symbol, inst.c.sector))
    sector_by_symbol = {
        str(r["symbol"]): str(r["sector"] or "unknown") for r in inst_df.to_dict(orient="records")
    }

    # {} unless meta_sizing_enabled (dark path, plan Phase 3): absent symbols
    # -> None -> the sizing tilt uses raw conviction, bit-identical to before.
    effective_probability = _meta_effective_probability(dynamo, as_of)
    sizing_candidates: dict[str, SizingCandidate] = {}
    for symbol in symbols:
        if symbol not in sigma_annual or symbol not in adv:
            continue  # fail closed: no vol/liquidity history, no same-day entry
        verdict = verdicts.get(symbol)
        quant = None
        qframe = repo.latest_quant_scores([symbol])
        if not qframe.empty:
            quant = qframe.iloc[-1]
        sizing_candidates[symbol] = SizingCandidate(
            symbol=symbol,
            sleeve=Sleeve.TACTICAL,
            conviction=verdict.conviction if verdict is not None else 0.5,
            sigma_annual=sigma_annual[symbol],
            adv_dollar_20d=adv[symbol],
            effective_probability=effective_probability.get(symbol),
            expected_excess_return=(
                float(quant["expected_excess_return"])
                if quant is not None and quant["expected_excess_return"] is not None
                else None
            ),
            vol_forecast_annual=(
                float(quant["vol_forecast"])
                if quant is not None and quant["vol_forecast"] is not None
                else None
            ),
            sector=sector_by_symbol.get(symbol),
        )

    fills = _fills_window(dynamo, as_of)
    entered_today = frozenset(
        {sym for sym, side, day in fills if side == "buy" and day == as_of}
        | set(_entries_today(dynamo, as_of))
    )
    result = select_intraday_entries(
        promoted_material=promoted_material,
        breakout_signals=breakout_signals,
        live_quotes=live_quotes,
        verdicts=verdicts,
        sizing_candidates=sizing_candidates,
        config=load_risk_config(settings),
        context=SizingContext(
            equity=state.equity,
            existing_weights=state.weights,
            sector_by_symbol=sector_by_symbol,
            vetoed=frozenset(v.symbol for v in risk.vetoes),
        ),
        entries_today=len(_entries_today(dynamo, as_of)),
        entered_today=entered_today,
        params=IntradayEntryParams(cap_per_day=settings.intraday_entry_cap_per_day),
    )
    for symbol, reason in result.skipped:
        context.log.info("intraday_entries: skipped %s (%s)", symbol, reason)
    if not result.entries:
        context.log.info("intraday_entries: nothing sized this cycle")
        return

    broker = _broker(dynamo)
    if broker is None:
        return
    manager = _order_manager(dynamo, broker)
    manager.startup_reconciliation()  # doc 09 §4: broker truth before submission
    dc_repo = DecisionContextRepository(dynamo.engine())
    clock = UtcClock()
    submitted = 0
    for entry in result.entries:
        qty = round(entry.weight * state.equity / entry.quote, 4)
        if qty <= 0:
            continue
        coid = client_order_id(
            _intraday_target_id(as_of),
            entry.symbol,
            OrderSide.BUY.value,
            intraday_ts_bucket(clock),
        )
        if manager.orders_store.get(coid) is not None:
            continue  # same slot re-run — intent already derived
        order = Order(
            client_order_id=coid,
            symbol=entry.symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.MARKETABLE_LIMIT,
            limit_price=round(marketable_limit_price(OrderSide.BUY, entry.quote, 0.0), 4),
            tif="day",
            state=OrderState.PROPOSED,
            sleeve=Sleeve.TACTICAL,
            created_at=clock.now(),
        )
        manager.orders_store.insert(order)
        verdict = verdicts.get(entry.symbol)
        dc = DecisionContext(
            id=f"dc-{coid}",
            created_at=utcnow(),
            env=settings.dynamo_env.value,
            symbol=entry.symbol,
            sleeve=Sleeve.TACTICAL,
            verdict_id=verdict.id if verdict is not None else None,
            target_portfolio_id=_intraday_target_id(as_of),
            payload={
                "kind": "intraday_entry",
                "trigger": entry.trigger,
                "signal_ref": entry.signal_ref,
                "quote": entry.quote,
                "weight": entry.weight,
                "rationale": entry.rationale,
                "risk_verdict_ref": verdict_id,
                "order": order.model_dump(mode="json"),
            },
            # Committee expectations denormalized from the verdict where one
            # exists; (0, 0, 20) means "no forecast" for verdict-less entries.
            expected_upside_pct=verdict.expected_upside_pct if verdict is not None else 0.0,
            expected_downside_pct=(verdict.expected_downside_pct if verdict is not None else 0.0),
            horizon_days=verdict.horizon_days if verdict is not None else 20,
        )
        dc_id = dc_repo.append(dc)
        manager.transition(coid, OrderState.RISK_CHECKED, "intraday sizing cascade passed")
        manager.attach_decision_context(coid, dc_id)
        manager.request_approval(coid)
        if settings.auto_approve:
            try:
                manager.submit(coid)
                submitted += 1
            except ExecutionError as exc:
                context.log.error("intraday_entries: %s not submitted: %s", coid, exc)
    context.log.info(
        "intraday_entries: %d triggered, %d submitted (cap %d/day)",
        len(result.entries),
        submitted,
        settings.intraday_entry_cap_per_day,
    )


@job
def intraday_entries_job() -> None:
    """Same-day tactical entries through the standard order path (ADR-0010)."""
    intraday_evaluate_entries()


INTRADAY_JOBS = [intraday_entries_job]
