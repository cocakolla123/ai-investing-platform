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
from orchestration.assets._macro import aligned_series
from orchestration.assets.research import _pydate, _read_df, _rows_to_models
from orchestration.resources import DynamoResource, utcnow

PORTFOLIO_GROUP = "portfolio"

# Paper bootstrap equity when neither equity_curve nor a broker account exists.
RETURNS_LOOKBACK_DAYS = 750
BENCHMARK_SYMBOLS = ("SPY", "TLT")


def consolidated_adv_20d(bars: pd.DataFrame) -> dict[str, float]:
    """Per-symbol 20d median dollar volume in CONSOLIDATED-EQUIVALENT dollars.

    The risk/sizing layer's units contract (``dynamo.risk.liquidity``) says
    every ``adv_dollar_20d`` it receives is consolidated-equivalent, yet the
    live assets fed it raw feed dollars: IEX volume for most equities (~15x
    under the tape) and Alpaca's own crypto venue volume (>= 1000x under the
    global spot market — the first campaign vetoed every crypto buy on a
    ~$4k "dollar volume" for AVAX). This helper is the conversion at the
    point of computation, using the ``volume_source`` label the canonical
    bars frame carries; frames without the column (tests, ad-hoc frames)
    scale by 1.0, the conservative assume-consolidated default.
    """
    from dynamo.domain.volume_source import dominant_source
    from dynamo.risk.liquidity import consolidated_equivalent_adv

    adv: dict[str, float] = {}
    if bars is None or bars.empty:
        return adv
    for sym, grp in bars.groupby("symbol"):
        tail = grp.sort_values("ts").tail(20)
        observed = float((tail["close"].astype(float) * tail["volume"].astype(float)).median())
        source = dominant_source(tail["volume_source"]) if "volume_source" in tail.columns else None
        adv[str(sym)] = consolidated_equivalent_adv(observed, source)
    return adv


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


def _ledger_cash_and_equity(dynamo: DynamoResource, positions: Any) -> tuple[float, float] | None:
    """(cash, equity) from the cash ledger, marked to the latest closes.

    None before the bootstrap snapshot exists. Cash used to be read from the
    latest ``equity_curve`` row, which the nightly snapshot wrote back
    unchanged, so fills never reduced it (dynamo.execution.cash_ledger).
    """
    from dynamo.execution.cash_ledger import mark_to_market
    from dynamo.persistence.repositories.execution import CashLedgerRepository

    cash = CashLedgerRepository(dynamo.engine()).cash()
    if cash is None:
        return None
    prices = _latest_prices(dynamo, sorted({p.symbol for p in positions}), utcnow().date())
    equity, _ = mark_to_market(cash, positions, prices)
    return cash, equity


def _portfolio_state(dynamo: DynamoResource) -> Any | None:
    """Internal portfolio state: positions table + ledger cash, marked to market
    (broker snapshot only to bootstrap the first equity_curve row)."""
    from dynamo.domain.models.portfolio import PortfolioState
    from dynamo.persistence.repositories.execution import PositionRepository

    positions = tuple(PositionRepository(dynamo.engine()).all_positions())
    ledger = _ledger_cash_and_equity(dynamo, positions)
    if ledger is not None:
        cash, equity = ledger
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

    from dynamo.config.risk_config import load_risk_config
    from dynamo.domain.models.risk import LadderRung
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
    adv = consolidated_adv_20d(bars)

    spy = wide.get("SPY")
    if spy is None or spy.dropna().empty:
        context.log.warning("risk_verdict: no SPY history — cannot build regime inputs")
        return MaterializeResult(metadata={"skipped": "no SPY history"})
    spy = spy.dropna()
    n = len(spy)
    m = schema.macro_series
    macro = _read_df(
        dynamo,
        select(m.c.series_id, m.c.ts, m.c.vintage_ts, m.c.value).where(
            m.c.ts >= as_of - timedelta(days=RETURNS_LOOKBACK_DAYS + 200)
        ),
    )

    def _aligned(series_id: str, default: float) -> Any:
        return aligned_series(macro, series_id, spy.index, default)

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

    # doc 06 §6: the Kupiec traffic light needs REALIZED exceptions. Nothing
    # counted them — var_exceptions_250d defaulted to 0, so the light was
    # permanently green and the 25% tightening was unreachable code.
    from dynamo.persistence.repositories.risk import VarBacktestReader
    from dynamo.risk.var import VarTrafficLight, backtest_var

    var_backtest = backtest_var(VarBacktestReader(dynamo.engine()).observations(today=as_of))

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
            config=load_risk_config(dynamo.settings()),
            as_of=utcnow(),
            session_index=n - 1,
            quarantined=quarantined,
            var_exceptions_250d=var_backtest.exceptions,
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

    if var_backtest.light is not VarTrafficLight.GREEN:
        # doc 12 §7: "VaR exception cluster | warn | Kupiec traffic light
        # leaves green". The tightening itself already happened in the engine.
        dynamo.alert(
            "warn",
            "var_exception_cluster",
            f"VaR traffic light {var_backtest.light.value}: {var_backtest.exceptions} "
            f"exceptions in {var_backtest.observations} sessions; "
            f"caps x{var_backtest.tighten_factor:.2f}",
        )

    ladder_event_id = _apply_ladder_transition(
        context,
        dynamo,
        prev_ladder.rung,
        verdict.ladder,
        verdict.metrics.drawdown_from_hwm,
        as_of,
    )

    return MaterializeResult(
        metadata={
            "id": verdict_id,
            "ladder": verdict.ladder.value,
            "new_risk_allowed": verdict.new_risk_allowed,
            "degraded": verdict.degraded,
            "vetoes": len(verdict.vetoes),
            "var_exceptions_250d": var_backtest.exceptions,
            "var_traffic_light": var_backtest.light.value,
            "var_cap_multiplier": var_backtest.tighten_factor,
            "ladder_event_id": -1 if ladder_event_id is None else ladder_event_id,
        }
    )


class _NoOpBroker:
    """No broker configured: the kill-switch file still gets written."""

    def open_orders(self) -> list[dict[str, object]]:
        return []

    def cancel(self, client_order_id: str) -> None:  # pragma: no cover - never called
        return None


def _apply_ladder_transition(
    context: AssetExecutionContext,
    dynamo: DynamoResource,
    from_rung: Any,
    to_rung: Any,
    drawdown: float,
    as_of: date,
) -> int | None:
    """Perform the obligations of a ladder move (doc 06 §3).

    Before this, ``RungEffects.kill_switch`` was read only by the backtester
    and ``requires_human_ack`` by nothing at all: in the live path a -20%
    drawdown reduced to ``gross_override = 0.0`` — no file, no alert, no
    freeze. Now DD20 writes the kill-switch file and cancels open orders
    through the SAME ``engage_kill_switch`` the API's big red button uses,
    and every ack-requiring rung leaves an unacked row that blocks new orders
    until a human clears it with a note.
    """
    from dynamo.execution.gate import engage_kill_switch
    from dynamo.persistence.repositories.risk import LadderEventRepository
    from dynamo.risk.ladder_response import respond_to_transition

    response = respond_to_transition(from_rung, to_rung, drawdown)
    if response is None:
        return None

    event_id = LadderEventRepository(dynamo.engine()).record(
        as_of=as_of,
        from_rung=response.from_rung.value,
        to_rung=response.to_rung.value,
        drawdown=drawdown,
        requires_ack=response.requires_human_ack,
        kill_switch_engaged=response.engage_kill_switch,
    )
    context.log.warning("ladder_transition: %s (event %s)", response.message, event_id)

    if response.engage_kill_switch:
        settings = dynamo.settings()
        broker = _broker(dynamo) or _NoOpBroker()
        try:
            engage_kill_switch(
                settings.kill_switch_file,
                broker,  # type: ignore[arg-type]  # _NoOpBroker covers the used subset
                lambda name, message: dynamo.alert("page", name, message, ackable=False),
                engaged_by=f"drawdown ladder {response.to_rung.value}",
                now=utcnow(),
            )
        except Exception as exc:  # the freeze must still be recorded and paged
            context.log.error("ladder_transition: kill-switch engage failed: %r", exc)
            dynamo.alert(
                "page",
                "kill_switch_engage_failed",
                f"DD20 reached but engaging the kill switch failed: {exc!r} — "
                f"halt trading MANUALLY (docs/runbooks/kill-switch.md)",
                ackable=False,
            )

    dynamo.alert(
        response.alert_severity,
        response.alert_name,
        f"{response.message} (event id {event_id})",
        # The ladder ack clears this, not an alert ack — never both.
        ackable=not response.requires_human_ack,
    )
    return event_id


# --------------------------------------------------------------------------
# risk_scenarios — doc 06 §6 weekly Monte Carlo + 7-scenario stress suite
# --------------------------------------------------------------------------

MONTE_CARLO_SEED = 20260101  # fixed: a stress number that moves on rerun is noise


def _returns_panel(dynamo: DynamoResource, symbols: list[str], as_of: date, days: int) -> Any:
    """Daily returns panel over ``symbols`` — the risk modules' input shape."""
    import numpy as np

    from dynamo.risk.types import ReturnsPanel

    bars = dynamo.bars_window(symbols, as_of, days)
    if bars.empty:
        return None
    wide = bars.pivot_table(index="ts", columns="symbol", values="close").sort_index()
    returns = wide.pct_change().dropna(how="all").fillna(0.0)
    if returns.empty:
        return None
    return ReturnsPanel(
        dates=tuple(_pydate(t) for t in returns.index),
        symbols=tuple(str(c) for c in returns.columns),
        values=returns.to_numpy(dtype=np.float64),
    )


@dagster_asset(name="risk_scenarios", group_name=PORTFOLIO_GROUP)
def risk_scenarios_asset(
    context: AssetExecutionContext, dynamo: DynamoResource
) -> MaterializeResult:
    """Weekly Monte Carlo + deterministic stress suite (doc 06 §6).

    ``risk/montecarlo.py`` (167 lines) and ``run_stress_suite`` had zero
    importers outside their own tests — no asset, no schedule, no table — and
    ``/risk/stress`` read a verdict key nothing ever wrote. This is the
    producer: it writes ``risk_scenarios``, which the endpoint now reads.

    Deliberately NOT part of the nightly risk verdict: the stress suite is a
    10k-path simulation over ~2y of history, and doc 06 §6 asks for it
    weekly. It is advisory — a scenario result never moves a cap on its own.
    A breach of the defensive floor pages, because that is the case where a
    human should look before the market makes the decision.
    """
    from dynamo.risk.montecarlo import run_monte_carlo
    from dynamo.risk.stress import DEFENSIVE_FLOOR_BREACH, run_stress_suite

    as_of = utcnow().date()
    state = _portfolio_state(dynamo)
    if state is None:
        context.log.warning("risk_scenarios: no portfolio state — skipping")
        return MaterializeResult(metadata={"skipped": "no portfolio state"})

    prices = _latest_prices(dynamo, sorted({p.symbol for p in state.positions}), as_of)
    weights = {
        p.symbol: p.qty * prices[p.symbol] / state.equity
        for p in state.positions
        if p.qty != 0.0 and prices.get(p.symbol, 0.0) > 0.0
    }
    if not weights:
        context.log.info("risk_scenarios: no priced positions — nothing to stress")
        return MaterializeResult(metadata={"skipped": "no positions"})

    symbols = sorted({*weights, *BENCHMARK_SYMBOLS})
    # The historical windows reach back to 2008; a 750-day panel cannot
    # replay them, so this asset reads the full available history.
    panel = _returns_panel(dynamo, symbols, as_of, days=365 * 20)
    if panel is None:
        context.log.warning("risk_scenarios: no return history — skipping")
        return MaterializeResult(metadata={"skipped": "no returns"})

    hwm = state.high_water_mark or state.equity
    rows: list[dict[str, Any]] = []

    stress = run_stress_suite(panel, weights, state.equity, hwm)
    for result in stress:
        rows.append(
            {
                "as_of": as_of,
                "kind": "stress",
                "scenario": result.scenario,
                "equity_impact_pct": result.equity_impact_pct,
                "ladder_implication": (
                    result.ladder_implication.value if result.ladder_implication else None
                ),
                "details": dict(result.details),
            }
        )

    mc = run_monte_carlo(
        panel, weights, equity=state.equity, high_water_mark=hwm, seed=MONTE_CARLO_SEED
    )
    if mc is not None:
        rows.append(
            {
                "as_of": as_of,
                "kind": "monte_carlo",
                "scenario": f"{mc.horizon_days}d_student_t_copula",
                "equity_impact_pct": -mc.expected_shortfall_95,
                "ladder_implication": None,
                "details": {
                    "n_paths": mc.n_paths,
                    "horizon_days": mc.horizon_days,
                    "seed": mc.seed,
                    "expected_shortfall_95": mc.expected_shortfall_95,
                    "loss_probabilities": {str(k): v for k, v in mc.loss_probabilities.items()},
                    "rung_hit_probabilities": {
                        rung.value: prob for rung, prob in mc.rung_hit_probabilities.items()
                    },
                    "df_by_symbol": mc.df_by_symbol,
                },
            }
        )

    from dynamo.persistence.repositories.risk import RiskScenarioRepository

    written = RiskScenarioRepository(dynamo.engine()).upsert(rows)

    breaches = [
        r["scenario"]
        for r in rows
        if r["kind"] == "stress" and float(r["equity_impact_pct"]) <= DEFENSIVE_FLOOR_BREACH
    ]
    if breaches:
        dynamo.alert(
            "warn",
            "stress_defensive_floor_breach",
            f"{len(breaches)} stress scenarios breach the -15% defensive floor: "
            f"{', '.join(sorted(breaches)[:7])}",
        )

    worst = min((float(r["equity_impact_pct"]) for r in rows), default=0.0)
    return MaterializeResult(
        metadata={
            "as_of": as_of.isoformat(),
            "rows": written,
            "scenarios": len(stress),
            "monte_carlo": mc is not None,
            "worst_impact_pct": worst,
            "defensive_floor_breaches": len(breaches),
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
    if not bars.empty:
        for sym, grp in bars.groupby("symbol"):
            closes = grp.sort_values("ts")["close"].astype(float)
            if len(closes) > 21:
                sigma_annual[str(sym)] = float(closes.pct_change().std() * (252**0.5))
    adv = consolidated_adv_20d(bars)
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
    from dynamo.execution.gate import (
        Readiness,
        kill_switch_engaged,
        live_orders_permitted,
        load_approval,
    )

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
    return any(not str(r.get("scope", "")).startswith(_FOREIGN_HALT_PREFIXES) for r in runs)


def _ladder_halt_active(dynamo: DynamoResource) -> bool:
    """An unacked ack-requiring ladder rung blocks new orders (doc 06 §3).

    DD8 (human ack) and DD15 (human review) were documented mitigations that
    read ``RungEffects.requires_human_ack`` — which nothing read. The halt is
    a database row for the same reason the reconciliation halt is: the
    15-minute op rebuilds every object each cycle, so an in-memory flag would
    evaporate before the next order was proposed.
    """
    from dynamo.persistence.repositories.risk import LadderEventRepository

    return LadderEventRepository(dynamo.engine()).is_halted()


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
        halt_check=lambda _order: (
            _equities_sticky_halt_active(dynamo) or _ladder_halt_active(dynamo)
        ),
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
    if _ladder_halt_active(dynamo):
        context.log.error(
            "execute_paper_orders: drawdown-ladder rung awaiting human acknowledgment — "
            "refusing all orders (POST /admin/ladder/{id}/ack to clear)"
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
        """(cash, equity) from the cash ledger, marked to the latest closes; the
        same numbers the portfolio state uses."""
        ledger = _ledger_cash_and_equity(self._dynamo, self.positions())
        return (0.0, 0.0) if ledger is None else ledger

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
        equity_tolerance_pct=s.reconciliation_equity_tolerance_pct,
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


PORTFOLIO_ASSETS = [risk_verdict_asset, risk_scenarios_asset, target_portfolio_asset]
EXECUTION_JOBS = [paper_execution_job, reconciliation_job, equity_fills_job, stops_job]
