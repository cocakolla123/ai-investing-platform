"""RiskEngine: state in, RiskVerdict out. Fail-closed, never fail-open.

Design: docs/design/06-risk-engine.md sections 1, 2, 8, 11. The verdict is a
hard constraint set produced independently of any proposal; if the engine
cannot compute (data gap, stale prices), the answer is "no new risk" with an
explanatory veto row — degraded=True.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from dynamo.domain.models.portfolio import PortfolioState
from dynamo.domain.models.research import Sleeve
from dynamo.domain.models.risk import (
    Regime,
    RegimeState,
    RiskConfig,
    RiskMetrics,
    RiskVerdict,
    Veto,
)
from dynamo.risk.correlation import correlation_clusters, shrunk_correlation
from dynamo.risk.ladder import LadderState, effects_for, step
from dynamo.risk.liquidity import max_entry_weight
from dynamo.risk.regime import RegimeHistory, latest_regime
from dynamo.risk.stops import StopInputs, recompute_stops
from dynamo.risk.types import ReturnsPanel, SymbolBars
from dynamo.risk.var import (
    VarResult,
    historical_var,
    kupiec_tighten_factor,
    kupiec_traffic_light,
)
from dynamo.risk.vol import atr, portfolio_realized_vol_annual, portfolio_return_series
from dynamo.risk.vol_target import exposure_scale

RISK_OFF_GROSS_MULTIPLIER = 0.8
CRISIS_STOP_TIGHTEN = 0.5  # "tighten all stops one notch": ATR multiple - 0.5
_MIN_ATR_MULTIPLE = 1.0
DEFAULT_MAX_STALE_DAYS = 7


@dataclass(frozen=True)
class RiskEngineInputs:
    """Everything the engine needs; adapters gather it, the engine stays pure."""

    portfolio: PortfolioState
    returns: ReturnsPanel  # ~2y daily adjusted returns incl. holdings (+SPY/TLT)
    prices: Mapping[str, float]  # current marks for held symbols
    sector_by_symbol: Mapping[str, str]
    adv_dollar_20d: Mapping[str, float]  # 20d median dollar volume per symbol
    regime_history: RegimeHistory
    prev_ladder: LadderState
    config: RiskConfig
    as_of: datetime
    session_index: int
    bars: Mapping[str, SymbolBars] = field(default_factory=dict[str, SymbolBars])
    quarantined: frozenset[str] = frozenset()
    var_exceptions_250d: int = 0
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS


def _finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _mark_weights(inputs: RiskEngineInputs, equity: float) -> dict[str, float] | None:
    """Held weights marked at current prices; None if a price is missing/bad."""
    weights: dict[str, float] = {}
    for position in inputs.portfolio.positions:
        if position.qty == 0.0:
            continue
        price = inputs.prices.get(position.symbol)
        if price is None or not _finite_positive(price):
            return None
        weights[position.symbol] = weights.get(position.symbol, 0.0) + position.qty * price / equity
    return weights


def _fallback_regime(inputs: RiskEngineInputs) -> RegimeState:
    """Conservative stand-in when regime inputs are unusable (fail closed)."""
    return RegimeState(
        as_of=inputs.as_of.date(),
        regime=Regime.RISK_OFF,
        composite_score=-1.0,
        inputs={},
        sessions_in_state=0,
    )


def _portfolio_beta(
    panel: ReturnsPanel, weights: Mapping[str, float], market_symbol: str = "SPY"
) -> float | None:
    market = panel.column(market_symbol)
    if market is None or not weights:
        return None
    series = portfolio_return_series(panel, weights)
    if series is None:
        return None
    lookback = min(252, int(series.shape[0]))
    x = market[-lookback:]
    y = series[-lookback:]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 60:
        return None
    x = x[mask]
    y = y[mask]
    var = float(np.var(x))
    if var <= 0.0:
        return None
    return float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))) / var)


class RiskEngine:
    """Pure verdict computation. No I/O, no clock reads, no unseeded randomness."""

    def compute_verdict(self, inputs: RiskEngineInputs) -> RiskVerdict:
        cfg = inputs.config
        portfolio = inputs.portfolio
        problems: list[str] = []

        equity = portfolio.equity
        equity_ok = _finite_positive(equity)
        if not equity_ok:
            problems.append(f"equity is not a positive finite number: {equity!r}")

        hwm = portfolio.high_water_mark
        if hwm is None or not _finite_positive(hwm):
            hwm = equity if equity_ok else float("nan")
        ladder_state = step(
            inputs.prev_ladder,
            equity if equity_ok else float("nan"),
            hwm,
            inputs.session_index,
            recovery_sessions=cfg.hysteresis_sessions,
        )

        if inputs.returns.n_sessions == 0:
            problems.append("returns history is empty")
        else:
            age_days = (inputs.as_of.date() - inputs.returns.dates[-1]).days
            if age_days > inputs.max_stale_days:
                problems.append(
                    f"returns history is stale ({age_days}d old > {inputs.max_stale_days}d limit)"
                )

        weights: dict[str, float] | None = None
        if equity_ok:
            weights = _mark_weights(inputs, equity)
            if weights is None:
                problems.append("missing or invalid mark price for a held position")
            else:
                for symbol in weights:
                    if inputs.returns.index_of(symbol) is None:
                        problems.append(f"held symbol {symbol} missing from returns history")

        regime = latest_regime(inputs.regime_history, cfg)
        if regime is None:
            problems.append("regime inputs insufficient (history shorter than warmup)")

        if problems or weights is None or regime is None:
            return self._degraded_verdict(inputs, ladder_state, regime, problems)
        return self._full_verdict(inputs, ladder_state, regime, weights, hwm)

    # ------------------------------------------------------------- fail-closed
    def _degraded_verdict(
        self,
        inputs: RiskEngineInputs,
        ladder_state: LadderState,
        regime: RegimeState | None,
        problems: list[str],
    ) -> RiskVerdict:
        cfg = inputs.config
        effects = effects_for(ladder_state.rung)
        gross_cap = cfg.gross_cap
        if effects.gross_override is not None:
            gross_cap = min(gross_cap, effects.gross_override)
        vetoes = tuple(Veto(symbol="*", rule="fail_closed", reason=reason) for reason in problems)
        equity = inputs.portfolio.equity
        cash = inputs.portfolio.cash
        cash_fraction = cash / equity if _finite_positive(equity) and math.isfinite(cash) else 1.0
        metrics = RiskMetrics(
            as_of=inputs.as_of,
            gross_exposure=0.0,
            cash_fraction=cash_fraction,
            drawdown_from_hwm=0.0,
            var_exceptions_250d=inputs.var_exceptions_250d,
        )
        return RiskVerdict(
            as_of=inputs.as_of,
            regime=regime if regime is not None else _fallback_regime(inputs),
            ladder=ladder_state.rung,
            new_risk_allowed=False,
            gross_exposure_cap=gross_cap,
            net_exposure_cap=gross_cap,
            cash_floor=max(cfg.cash_floor, 1.0 - gross_cap),
            per_position_cap={},
            sector_caps={},
            cluster_caps={},
            clusters={},
            stop_updates=(),
            vetoes=vetoes,
            metrics=metrics,
            degraded=True,
            config=cfg,  # what RAN, so /risk/limits stops reporting defaults
        )

    # --------------------------------------------------------------- full path
    def _full_verdict(
        self,
        inputs: RiskEngineInputs,
        ladder_state: LadderState,
        regime: RegimeState,
        weights: dict[str, float],
        hwm: float,
    ) -> RiskVerdict:
        cfg = inputs.config
        equity = inputs.portfolio.equity
        panel = inputs.returns
        effects = effects_for(ladder_state.rung)

        gross_exposure = float(sum(abs(w) for w in weights.values()))
        cash_fraction = (
            inputs.portfolio.cash / equity if math.isfinite(inputs.portfolio.cash) else 0.0
        )
        drawdown_from_hwm = equity / hwm - 1.0

        tighten = kupiec_tighten_factor(kupiec_traffic_light(inputs.var_exceptions_250d))

        corr = shrunk_correlation(panel)
        clusters = correlation_clusters(corr, panel.symbols, cfg.correlation_cluster_threshold)

        var_result: VarResult | None = None
        portfolio_vol: float | None = None
        vol_scale = 1.0
        if weights:
            var_result = historical_var(panel, weights)
            realized = portfolio_realized_vol_annual(panel, weights)
            if math.isfinite(realized):
                portfolio_vol = realized
                vol_scale = exposure_scale(realized, cfg)

        var_breach = var_result is not None and var_result.var95 > cfg.var95_cap

        # --- vetoes -------------------------------------------------------
        vetoes: list[Veto] = []
        for symbol in sorted(inputs.quarantined):
            vetoes.append(
                Veto(
                    symbol=symbol,
                    rule="quarantine",
                    reason="symbol is quarantined; no new risk permitted",
                )
            )
        for symbol in panel.symbols:
            if symbol in inputs.quarantined:
                continue
            liquidity_cap = max_entry_weight(
                inputs.adv_dollar_20d.get(symbol, float("nan")), equity, cfg
            )
            if liquidity_cap < cfg.min_position_weight:
                vetoes.append(
                    Veto(
                        symbol=symbol,
                        rule="liquidity",
                        reason=(
                            "estimated exit exceeds "
                            f"{cfg.liquidity_max_days:g} days at "
                            f"{cfg.liquidity_adv_participation:.0%} ADV; no new buys"
                        ),
                        inputs={"max_entry_weight": liquidity_cap},
                    )
                )
        if var_breach and var_result is not None:
            vetoes.append(
                Veto(
                    symbol="*",
                    rule="var_limit",
                    reason=(
                        f"portfolio 95% 1-day VaR {var_result.var95:.4f} exceeds "
                        f"cap {cfg.var95_cap:.4f}; no new risk"
                    ),
                    inputs={"var95_1d": var_result.var95, "cap": cfg.var95_cap},
                )
            )

        # --- effective per-symbol caps ------------------------------------
        per_position_cap: dict[str, float] = {}
        for symbol in panel.symbols:
            if symbol in inputs.quarantined:
                per_position_cap[symbol] = 0.0
                continue
            cap = cfg.per_position_cap
            sector = inputs.sector_by_symbol.get(symbol)
            if sector is not None:
                used = sum(
                    w
                    for other, w in weights.items()
                    if other != symbol and inputs.sector_by_symbol.get(other) == sector and w > 0.0
                )
                cap = min(cap, max(cfg.sector_cap - used, 0.0))
            cluster_id = clusters.get(symbol)
            if cluster_id is not None:
                used = sum(
                    w
                    for other, w in weights.items()
                    if other != symbol and clusters.get(other) == cluster_id and w > 0.0
                )
                cap = min(cap, max(cfg.cluster_cap - used, 0.0))
            cap *= tighten
            per_position_cap[symbol] = float(min(max(cap, 0.0), cfg.per_position_hard))

        sectors_seen = {
            s for s in (inputs.sector_by_symbol.get(sym) for sym in panel.symbols) if s is not None
        }
        sector_caps = {s: cfg.sector_cap * tighten for s in sorted(sectors_seen)}
        cluster_caps = {c: cfg.cluster_cap * tighten for c in sorted(set(clusters.values()))}

        # --- exposure caps ------------------------------------------------
        gross_cap = cfg.gross_cap
        if effects.gross_override is not None:
            gross_cap = min(gross_cap, effects.gross_override)
        if regime.regime is Regime.RISK_OFF:
            gross_cap *= RISK_OFF_GROSS_MULTIPLIER
        if vol_scale < 1.0:
            gross_cap *= vol_scale
        gross_cap = max(gross_cap, 0.0)
        cash_floor = max(cfg.cash_floor, 1.0 - gross_cap)

        new_risk_allowed = (
            effects.new_positions_allowed and regime.regime is not Regime.CRISIS and not var_breach
        )

        # --- stops --------------------------------------------------------
        atr_override = (
            max(cfg.atr_stop_multiple - CRISIS_STOP_TIGHTEN, _MIN_ATR_MULTIPLE)
            if regime.regime is Regime.CRISIS
            else None
        )
        stop_inputs: list[StopInputs] = []
        for position in inputs.portfolio.positions:
            if position.qty == 0.0:
                continue
            price = inputs.prices.get(position.symbol, float("nan"))
            symbol_bars = inputs.bars.get(position.symbol)
            atr14 = atr(symbol_bars) if symbol_bars is not None else float("nan")
            sleeve = position.lots[0].sleeve if position.lots else Sleeve.TACTICAL
            stop_inputs.append(
                StopInputs(
                    symbol=position.symbol,
                    sleeve=sleeve,
                    entry_price=position.avg_entry_price,
                    current_price=price,
                    atr14=atr14,
                    highest_close_since_entry=None,
                    prev_stop=position.current_stop,
                )
            )
        stop_updates = tuple(recompute_stops(stop_inputs, cfg, atr_override))

        # --- metrics ------------------------------------------------------
        metrics = RiskMetrics(
            as_of=inputs.as_of,
            portfolio_vol_annual=portfolio_vol,
            var95_1d=var_result.var95 if var_result is not None else None,
            var99_1d=var_result.var99 if var_result is not None else None,
            cvar95_1d=var_result.cvar95 if var_result is not None else None,
            net_beta=_portfolio_beta(panel, weights),
            gross_exposure=gross_exposure,
            cash_fraction=cash_fraction,
            drawdown_from_hwm=drawdown_from_hwm,
            var_exceptions_250d=inputs.var_exceptions_250d,
        )
        return RiskVerdict(
            as_of=inputs.as_of,
            regime=regime,
            ladder=ladder_state.rung,
            new_risk_allowed=new_risk_allowed,
            gross_exposure_cap=gross_cap,
            net_exposure_cap=gross_cap,
            cash_floor=cash_floor,
            per_position_cap=per_position_cap,
            sector_caps=sector_caps,
            cluster_caps=cluster_caps,
            clusters=clusters,
            stop_updates=stop_updates,
            vetoes=tuple(vetoes),
            metrics=metrics,
            degraded=False,
            config=cfg,  # what RAN, so /risk/limits stops reporting defaults
        )
