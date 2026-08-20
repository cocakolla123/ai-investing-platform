"""Event-driven backtest validator (doc 08 section 1).

A thin daily event loop that imports the actual production ``risk/`` and
``portfolio/`` modules: the same drawdown ladder, regime states, sizing
cascade, and ``TargetPortfolio`` construction that run in paper/live. The
validator owns only the loop, fills, and accounting; decision logic is
reused, never reimplemented.

Sequence per simulated session (doc 08 section 1):
  1. fill yesterday's orders at TODAY's open with the section-3 cost model
     (no same-bar fills, ever);
  2. mark to close; update equity, high-water mark, drawdown ladder;
  3. build the session RiskVerdict from production components (ladder step,
     regime state, caps);
  4. call production RuleBasedConstructor -> TargetPortfolio;
  5. diff -> orders queued for the NEXT open.

Survivorship: the point-in-time universe is a REQUIRED input — a symbol is
only tradable on dates it was actually a member; there is no API to run
without the membership join. Committee verdicts are replayed as recorded
history via the advisory provider — no retroactive Claude calls, ever.

Accounting is tracked to the cent (all cash flows rounded to 2 decimals).
Pure and deterministic: no I/O, no wall clock, sorted iteration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dynamo.backtest.costs import (
    CostParams,
    order_cost,
    order_skipped,
    whole_share_qty,
)
from dynamo.domain.models.portfolio import Lot, PortfolioState, Position, TargetPortfolio
from dynamo.domain.models.research import CommitteeVerdict, Signal, Sleeve, WatchlistRow
from dynamo.domain.models.risk import (
    Regime,
    RegimeState,
    RiskConfig,
    RiskMetrics,
    RiskVerdict,
    Veto,
)
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.constructor import AdvisoryInputs, RuleBasedConstructor
from dynamo.risk.engine import RISK_OFF_GROSS_MULTIPLIER
from dynamo.risk.ladder import LadderState, effects_for, step

_SIGMA_WINDOW = 60
_ANNUALIZE = math.sqrt(252.0)


class BacktestConfig(BaseModel):
    """Frozen, hashable run configuration (doc 08 section 8 reproducibility)."""

    model_config = ConfigDict(frozen=True)

    initial_equity: float = Field(default=100_000.0, gt=0.0)
    seed: int = 0
    impact_k: float = 1.0
    flat_cost_bps: float | None = None  # parity runs: matched flat costs
    apply_ladder: bool = True  # parity runs disable constraints entirely
    label: str = ""
    # Bankroll realism (backtest-campaign Phase 3) — defaults preserve prior
    # behavior bit-for-bit. Crypto pairs are exempt from whole-share flooring.
    whole_shares: bool = False
    min_order_notional: float = Field(default=0.0, ge=0.0)
    commission_per_order: float = Field(default=0.0, ge=0.0)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()

    @property
    def cost_params(self) -> CostParams:
        return CostParams(
            impact_k=self.impact_k,
            flat_cost_bps=self.flat_cost_bps,
            commission_per_order=self.commission_per_order,
            min_order_notional=self.min_order_notional,
        )


@dataclass(frozen=True)
class BacktestData:
    """Point-in-time inputs. ``universe`` is REQUIRED (survivorship defense)."""

    opens: pd.DataFrame  # index: sessions (date-like), columns: symbols
    closes: pd.DataFrame
    universe: Mapping[str, tuple[tuple[date, date | None], ...]]  # symbol -> windows
    adv_dollar_20d: Mapping[str, float] = field(default_factory=dict)
    sectors: Mapping[str, str] = field(default_factory=dict)
    regime_states: Mapping[date, RegimeState] = field(default_factory=dict)
    benchmark_closes: pd.Series | None = None  # SPY, for the report
    sigma_annual_override: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.opens.index.equals(self.closes.index):
            raise ValueError("opens and closes must share the same session index")
        if sorted(self.opens.columns) != sorted(self.closes.columns):
            raise ValueError("opens and closes must share the same symbol columns")


class AdvisoryProvider(Protocol):
    """Replays recorded/simulated research for one session, point-in-time.

    ``visible_closes`` is truncated so nothing after ``as_of`` is visible;
    the look-ahead audit reruns with this window lagged one extra session.
    """

    def advise(
        self,
        as_of: date,
        visible_closes: pd.DataFrame,
        holdings: Mapping[str, float],
    ) -> SessionAdvice: ...


@dataclass(frozen=True)
class SessionAdvice:
    watchlist: tuple[WatchlistRow, ...] = ()
    verdicts: tuple[CommitteeVerdict, ...] = ()
    signals: tuple[Signal, ...] = ()


@dataclass(frozen=True)
class FillRecord:
    session: date
    symbol: str
    side: str  # buy | sell
    qty: float
    price: float
    trade_value: float  # cents
    cost: float  # cents
    realized_pnl: float | None  # sells only (average-cost basis)


@dataclass(frozen=True)
class SessionRecord:
    session: date
    equity: float
    cash: float
    gross_exposure: float
    cash_fraction: float
    regime: str
    ladder: str
    turnover_used_mtd: float
    violations: tuple[str, ...]


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series  # indexed by session date, cents
    fills: tuple[FillRecord, ...]
    sessions: tuple[SessionRecord, ...]
    targets: tuple[TargetPortfolio, ...]
    excluded_out_of_universe: tuple[tuple[date, str], ...]
    one_way_turnover_total: float
    constraint_violation_count: int
    config_hash: str

    @property
    def terminal_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else 0.0

    def realized_pnls(self) -> list[tuple[str, str, float]]:
        return [
            (f.session.isoformat(), f.symbol, f.realized_pnl)
            for f in self.fills
            if f.realized_pnl is not None
        ]


def in_universe(
    universe: Mapping[str, tuple[tuple[date, date | None], ...]],
    symbol: str,
    on: date,
) -> bool:
    for start, end in universe.get(symbol, ()):
        if start <= on and (end is None or on <= end):
            return True
    return False


def _cents(x: float) -> float:
    return round(x, 2)


def _as_date(ts: object) -> date:
    if isinstance(ts, pd.Timestamp):
        return ts.date()
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    parsed = pd.Timestamp(str(ts)).to_pydatetime()
    assert isinstance(parsed, datetime)
    return parsed.date()


_NEUTRAL_SCORE = 0.0


def _default_regime(d: date) -> RegimeState:
    return RegimeState(as_of=d, regime=Regime.NEUTRAL, composite_score=_NEUTRAL_SCORE)


class BacktestEngine:
    """The event loop. All decision logic is imported from production modules."""

    def __init__(
        self,
        config: BacktestConfig,
        construction_config: ConstructionConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.config = config
        # Idle-cash ballast rides through here untouched: pass
        # ConstructionConfig(ballast_symbol="SPY") and the production
        # constructor's overlay applies. BacktestConfig itself carries no
        # ballast field — replay is the campaign-grade harness for ballast
        # baselines; a dedicated backtest knob is deferred until a backtest
        # campaign needs one.
        self.construction_config = construction_config or ConstructionConfig()
        self.risk_config = risk_config or RiskConfig()
        self._constructor = RuleBasedConstructor()

    # ------------------------------------------------------------------ run
    def run(self, data: BacktestData, provider: AdvisoryProvider) -> BacktestResult:
        sessions = [_as_date(ts) for ts in data.opens.index]
        symbols = sorted(str(c) for c in data.opens.columns)

        cash = _cents(self.config.initial_equity)
        shares: dict[str, float] = {}
        avg_cost: dict[str, float] = {}
        sleeve_of: dict[str, Sleeve] = {}  # attribution tag carried from targets
        ladder = LadderState()
        hwm = self.config.initial_equity
        turnover_mtd = 0.0
        current_month: tuple[int, int] | None = None

        pending: TargetPortfolio | None = None
        pending_equity = self.config.initial_equity
        pending_closes: dict[str, float] = {}

        equity_points: list[float] = []
        fills: list[FillRecord] = []
        session_records: list[SessionRecord] = []
        targets: list[TargetPortfolio] = []
        excluded: list[tuple[date, str]] = []
        turnover_total = 0.0
        violation_count = 0

        for i, d in enumerate(sessions):
            if current_month != (d.year, d.month):
                current_month = (d.year, d.month)
                turnover_mtd = 0.0

            opens_row = data.opens.iloc[i]
            closes_row = data.closes.iloc[i]

            # 1. Fill yesterday's target at TODAY's open (never same-bar).
            if pending is not None:
                cash, fill_batch, traded = self._execute(
                    pending,
                    pending_equity,
                    pending_closes,
                    opens_row,
                    d,
                    cash,
                    shares,
                    avg_cost,
                    data,
                )
                fills.extend(fill_batch)
                turnover = traded / pending_equity / 2.0 if pending_equity > 0 else 0.0
                turnover_mtd += turnover
                turnover_total += turnover
                pending = None

            # 2. Mark to close; ladder + HWM from production risk code.
            position_value = 0.0
            closes_now: dict[str, float] = {}
            for s in symbols:
                px = float(closes_row[s])
                closes_now[s] = px
                held = shares.get(s, 0.0)
                if held != 0.0 and math.isfinite(px):
                    position_value += _cents(held * px)
            equity = _cents(cash + position_value)
            hwm = max(hwm, equity)
            if self.config.apply_ladder:
                ladder = step(ladder, equity, hwm, i)
            equity_points.append(equity)

            gross = position_value / equity if equity > 0 else 0.0
            cash_frac = cash / equity if equity > 0 else 1.0

            # 3. Session verdict from production components.
            regime = data.regime_states.get(d) or _default_regime(d)
            tradable = [s for s in symbols if in_universe(data.universe, s, d)]
            verdict = self._session_verdict(
                d, regime, ladder, equity, hwm, gross, cash_frac, tradable
            )

            # 4. Production portfolio construction on point-in-time advice.
            visible = data.closes.iloc[: i + 1]
            holdings = {
                s: _cents(shares[s] * closes_now[s]) / equity
                for s in sorted(shares)
                if shares[s] != 0.0 and equity > 0
            }
            advice = provider.advise(d, visible, holdings)
            advice, dropped = self._enforce_universe(advice, tradable)
            excluded.extend((d, s) for s in dropped)

            advisory = AdvisoryInputs(
                as_of=datetime(d.year, d.month, d.day),
                watchlist=advice.watchlist,
                verdicts={v.symbol: v for v in advice.verdicts},
                signals=advice.signals,
                sigma_annual=self._sigmas(data, visible, tradable),
                adv_dollar_20d=data.adv_dollar_20d,
                sector_by_symbol=data.sectors,
            )
            positions = tuple(
                Position(
                    symbol=s,
                    qty=shares[s],
                    avg_entry_price=avg_cost.get(s, closes_now.get(s, 0.0)),
                    lots=(
                        Lot(
                            symbol=s,
                            sleeve=sleeve_of.get(s, Sleeve.LONG_TERM),
                            qty=shares[s],
                            entry_price=avg_cost.get(s, closes_now.get(s, 0.0)),
                            entry_date=d,
                        ),
                    ),
                )
                for s in sorted(shares)
                if shares[s] != 0.0
            )
            state = PortfolioState(
                as_of=datetime(d.year, d.month, d.day),
                equity=equity,
                cash=cash,
                positions=positions,
                high_water_mark=hwm,
            )
            target = self._constructor.construct(
                advisory,
                verdict,
                state,
                self.construction_config,
                risk_config=self.risk_config,
                current_weights=holdings,
                turnover_used_mtd=turnover_mtd,
            )
            targets.append(target)
            for w in target.weights:
                sleeve_of.setdefault(w.symbol, w.sleeve)

            # Constraint audit: the constructed target must satisfy the verdict.
            violations = compliance_violations(target, verdict)
            violation_count += len(violations)

            session_records.append(
                SessionRecord(
                    session=d,
                    equity=equity,
                    cash=cash,
                    gross_exposure=gross,
                    cash_fraction=cash_frac,
                    regime=regime.regime.value,
                    ladder=ladder.rung.value,
                    turnover_used_mtd=turnover_mtd,
                    violations=violations,
                )
            )

            # 5. Queue for next open.
            if i < len(sessions) - 1:
                pending = target
                pending_equity = equity
                pending_closes = closes_now

        curve = pd.Series(equity_points, index=data.opens.index, name="equity")
        return BacktestResult(
            equity_curve=curve,
            fills=tuple(fills),
            sessions=tuple(session_records),
            targets=tuple(targets),
            excluded_out_of_universe=tuple(excluded),
            one_way_turnover_total=turnover_total,
            constraint_violation_count=violation_count,
            config_hash=self.config.config_hash,
        )

    # ---------------------------------------------------------------- fills
    def _execute(
        self,
        target: TargetPortfolio,
        decision_equity: float,
        decision_closes: Mapping[str, float],
        opens_row: pd.Series,
        d: date,
        cash: float,
        shares: dict[str, float],
        avg_cost: dict[str, float],
        data: BacktestData,
    ) -> tuple[float, list[FillRecord], float]:
        target_by_symbol: dict[str, float] = {}
        for w in target.weights:
            target_by_symbol[w.symbol] = target_by_symbol.get(w.symbol, 0.0) + w.weight

        desired_shares: dict[str, float] = {}
        for symbol in sorted(set(target_by_symbol) | {s for s in shares if shares[s]}):
            weight = target_by_symbol.get(symbol, 0.0)
            close = decision_closes.get(symbol, float("nan"))
            if weight > 0.0 and math.isfinite(close) and close > 0.0:
                desired_shares[symbol] = whole_share_qty(
                    symbol,
                    _cents(weight * decision_equity) / close,
                    self.config.whole_shares,
                )
            else:
                desired_shares[symbol] = 0.0

        fills: list[FillRecord] = []
        traded_value = 0.0
        # Sells first so the freed cash funds the buys deterministically.
        deltas = sorted(
            ((s, q - shares.get(s, 0.0)) for s, q in desired_shares.items()),
            key=lambda t: (t[1] >= 0.0, t[0]),
        )
        for symbol, delta in deltas:
            if abs(delta) < 1e-12:
                continue
            raw_open = opens_row.get(symbol)
            open_px = float(raw_open) if raw_open is not None else float("nan")
            if not (math.isfinite(open_px) and open_px > 0.0):
                continue  # no fill possible; position unchanged
            qty = abs(delta)
            value = _cents(qty * open_px)
            if value < 0.01:
                continue  # sub-cent rounding residue, not a trade
            params = self.config.cost_params
            if order_skipped(value, params):
                continue  # below the minimum order notional — never filled
            sigma_daily = self._sigma_daily(data, symbol)
            cost = _cents(
                order_cost(
                    value,
                    sigma_daily,
                    data.adv_dollar_20d.get(symbol, float("nan")),
                    params,
                )
            )
            if delta > 0.0:  # buy
                cash = _cents(cash - value - cost)
                prev_qty = shares.get(symbol, 0.0)
                prev_cost = avg_cost.get(symbol, 0.0)
                new_qty = prev_qty + qty
                avg_cost[symbol] = (
                    (prev_qty * prev_cost + value + cost) / new_qty if new_qty else 0.0
                )
                shares[symbol] = new_qty
                realized = None
                side = "buy"
            else:  # sell
                cash = _cents(cash + value - cost)
                basis = avg_cost.get(symbol, 0.0)
                realized = _cents(value - cost - qty * basis)
                shares[symbol] = shares.get(symbol, 0.0) - qty
                if abs(shares[symbol]) < 1e-9:
                    shares.pop(symbol, None)
                    avg_cost.pop(symbol, None)
                side = "sell"
            traded_value += value
            fills.append(
                FillRecord(
                    session=d,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price=open_px,
                    trade_value=value,
                    cost=cost,
                    realized_pnl=realized,
                )
            )
        return cash, fills, traded_value

    # ------------------------------------------------------- session verdict
    def _session_verdict(
        self,
        d: date,
        regime: RegimeState,
        ladder: LadderState,
        equity: float,
        hwm: float,
        gross: float,
        cash_frac: float,
        tradable: Sequence[str],
    ) -> RiskVerdict:
        cfg = self.risk_config
        effects = effects_for(ladder.rung)
        gross_cap = cfg.gross_cap
        if effects.gross_override is not None:
            gross_cap = min(gross_cap, effects.gross_override)
        if regime.regime is Regime.RISK_OFF:
            gross_cap *= RISK_OFF_GROSS_MULTIPLIER
        gross_cap = max(gross_cap, 0.0)
        new_risk = effects.new_positions_allowed and regime.regime is not Regime.CRISIS
        as_of = datetime(d.year, d.month, d.day)
        vetoes: tuple[Veto, ...] = ()
        if effects.kill_switch:
            vetoes = (Veto(symbol="*", rule="kill_switch", reason="ladder dd20: full stop"),)
        return RiskVerdict(
            as_of=as_of,
            regime=regime,
            ladder=ladder.rung,
            new_risk_allowed=new_risk,
            gross_exposure_cap=gross_cap,
            net_exposure_cap=gross_cap,
            cash_floor=max(cfg.cash_floor, 1.0 - gross_cap),
            per_position_cap={s: cfg.per_position_cap for s in sorted(tradable)},
            sector_caps={},
            cluster_caps={},
            clusters={},
            stop_updates=(),
            vetoes=vetoes,
            metrics=RiskMetrics(
                as_of=as_of,
                gross_exposure=gross,
                cash_fraction=cash_frac,
                drawdown_from_hwm=equity / hwm - 1.0 if hwm > 0 else 0.0,
            ),
            degraded=False,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _enforce_universe(
        advice: SessionAdvice, tradable: Sequence[str]
    ) -> tuple[SessionAdvice, list[str]]:
        """Refuse symbols outside point-in-time universe membership."""
        allowed = set(tradable)
        dropped: set[str] = set()
        watchlist = tuple(r for r in advice.watchlist if r.symbol in allowed)
        dropped |= {r.symbol for r in advice.watchlist if r.symbol not in allowed}
        verdicts = tuple(v for v in advice.verdicts if v.symbol in allowed)
        dropped |= {v.symbol for v in advice.verdicts if v.symbol not in allowed}
        signals = tuple(s for s in advice.signals if s.symbol in allowed)
        dropped |= {s.symbol for s in advice.signals if s.symbol not in allowed}
        return (
            SessionAdvice(watchlist=watchlist, verdicts=verdicts, signals=signals),
            sorted(dropped),
        )

    def _sigmas(
        self,
        data: BacktestData,
        visible_closes: pd.DataFrame,
        tradable: Sequence[str],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for symbol in sorted(tradable):
            override = data.sigma_annual_override.get(symbol)
            if override is not None:
                out[symbol] = override
                continue
            series = visible_closes[symbol].to_numpy(dtype=np.float64)
            series = series[np.isfinite(series)]
            if series.shape[0] < 3:
                out[symbol] = float("nan")
                continue
            window = series[-(_SIGMA_WINDOW + 1) :]
            rets = window[1:] / window[:-1] - 1.0
            rets = rets[np.isfinite(rets)]
            if rets.shape[0] < 2:
                out[symbol] = float("nan")
                continue
            out[symbol] = float(np.std(rets, ddof=1)) * _ANNUALIZE
        return out

    def _sigma_daily(self, data: BacktestData, symbol: str) -> float:
        override = data.sigma_annual_override.get(symbol)
        if override is not None and math.isfinite(override):
            return override / _ANNUALIZE
        return float("nan")  # cost model falls back to its conservative default


# ---------------------------------------------------------------- auditing
def compliance_violations(target: TargetPortfolio, verdict: RiskVerdict) -> tuple[str, ...]:
    """Constraint check for a constructed target; must always come back empty."""
    out: list[str] = []
    by_symbol: dict[str, float] = {}
    for w in target.weights:
        by_symbol[w.symbol] = by_symbol.get(w.symbol, 0.0) + w.weight
    total = math.fsum(by_symbol.values())
    tol = 1e-9
    if abs(total + target.cash - 1.0) > tol:
        out.append(f"weights+cash != 1 ({total + target.cash:.12f})")
    if target.cash < verdict.cash_floor - tol:
        out.append(f"cash {target.cash:.6f} < floor {verdict.cash_floor:.6f}")
    if total > verdict.gross_exposure_cap + tol:
        out.append(f"gross {total:.6f} > cap {verdict.gross_exposure_cap:.6f}")
    vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}
    for symbol in sorted(by_symbol):
        cap = verdict.per_position_cap.get(symbol)
        if cap is not None and by_symbol[symbol] > cap + tol:
            out.append(f"{symbol} weight {by_symbol[symbol]:.6f} > cap {cap:.6f}")
        if symbol in vetoed:
            out.append(f"vetoed symbol {symbol} in target")
    return tuple(out)
