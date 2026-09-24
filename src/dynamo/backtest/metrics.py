"""Backtest metrics and the JSON-serializable report (doc 08 section 5).

No single-number verdicts — the report is a document, not a scalar. Includes
the deflated Sharpe ratio (Bailey & Lopez de Prado) computed against the
persistent trial count, and SPY benchmark comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

TRADING_DAYS = 252
_EULER_GAMMA = 0.5772156649015329


class MetricsBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_sessions: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_sessions: int = 0
    volatility_annual: float = 0.0


class TradeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_closed: int = 0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    worst_trades: tuple[dict[str, float | str], ...] = ()  # worst-5 with dates


class BehaviorStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    avg_gross_exposure: float = 0.0
    max_gross_exposure: float = 0.0
    avg_cash_fraction: float = 1.0
    one_way_turnover_total: float = 0.0
    one_way_turnover_monthly_avg: float = 0.0


class BenchmarkComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark: str = "SPY"
    benchmark_total_return: float = 0.0
    benchmark_cagr: float = 0.0
    benchmark_sharpe: float = 0.0
    alpha_annual: float = 0.0
    beta: float = 0.0
    excess_total_return: float = 0.0
    # Skill net of market exposure (awesome-systematic-trading's core caution:
    # a large slice of published "edge" is just index beta). Residual =
    # strategy return minus beta x benchmark return, per aligned session.
    tracking_error_annual: float = 0.0
    information_ratio: float = 0.0  # alpha_annual / tracking_error_annual
    alpha_t_stat: float = 0.0  # information_ratio * sqrt(years)
    # (1.96 / IR)^2: years of track record needed for this IR to clear
    # t = 1.96. None when IR <= 0 (no amount of time proves a negative edge).
    years_to_significance: float | None = None
    n_aligned: int = 0  # sessions where both series had a return


class BacktestReport(BaseModel):
    """Full-window report; JSON-serializable via ``model_dump_json``."""

    model_config = ConfigDict(frozen=True)

    run_id: str = ""
    engine: str = "event_driven"
    strategy_ref: str = ""
    window_start: str = ""
    window_end: str = ""
    config_hash: str = ""
    seed: int = 0
    trial_number: int = 1
    metrics: MetricsBlock = MetricsBlock()
    trades: TradeStats = TradeStats()
    behavior: BehaviorStats = BehaviorStats()
    per_regime: dict[str, MetricsBlock] = {}
    deflated_sharpe: float = 0.0
    benchmark: BenchmarkComparison | None = None
    constraint_violations: int = 0
    look_ahead_audit_passed: bool | None = None


# ------------------------------------------------------------------ metrics
def _returns(equity: pd.Series) -> np.ndarray:
    values = np.asarray(equity, dtype=np.float64)
    if values.shape[0] < 2:
        return np.empty(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = values[1:] / values[:-1] - 1.0
    return r[np.isfinite(r)]


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest underwater stretch in sessions)."""
    values = np.asarray(equity, dtype=np.float64)
    if values.shape[0] == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = values / peaks - 1.0
    max_dd = float(np.min(dd)) if dd.size else 0.0
    longest = current = 0
    for x in dd:
        if x < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return max_dd, longest


def sharpe_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return float("inf") if float(np.mean(returns)) > 0.0 else 0.0
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if dd <= 0.0:
        return 0.0
    return float(np.mean(returns) / dd * math.sqrt(TRADING_DAYS))


def metrics_block(equity: pd.Series) -> MetricsBlock:
    r = _returns(equity)
    n = len(equity)
    if n < 2 or float(equity.iloc[0]) <= 0.0:
        return MetricsBlock(n_sessions=n)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((n - 1) / TRADING_DAYS, 1e-9)
    growth = float(equity.iloc[-1] / equity.iloc[0])
    cagr = growth ** (1.0 / years) - 1.0 if growth > 0.0 else -1.0
    max_dd, dd_sessions = max_drawdown(equity)
    vol = float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS)) if r.size >= 2 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else 0.0
    return MetricsBlock(
        n_sessions=n,
        total_return=total,
        cagr=float(cagr),
        sharpe=sharpe_ratio(r),
        sortino=sortino_ratio(r),
        calmar=float(calmar),
        max_drawdown=max_dd,
        max_drawdown_sessions=dd_sessions,
        volatility_annual=vol,
    )


def per_regime_blocks(
    equity: pd.Series, regime_by_session: Sequence[str]
) -> dict[str, MetricsBlock]:
    """Metrics per regime label; session i's return is attributed to its regime."""
    r = _returns(equity)
    labels = list(regime_by_session)[1 : len(equity)]
    out: dict[str, MetricsBlock] = {}
    for regime in sorted(set(labels)):
        mask = np.array([lab == regime for lab in labels], dtype=bool)
        seg = r[mask[: r.size]]
        if seg.size == 0:
            out[regime] = MetricsBlock(n_sessions=0)
            continue
        pseudo_equity = pd.Series(np.cumprod(np.concatenate(([1.0], 1.0 + seg))))
        out[regime] = metrics_block(pseudo_equity)
    return out


def trade_stats(realized_pnls: Sequence[tuple[str, str, float]]) -> TradeStats:
    """From (date_iso, symbol, realized_pnl) rows for closed (selling) trades."""
    pnls = [p for _, _, p in realized_pnls]
    if not pnls:
        return TradeStats()
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]
    gross_win = float(sum(wins))
    gross_loss = float(-sum(losses))
    worst = sorted(realized_pnls, key=lambda t: (t[2], t[0], t[1]))[:5]
    return TradeStats(
        n_closed=len(pnls),
        profit_factor=gross_win / gross_loss if gross_loss > 0.0 else float("inf"),
        win_rate=len(wins) / len(pnls),
        avg_win=gross_win / len(wins) if wins else 0.0,
        avg_loss=-gross_loss / len(losses) if losses else 0.0,
        worst_trades=tuple({"date": d, "symbol": s, "pnl": p} for d, s, p in worst),
    )


def benchmark_comparison(
    equity: pd.Series, benchmark_closes: pd.Series, name: str = "SPY"
) -> BenchmarkComparison:
    bench = benchmark_closes.dropna()
    if len(bench) < 2 or len(equity) < 2:
        return BenchmarkComparison(benchmark=name)
    b_equity = bench / float(bench.iloc[0])
    b_block = metrics_block(pd.Series(b_equity.to_numpy()))
    s_block = metrics_block(equity)
    x, y = _aligned_returns(bench, equity)
    n = int(x.size)
    beta = alpha = te = ir = t_stat = 0.0
    years_needed: float | None = None
    if n >= 2:
        var = float(np.var(x))
        if var > 0.0:
            beta = float(np.cov(x, y, ddof=0)[0, 1] / var)
            alpha = float((np.mean(y) - beta * np.mean(x)) * TRADING_DAYS)
            resid = y - beta * x
            if n >= 3:
                te = float(np.std(resid, ddof=1) * math.sqrt(TRADING_DAYS))
            if te > 0.0:
                ir = alpha / te
                t_stat = ir * math.sqrt(n / TRADING_DAYS)
                years_needed = (1.96 / ir) ** 2 if ir > 0.0 else None
    return BenchmarkComparison(
        benchmark=name,
        benchmark_total_return=b_block.total_return,
        benchmark_cagr=b_block.cagr,
        benchmark_sharpe=b_block.sharpe,
        alpha_annual=alpha,
        beta=beta,
        excess_total_return=s_block.total_return - b_block.total_return,
        tracking_error_annual=te,
        information_ratio=ir,
        alpha_t_stat=t_stat,
        years_to_significance=years_needed,
        n_aligned=n,
    )


def _aligned_returns(bench: pd.Series, equity: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(benchmark, strategy) simple returns over the SAME sessions.

    With dated indexes the two series are joined on date before differencing,
    so a session one side lacks can never pair a return with the wrong day.
    (Positional truncation was the old rule: correct only when both series
    covered exactly the same sessions.) Undated series fall back to position.
    """
    if _is_dated(bench.index) and _is_dated(equity.index):
        joint = pd.concat(
            [
                pd.Series(bench.to_numpy(dtype=float), index=_day_index(bench.index)),
                pd.Series(equity.to_numpy(dtype=float), index=_day_index(equity.index)),
            ],
            axis=1,
            join="inner",
        ).sort_index()
        joint = joint[~joint.index.duplicated(keep="last")]
        r = (joint / joint.shift(1) - 1.0).iloc[1:]
        r = r[np.isfinite(r).all(axis=1)]
        return r.iloc[:, 0].to_numpy(dtype=float), r.iloc[:, 1].to_numpy(dtype=float)
    rb, rs = _returns(bench), _returns(equity)
    n = min(rb.size, rs.size)
    return rb[:n], rs[:n]


def _is_dated(index: pd.Index) -> bool:
    if len(index) == 0:
        return False
    if isinstance(index, pd.DatetimeIndex):
        return True
    sample = index[0]
    return hasattr(sample, "year") and hasattr(sample, "month") and hasattr(sample, "day")


def _day_index(index: pd.Index) -> pd.DatetimeIndex:
    """Calendar days, tz-naive: aware stamps are read in UTC, naive as-is."""
    out: list[pd.Timestamp] = []
    for t in index:
        ts = cast(pd.Timestamp, pd.Timestamp(t))
        if ts.tzinfo is not None:
            ts = cast(pd.Timestamp, ts.tz_convert("UTC").tz_localize(None))
        out.append(cast(pd.Timestamp, pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)))
    return pd.DatetimeIndex(out)


# --------------------------------------------------- deflated Sharpe (BLdP)
def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """E[max SR] over N independent trials (Bailey & Lopez de Prado)."""
    if trials <= 1 or sharpe_variance <= 0.0:
        return 0.0
    nd = NormalDist()
    n = float(trials)
    return math.sqrt(sharpe_variance) * (
        (1.0 - _EULER_GAMMA) * nd.inv_cdf(1.0 - 1.0 / n)
        + _EULER_GAMMA * nd.inv_cdf(1.0 - 1.0 / (n * math.e))
    )


def deflated_sharpe_ratio(
    sharpe_annual: float,
    n_obs: int,
    trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_variance_across_trials: float | None = None,
) -> float:
    """P[true SR > 0] after deflating for multiple testing and non-normality.

    Monotone decreasing in the trial count: running 200 variants and
    reporting the best is only cheating if you hide the 200.
    """
    if n_obs < 2:
        return 0.0
    sr = sharpe_annual / math.sqrt(TRADING_DAYS)  # per-observation SR
    var_hat = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr) / (n_obs - 1)
    var_hat = max(var_hat, 1e-12)
    v_trials = (
        sharpe_variance_across_trials if sharpe_variance_across_trials is not None else var_hat
    )
    sr0 = expected_max_sharpe(trials, v_trials)
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(max(var_hat * (n_obs - 1), 1e-12))
    return float(NormalDist().cdf(z))


# ------------------------------------------------------------------- report
def build_report(
    *,
    run_id: str,
    strategy_ref: str,
    equity: pd.Series,
    session_dates: Sequence[str],
    regime_by_session: Sequence[str],
    gross_by_session: Sequence[float],
    cash_by_session: Sequence[float],
    one_way_turnover_total: float,
    realized_pnls: Sequence[tuple[str, str, float]],
    trials: int,
    config_hash: str = "",
    seed: int = 0,
    benchmark_closes: pd.Series | None = None,
    constraint_violations: int = 0,
    look_ahead_audit_passed: bool | None = None,
    extra: Mapping[str, object] | None = None,
) -> BacktestReport:
    del extra  # reserved for stress blocks and fold metrics (walkforward owner)
    block = metrics_block(equity)
    n_months = max(len(equity) / 21.0, 1e-9)
    gross = np.asarray(list(gross_by_session), dtype=np.float64)
    cash = np.asarray(list(cash_by_session), dtype=np.float64)
    behavior = BehaviorStats(
        avg_gross_exposure=float(np.mean(gross)) if gross.size else 0.0,
        max_gross_exposure=float(np.max(gross)) if gross.size else 0.0,
        avg_cash_fraction=float(np.mean(cash)) if cash.size else 1.0,
        one_way_turnover_total=one_way_turnover_total,
        one_way_turnover_monthly_avg=one_way_turnover_total / n_months,
    )
    r = _returns(equity)
    skew = float(pd.Series(r).skew()) if r.size >= 3 else 0.0
    kurt = float(pd.Series(r).kurt() + 3.0) if r.size >= 4 else 3.0
    dsr = deflated_sharpe_ratio(block.sharpe, r.size + 1, trials, skew, kurt)
    bench = benchmark_comparison(equity, benchmark_closes) if benchmark_closes is not None else None
    return BacktestReport(
        run_id=run_id,
        engine="event_driven",
        strategy_ref=strategy_ref,
        window_start=str(session_dates[0]) if len(session_dates) else "",
        window_end=str(session_dates[-1]) if len(session_dates) else "",
        config_hash=config_hash,
        seed=seed,
        trial_number=trials,
        metrics=block,
        trades=trade_stats(realized_pnls),
        behavior=behavior,
        per_regime=per_regime_blocks(equity, regime_by_session),
        deflated_sharpe=dsr,
        benchmark=bench,
        constraint_violations=constraint_violations,
        look_ahead_audit_passed=look_ahead_audit_passed,
    )
