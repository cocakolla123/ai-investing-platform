"""Feature catalog — doc 04 section 2.2. ~51 features across 10 groups.

Every feature is a pure function of data available at ``as_of``:

* prices/volume: bar ``ts`` <= as_of
* fundamentals: only ``xbrl_facts`` rows with ``filed_at`` <= as_of (TTM from
  the four most recent *filed* quarters)
* insider: Form 4 ``filed_at`` <= as_of
* macro: FRED vintages with ``vintage_ts`` <= as_of
* estimates/calendar: snapshot rows <= as_of

Missing inputs produce NaN — feature functions never raise. Groups 1-8 are
cross-sectionally normalized downstream (features.py); groups 9-10 (macro,
calendar) stay raw.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Macro series ids (FRED). Missing series simply yield NaN.
HY_OAS_SERIES = "BAMLH0A0HYM2"
CURVE_SERIES = "T10Y2Y"
CURVE_LONG_SERIES = "DGS10"
CURVE_SHORT_SERIES = "DGS2"

# Days-to-exit heuristic: notional / (participation x 20d average dollar volume).
POSITION_NOTIONAL_USD = 25_000.0
EXIT_PARTICIPATION = 0.10

TAX_RATE = 0.21

SHARES_TAGS = (
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
)

# Groups whose features stay raw (context shared across names) — doc 04 section 3.
RAW_GROUPS = frozenset({"macro", "calendar"})

GROUP_TARGETS: dict[str, int] = {
    "momentum": 8,
    "meanrev": 5,
    "volatility": 6,
    "volume": 5,
    "value": 6,
    "quality": 7,
    "estimates": 3,
    "insider": 3,
    "macro": 5,
    "calendar": 3,
}


@dataclass(frozen=True)
class FeatureInputs:
    """Point-in-time input frames. Any frame may be None or empty.

    bars: long df [symbol, ts, open, high, low, close, adj_close, volume]
    fundamentals: long df of xbrl facts [symbol, tag, period_end, value, filed_at]
    insider: [symbol, insider_name, tx_type, shares, price, filed_at]
    estimates: [symbol, fetched_at, eps_est] (append-only snapshots)
    macro: [series_id, ts, value, vintage_ts]
    regime: Series of regime labels indexed by date
    earnings_calendar: [symbol, event_date]
    sector_map: symbol -> sector ETF ticker (also the peer-group key)
    """

    bars: pd.DataFrame | None = None
    fundamentals: pd.DataFrame | None = None
    insider: pd.DataFrame | None = None
    estimates: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    regime: pd.Series | None = None
    earnings_calendar: pd.DataFrame | None = None
    sector_map: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FeatureContext:
    """Panels prepared once per (inputs, as_of) so ~50 features share pivots."""

    inputs: FeatureInputs
    as_of: pd.Timestamp
    universe: tuple[str, ...]
    price: pd.DataFrame  # adjusted close, ts x symbol, ts <= as_of
    close: pd.DataFrame  # raw close
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame


FeatureFn = Callable[[FeatureContext], pd.Series]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    lookback_days: int
    fn: FeatureFn


def to_ts(value: object) -> pd.Timestamp:
    """pd.Timestamp with a non-NaT static type (inputs are always concrete)."""
    return cast(pd.Timestamp, pd.Timestamp(value))  # type: ignore[arg-type]


def _rows(df: pd.DataFrame, mask: object) -> pd.DataFrame:
    """Boolean-mask row filter with a stable DataFrame static type."""
    return df.loc[cast(pd.Series, mask)]


def _roll(
    df: pd.DataFrame, window: int, agg: str, min_periods: int | None = None
) -> pd.DataFrame:
    """rolling(...).<agg>() with a stable DataFrame static type."""
    r = df.rolling(window, min_periods=window if min_periods is None else min_periods)
    return cast(pd.DataFrame, getattr(r, agg)())


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float)


def _pivot(bars: pd.DataFrame, column: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if column not in bars.columns:
        return _empty_panel()
    df = _rows(bars, pd.to_datetime(bars["ts"]) <= as_of).loc[:, ["symbol", "ts", column]].copy()
    if df.empty:
        return _empty_panel()
    df["ts"] = pd.to_datetime(df["ts"])
    panel = df.pivot_table(index="ts", columns="symbol", values=column, aggfunc="last")
    return panel.sort_index()


def build_context(
    inputs: FeatureInputs, as_of: date | pd.Timestamp, universe: Sequence[str]
) -> FeatureContext:
    ts_as_of = to_ts(as_of)
    bars = inputs.bars
    if bars is None or bars.empty or "ts" not in bars.columns or "symbol" not in bars.columns:
        empty = _empty_panel()
        return FeatureContext(
            inputs=inputs,
            as_of=ts_as_of,
            universe=tuple(universe),
            price=empty,
            close=empty.copy(),
            high=empty.copy(),
            low=empty.copy(),
            volume=empty.copy(),
        )
    close = _pivot(bars, "close", ts_as_of)
    adj = _pivot(bars, "adj_close", ts_as_of)
    price = cast(pd.DataFrame, adj.combine_first(close)) if not adj.empty else close
    return FeatureContext(
        inputs=inputs,
        as_of=ts_as_of,
        universe=tuple(universe),
        price=price,
        close=close,
        high=_pivot(bars, "high", ts_as_of),
        low=_pivot(bars, "low", ts_as_of),
        volume=_pivot(bars, "volume", ts_as_of),
    )


def _nan_series(ctx: FeatureContext) -> pd.Series:
    return pd.Series(np.nan, index=list(ctx.universe), dtype=float)


def _last_row(panel: pd.DataFrame, ctx: FeatureContext) -> pd.Series:
    if panel.empty:
        return _nan_series(ctx)
    return panel.iloc[-1].reindex(ctx.universe).astype(float)


def _log_returns(price: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = cast(pd.DataFrame, np.log(price))
    return logged.diff()


# ---------------------------------------------------------------------------
# Group 1 — momentum / trend
# ---------------------------------------------------------------------------


def _ret(days: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) <= days:
            return _nan_series(ctx)
        return (p.iloc[-1] / p.iloc[-1 - days] - 1.0).reindex(ctx.universe)

    return fn


def _mom_12_1(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) <= TRADING_DAYS_PER_YEAR:
        return _nan_series(ctx)
    return (p.iloc[-22] / p.iloc[-1 - TRADING_DAYS_PER_YEAR] - 1.0).reindex(ctx.universe)


def _dist_ma(window: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) < window:
            return _nan_series(ctx)
        ma = _roll(p, window, "mean")
        return (p.iloc[-1] / ma.iloc[-1] - 1.0).reindex(ctx.universe)

    return fn


def _ma50_slope(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 70:
        return _nan_series(ctx)
    ma = _roll(p, 50, "mean")
    return ((ma.iloc[-1] - ma.iloc[-21]) / ma.iloc[-21]).reindex(ctx.universe)


def _rs_20d_sector(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) <= 20:
        return _nan_series(ctx)
    ret20 = p.iloc[-1] / p.iloc[-21] - 1.0
    out = {}
    for sym in ctx.universe:
        etf = ctx.inputs.sector_map.get(sym)
        if etf is None or etf not in ret20.index or sym not in ret20.index:
            out[sym] = np.nan
        else:
            out[sym] = float(ret20[sym] - ret20[etf])
    return pd.Series(out, dtype=float).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 2 — mean-reversion
# ---------------------------------------------------------------------------


def _ret_z(days: int, window: int = 120) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) <= days + 30:
            return _nan_series(ctx)
        r = p / p.shift(days) - 1.0
        mu = _roll(r, window, "mean", 30)
        sd = _roll(r, window, "std", 30)
        z = (r.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1]
        return z.replace([np.inf, -np.inf], np.nan).reindex(ctx.universe)

    return fn


def _rsi_14(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 15:
        return _nan_series(ctx)
    diff = p.diff()
    gain = _roll(diff.clip(lower=0.0), 14, "mean")
    loss = _roll((-diff).clip(lower=0.0), 14, "mean")
    denom = gain.iloc[-1] + loss.iloc[-1]
    rsi = 100.0 * gain.iloc[-1] / denom.where(denom > 0)
    rsi = rsi.where(denom > 0, 50.0)
    return rsi.reindex(ctx.universe)


def _dist_20d_mean_sigma(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 20:
        return _nan_series(ctx)
    mu = _roll(p, 20, "mean").iloc[-1]
    sd = _roll(p, 20, "std").iloc[-1]
    z = (p.iloc[-1] - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).reindex(ctx.universe)


def _drawdown_20d_high(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 20:
        return _nan_series(ctx)
    hi = _roll(p, 20, "max").iloc[-1]
    return (p.iloc[-1] / hi - 1.0).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 3 — volatility
# ---------------------------------------------------------------------------


def _vol_cc(window: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) <= window:
            return _nan_series(ctx)
        lr = _log_returns(p)
        vol = _roll(lr, window, "std").iloc[-1] * math.sqrt(TRADING_DAYS_PER_YEAR)
        return vol.reindex(ctx.universe)

    return fn


def _vol_parkinson_20d(ctx: FeatureContext) -> pd.Series:
    h, lo = ctx.high, ctx.low
    if h.empty or lo.empty or len(h) < 20:
        return _nan_series(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = cast(pd.DataFrame, np.log(h / lo)) ** 2 / (4.0 * math.log(2.0))
    var = _roll(hl, 20, "mean").iloc[-1]
    return (var**0.5 * math.sqrt(TRADING_DAYS_PER_YEAR)).reindex(ctx.universe)


def _vol_ewma_60d(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 30:
        return _nan_series(ctx)
    lr = _log_returns(p)
    ewm_std = cast(pd.DataFrame, lr.ewm(span=60, min_periods=20).std())
    vol = ewm_std.iloc[-1] * math.sqrt(TRADING_DAYS_PER_YEAR)
    return vol.reindex(ctx.universe)


def _vol20_series(p: pd.DataFrame) -> pd.DataFrame:
    lr = _log_returns(p)
    return _roll(lr, 20, "std") * math.sqrt(TRADING_DAYS_PER_YEAR)


def _vol_of_vol_60d(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 80:
        return _nan_series(ctx)
    v = _vol20_series(p)
    return _roll(v, 60, "std", 40).iloc[-1].reindex(ctx.universe)


def _vol_pctile_2y(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 120:
        return _nan_series(ctx)
    v = _vol20_series(p).tail(2 * TRADING_DAYS_PER_YEAR)
    out = {}
    for sym in ctx.universe:
        if sym not in v.columns:
            out[sym] = np.nan
            continue
        s = v[sym].dropna()
        if len(s) < 60:
            out[sym] = np.nan
        else:
            out[sym] = float((s <= s.iloc[-1]).mean())
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _true_range(ctx: FeatureContext) -> pd.DataFrame:
    h, lo, c = ctx.high, ctx.low, ctx.close
    if h.empty or lo.empty or c.empty:
        return _empty_panel()
    prev_c = c.shift(1)
    stacked = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], keys=["hl", "hc", "lc"])
    return cast(pd.DataFrame, stacked.groupby(level=1).max())


def _atr14_over_price(ctx: FeatureContext) -> pd.Series:
    tr = _true_range(ctx)
    if tr.empty or len(tr) < 15:
        return _nan_series(ctx)
    atr = _roll(tr, 14, "mean").iloc[-1]
    last_close = _last_row(ctx.close, ctx)
    return (atr.reindex(ctx.universe) / last_close).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Group 4 — volume / liquidity
# ---------------------------------------------------------------------------


def _dollar_panel(ctx: FeatureContext) -> pd.DataFrame:
    if ctx.close.empty or ctx.volume.empty:
        return _empty_panel()
    return ctx.close * ctx.volume


def _dollar_vol_20d(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    if dv.empty or len(dv) < 20:
        return _nan_series(ctx)
    return _roll(dv, 20, "mean").iloc[-1].reindex(ctx.universe)


def _dollar_vol_z_60d(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    if dv.empty or len(dv) < 60:
        return _nan_series(ctx)
    mu = _roll(dv, 60, "mean").iloc[-1]
    sd = _roll(dv, 60, "std").iloc[-1]
    z = (dv.iloc[-1] - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).reindex(ctx.universe)


def _volume_trend_20_60(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    if dv.empty or len(dv) < 60:
        return _nan_series(ctx)
    adv20 = _roll(dv, 20, "mean").iloc[-1]
    adv60 = _roll(dv, 60, "mean").iloc[-1]
    return (adv20 / adv60 - 1.0).replace([np.inf, -np.inf], np.nan).reindex(ctx.universe)


def _amihud_20d(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    p = ctx.price
    if dv.empty or len(p) < 21:
        return _nan_series(ctx)
    ret = (p / p.shift(1) - 1.0).abs()
    illiq = (ret / dv.where(dv > 0)) * 1e6
    return _roll(illiq, 20, "mean").iloc[-1].reindex(ctx.universe)


def _days_to_exit(ctx: FeatureContext) -> pd.Series:
    adv20 = _dollar_vol_20d(ctx)
    denom = EXIT_PARTICIPATION * adv20.where(adv20 > 0)
    return cast(pd.Series, POSITION_NOTIONAL_USD / denom)


# ---------------------------------------------------------------------------
# Fundamentals — point-in-time helpers (shared with research/quality.py)
# ---------------------------------------------------------------------------


def pit_facts(fundamentals: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows with filed_at <= as_of; the only accessor the feature path uses."""
    cols = ["symbol", "tag", "period_end", "value", "filed_at"]
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=pd.Index(cols))
    missing = [c for c in cols if c not in fundamentals.columns]
    if missing:
        return pd.DataFrame(columns=pd.Index(cols))
    df = fundamentals.loc[:, cols].copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return _rows(df, df["filed_at"] <= as_of)


def quarterly_series(
    facts: pd.DataFrame, symbol: str, tag: str, as_of: pd.Timestamp
) -> pd.Series:
    """Latest filed value per period_end, sorted by period_end ascending."""
    sub = _rows(facts, (facts["symbol"] == symbol) & (facts["tag"] == tag))
    if sub.empty:
        return pd.Series(dtype=float)
    latest = cast(
        pd.DataFrame,
        sub.sort_values(["period_end", "filed_at"]).groupby("period_end", sort=True).last(),
    )
    return cast(pd.Series, latest["value"]).astype(float)


def ttm_sum(
    facts: pd.DataFrame,
    symbol: str,
    tag: str,
    as_of: pd.Timestamp,
    offset_quarters: int = 0,
) -> float:
    """Sum of the four most recent filed quarters, offset back by N quarters."""
    q = quarterly_series(facts, symbol, tag, as_of)
    hi = len(q) - offset_quarters
    lo = hi - 4
    if lo < 0:
        return float("nan")
    return float(q.iloc[lo:hi].sum())


def latest_value(facts: pd.DataFrame, symbol: str, tag: str, as_of: pd.Timestamp) -> float:
    """Most recently filed value (balance-sheet style tags)."""
    sub = _rows(facts, (facts["symbol"] == symbol) & (facts["tag"] == tag))
    if sub.empty:
        return float("nan")
    ordered = sub.sort_values(["filed_at", "period_end"])
    return float(ordered["value"].iloc[-1])


def n_filed_quarters(facts: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> int:
    return len(quarterly_series(facts, symbol, "Revenues", as_of))


def shares_outstanding(facts: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> float:
    for tag in SHARES_TAGS:
        v = latest_value(facts, symbol, tag, as_of)
        if not math.isnan(v) and v > 0:
            return v
    return float("nan")


def pit_close(price: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> float:
    if price.empty or symbol not in price.columns:
        return float("nan")
    s = price.loc[price.index <= as_of, symbol].dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _market_cap(ctx: FeatureContext, facts: pd.DataFrame, sym: str) -> float:
    sh = shares_outstanding(facts, sym, ctx.as_of)
    px = pit_close(ctx.close, sym, ctx.as_of)
    return sh * px


def _enterprise_value(ctx: FeatureContext, facts: pd.DataFrame, sym: str) -> float:
    mcap = _market_cap(ctx, facts, sym)
    if math.isnan(mcap):
        return float("nan")
    ltd = latest_value(facts, sym, "LongTermDebt", ctx.as_of)
    cash = latest_value(facts, sym, "CashAndCashEquivalentsAtCarryingValue", ctx.as_of)
    ltd = 0.0 if math.isnan(ltd) else ltd
    cash = 0.0 if math.isnan(cash) else cash
    return mcap + ltd - cash


def _fcf_ttm(facts: pd.DataFrame, sym: str, as_of: pd.Timestamp, offset: int = 0) -> float:
    cfo = ttm_sum(facts, sym, "NetCashProvidedByUsedInOperatingActivities", as_of, offset)
    capex = ttm_sum(facts, sym, "PaymentsToAcquirePropertyPlantAndEquipment", as_of, offset)
    if math.isnan(cfo):
        return float("nan")
    return cfo - (0.0 if math.isnan(capex) else capex)


def _per_symbol(ctx: FeatureContext, fn: Callable[[pd.DataFrame, str], float]) -> pd.Series:
    facts = pit_facts(ctx.inputs.fundamentals, ctx.as_of)
    out = {}
    for sym in ctx.universe:
        try:
            out[sym] = fn(facts, sym)
        except Exception:  # missing inputs -> NaN, never raise
            out[sym] = float("nan")
    return pd.Series(out, dtype=float).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 5 — fundamental value
# ---------------------------------------------------------------------------


def _ev_ebit(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        ev = _enterprise_value(ctx, facts, sym)
        ebit = ttm_sum(facts, sym, "OperatingIncomeLoss", ctx.as_of)
        if math.isnan(ev) or math.isnan(ebit) or ebit <= 0:
            return float("nan")
        return ev / ebit

    return _per_symbol(ctx, fn)


def _fcf_yield(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        ev = _enterprise_value(ctx, facts, sym)
        fcf = _fcf_ttm(facts, sym, ctx.as_of)
        if math.isnan(ev) or ev <= 0 or math.isnan(fcf):
            return float("nan")
        return fcf / ev

    return _per_symbol(ctx, fn)


def _ps_ttm(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        mcap = _market_cap(ctx, facts, sym)
        rev = ttm_sum(facts, sym, "Revenues", ctx.as_of)
        if math.isnan(mcap) or math.isnan(rev) or rev <= 0:
            return float("nan")
        return mcap / rev

    return _per_symbol(ctx, fn)


def _sector_pctile(values: pd.Series, sector_map: Mapping[str, str]) -> pd.Series:
    sectors = pd.Series({str(s): sector_map.get(str(s), "_none") for s in values.index})
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in sectors.groupby(sectors).groups.items():
        grp = values.loc[list(idx)].dropna()
        if grp.empty:
            continue
        out.loc[grp.index] = grp.rank(method="average", pct=True)
    return out


def _ps_sector_pctile(ctx: FeatureContext) -> pd.Series:
    return _sector_pctile(_ps_ttm(ctx), ctx.inputs.sector_map)


def _earnings_yield(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        mcap = _market_cap(ctx, facts, sym)
        ni = ttm_sum(facts, sym, "NetIncomeLoss", ctx.as_of)
        if math.isnan(mcap) or mcap <= 0 or math.isnan(ni):
            return float("nan")
        return ni / mcap

    return _per_symbol(ctx, fn)


def _fcf_yield_history(
    ctx: FeatureContext, facts: pd.DataFrame, sym: str, n_quarters: int = 12
) -> list[float]:
    """FCF yield replayed at each of the last N quarter filing dates."""
    q = quarterly_series(facts, sym, "Revenues", ctx.as_of)
    if q.empty:
        return []
    filings = (
        _rows(facts, (facts["symbol"] == sym) & (facts["tag"] == "Revenues"))
        .sort_values(["period_end", "filed_at"])
        .groupby("period_end", sort=True)
        .last()["filed_at"]
    )
    vals: list[float] = []
    for filed_at in filings.iloc[-n_quarters:]:
        at = to_ts(filed_at)
        sub = _rows(facts, facts["filed_at"] <= at)
        sh = shares_outstanding(sub, sym, at)
        px = pit_close(ctx.close, sym, at)
        fcf = _fcf_ttm(sub, sym, at)
        ltd = latest_value(sub, sym, "LongTermDebt", at)
        cash = latest_value(sub, sym, "CashAndCashEquivalentsAtCarryingValue", at)
        ev = sh * px + (0.0 if math.isnan(ltd) else ltd) - (0.0 if math.isnan(cash) else cash)
        if not math.isnan(ev) and ev > 0 and not math.isnan(fcf):
            vals.append(fcf / ev)
    return vals


def _fcf_yield_vs_3y(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        hist = _fcf_yield_history(ctx, facts, sym)
        if len(hist) < 8:
            return float("nan")
        cur = hist[-1]
        return float(np.mean([cur >= h for h in hist]))

    return _per_symbol(ctx, fn)


# ---------------------------------------------------------------------------
# Group 6 — fundamental quality / growth
# ---------------------------------------------------------------------------


def _gross_margin_ttm(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        gp = ttm_sum(facts, sym, "GrossProfit", ctx.as_of)
        rev = ttm_sum(facts, sym, "Revenues", ctx.as_of)
        if math.isnan(gp) or math.isnan(rev) or rev <= 0:
            return float("nan")
        return gp / rev

    return _per_symbol(ctx, fn)


def gross_margin_stability(facts: pd.DataFrame, sym: str, as_of: pd.Timestamp) -> float:
    """Negative sigma of quarterly gross margin over up to 12 quarters (>=8 needed)."""
    gp = quarterly_series(facts, sym, "GrossProfit", as_of).iloc[-12:]
    rev = quarterly_series(facts, sym, "Revenues", as_of).iloc[-12:]
    gm = (gp / rev.where(rev > 0)).dropna()
    if len(gm) < 8:
        return float("nan")
    return -float(gm.std())


def _gm_stability_feature(ctx: FeatureContext) -> pd.Series:
    return _per_symbol(ctx, lambda facts, sym: gross_margin_stability(facts, sym, ctx.as_of))


def roic_ttm(facts: pd.DataFrame, sym: str, as_of: pd.Timestamp) -> float:
    ebit = ttm_sum(facts, sym, "OperatingIncomeLoss", as_of)
    se = latest_value(facts, sym, "StockholdersEquity", as_of)
    ltd = latest_value(facts, sym, "LongTermDebt", as_of)
    cash = latest_value(facts, sym, "CashAndCashEquivalentsAtCarryingValue", as_of)
    if math.isnan(ebit) or math.isnan(se):
        return float("nan")
    invested = se + (0.0 if math.isnan(ltd) else ltd) - (0.0 if math.isnan(cash) else cash)
    if invested <= 0:
        return float("nan")
    return ebit * (1.0 - TAX_RATE) / invested


def _roic_feature(ctx: FeatureContext) -> pd.Series:
    return _per_symbol(ctx, lambda facts, sym: roic_ttm(facts, sym, ctx.as_of))


def _rev_growth_yoy(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        cur = ttm_sum(facts, sym, "Revenues", ctx.as_of)
        prev = ttm_sum(facts, sym, "Revenues", ctx.as_of, offset_quarters=4)
        if math.isnan(cur) or math.isnan(prev) or prev <= 0:
            return float("nan")
        return cur / prev - 1.0

    return _per_symbol(ctx, fn)


def _rev_cagr_3y(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        cur = ttm_sum(facts, sym, "Revenues", ctx.as_of)
        old = ttm_sum(facts, sym, "Revenues", ctx.as_of, offset_quarters=12)
        if math.isnan(cur) or math.isnan(old) or old <= 0 or cur <= 0:
            return float("nan")
        return (cur / old) ** (1.0 / 3.0) - 1.0

    return _per_symbol(ctx, fn)


def _accruals_ratio(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        ni = ttm_sum(facts, sym, "NetIncomeLoss", ctx.as_of)
        cfo = ttm_sum(facts, sym, "NetCashProvidedByUsedInOperatingActivities", ctx.as_of)
        se = latest_value(facts, sym, "StockholdersEquity", ctx.as_of)
        if math.isnan(ni) or math.isnan(cfo) or math.isnan(se) or se == 0:
            return float("nan")
        return (ni - cfo) / abs(se)

    return _per_symbol(ctx, fn)


def _fcf_conversion(ctx: FeatureContext) -> pd.Series:
    def fn(facts: pd.DataFrame, sym: str) -> float:
        fcf = _fcf_ttm(facts, sym, ctx.as_of)
        ni = ttm_sum(facts, sym, "NetIncomeLoss", ctx.as_of)
        if math.isnan(fcf) or math.isnan(ni) or ni <= 0:
            return float("nan")
        return fcf / ni

    return _per_symbol(ctx, fn)


# ---------------------------------------------------------------------------
# Group 7 — estimate revisions (NaN when no estimates snapshots supplied)
# ---------------------------------------------------------------------------


def _estimate_at(est: pd.DataFrame, sym: str, at: pd.Timestamp) -> float:
    sub = _rows(est, (est["symbol"] == sym) & (est["fetched_at"] <= at))
    if sub.empty:
        return float("nan")
    return float(sub.sort_values("fetched_at")["eps_est"].iloc[-1])


def _est_revision(days: int, mode: str) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        est = ctx.inputs.estimates
        if est is None or est.empty or not {"symbol", "fetched_at", "eps_est"} <= set(
            est.columns
        ):
            return _nan_series(ctx)
        e = est.copy()
        e["fetched_at"] = pd.to_datetime(e["fetched_at"])
        out = {}
        for sym in ctx.universe:
            cur = _estimate_at(e, sym, ctx.as_of)
            old = _estimate_at(e, sym, to_ts(ctx.as_of - pd.Timedelta(days=days)))
            if math.isnan(cur) or math.isnan(old) or old == 0:
                out[sym] = np.nan
            elif mode == "sign":
                out[sym] = float(np.sign(cur - old))
            else:
                out[sym] = (cur - old) / abs(old)
        return pd.Series(out, dtype=float).reindex(ctx.universe)

    return fn


# ---------------------------------------------------------------------------
# Group 8 — insider activity (Form 4s count when filed, not when traded)
# ---------------------------------------------------------------------------


def _insider_window(
    insider: pd.DataFrame | None, as_of: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    cols = {"symbol", "insider_name", "tx_type", "shares", "filed_at"}
    if insider is None or insider.empty or not cols <= set(insider.columns):
        return pd.DataFrame(columns=pd.Index(sorted(cols | {"price"})))
    df = insider.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    lo = as_of - pd.Timedelta(days=window_days)
    return _rows(df, (df["filed_at"] <= as_of) & (df["filed_at"] > lo))


def insider_net_buy_signed(
    insider: pd.DataFrame | None, symbol: str, as_of: pd.Timestamp, window_days: int = 90
) -> float:
    window = _insider_window(insider, as_of, window_days)
    df = _rows(window, window["symbol"] == symbol)
    if df.empty:
        return float("nan")
    px = df["price"] if "price" in df.columns else pd.Series(np.nan, index=df.index)
    dollars = cast(pd.Series, df["shares"].astype(float) * px.astype(float)).fillna(0.0)
    tx = cast(pd.Series, df["tx_type"].astype(str)).str.upper()
    buys = dollars[tx.str.startswith("P")].sum()
    sells = dollars[tx.str.startswith("S")].sum()
    net = float(buys - sells)
    return math.copysign(math.log1p(abs(net)), net) if net != 0 else 0.0


def _insider_net_buy_90d(ctx: FeatureContext) -> pd.Series:
    out = {
        sym: insider_net_buy_signed(ctx.inputs.insider, sym, ctx.as_of, 90)
        for sym in ctx.universe
    }
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _insider_buyer_count_90d(ctx: FeatureContext) -> pd.Series:
    df = _insider_window(ctx.inputs.insider, ctx.as_of, 90)
    if df.empty:
        return _nan_series(ctx)
    tx = cast(pd.Series, df["tx_type"].astype(str)).str.upper()
    buys = _rows(df, tx.str.startswith("P"))
    counts = cast(pd.Series, buys.groupby("symbol")["insider_name"].nunique())
    return counts.reindex(ctx.universe).astype(float)


def _insider_cluster_buy(ctx: FeatureContext) -> pd.Series:
    df = _insider_window(ctx.inputs.insider, ctx.as_of, 90)
    if df.empty:
        return _nan_series(ctx)
    tx = cast(pd.Series, df["tx_type"].astype(str)).str.upper()
    buys = _rows(df, tx.str.startswith("P"))
    out = {}
    for sym in ctx.universe:
        sub = _rows(buys, buys["symbol"] == sym)
        if sub.empty:
            out[sym] = 0.0
            continue
        flag = 0.0
        for _, row in sub.iterrows():
            end = row["filed_at"]
            window_lo = end - pd.Timedelta(days=30)
            near = _rows(sub, (sub["filed_at"] <= end) & (sub["filed_at"] > window_lo))
            if near["insider_name"].nunique() >= 3:
                flag = 1.0
                break
        out[sym] = flag
    return pd.Series(out, dtype=float).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 9 — macro / regime (raw; same value replicated per symbol)
# ---------------------------------------------------------------------------


def _macro_series(ctx: FeatureContext, series_id: str) -> pd.Series:
    m = ctx.inputs.macro
    cols = {"series_id", "ts", "value", "vintage_ts"}
    if m is None or m.empty or not cols <= set(m.columns):
        return pd.Series(dtype=float)
    df = _rows(m, m["series_id"] == series_id).copy()
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"])
    df["vintage_ts"] = pd.to_datetime(df["vintage_ts"])
    df = _rows(df, (df["vintage_ts"] <= ctx.as_of) & (df["ts"] <= ctx.as_of))
    if df.empty:
        return pd.Series(dtype=float)
    ordered = df.sort_values(["ts", "vintage_ts"])
    latest = cast(pd.DataFrame, ordered.groupby("ts", sort=True).last())
    return cast(pd.Series, latest["value"]).astype(float)


def _macro_last(ctx: FeatureContext, series_id: str) -> float:
    s = _macro_series(ctx, series_id)
    return float(s.iloc[-1]) if len(s) else float("nan")


def _broadcast(ctx: FeatureContext, value: float) -> pd.Series:
    return pd.Series(value, index=list(ctx.universe), dtype=float)


def _regime_code(ctx: FeatureContext) -> pd.Series:
    r = ctx.inputs.regime
    if r is None or len(r) == 0:
        return _nan_series(ctx)
    s = r.copy()
    s.index = pd.to_datetime(s.index)
    s = cast(pd.Series, s[s.index <= ctx.as_of])
    if s.empty:
        return _nan_series(ctx)
    val = s.iloc[-1]
    if isinstance(val, (int | float | np.floating | np.integer)):
        return _broadcast(ctx, float(val))
    labels = sorted({str(v) for v in r.dropna().unique()})
    return _broadcast(ctx, float(labels.index(str(val))))


def _hy_oas_level(ctx: FeatureContext) -> pd.Series:
    return _broadcast(ctx, _macro_last(ctx, HY_OAS_SERIES))


def _hy_oas_chg_60d(ctx: FeatureContext) -> pd.Series:
    s = _macro_series(ctx, HY_OAS_SERIES)
    if len(s) < 61:
        return _nan_series(ctx)
    return _broadcast(ctx, float(s.iloc[-1] - s.iloc[-61]))


def _yield_curve_slope(ctx: FeatureContext) -> pd.Series:
    direct = _macro_last(ctx, CURVE_SERIES)
    if not math.isnan(direct):
        return _broadcast(ctx, direct)
    long_ = _macro_last(ctx, CURVE_LONG_SERIES)
    short = _macro_last(ctx, CURVE_SHORT_SERIES)
    return _broadcast(ctx, long_ - short)


def _breadth_above_200dma(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 200:
        return _nan_series(ctx)
    ma = _roll(p, 200, "mean").iloc[-1]
    last = p.iloc[-1]
    flags = (last > ma)[ma.notna() & last.notna()]
    if flags.empty:
        return _nan_series(ctx)
    return _broadcast(ctx, float(flags.mean()))


# ---------------------------------------------------------------------------
# Group 10 — calendar (raw)
# ---------------------------------------------------------------------------


def _calendar_frame(ctx: FeatureContext) -> pd.DataFrame:
    cal = ctx.inputs.earnings_calendar
    if cal is None or cal.empty or not {"symbol", "event_date"} <= set(cal.columns):
        return pd.DataFrame(columns=pd.Index(["symbol", "event_date"]))
    df = cal.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def _days_to_next_earnings(ctx: FeatureContext) -> pd.Series:
    df = _calendar_frame(ctx)
    if df.empty:
        return _nan_series(ctx)
    fut = _rows(df, df["event_date"] >= ctx.as_of)
    nxt = cast(pd.Series, fut.groupby("symbol")["event_date"].min())
    days = cast(pd.Series, nxt - ctx.as_of).dt.days
    return days.reindex(ctx.universe).astype(float)


def _days_since_last_earnings(ctx: FeatureContext) -> pd.Series:
    df = _calendar_frame(ctx)
    if df.empty:
        return _nan_series(ctx)
    past = _rows(df, df["event_date"] < ctx.as_of)
    prev = cast(pd.Series, past.groupby("symbol")["event_date"].max())
    days = cast(pd.Series, ctx.as_of - prev).dt.days
    return days.reindex(ctx.universe).astype(float)


def _month_of_year(ctx: FeatureContext) -> pd.Series:
    return _broadcast(ctx, float(ctx.as_of.month))


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def _specs() -> list[FeatureSpec]:
    s = FeatureSpec
    return [
        # 1 — momentum / trend (8)
        s("ret_20d", "momentum", 20, _ret(20)),
        s("ret_60d", "momentum", 60, _ret(60)),
        s("ret_120d", "momentum", 120, _ret(120)),
        s("mom_12_1", "momentum", 252, _mom_12_1),
        s("dist_ma50", "momentum", 50, _dist_ma(50)),
        s("dist_ma200", "momentum", 200, _dist_ma(200)),
        s("ma50_slope", "momentum", 70, _ma50_slope),
        s("rs_20d_sector", "momentum", 20, _rs_20d_sector),
        # 2 — mean-reversion (5)
        s("ret_5d_z", "meanrev", 125, _ret_z(5)),
        s("ret_10d_z", "meanrev", 130, _ret_z(10)),
        s("rsi_14", "meanrev", 14, _rsi_14),
        s("dist_20d_mean_sigma", "meanrev", 20, _dist_20d_mean_sigma),
        s("drawdown_20d_high", "meanrev", 20, _drawdown_20d_high),
        # 3 — volatility (6)
        s("vol_cc_20d", "volatility", 20, _vol_cc(20)),
        s("vol_parkinson_20d", "volatility", 20, _vol_parkinson_20d),
        s("vol_ewma_60d", "volatility", 60, _vol_ewma_60d),
        s("vol_of_vol_60d", "volatility", 80, _vol_of_vol_60d),
        s("vol_pctile_2y", "volatility", 2 * TRADING_DAYS_PER_YEAR, _vol_pctile_2y),
        s("atr14_over_price", "volatility", 15, _atr14_over_price),
        # 4 — volume / liquidity (5)
        s("dollar_vol_20d", "volume", 20, _dollar_vol_20d),
        s("dollar_vol_z_60d", "volume", 60, _dollar_vol_z_60d),
        s("volume_trend_20_60", "volume", 60, _volume_trend_20_60),
        s("amihud_20d", "volume", 21, _amihud_20d),
        s("days_to_exit", "volume", 20, _days_to_exit),
        # 5 — fundamental value (6)
        s("ev_ebit", "value", 400, _ev_ebit),
        s("fcf_yield", "value", 400, _fcf_yield),
        s("ps_ttm", "value", 400, _ps_ttm),
        s("ps_sector_pctile", "value", 400, _ps_sector_pctile),
        s("earnings_yield", "value", 400, _earnings_yield),
        s("fcf_yield_vs_3y", "value", 3 * 365, _fcf_yield_vs_3y),
        # 6 — fundamental quality / growth (7)
        s("gross_margin_ttm", "quality", 400, _gross_margin_ttm),
        s("gross_margin_stability_12q", "quality", 3 * 365, _gm_stability_feature),
        s("roic_ttm", "quality", 400, _roic_feature),
        s("rev_growth_yoy", "quality", 2 * 365, _rev_growth_yoy),
        s("rev_cagr_3y", "quality", 4 * 365, _rev_cagr_3y),
        s("accruals_ratio", "quality", 400, _accruals_ratio),
        s("fcf_conversion", "quality", 400, _fcf_conversion),
        # 7 — estimate revisions (3) — NaN when no estimates data
        s("est_rev_breadth_30d", "estimates", 30, _est_revision(30, "sign")),
        s("est_rev_magnitude_90d", "estimates", 90, _est_revision(90, "rel")),
        s("fwd_estimate_momentum", "estimates", 30, _est_revision(30, "rel")),
        # 8 — insider activity (3)
        s("insider_net_buy_90d", "insider", 90, _insider_net_buy_90d),
        s("insider_buyer_count_90d", "insider", 90, _insider_buyer_count_90d),
        s("insider_cluster_buy", "insider", 90, _insider_cluster_buy),
        # 9 — macro / regime (5, raw)
        s("regime_code", "macro", 1, _regime_code),
        s("hy_oas_level", "macro", 1, _hy_oas_level),
        s("hy_oas_chg_60d", "macro", 90, _hy_oas_chg_60d),
        s("yield_curve_slope", "macro", 1, _yield_curve_slope),
        s("breadth_above_200dma", "macro", 200, _breadth_above_200dma),
        # 10 — calendar (3, raw)
        s("days_to_next_earnings", "calendar", 1, _days_to_next_earnings),
        s("days_since_last_earnings", "calendar", 365, _days_since_last_earnings),
        s("month_of_year", "calendar", 1, _month_of_year),
    ]


FEATURE_CATALOG: dict[str, FeatureSpec] = {spec.name: spec for spec in _specs()}

NORMALIZED_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in FEATURE_CATALOG.items() if spec.group not in RAW_GROUPS
)
RAW_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in FEATURE_CATALOG.items() if spec.group in RAW_GROUPS
)
