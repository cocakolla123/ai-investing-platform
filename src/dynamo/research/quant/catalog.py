"""Feature catalog — doc 04 section 2.2, plus five groups this repo added.

69 features across 15 groups. Groups 1-10 are doc 04's original set;
11-15 came out of the strategy audit that found the catalog was price and
volume only, on a single timeframe:

11. ``weekly``           — multi-timeframe structure (weekly resamples)
12. ``accumulation``     — signed-volume direction and sustained expansion
13. ``momentum_quality`` — the PATH of a move, not just its total
14. ``attention``        — how closely a name is followed (news coverage)
15. ``shortinterest``    — FINRA bi-monthly cycles (``dynamo.data.finra``)

Every feature is a pure function of data available at ``as_of``:

* prices/volume: bar ``ts`` <= as_of
* fundamentals: only ``xbrl_facts`` rows with ``filed_at`` <= as_of (TTM from
  the four most recent *filed* quarters)
* insider: Form 4 ``filed_at`` <= as_of
* macro: FRED vintages with ``vintage_ts`` <= as_of
* estimates/calendar: snapshot rows <= as_of
* news: ``published_at`` <= as_of
* short interest: ``publication_date`` <= as_of — NOT ``settlement_date``,
  which precedes publication by ~8 days (see :mod:`dynamo.data.finra`)

Missing inputs produce NaN — feature functions never raise. Every group
except ``macro`` and ``calendar`` is cross-sectionally normalized downstream
(features.py); those two stay raw because they are context shared across
names rather than per-name readings.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd

from dynamo.domain.symbols import is_crypto_symbol

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

# breadth_above_200dma is a market-state reading, not a per-name feature: it
# must be computed from a FIXED equity reference set (see
# :func:`_equity_reference_set`), never from "whichever names happen to have a
# valid 200-day MA today". Below this many usable reference names the reading
# is not a market breadth number and the feature stays NaN.
BREADTH_MIN_NAMES = 10

# Multi-timeframe (group 11). Weekly bars are resampled PER SYMBOL over that
# symbol's OWN sessions (never over a union calendar) and labelled by the
# week's Friday, so an equity's 5 sessions and a crypto pair's 7 both collapse
# to exactly one bar per calendar week.
WEEK_END_RULE = "W-FRI"
WEEKLY_TIGHT_WEEKS = 8  # the "recent" leg of the range-contraction ratio
WEEKLY_WIDE_WEEKS = 26  # the half-year leg it is measured against
DAILY_BREAKOUT_SESSIONS = 20  # a daily breakout is a 20-session closing high
WEEKLY_BREAKOUT_WEEKS = 26  # ...confirmed only when it is a 26-week high too

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
    "weekly": 7,
    "accumulation": 4,
    "momentum_quality": 3,
    "attention": 1,
    "shortinterest": 3,
}

# Accumulation / distribution (group 12).
OBV_SLOPE_WINDOWS = (20, 60)
SUSTAINED_NEAR_SESSIONS = 20  # "is volume elevated NOW"
SUSTAINED_FAR_SESSIONS = 120  # "...against the name's own prior regime"
UNUSUAL_VOLUME_WINDOW = 60  # sessions scanned for quiet-but-heavy days
UNUSUAL_VOLUME_MULT = 2.0  # a session is unusual above 2x its own baseline
UNUSUAL_VOLUME_BASELINE = 60  # sessions of median volume forming that baseline
NEWS_BLACKOUT_DAYS = 1  # a session counts as "with news" within +/- 1 day
# A symbol is only credited with OBSERVED silence if we have seen at least one
# story about it in this window; below that, absence is missing coverage, not
# quiet — see :func:`_unusual_volume_no_news`.
NEWS_COVERAGE_EVIDENCE_DAYS = 365
NEWS_WINDOW_DAYS = 60  # attention proxy: stories in the trailing 60 days

# Momentum quality (group 13): the window over which a total move's PATH is
# judged. 60 sessions is the same horizon as ret_60d, deliberately — these
# features exist to tell two names with the same ret_60d apart.
MOMENTUM_QUALITY_WINDOW = 60


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
    news: [symbol | symbols, published_at] (either one symbol per row or the
        stored ARRAY column, which is exploded) — attention proxy and the
        "was this volume explained?" input
    short_interest: [symbol, settlement_date, publication_date, shares_short,
        previous_shares_short, avg_daily_volume, days_to_cover] — FINRA
        cycles; read on publication_date, NEVER on settlement_date
    sector_map: symbol -> sector ETF ticker (also the peer-group key)
    """

    bars: pd.DataFrame | None = None
    fundamentals: pd.DataFrame | None = None
    insider: pd.DataFrame | None = None
    estimates: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    regime: pd.Series | None = None
    earnings_calendar: pd.DataFrame | None = None
    news: pd.DataFrame | None = None
    short_interest: pd.DataFrame | None = None
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
    # Weekly resamples, built lazily once per (inputs, as_of) and shared by the
    # whole multi-timeframe group — see :meth:`weekly`.
    weekly_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def weekly(self, kind: str) -> pd.DataFrame:
        """Weekly panel for ``kind`` in ``price|close|high|low|volume``.

        Memoized: six weekly features share one resample pass. The resample
        itself is per-symbol (:func:`_weekly_panel`), so the result is
        invariant to who else is in the universe — the same property
        :func:`_apply_per_symbol` gives the daily features.
        """
        cached = self.weekly_cache.get(kind)
        if cached is None:
            source = {
                "price": self.price,
                "close": self.close,
                "high": self.high,
                "low": self.low,
                "volume": self.volume,
            }[kind]
            how = {
                "price": "last",
                "close": "last",
                "high": "max",
                "low": "min",
                "volume": "sum",
            }[kind]
            cached = _weekly_panel(source, how)
            self.weekly_cache[kind] = cached
        return cached


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


def dt_naive_utc(values: object) -> pd.Series:
    """Datetime series normalized to tz-naive UTC.

    Repository frames carry tz-aware timestamptz columns while fixtures and
    as-of Timestamps are naive; comparing the two raises in pandas. All stored
    times are UTC, so naive values are read as UTC and aware values are
    converted — one comparable dtype everywhere.
    """
    converted = pd.to_datetime(cast(Any, values), utc=True)
    if isinstance(converted, pd.DatetimeIndex):
        return pd.Series(converted.tz_localize(None))
    return cast(pd.Series, converted.dt.tz_localize(None))


def _rows(df: pd.DataFrame, mask: object) -> pd.DataFrame:
    """Boolean-mask row filter with a stable DataFrame static type."""
    return df.loc[cast(pd.Series, mask)]


def _apply_per_symbol(df: pd.DataFrame, fn: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    """Apply ``fn`` to each column's OWN non-NaN observation sequence.

    :func:`_pivot` indexes the panel by the UNION of every symbol's bar
    timestamps. Crypto trades 7 days a week and equities 5, so in a mixed
    universe roughly 2 of every 7 panel rows are empty for an equity — at
    2026-03-03 the panel had 421 calendar rows where equities had 288 real
    ones. Anything positional on that shared index (``rolling``, ``shift``,
    ``diff``, ``iloc[-k]``) therefore measures CALENDAR days, not sessions:
    a 50-row rolling mean requiring 50 observations never filled for a 5-day
    name in a 7-day panel, which is exactly how 16 features became NaN for
    every equity while the 5 crypto pairs kept working.

    Rolling/shifting per symbol makes "N periods" mean "N sessions of THIS
    symbol's own history", which additionally makes every derived feature
    invariant to which other symbols happen to share the universe — the
    property :mod:`tests.unit.quant.test_calendar_invariance` pins down.
    Results are reindexed back onto the panel index, so a symbol has no
    value on a date it did not trade (NaN), never a stale carry-forward.
    """
    if df.empty or len(df.columns) == 0:
        return df.copy()
    out: dict[Hashable, pd.Series] = {}
    for col in df.columns:
        own = cast(pd.Series, df[col]).dropna()
        out[col] = (
            pd.Series(np.nan, index=df.index, dtype=float)
            if own.empty
            else cast(pd.Series, fn(own)).reindex(df.index)
        )
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _roll_per_symbol(
    df: pd.DataFrame, window: int, agg: str, min_periods: int | None = None
) -> pd.DataFrame:
    """``rolling(window).<agg>()`` over each symbol's own session sequence."""
    mp = window if min_periods is None else min_periods

    def rolled(s: pd.Series) -> pd.Series:
        return cast(pd.Series, getattr(s.rolling(window, min_periods=mp), agg)())

    return _apply_per_symbol(df, rolled)


def _shift_per_symbol(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """``shift(periods)`` over each symbol's own session sequence.

    The positional replacement for ``panel.iloc[-1 - k]``: "k sessions ago"
    for THIS symbol, not "k calendar rows ago in a union-calendar panel"
    (where an equity's 20-day return was really 21 calendar days and NaN'd
    outright whenever the offset landed on a weekend)."""
    return _apply_per_symbol(df, lambda s: cast(pd.Series, s.shift(periods)))


def _diff_per_symbol(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """``diff(periods)`` over each symbol's own session sequence."""
    return _apply_per_symbol(df, lambda s: s.diff(periods))


def _weekly_panel(daily: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """Weekly resample of a daily panel, done PER SYMBOL.

    Same discipline as :func:`_apply_per_symbol`, one level up: each column is
    resampled over its OWN non-NaN sessions, so a week is aggregated from the
    5 sessions an equity actually traded and the 7 a crypto pair did — never
    from the union calendar's empty weekend rows (which would make an equity's
    weekly volume sum look identical while silently changing which daily rows
    a ``high``/``low`` aggregate saw).

    ``how`` is the within-week aggregation (``last`` for closes, ``max``/
    ``min`` for highs/lows, ``sum`` for volume). Weeks a symbol did not trade
    are dropped for that symbol (NaN in the union weekly index), so the
    downstream ``_roll_per_symbol``/``_shift_per_symbol`` helpers keep counting
    in that symbol's own weeks.

    The final week is normally partial (``as_of`` is rarely a Friday). That is
    intentional and point-in-time honest: it contains only sessions <= as_of,
    it is identical for every symbol, and it is what "this week so far" means
    to a human reading a weekly chart mid-week.
    """
    if daily.empty or len(daily.columns) == 0:
        return _empty_panel()
    out: dict[Hashable, pd.Series] = {}
    for col in daily.columns:
        own = cast(pd.Series, daily[col]).dropna()
        if own.empty:
            continue
        grouped = own.resample(WEEK_END_RULE)
        # ``sum`` of an empty week is 0.0, not NaN — min_count=1 keeps a week
        # the symbol did not trade out of its own weekly sequence.
        weekly = cast(
            pd.Series,
            grouped.sum(min_count=1) if how == "sum" else getattr(grouped, how)(),
        )
        out[col] = weekly.dropna()
    if not out:
        return _empty_panel()
    frame = pd.DataFrame(out)
    return cast(pd.DataFrame, frame.reindex(columns=daily.columns).sort_index())


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]), dtype=float)


def _pivot(bars: pd.DataFrame, column: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if column not in bars.columns:
        return _empty_panel()
    df = _rows(bars, dt_naive_utc(bars["ts"]) <= as_of).loc[:, ["symbol", "ts", column]].copy()
    if df.empty:
        return _empty_panel()
    df["ts"] = dt_naive_utc(df["ts"])
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
    """Daily log returns on each symbol's OWN session sequence.

    A calendar-panel ``diff()`` differences Monday against Sunday for an
    equity (an empty row), so ~1 in 5 equity returns came back NaN before
    any rolling window even ran."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = cast(pd.DataFrame, np.log(price))
    return _diff_per_symbol(logged)


# ---------------------------------------------------------------------------
# Group 1 — momentum / trend
# ---------------------------------------------------------------------------


def _finite(s: pd.Series) -> pd.Series:
    return cast(pd.Series, s.replace([np.inf, -np.inf], np.nan))


def _ret(days: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) <= days:
            return _nan_series(ctx)
        past = _shift_per_symbol(p, days).iloc[-1]
        return _finite(p.iloc[-1] / past - 1.0).reindex(ctx.universe)

    return fn


def _mom_12_1(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) <= TRADING_DAYS_PER_YEAR:
        return _nan_series(ctx)
    # 12-1: price 21 sessions ago over price 252 sessions ago — both counted
    # in the symbol's OWN sessions.
    recent = _shift_per_symbol(p, 21).iloc[-1]
    old = _shift_per_symbol(p, TRADING_DAYS_PER_YEAR).iloc[-1]
    return _finite(recent / old - 1.0).reindex(ctx.universe)


def _dist_ma(window: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) < window:
            return _nan_series(ctx)
        ma = _roll_per_symbol(p, window, "mean")
        return _finite(p.iloc[-1] / ma.iloc[-1] - 1.0).reindex(ctx.universe)

    return fn


def _ma50_slope(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 70:
        return _nan_series(ctx)
    ma = _roll_per_symbol(p, 50, "mean")
    prev = _shift_per_symbol(ma, 21).iloc[-1]
    return _finite((ma.iloc[-1] - prev) / prev).reindex(ctx.universe)


def _rs_20d_sector(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) <= 20:
        return _nan_series(ctx)
    ret20 = _finite(p.iloc[-1] / _shift_per_symbol(p, 20).iloc[-1] - 1.0)
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
        r = p / _shift_per_symbol(p, days) - 1.0
        mu = _roll_per_symbol(r, window, "mean", 30)
        sd = _roll_per_symbol(r, window, "std", 30)
        z = (r.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1]
        return _finite(z).reindex(ctx.universe)

    return fn


def _rsi_14(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 15:
        return _nan_series(ctx)
    diff = _diff_per_symbol(p)
    gain = _roll_per_symbol(diff.clip(lower=0.0), 14, "mean").iloc[-1]
    loss = _roll_per_symbol((-diff).clip(lower=0.0), 14, "mean").iloc[-1]
    denom = gain + loss
    rsi = 100.0 * gain / denom.where(denom > 0)
    # 50.0 ONLY where the window is genuinely flat (denominator exactly 0),
    # never where the denominator is NaN. The old `where(denom > 0, 50.0)`
    # also fired on NaN, so all 50 equities reported a fabricated RSI of
    # exactly 50 that then counted as "covered" in feature_coverage.
    return cast(pd.Series, rsi.mask(denom == 0.0, 50.0)).reindex(ctx.universe)


def _dist_20d_mean_sigma(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 20:
        return _nan_series(ctx)
    mu = _roll_per_symbol(p, 20, "mean").iloc[-1]
    sd = _roll_per_symbol(p, 20, "std").iloc[-1]
    z = (p.iloc[-1] - mu) / sd
    return _finite(z).reindex(ctx.universe)


def _drawdown_20d_high(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 20:
        return _nan_series(ctx)
    hi = _roll_per_symbol(p, 20, "max").iloc[-1]
    return _finite(p.iloc[-1] / hi - 1.0).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 3 — volatility
# ---------------------------------------------------------------------------


def _vol_cc(window: int) -> FeatureFn:
    def fn(ctx: FeatureContext) -> pd.Series:
        p = ctx.price
        if len(p) <= window:
            return _nan_series(ctx)
        lr = _log_returns(p)
        vol = _roll_per_symbol(lr, window, "std").iloc[-1] * math.sqrt(TRADING_DAYS_PER_YEAR)
        return vol.reindex(ctx.universe)

    return fn


def _vol_parkinson_20d(ctx: FeatureContext) -> pd.Series:
    h, lo = ctx.high, ctx.low
    if h.empty or lo.empty or len(h) < 20:
        return _nan_series(ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = cast(pd.DataFrame, np.log(h / lo)) ** 2 / (4.0 * math.log(2.0))
    var = _roll_per_symbol(hl, 20, "mean").iloc[-1]
    return (var**0.5 * math.sqrt(TRADING_DAYS_PER_YEAR)).reindex(ctx.universe)


def _vol_ewma_60d(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 30:
        return _nan_series(ctx)
    lr = _log_returns(p)
    # span/min_periods count the SYMBOL'S own return observations (an EWMA
    # over calendar rows decays ~7/5 too fast for a 5-day-a-week name).
    ewm_std = _apply_per_symbol(lr, lambda s: cast(pd.Series, s.ewm(span=60, min_periods=20).std()))
    vol = ewm_std.iloc[-1] * math.sqrt(TRADING_DAYS_PER_YEAR)
    return vol.reindex(ctx.universe)


def _vol20_series(p: pd.DataFrame) -> pd.DataFrame:
    lr = _log_returns(p)
    return _roll_per_symbol(lr, 20, "std") * math.sqrt(TRADING_DAYS_PER_YEAR)


def _vol_of_vol_60d(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 80:
        return _nan_series(ctx)
    v = _vol20_series(p)
    return _roll_per_symbol(v, 60, "std", 40).iloc[-1].reindex(ctx.universe)


def _vol_pctile_2y(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 120:
        return _nan_series(ctx)
    v = _vol20_series(p)
    out = {}
    for sym in ctx.universe:
        if sym not in v.columns:
            out[sym] = np.nan
            continue
        # The 2-year window is 2 x 252 of the SYMBOL'S OWN sessions, not the
        # last 504 calendar rows of a union-calendar panel.
        s = cast(pd.Series, v[sym]).dropna().iloc[-2 * TRADING_DAYS_PER_YEAR :]
        if len(s) < 60:
            out[sym] = np.nan
        else:
            out[sym] = float((s <= s.iloc[-1]).mean())
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _true_range(ctx: FeatureContext) -> pd.DataFrame:
    h, lo, c = ctx.high, ctx.low, ctx.close
    if h.empty or lo.empty or c.empty:
        return _empty_panel()
    prev_c = _shift_per_symbol(c, 1)
    stacked = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], keys=["hl", "hc", "lc"])
    return cast(pd.DataFrame, stacked.groupby(level=1).max())


def _atr14_over_price(ctx: FeatureContext) -> pd.Series:
    tr = _true_range(ctx)
    if tr.empty or len(tr) < 15:
        return _nan_series(ctx)
    atr = _roll_per_symbol(tr, 14, "mean").iloc[-1]
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
    return _roll_per_symbol(dv, 20, "mean").iloc[-1].reindex(ctx.universe)


def _dollar_vol_z_60d(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    if dv.empty or len(dv) < 60:
        return _nan_series(ctx)
    mu = _roll_per_symbol(dv, 60, "mean").iloc[-1]
    sd = _roll_per_symbol(dv, 60, "std").iloc[-1]
    z = (dv.iloc[-1] - mu) / sd
    return _finite(z).reindex(ctx.universe)


def _volume_trend_20_60(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    if dv.empty or len(dv) < 60:
        return _nan_series(ctx)
    adv20 = _roll_per_symbol(dv, 20, "mean").iloc[-1]
    adv60 = _roll_per_symbol(dv, 60, "mean").iloc[-1]
    return _finite(adv20 / adv60 - 1.0).reindex(ctx.universe)


def _amihud_20d(ctx: FeatureContext) -> pd.Series:
    dv = _dollar_panel(ctx)
    p = ctx.price
    if dv.empty or len(p) < 21:
        return _nan_series(ctx)
    ret = (p / _shift_per_symbol(p, 1) - 1.0).abs()
    illiq = (ret / dv.where(dv > 0)) * 1e6
    return _roll_per_symbol(illiq, 20, "mean").iloc[-1].reindex(ctx.universe)


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
    df["filed_at"] = dt_naive_utc(df["filed_at"])
    df["period_end"] = dt_naive_utc(df["period_end"])
    return _rows(df, df["filed_at"] <= as_of)


def quarterly_series(facts: pd.DataFrame, symbol: str, tag: str, as_of: pd.Timestamp) -> pd.Series:
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
        if est is None or est.empty or not {"symbol", "fetched_at", "eps_est"} <= set(est.columns):
            return _nan_series(ctx)
        e = est.copy()
        e["fetched_at"] = dt_naive_utc(e["fetched_at"])
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
    df["filed_at"] = dt_naive_utc(df["filed_at"])
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
        sym: insider_net_buy_signed(ctx.inputs.insider, sym, ctx.as_of, 90) for sym in ctx.universe
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
    df["ts"] = dt_naive_utc(df["ts"])
    df["vintage_ts"] = dt_naive_utc(df["vintage_ts"])
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
    s.index = pd.DatetimeIndex(dt_naive_utc(s.index))
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


def _equity_reference_set(ctx: FeatureContext) -> list[str]:
    """The universe's operating companies — breadth's fixed denominator.

    Market breadth means "what share of THE MARKET is above its 200-day
    average". Measuring it over "whichever names have a valid 200-day MA
    today" made it a reading of the 5 crypto pairs (the only names whose MA
    survived the calendar bug) broadcast to all 55 rows; it read 0.0 on
    every sampled day."""
    return [
        sym
        for sym in ctx.universe
        if asset_class_of(sym, ctx.inputs.sector_map) == ASSET_CLASS_EQUITY
    ]


def _breadth_above_200dma(ctx: FeatureContext) -> pd.Series:
    p = ctx.price
    if len(p) < 200:
        return _nan_series(ctx)
    reference = [sym for sym in _equity_reference_set(ctx) if sym in p.columns]
    if len(reference) < BREADTH_MIN_NAMES:
        return _nan_series(ctx)
    sub = p.loc[:, reference]
    ma = _roll_per_symbol(sub, 200, "mean").iloc[-1]
    last = sub.iloc[-1]
    usable = ma.notna() & last.notna()
    if int(usable.sum()) < BREADTH_MIN_NAMES:
        return _nan_series(ctx)
    flags = (last > ma)[usable]
    return _broadcast(ctx, float(flags.mean()))


# ---------------------------------------------------------------------------
# Group 10 — calendar (raw)
# ---------------------------------------------------------------------------


def _calendar_frame(ctx: FeatureContext) -> pd.DataFrame:
    cal = ctx.inputs.earnings_calendar
    if cal is None or cal.empty or not {"symbol", "event_date"} <= set(cal.columns):
        return pd.DataFrame(columns=pd.Index(["symbol", "event_date"]))
    df = cal.copy()
    df["event_date"] = dt_naive_utc(df["event_date"])
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
# Group 11 — multi-timeframe (weekly structure)
# ---------------------------------------------------------------------------
#
# Every other group in this catalog reads daily bars. Weekly structure is a
# genuinely different lens rather than a smoothed copy of the daily one: a name
# can be choppy day to day and still be carving a clean weekly uptrend, and a
# multi-week range contraction ("quiet accumulation") is invisible to a 20-day
# window. The confirmation feature — a daily breakout that is ALSO a weekly
# breakout — is standard multi-timeframe practice and was simply not
# expressible before, because no weekly series existed anywhere in the repo.
#
# All six read :meth:`FeatureContext.weekly`, whose panels are resampled per
# symbol (:func:`_weekly_panel`) and then rolled/shifted with the same
# per-symbol helpers the daily features use, so they inherit the
# calendar-invariance property that ``tests/unit/quant/test_calendar_invariance``
# pins down.


def _weekly_ret(weeks: int) -> FeatureFn:
    """Total return over ``weeks`` of the symbol's OWN weekly bars."""

    def fn(ctx: FeatureContext) -> pd.Series:
        w = ctx.weekly("price")
        if w.empty or len(w) <= weeks:
            return _nan_series(ctx)
        past = _shift_per_symbol(w, weeks).iloc[-1]
        return _finite(w.iloc[-1] / past - 1.0).reindex(ctx.universe)

    return fn


def _weekly_dist_ma(window: int) -> FeatureFn:
    """Distance of the latest weekly close from its ``window``-week average."""

    def fn(ctx: FeatureContext) -> pd.Series:
        w = ctx.weekly("price")
        if w.empty or len(w) < window:
            return _nan_series(ctx)
        ma = _roll_per_symbol(w, window, "mean")
        return _finite(w.iloc[-1] / ma.iloc[-1] - 1.0).reindex(ctx.universe)

    return fn


def _weekly_range_contraction(ctx: FeatureContext) -> pd.Series:
    """How much tighter the last 8 weeks are than the last 26.

    ``1 - (8-week high/low span) / (26-week high/low span)``: 0 when the recent
    range is as wide as the half-year range, approaching 1 when the name has
    gone quiet inside it. Higher = more contracted — the multi-week
    consolidation shape a daily 20-day window cannot see.
    """
    hi, lo = ctx.weekly("high"), ctx.weekly("low")
    if hi.empty or lo.empty or len(hi) < WEEKLY_WIDE_WEEKS:
        return _nan_series(ctx)
    tight_hi = _roll_per_symbol(hi, WEEKLY_TIGHT_WEEKS, "max").iloc[-1]
    tight_lo = _roll_per_symbol(lo, WEEKLY_TIGHT_WEEKS, "min").iloc[-1]
    wide_hi = _roll_per_symbol(hi, WEEKLY_WIDE_WEEKS, "max").iloc[-1]
    wide_lo = _roll_per_symbol(lo, WEEKLY_WIDE_WEEKS, "min").iloc[-1]
    tight = tight_hi / tight_lo.where(tight_lo > 0) - 1.0
    wide = wide_hi / wide_lo.where(wide_lo > 0) - 1.0
    return _finite(1.0 - tight / wide.where(wide > 0)).reindex(ctx.universe)


def _weekly_confirms_daily_breakout(ctx: FeatureContext) -> pd.Series:
    """1.0 when a 20-session daily breakout is ALSO a 26-week weekly breakout.

    Both legs must agree: a 20-day high inside a six-month downtrend scores
    0.0, exactly like a name that broke out on neither timeframe. NaN (not 0.0)
    whenever either timeframe lacks the history to answer — an unknown
    confirmation is never reported as a failed one.
    """
    p, w = ctx.price, ctx.weekly("price")
    if p.empty or w.empty:
        return _nan_series(ctx)
    if len(p) < DAILY_BREAKOUT_SESSIONS or len(w) < WEEKLY_BREAKOUT_WEEKS:
        return _nan_series(ctx)
    daily_hi = _roll_per_symbol(p, DAILY_BREAKOUT_SESSIONS, "max").iloc[-1]
    weekly_hi = _roll_per_symbol(w, WEEKLY_BREAKOUT_WEEKS, "max").iloc[-1]
    last_daily, last_weekly = p.iloc[-1], w.iloc[-1]
    known = daily_hi.notna() & weekly_hi.notna() & last_daily.notna() & last_weekly.notna()
    both = (last_daily >= daily_hi) & (last_weekly >= weekly_hi)
    flags = cast(pd.Series, both.astype(float).where(known))
    return flags.reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 12 — accumulation / distribution
# ---------------------------------------------------------------------------
#
# The existing volume group measures SIZE (dollar volume, Amihud) and one-day
# surprise (dollar_vol_z_60d). The read-only detector pilot measured the
# one-day-surprise shape at -3.03% forward 20-session excess return, i.e.
# contra-indicative and most likely IEX routing noise. What it never measured
# is DIRECTION accumulated over time, or elevation SUSTAINED over weeks —
# which is what "someone is quietly building a position" actually looks like.


def _obv_panel(ctx: FeatureContext) -> pd.DataFrame:
    """Cumulative signed volume (on-balance volume) per symbol.

    ``+volume`` on an up session, ``-volume`` on a down session, running sum
    over the symbol's OWN sessions. The first session of each symbol has no
    prior close, so it contributes NaN rather than an arbitrary sign.
    """
    p, v = ctx.price, ctx.volume
    if p.empty or v.empty:
        return _empty_panel()
    direction = cast(pd.DataFrame, np.sign(_diff_per_symbol(p)))
    signed = direction * v
    return _apply_per_symbol(signed, lambda s: cast(pd.Series, s.cumsum()))


def _obv_slope(window: int) -> FeatureFn:
    """OBV change over ``window`` sessions as a fraction of volume traded.

    Dividing by ``window x average volume`` puts every name on the same
    scale: +1.0 would mean every share over the window changed hands on an up
    session, 0.0 means buying and selling volume cancelled exactly. Raw OBV
    levels are meaningless cross-sectionally (they scale with float and with
    how long the symbol has been in the panel); this slope is not.
    """

    def fn(ctx: FeatureContext) -> pd.Series:
        obv = _obv_panel(ctx)
        if obv.empty or len(obv) <= window:
            return _nan_series(ctx)
        change = obv.iloc[-1] - _shift_per_symbol(obv, window).iloc[-1]
        traded = _roll_per_symbol(ctx.volume, window, "mean").iloc[-1] * window
        denom = traded.where(traded > 0)
        return _finite(change / denom).reindex(ctx.universe)

    return fn


def _sustained_volume_expansion(ctx: FeatureContext) -> pd.Series:
    """Median volume of the last 20 sessions vs the 120 sessions BEFORE them.

    Deliberately median-of-many, not max-of-one: a single 10x print barely
    moves a 20-session median, while twenty sessions at 1.5x moves it 50%.
    That is the whole distinction from ``dollar_vol_z_60d`` — persistent
    elevation rather than a one-day pop, which the detector pilot scored at
    -3.03% and which looks like routing noise. The baseline window is shifted
    clear of the recent one so a name is compared against its own prior
    regime, not against a window it is itself inflating.
    """
    v = ctx.volume
    if v.empty or len(v) < SUSTAINED_NEAR_SESSIONS + SUSTAINED_FAR_SESSIONS:
        return _nan_series(ctx)
    near = _roll_per_symbol(v, SUSTAINED_NEAR_SESSIONS, "median").iloc[-1]
    prior = _shift_per_symbol(v, SUSTAINED_NEAR_SESSIONS)
    far = _roll_per_symbol(prior, SUSTAINED_FAR_SESSIONS, "median").iloc[-1]
    denom = far.where(far > 0)
    return _finite(near / denom - 1.0).reindex(ctx.universe)


def _news_frame(ctx: FeatureContext) -> pd.DataFrame:
    """``[symbol, published_at]`` with ``published_at <= as_of``.

    Accepts either one symbol per row or the stored ``symbols`` ARRAY column
    (exploded). Empty frame when there is no news input at all.
    """
    news = ctx.inputs.news
    cols = ["symbol", "published_at"]
    empty = pd.DataFrame(columns=pd.Index(cols))
    if news is None or news.empty or "published_at" not in news.columns:
        return empty
    df = news.copy()
    if "symbol" not in df.columns:
        if "symbols" not in df.columns:
            return empty
        df = df.explode("symbols").rename(columns={"symbols": "symbol"})
    df = df.loc[:, cols].dropna(subset=["symbol"])
    if df.empty:
        return empty
    df["symbol"] = df["symbol"].astype(str)
    df["published_at"] = dt_naive_utc(df["published_at"])
    return _rows(df, df["published_at"] <= ctx.as_of)


def _unusual_volume_no_news(ctx: FeatureContext) -> pd.Series:
    """Fraction of the last 60 sessions that were heavy AND unexplained.

    A session is heavy above ``2x`` the symbol's own trailing 60-session
    median volume, and unexplained when no story about the symbol was
    published within a day either side. That combination — real volume, no
    visible reason — is the "someone knows something" shape.

    Graceful degradation is the point of the guard, not an aside. News
    coverage is sparse for tier-2 names, so a symbol we have simply never
    collected a story about would otherwise score a perfect 1.0 on every
    heavy day and flood the screen with false positives. The feature is
    therefore NaN unless we have seen at least one story about that symbol in
    the trailing year: silence only counts when it is OBSERVED silence.
    """
    v = ctx.volume
    news = _news_frame(ctx)
    if v.empty or news.empty or len(v) < UNUSUAL_VOLUME_BASELINE:
        return _nan_series(ctx)
    baseline = _shift_per_symbol(_roll_per_symbol(v, UNUSUAL_VOLUME_BASELINE, "median"), 1)
    heavy = v > UNUSUAL_VOLUME_MULT * baseline.where(baseline > 0)
    evidence_lo = ctx.as_of - pd.Timedelta(days=NEWS_COVERAGE_EVIDENCE_DAYS)
    covered = set(_rows(news, news["published_at"] >= evidence_lo)["symbol"].unique())
    blackout = np.timedelta64(NEWS_BLACKOUT_DAYS, "D")
    out: dict[str, float] = {}
    for sym in ctx.universe:
        if sym not in v.columns or sym not in covered:
            out[sym] = float("nan")
            continue
        own = cast(pd.Series, heavy[sym]).where(cast(pd.Series, v[sym]).notna()).dropna()
        window = own.iloc[-UNUSUAL_VOLUME_WINDOW:]
        if len(window) < UNUSUAL_VOLUME_WINDOW:
            out[sym] = float("nan")
            continue
        published = cast(pd.Series, _rows(news, news["symbol"] == sym)["published_at"]).to_numpy()
        heavy_days = cast(pd.DatetimeIndex, window.index[window.astype(bool)])
        quiet = 0
        for day in heavy_days:
            near = np.abs(published - np.datetime64(day)) <= blackout
            if not bool(near.any()):
                quiet += 1
        out[sym] = quiet / float(UNUSUAL_VOLUME_WINDOW)
    return pd.Series(out, dtype=float).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 13 — momentum quality (the PATH, not just the total)
# ---------------------------------------------------------------------------
#
# ret_60d cannot tell a name that ground out +30% over three months from one
# that gapped +30% on a single headline and went sideways. They are different
# animals with different forward distributions, and today they score
# identically. These three read the path.


def _quality_paths(ctx: FeatureContext) -> dict[str, pd.Series]:
    """Each symbol's last ``MOMENTUM_QUALITY_WINDOW`` daily log returns.

    Taken from the symbol's OWN session sequence, so "60 sessions" means the
    same thing for a 5-day equity and a 7-day crypto pair.
    """
    p = ctx.price
    out: dict[str, pd.Series] = {}
    if p.empty:
        return out
    for sym in ctx.universe:
        if sym not in p.columns:
            continue
        own = cast(pd.Series, p[sym]).dropna()
        if len(own) < MOMENTUM_QUALITY_WINDOW + 1:
            continue
        path = own.iloc[-(MOMENTUM_QUALITY_WINDOW + 1) :]
        if bool((path <= 0).any()):
            continue
        lr = cast(pd.Series, np.log(path)).diff().dropna()
        if len(lr) == MOMENTUM_QUALITY_WINDOW:
            out[sym] = lr
    return out


def _up_session_fraction_60d(ctx: FeatureContext) -> pd.Series:
    """Share of the last 60 sessions that closed up — smoothness, plainly."""
    paths = _quality_paths(ctx)
    return pd.Series(
        {sym: float((lr > 0).mean()) for sym, lr in paths.items()}, dtype=float
    ).reindex(ctx.universe)


def _max_session_share_60d(ctx: FeatureContext) -> pd.Series:
    """Largest single session's share of the window's total absolute move.

    ``max(log return) / sum(|log return|)``. Near 1.0 means one gap did
    essentially all of it (a lottery ticket); near ``1/60`` means the move
    was earned across the whole window (a climber). Defined for fallers too —
    a negative reading is a name whose best day was still a loss.
    """
    paths = _quality_paths(ctx)
    out: dict[str, float] = {}
    for sym, lr in paths.items():
        total = float(lr.abs().sum())
        out[sym] = float(lr.max()) / total if total > 0 else float("nan")
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _drawdown_adj_momentum_60d(ctx: FeatureContext) -> pd.Series:
    """60-session return divided by ``1 + `` its deepest drawdown en route.

    Two names arriving at the same +30% score very differently when one did
    it without ever giving back 25%. This is the path-risk-adjusted version
    of ``ret_60d``, not a replacement for it.
    """
    paths = _quality_paths(ctx)
    out: dict[str, float] = {}
    for sym, lr in paths.items():
        equity = cast(pd.Series, np.exp(lr.cumsum()))
        drawdown = float((1.0 - equity / equity.cummax()).max())
        out[sym] = float(equity.iloc[-1] - 1.0) / (1.0 + drawdown)
    return pd.Series(out, dtype=float).reindex(ctx.universe)


# ---------------------------------------------------------------------------
# Group 14 — attention
# ---------------------------------------------------------------------------


def _news_count_60d(ctx: FeatureContext) -> pd.Series:
    """Stories mentioning the symbol in the trailing 60 days.

    The low-attention proxy the sleeper screen is built on: an under-followed
    name is one nobody is writing about. Unlike
    :func:`_unusual_volume_no_news` a zero here IS reported as 0.0 (once any
    news at all has been collected) — "nobody wrote about it" is precisely
    the reading we want, and the screen treats it as a positive, not as a
    trigger. The whole feature is NaN only when no news was collected at all.
    """
    news = _news_frame(ctx)
    if news.empty:
        return _nan_series(ctx)
    lo = ctx.as_of - pd.Timedelta(days=NEWS_WINDOW_DAYS)
    recent = _rows(news, news["published_at"] >= lo)
    counts = cast(pd.Series, recent.groupby("symbol").size().astype(float))
    return cast(pd.Series, counts.reindex(ctx.universe).fillna(0.0))


# ---------------------------------------------------------------------------
# Group 15 — short interest (FINRA, bi-monthly)
# ---------------------------------------------------------------------------
#
# New data, not a new transform: no short-interest series existed anywhere in
# the platform before ``dynamo.data.finra``. Crowding and the mechanics of
# covering are a genuinely different axis from price and volume — and rising
# short interest INTO a rising price is a specific, well-known setup that no
# existing feature can express.
#
# THE POINT-IN-TIME RULE, RESTATED WHERE IT IS ENFORCED. A cycle describes
# ``settlement_date`` but is not disseminated until ``publication_date``, some
# eight days later. Everything below filters on the PUBLICATION date. Filtering
# on the settlement date would hand a backtest a week and a half of the future
# on every cycle, twice a month, for free — the single easiest way to
# manufacture a beautiful and completely fake short-interest signal.


def _short_interest_latest(ctx: FeatureContext) -> pd.DataFrame:
    """The most recent PUBLISHED cycle per symbol, indexed by symbol.

    One WHOLE ROW per symbol, not a column-wise last-non-null. ``groupby(
    "symbol").last()`` would look right and be wrong: it skips NaN per column,
    so a cycle that reports no ``days_to_cover`` would silently inherit the
    previous cycle's figure while ``shares_short`` came from the current one —
    a Frankenstein row stitched from two different fortnights.
    """
    si = ctx.inputs.short_interest
    required = {"symbol", "settlement_date", "publication_date", "shares_short"}
    empty = pd.DataFrame(columns=pd.Index(sorted(required)))
    if si is None or si.empty or not required <= set(si.columns):
        return empty
    df = si.copy()
    df["settlement_date"] = dt_naive_utc(df["settlement_date"])
    df["publication_date"] = dt_naive_utc(df["publication_date"])
    visible = _rows(df, df["publication_date"] <= ctx.as_of)
    if visible.empty:
        return empty
    ordered = visible.sort_values(["symbol", "settlement_date", "publication_date"])
    latest = ordered.drop_duplicates(subset="symbol", keep="last")
    return cast(pd.DataFrame, latest.set_index("symbol"))


def _short_interest_column(latest: pd.DataFrame, name: str) -> pd.Series:
    if name not in latest.columns:
        return pd.Series(dtype=float)
    return cast(pd.Series, latest[name]).astype(float)


def _short_interest_pct_float(ctx: FeatureContext) -> pd.Series:
    """Shares short as a fraction of shares outstanding.

    Shares outstanding come from the same point-in-time XBRL path every other
    fundamental feature uses (``filed_at <= as_of``), so both legs of the
    ratio are honest. True free float would be better than shares outstanding
    — insider and strategic holdings are not really shortable — but no free
    float source is wired up, and the difference is a roughly constant
    per-name scale factor that cross-sectional normalization largely absorbs.
    """
    latest = _short_interest_latest(ctx)
    if latest.empty:
        return _nan_series(ctx)
    shorts = _short_interest_column(latest, "shares_short")
    facts = pit_facts(ctx.inputs.fundamentals, ctx.as_of)
    out: dict[str, float] = {}
    for sym in ctx.universe:
        if sym not in shorts.index:
            out[sym] = float("nan")
            continue
        shares = shares_outstanding(facts, sym, ctx.as_of)
        short = float(shorts[sym])
        out[sym] = short / shares if shares > 0 and math.isfinite(short) else float("nan")
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _days_to_cover(ctx: FeatureContext) -> pd.Series:
    """Sessions of average volume needed to buy the short position back.

    FINRA computes and publishes this; it is recomputed from the raw
    quantities only when the published field is missing or the file reports a
    zero average daily volume (FINRA then emits a 999.99 sentinel, which is
    not a number of days and is dropped rather than trusted).
    """
    latest = _short_interest_latest(ctx)
    if latest.empty:
        return _nan_series(ctx)
    published = _short_interest_column(latest, "days_to_cover")
    shorts = _short_interest_column(latest, "shares_short")
    adv = _short_interest_column(latest, "avg_daily_volume")
    out: dict[str, float] = {}
    for sym in ctx.universe:
        if sym not in shorts.index:
            out[sym] = float("nan")
            continue
        volume = float(adv[sym]) if sym in adv.index else float("nan")
        if not (math.isfinite(volume) and volume > 0):
            out[sym] = float("nan")  # 999.99 sentinel territory
            continue
        value = float(published[sym]) if sym in published.index else float("nan")
        out[sym] = value if math.isfinite(value) else float(shorts[sym]) / volume
    return pd.Series(out, dtype=float).reindex(ctx.universe)


def _short_interest_change(ctx: FeatureContext) -> pd.Series:
    """Change in the short position versus the previous cycle.

    Read off the row itself (FINRA publishes the prior cycle's unadjusted
    position alongside the current one), so both numbers were disseminated at
    the same instant and the feature needs no second point-in-time join.
    """
    latest = _short_interest_latest(ctx)
    if latest.empty:
        return _nan_series(ctx)
    current = _short_interest_column(latest, "shares_short")
    previous = _short_interest_column(latest, "previous_shares_short")
    if previous.empty:
        return _nan_series(ctx)
    prior = previous.where(previous > 0)
    return _finite(current / prior - 1.0).reindex(ctx.universe)


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
        # 11 — multi-timeframe / weekly (7)
        s("weekly_ret_4w", "weekly", 35, _weekly_ret(4)),
        s("weekly_ret_13w", "weekly", 100, _weekly_ret(13)),
        s("weekly_ret_26w", "weekly", 190, _weekly_ret(26)),
        s("weekly_dist_ma10w", "weekly", 80, _weekly_dist_ma(10)),
        s("weekly_dist_ma30w", "weekly", 220, _weekly_dist_ma(30)),
        s(
            "weekly_range_contraction_8w",
            "weekly",
            7 * WEEKLY_WIDE_WEEKS,
            _weekly_range_contraction,
        ),
        s(
            "weekly_confirms_daily_breakout",
            "weekly",
            7 * WEEKLY_BREAKOUT_WEEKS,
            _weekly_confirms_daily_breakout,
        ),
        # 12 — accumulation / distribution (4)
        s("obv_slope_20d", "accumulation", 21, _obv_slope(20)),
        s("obv_slope_60d", "accumulation", 61, _obv_slope(60)),
        s(
            "sustained_volume_expansion",
            "accumulation",
            SUSTAINED_NEAR_SESSIONS + SUSTAINED_FAR_SESSIONS,
            _sustained_volume_expansion,
        ),
        s(
            "unusual_volume_no_news",
            "accumulation",
            NEWS_COVERAGE_EVIDENCE_DAYS,
            _unusual_volume_no_news,
        ),
        # 13 — momentum quality (3)
        s(
            "up_session_fraction_60d",
            "momentum_quality",
            MOMENTUM_QUALITY_WINDOW + 1,
            _up_session_fraction_60d,
        ),
        s(
            "max_session_share_60d",
            "momentum_quality",
            MOMENTUM_QUALITY_WINDOW + 1,
            _max_session_share_60d,
        ),
        s(
            "drawdown_adj_momentum_60d",
            "momentum_quality",
            MOMENTUM_QUALITY_WINDOW + 1,
            _drawdown_adj_momentum_60d,
        ),
        # 14 — attention (1)
        s("news_count_60d", "attention", NEWS_WINDOW_DAYS, _news_count_60d),
        # 15 — short interest (3) — bi-monthly cycles, read on publication date
        s("short_interest_pct_float", "shortinterest", 60, _short_interest_pct_float),
        s("days_to_cover", "shortinterest", 60, _days_to_cover),
        s("short_interest_change", "shortinterest", 60, _short_interest_change),
    ]


FEATURE_CATALOG: dict[str, FeatureSpec] = {spec.name: spec for spec in _specs()}

NORMALIZED_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in FEATURE_CATALOG.items() if spec.group not in RAW_GROUPS
)
RAW_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in FEATURE_CATALOG.items() if spec.group in RAW_GROUPS
)


# ---------------------------------------------------------------------------
# Asset-class coverage buckets
# ---------------------------------------------------------------------------
#
# ``feature_coverage`` used to be "fraction of all ~51 columns that are
# non-NaN", which quietly measures the wrong thing: an index fund files no
# XBRL, so its 13 value/quality columns can never be anything but NaN, and it
# scored 0.219 coverage for being an ETF. Coverage is only meaningful against
# an asset-class-appropriate denominator.
#
# The three buckets and what each is expected to produce:
#
# * ``equity``  — an operating company: ALL ten feature groups apply.
# * ``etf``     — an index/sector fund: no XBRL filings (no ``value``, no
#                 ``quality``), no Form 4s (``insider``), no sell-side EPS
#                 estimates (``estimates``), no earnings date, and no sector
#                 ETF of its own to be relatively strong against.
# * ``crypto``  — a crypto pair: same exclusions as ``etf``; it is likewise
#                 not an issuer and has no earnings calendar or sector peer.
#
# Everything else — momentum, mean-reversion, volatility, volume, macro, and
# the month-of-year calendar column — applies to all three classes and is
# counted for all three.

ASSET_CLASS_EQUITY = "equity"
ASSET_CLASS_ETF = "etf"
ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASSES: tuple[str, ...] = (ASSET_CLASS_EQUITY, ASSET_CLASS_ETF, ASSET_CLASS_CRYPTO)

_NON_ISSUER_GROUPS = frozenset({"value", "quality", "estimates", "insider"})
_NON_ISSUER_FEATURES = frozenset(
    {"days_to_next_earnings", "days_since_last_earnings", "rs_20d_sector"}
)
# The news feed is an equity/ETF headline feed; a crypto pair is not covered by
# it, so the two attention-dependent columns are structurally NaN for crypto
# rather than a data gap it should be charged for.
_NO_NEWS_FEED_FEATURES = frozenset({"news_count_60d", "unusual_volume_no_news"})
# An index fund files no XBRL, so it has no share count to express short
# interest as a percentage of — but it IS in FINRA's file, so days-to-cover
# and the change in short interest apply to it normally.
_NO_SHARE_COUNT_FEATURES = frozenset({"short_interest_pct_float"})

INAPPLICABLE_GROUPS: Mapping[str, frozenset[str]] = {
    ASSET_CLASS_EQUITY: frozenset(),
    ASSET_CLASS_ETF: _NON_ISSUER_GROUPS,
    # A crypto pair is not an exchange-listed security: it appears in no FINRA
    # short-interest file at all, so the whole group is structurally absent.
    ASSET_CLASS_CRYPTO: _NON_ISSUER_GROUPS | frozenset({"shortinterest"}),
}
INAPPLICABLE_FEATURES: Mapping[str, frozenset[str]] = {
    ASSET_CLASS_EQUITY: frozenset(),
    ASSET_CLASS_ETF: _NON_ISSUER_FEATURES | _NO_SHARE_COUNT_FEATURES,
    ASSET_CLASS_CRYPTO: _NON_ISSUER_FEATURES | _NO_NEWS_FEED_FEATURES,
}


def asset_class_of(
    symbol: str,
    sector_map: Mapping[str, str],
    etf_symbols: Collection[str] = (),
) -> str:
    """``equity`` | ``etf`` | ``crypto`` for one symbol.

    ``sector_map`` is the symbol -> sector-ETF map the feature path already
    carries (:class:`FeatureInputs`): its KEYS are the operating companies
    with a GICS sector, its VALUES are the sector SPDRs. A non-crypto symbol
    that is a sector-ETF value, or that has no sector classification at all
    (SPY, QQQ, DIA, IWM, TLT), is treated as a fund. ``etf_symbols`` lets a
    caller with better information (``instruments.is_etf``) override the
    inference; symbols listed there are always ``etf``.
    """
    if is_crypto_symbol(symbol):
        return ASSET_CLASS_CRYPTO
    if symbol in etf_symbols:
        return ASSET_CLASS_ETF
    if symbol in sector_map:
        return ASSET_CLASS_EQUITY
    return ASSET_CLASS_ETF if sector_map else ASSET_CLASS_EQUITY


def asset_classes_for(
    symbols: Iterable[str],
    sector_map: Mapping[str, str],
    etf_symbols: Collection[str] = (),
) -> dict[str, str]:
    """:func:`asset_class_of` for many symbols."""
    return {sym: asset_class_of(sym, sector_map, etf_symbols) for sym in symbols}


def applicable_features(asset_class: str, names: Iterable[str]) -> list[str]:
    """The subset of ``names`` this asset class is EXPECTED to produce.

    Columns unknown to the catalog (synthetic panels, ad-hoc frames) are
    always applicable — this only ever removes structurally-impossible
    columns, never invents new ones."""
    groups = INAPPLICABLE_GROUPS.get(asset_class, frozenset())
    features = INAPPLICABLE_FEATURES.get(asset_class, frozenset())
    out: list[str] = []
    for name in names:
        if name in features:
            continue
        spec = FEATURE_CATALOG.get(name)
        if spec is not None and spec.group in groups:
            continue
        out.append(name)
    return out
