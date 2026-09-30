"""Range-floor strategy: buy at the bottom of an established range, sell near the top, repeat.

The user's pattern, made mechanical:

* a stock trades sideways for a year in a channel,
* its lows keep returning to the same floor, near its 52-week low, and the
  floor holds,
* every return to the floor is followed by a bounce.

The strategy buys each return to the floor and sells the bounce, on repeat.

POINT-IN-TIME. Every decision on day t uses only bars dated <= t; orders fill
at the NEXT session's open. The range is re-derived every day from the prior
year only. Screening a year for stocks that bounced and then "backtesting"
bounces over that same year is guaranteed to look good, so the screen
(`--screen`) is reported separately and labelled as hindsight.

Baseline: identical entries/exits, but on ANY stock touching its 52-week low
with no range requirement — tells whether the range filter adds anything.

Usage:
    python scripts/research/range_floor_backtest.py --start 2025-09-29 --end 2026-09-29
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.domain.symbols import is_crypto_symbol
from dynamo.domain.volume_source import volume_source_share
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars


@dataclass(frozen=True)
class Params:
    lookback: int = 252  # sessions that define the range (the "52 weeks")
    floor_zone: float = 0.03  # a "touch" = low within 3% above the floor
    min_prior_bounces: int = 2  # floor must have held and bounced at least this often before
    bounce_min: float = 0.08  # a bounce = close >= touch low * (1 + 8%) ...
    bounce_within: int = 20  # ... within this many sessions of the touch
    touch_gap: int = 10  # touches closer than this are one test
    max_trend: float = 0.25  # |return over the lookback| must be below this (sideways)
    min_range_width: float = 0.15  # (high/low - 1) at least this: room to profit
    max_range_width: float = 1.00  # ... and at most this: not a crash or a rocket
    target_frac: float = 0.60  # sell when close >= floor + 60% of the range height
    stop: float = 0.05  # sell when close < floor * (1 - 5%): the floor broke
    max_hold: int = 30  # sessions; give up if it goes nowhere
    close_holds_floor: bool = False  # entry day must CLOSE at/above the floor
    target_pct: float | None = None  # if set, sell at entry * (1 + target_pct) instead
    cost_bps: float = 15.0  # per side: spread + slippage (conservative for mid caps)
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0  # consolidated-equivalent, median over lookback


def load_bars(start_bars: date, end: date) -> pd.DataFrame:
    engine = make_engine(Settings().database_url)
    sql = text(
        "select symbol, ts, open, high, low, close, adj_close, volume, source from bars_1d "
        "where ts >= :s and ts <= :e and source in ('alpaca','tiingo')"
    )
    with engine.connect() as conn:
        raw = pd.read_sql(sql, conn, params={"s": start_bars, "e": end})
    raw = raw[~raw["symbol"].map(is_crypto_symbol)]
    bars = canonical_bars(raw)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    return bars.sort_values(["symbol", "ts"]).reset_index(drop=True)


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry: float
    exit_date: str
    exit: float
    reason: str
    ret: float  # net of costs
    held: int
    spy_ret: float
    floor: float
    ceiling: float
    prior_bounces: int


def count_bounces(low: np.ndarray, close: np.ndarray, floor: float, p: Params) -> tuple[int, int]:
    """(tests, bounces) of `floor` within the given window arrays (already point-in-time)."""
    near = np.where(low <= floor * (1 + p.floor_zone))[0]
    tests = bounces = 0
    last = -(10**9)
    for i in near:
        if i - last < p.touch_gap:
            continue
        last = i
        tests += 1
        after = close[i + 1 : i + 1 + p.bounce_within]
        if after.size and after.max() >= low[i] * (1 + p.bounce_min):
            bounces += 1
    return tests, bounces


def simulate_symbol(
    g: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    p: Params,
    spy: pd.Series,
    require_range: bool,
) -> list[Trade]:
    o, h, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = g["volume"].to_numpy(float)
    share = volume_source_share(str(g["volume_source"].iloc[-1])) if "volume_source" in g else 1.0
    dates = g["ts"].to_numpy()
    n = len(g)
    trades: list[Trade] = []
    i = p.lookback
    while i < n - 1:
        d = pd.Timestamp(dates[i])
        if d < test_start:
            i += 1
            continue
        if d > test_end:
            break
        w0, w1 = i - p.lookback, i  # window strictly BEFORE today
        wl, wc, wh = lo[w0:w1], c[w0:w1], h[w0:w1]
        floor, ceiling = float(wl.min()), float(wh.max())
        px = c[i]
        dv = float(np.median((c[w0:w1] * v[w0:w1])[-60:])) / max(share, 1e-9)
        hold_ok = c[i] >= floor if p.close_holds_floor else c[i] >= floor * (1 - p.stop)
        touch = lo[i] <= floor * (1 + p.floor_zone) and hold_ok
        if not (touch and px >= p.min_price and dv >= p.min_dollar_volume and floor > 0):
            i += 1
            continue
        width = ceiling / floor - 1.0
        trend = wc[-1] / wc[0] - 1.0
        _tests, bounces = count_bounces(wl, wc, floor, p)
        if require_range and not (
            bounces >= p.min_prior_bounces
            and abs(trend) <= p.max_trend
            and p.min_range_width <= width <= p.max_range_width
        ):
            i += 1
            continue
        # enter next open
        e = i + 1
        entry = o[e] if o[e] > 0 else c[e]
        target = (
            entry * (1 + p.target_pct)
            if p.target_pct is not None
            else floor + p.target_frac * (ceiling - floor)
        )
        j, reason = e, "time"
        while j < n and j - e < p.max_hold:
            if c[j] >= target:
                reason = "target"
                break
            if c[j] < floor * (1 - p.stop):
                reason = "stop"
                break
            j += 1
        j = min(j, n - 1)
        if pd.Timestamp(dates[j]) > test_end and reason == "time":
            reason = "open_at_end"
        exit_px = c[j]
        cost = 2 * p.cost_bps / 1e4
        ret = exit_px / entry - 1.0 - cost
        s0, s1 = spy.asof(pd.Timestamp(dates[e])), spy.asof(pd.Timestamp(dates[j]))
        trades.append(
            Trade(
                symbol=str(g["symbol"].iloc[0]),
                entry_date=str(pd.Timestamp(dates[e]).date()),
                entry=float(entry),
                exit_date=str(pd.Timestamp(dates[j]).date()),
                exit=float(exit_px),
                reason=reason,
                ret=float(ret),
                held=int(j - e),
                spy_ret=float(s1 / s0 - 1.0),
                floor=floor,
                ceiling=ceiling,
                prior_bounces=bounces,
            )
        )
        i = j + 1  # flat again; the next return to the floor can trigger a new trade
    return trades


def summarize(trades: list[Trade], label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0}
    df = pd.DataFrame([asdict(t) for t in trades])
    closed = df[df["reason"] != "open_at_end"]
    excess = df["ret"] - df["spy_ret"]
    # per-trade t-stat, trades clustered by symbol are not independent — report
    # both the naive t and the symbol count so nobody over-reads it.
    t = excess.mean() / (excess.std(ddof=1) / math.sqrt(len(df))) if len(df) > 1 else float("nan")
    return {
        "label": label,
        "trades": len(df),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": float((closed["ret"] > 0).mean()) if len(closed) else float("nan"),
        "avg_ret": float(df["ret"].mean()),
        "median_ret": float(df["ret"].median()),
        "avg_spy_same_days": float(df["spy_ret"].mean()),
        "avg_excess_vs_spy": float(excess.mean()),
        "excess_t_naive": float(t),
        "avg_hold_sessions": float(df["held"].mean()),
        "exits": {k: int(v) for k, v in df["reason"].value_counts().items()},
        "worst": float(df["ret"].min()),
        "best": float(df["ret"].max()),
    }


def screen(bars: pd.DataFrame, end: pd.Timestamp, p: Params) -> pd.DataFrame:
    """HINDSIGHT list: stocks that fit the pattern over the trailing year to `end`."""
    rows = []
    for sym, g in bars.groupby("symbol", sort=False):
        g = g[g["ts"] <= end].tail(p.lookback)
        if len(g) < p.lookback:
            continue
        lo, c, h, v = (g[k].to_numpy(float) for k in ("low", "close", "high", "volume"))
        share = volume_source_share(str(g["volume_source"].iloc[-1]))
        dv = float(np.median((c * v)[-60:])) / max(share, 1e-9)
        if c[-1] < p.min_price or dv < p.min_dollar_volume:
            continue
        floor, ceiling = float(lo.min()), float(h.max())
        width, trend = ceiling / floor - 1.0, c[-1] / c[0] - 1.0
        if not (p.min_range_width <= width <= p.max_range_width and abs(trend) <= p.max_trend):
            continue
        tests, bounces = count_bounces(lo, c, floor, p)
        if bounces < 3:
            continue
        rows.append(
            {
                "symbol": sym,
                "floor": round(floor, 2),
                "ceiling": round(ceiling, 2),
                "range_width_pct": round(100 * width, 1),
                "year_return_pct": round(100 * trend, 1),
                "floor_tests": tests,
                "bounces": bounces,
                "last_close": round(float(c[-1]), 2),
                "pct_above_floor": round(100 * (c[-1] / floor - 1.0), 1),
                "dollar_volume_m": round(dv / 1e6, 1),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["bounces", "range_width_pct"], ascending=False) if len(out) else out


def run_grid(
    bars: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, spy: pd.Series, out: Path
) -> None:
    """Pre-declared 16-variant grid. ALL rows are reported; picking the best
    one and quoting it would be exactly the selection this study avoids."""
    import itertools
    from dataclasses import replace

    groups = [(sym, g.reset_index(drop=True)) for sym, g in bars.groupby("symbol", sort=False)]
    groups = [(sym, g) for sym, g in groups if len(g) > Params().lookback + 1]
    rows = []
    for holds, tgt, nb, stop in itertools.product(
        [False, True], [None, 0.08], [2, 3], [0.05, 0.10]
    ):
        p = replace(
            Params(), close_holds_floor=holds, target_pct=tgt, min_prior_bounces=nb, stop=stop
        )
        trades: list[Trade] = []
        for _sym, g in groups:
            trades += simulate_symbol(g, t0, t1, p, spy, require_range=True)
        sm = summarize(trades, "")
        rows.append(
            {
                "floor_must_hold": holds,
                "target": "range 60%" if tgt is None else f"+{tgt:.0%}",
                "min_bounces": nb,
                "stop": f"{stop:.0%}",
                "trades": sm.get("trades", 0),
                "win_rate": round(100 * sm.get("win_rate", float("nan")), 1),
                "avg_ret_pct": round(100 * sm.get("avg_ret", float("nan")), 2),
                "spy_same_days_pct": round(100 * sm.get("avg_spy_same_days", float("nan")), 2),
                "excess_pct": round(100 * sm.get("avg_excess_vs_spy", float("nan")), 2),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "grid.csv", index=False)

    # Hindsight demonstration: the SAME default strategy, but only on the
    # symbols the year-end screen picked (i.e. chosen because they bounced).
    scr = screen(bars, t1, Params())
    picked = set(scr["symbol"]) if len(scr) else set()
    hind: list[Trade] = []
    for sym, g in groups:
        if sym in picked:
            hind += simulate_symbol(g, t0, t1, Params(), spy, require_range=True)
    (out / "hindsight.json").write_text(
        json.dumps(summarize(hind, "hindsight: screened symbols"), indent=2)
    )
    print("HINDSIGHT", json.dumps(summarize(hind, "hindsight: screened symbols")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2025, 9, 29))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 29))
    ap.add_argument("--out", default="artifacts/research/range_floor")
    ap.add_argument("--grid", action="store_true", help="run the pre-declared robustness grid")
    args = ap.parse_args()
    p = Params()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bars = load_bars(date(args.start.year - 2, args.start.month, 1), args.end)
    test_start, test_end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    spy_g = bars[bars["symbol"] == "SPY"]
    spy = pd.Series(spy_g["close"].to_numpy(float), index=pd.DatetimeIndex(spy_g["ts"]))
    print(f"bars: {len(bars):,} rows, {bars['symbol'].nunique():,} symbols")

    strat: list[Trade] = []
    base: list[Trade] = []
    for _sym, g in bars.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        if len(g) <= p.lookback + 1:
            continue
        strat += simulate_symbol(g, test_start, test_end, p, spy, require_range=True)
        base += simulate_symbol(g, test_start, test_end, p, spy, require_range=False)

    s, b = summarize(strat, "range-floor strategy"), summarize(base, "baseline: any 52w-low touch")
    if args.grid:
        run_grid(bars, test_start, test_end, spy, out)
    pd.DataFrame([asdict(t) for t in strat]).to_csv(out / "trades_strategy.csv", index=False)
    pd.DataFrame([asdict(t) for t in base]).to_csv(out / "trades_baseline.csv", index=False)
    scr = screen(bars, test_end, p)
    scr.to_csv(out / "screen_hindsight.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {"params": asdict(p), "strategy": s, "baseline": b, "screen_count": len(scr)}, indent=2
        )
    )
    for block in (s, b):
        print(json.dumps(block, indent=2))
    print(f"screen (hindsight): {len(scr)} symbols -> {out / 'screen_hindsight.csv'}")


if __name__ == "__main__":
    main()
