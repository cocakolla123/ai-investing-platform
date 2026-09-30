"""Short-cycle oscillators: buy the bottom of a recent range, sell the swing back up, repeat.

The "fluctuates daily/weekly" version of the range idea, and the honest test
of the market scanner's existing ``range_bound`` detector
(``dynamo.data.detectors.range_bound``). The oscillator filter mirrors that
detector's conditions, all computed from data up to and including day t:

1. mean-reverting: variance ratio Var(10-day log return) / (10 x Var(1-day))
   over the last 120 sessions <= 0.90 (a random walk is 1.0);
2. two-sided: >= 2 separate closes beyond BOTH the upper and the lower
   20-day 2-sigma Bollinger band in the last 60 sessions;
3. liquid: median consolidated-equivalent dollar volume (60d) >= $5M, price >= $5.

Optional extra (variant): lag-1 autocorrelation of daily returns over 60
sessions < 0 — "down days tend to be followed by up days".

Entry signal at the close of day t, filled at the next open. One position per
symbol at a time; after an exit the next signal can re-enter (the "repeat").
Baseline = the same entry/exit rules on every liquid stock, no oscillator
filter. All variants in the pre-declared grid are reported.

Usage:
    python scripts/research/oscillator_backtest.py --start 2025-09-29 --end 2026-09-29
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dynamo.domain.volume_source import volume_source_share
from scripts.research.range_floor_backtest import load_bars


@dataclass(frozen=True)
class Params:
    vr_window: int = 120
    vr_lag: int = 10
    max_vr: float = 0.90
    band_window: int = 20
    band_sigma: float = 2.0
    touch_window: int = 60
    min_touches: int = 2
    range_window: int = 60
    require_neg_autocorr: bool = False
    entry: str = (
        "range_low"  # "range_low": close in bottom 25% of 60d range; "band": close < lower band
    )
    exit: str = "mean"  # "mean": close >= 20d mean; "range_high": close in top 25% of 60d range
    stop: float = 0.08
    max_hold: int = 10
    cost_bps: float = 15.0
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0


def _runs_started(flag: pd.Series) -> pd.Series:
    return flag & ~flag.shift(1, fill_value=False)


def signals_for(g: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Per-day indicators for one symbol (all trailing, point-in-time)."""
    c = g["close"].astype(float)
    hi, lo = g["high"].astype(float), g["low"].astype(float)
    r = np.log(c).diff()
    var1 = r.rolling(p.vr_window).var()
    rk = np.log(c).diff(p.vr_lag)
    vr = rk.rolling(p.vr_window).var() / (p.vr_lag * var1)
    ma = c.rolling(p.band_window).mean()
    sd = c.rolling(p.band_window).std()
    upper, lower = ma + p.band_sigma * sd, ma - p.band_sigma * sd
    up_runs = _runs_started(c > upper).rolling(p.touch_window).sum()
    dn_runs = _runs_started(c < lower).rolling(p.touch_window).sum()
    rmin, rmax = lo.rolling(p.range_window).min(), hi.rolling(p.range_window).max()
    pos = (c - rmin) / (rmax - rmin)
    ac1 = r.rolling(p.touch_window).corr(r.shift(1))
    share = volume_source_share(str(g["volume_source"].iloc[-1]))
    dv = (c * g["volume"].astype(float)).rolling(60).median() / max(share, 1e-9)
    oscillator = (vr <= p.max_vr) & (up_runs >= p.min_touches) & (dn_runs >= p.min_touches)
    if p.require_neg_autocorr:
        oscillator &= ac1 < 0
    liquid = (c >= p.min_price) & (dv >= p.min_dollar_volume)
    trigger = (pos <= 0.25) if p.entry == "range_low" else (c < lower)
    return pd.DataFrame(
        {"oscillator": oscillator, "liquid": liquid, "trigger": trigger, "ma": ma, "pos": pos}
    )


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    ret: float
    spy_ret: float
    reason: str
    held: int


def simulate(
    g: pd.DataFrame,
    sig: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    p: Params,
    spy: pd.Series,
    filtered: bool,
) -> list[Trade]:
    o, c = g["open"].to_numpy(float), g["close"].to_numpy(float)
    dates = pd.DatetimeIndex(g["ts"])
    ok = (sig["trigger"] & sig["liquid"]).to_numpy(bool)
    if filtered:
        ok &= sig["oscillator"].fillna(False).to_numpy(bool)
    ma, pos = sig["ma"].to_numpy(float), sig["pos"].to_numpy(float)
    trades: list[Trade] = []
    n, i = len(g), 0
    while i < n - 1:
        if dates[i] < t0 or not ok[i]:
            i += 1
            continue
        if dates[i] > t1:
            break
        e = i + 1
        entry = o[e] if o[e] > 0 else c[e]
        j, reason = e, "time"
        while j < n and j - e < p.max_hold:
            done = c[j] >= ma[j] if p.exit == "mean" else pos[j] >= 0.75
            if done and j > e - 1:
                reason = "target"
                break
            if c[j] <= entry * (1 - p.stop):
                reason = "stop"
                break
            j += 1
        j = min(j, n - 1)
        ret = c[j] / entry - 1.0 - 2 * p.cost_bps / 1e4
        s0, s1 = spy.asof(dates[e]), spy.asof(dates[j])
        trades.append(
            Trade(
                str(g["symbol"].iloc[0]),
                str(dates[e].date()),
                str(dates[j].date()),
                float(ret),
                float(s1 / s0 - 1.0),
                reason,
                int(j - e),
            )
        )
        i = j + 1
    return trades


def summarize(tr: list[Trade]) -> dict:
    if not tr:
        return {"trades": 0}
    df = pd.DataFrame([asdict(t) for t in tr])
    ex = df["ret"] - df["spy_ret"]
    return {
        "trades": len(df),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": round(100 * float((df["ret"] > 0).mean()), 1),
        "avg_ret_pct": round(100 * float(df["ret"].mean()), 3),
        "spy_same_days_pct": round(100 * float(df["spy_ret"].mean()), 3),
        "excess_pct": round(100 * float(ex.mean()), 3),
        "excess_t_naive": round(float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(df)))), 2),
        "avg_hold": round(float(df["held"].mean()), 1),
        "exits": {k: int(v) for k, v in df["reason"].value_counts().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2025, 9, 29))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 29))
    ap.add_argument("--out", default="artifacts/research/oscillator")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bars = load_bars(date(args.start.year - 1, args.start.month, 1), args.end)
    t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
    spy_g = bars[bars["symbol"] == "SPY"]
    spy = pd.Series(spy_g["close"].to_numpy(float), index=pd.DatetimeIndex(spy_g["ts"]))
    groups = [
        g.reset_index(drop=True) for _, g in bars.groupby("symbol", sort=False) if len(g) > 150
    ]
    print(f"{len(groups):,} symbols with enough history", flush=True)

    rows, best_trades = [], {}
    grid = itertools.product([False, True], ["range_low", "band"], ["mean", "range_high"])
    for neg_ac, entry, exit_ in grid:
        p = replace(Params(), require_neg_autocorr=neg_ac, entry=entry, exit=exit_)
        filt: list[Trade] = []
        base: list[Trade] = []
        for g in groups:
            sig = signals_for(g, p)
            filt += simulate(g, sig, t0, t1, p, spy, filtered=True)
            if not neg_ac:  # baseline ignores the filter, so it only varies with entry/exit
                base += simulate(g, sig, t0, t1, p, spy, filtered=False)
        f, b = summarize(filt), summarize(base) if base else None
        row = {
            "neg_autocorr": neg_ac,
            "entry": entry,
            "exit": exit_,
            **{f"osc_{k}": v for k, v in f.items() if k != "exits"},
        }
        if b:
            row.update(
                {
                    f"all_{k}": v
                    for k, v in b.items()
                    if k in ("trades", "win_rate", "avg_ret_pct", "excess_pct")
                }
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not neg_ac and entry == "range_low" and exit_ == "mean":
            best_trades = {"oscillator": filt, "all": base}
    pd.DataFrame(rows).to_csv(out / "grid.csv", index=False)
    for k, tr in best_trades.items():
        pd.DataFrame([asdict(t) for t in tr]).to_csv(out / f"trades_default_{k}.csv", index=False)


if __name__ == "__main__":
    main()
