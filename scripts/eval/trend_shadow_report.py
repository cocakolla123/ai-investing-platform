"""Forward-shadow report for the trend rule (docs/preregistration_trend_shadow.md).

Run any time. Output is labeled INTERIM until 60 scored days have complete
20-session outcomes; only then does it print the pre-registered decision.
Writes evals/trend_shadow/report.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars

MODEL = "trend-rule-v1"
HORIZON = 20
DECISION_DAYS = 60
NW_LAGS = 19
OUT = Path("evals/trend_shadow")


def newey_west_t(x: np.ndarray, lags: int) -> float:
    n = x.size
    if n < 3:
        return math.nan
    d = x - x.mean()
    var = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        var += 2.0 * w * float(d[k:] @ d[:-k]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else math.nan


def main() -> None:
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        scores = pd.read_sql(
            text(
                "select symbol, as_of, expected_excess_return from shadow_quant_scores "
                "where model_version = :m and expected_excess_return is not null"
            ),
            c,
            params={"m": MODEL},
        )
    scores = cast(pd.DataFrame, scores[~scores["symbol"].str.contains("/")])  # equities only
    if scores.empty:
        print("no shadow scores yet")
        return
    syms = sorted({*scores["symbol"], "SPY"})
    start = pd.Timestamp(str(scores["as_of"].min())) - pd.Timedelta(days=5)
    with engine.connect() as c:
        raw = pd.read_sql(
            text("select * from bars_1d where symbol = any(:s) and ts >= :t"),
            c,
            params={"s": syms, "t": start.date()},
        )
    bars = canonical_bars(raw)
    bars["d"] = pd.to_datetime(bars["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    sessions = sorted(bars.loc[bars["symbol"] == "SPY", "d"].unique())
    o = bars.pivot_table(index="d", columns="symbol", values="open", aggfunc="last")
    cl = bars.pivot_table(index="d", columns="symbol", values="close", aggfunc="last")

    daily = []
    for as_of_key, group in scores.groupby("as_of"):
        as_of = str(as_of_key)
        g = cast(pd.DataFrame, group)
        after = [s for s in sessions if s >= pd.Timestamp(as_of)]
        if len(after) < HORIZON:
            continue  # outcomes not complete yet
        d1, d20 = after[0], after[HORIZON - 1]
        spy = cl["SPY"].loc[d20] / o["SPY"].loc[d1] - 1.0
        ex = {}
        for sym in g["symbol"]:
            if sym in o.columns and d1 in o.index and d20 in cl.index:
                po, pc = o.at[d1, sym], cl.at[d20, sym]
                if pd.notna(po) and pd.notna(pc) and po > 0:
                    ex[sym] = pc / po - 1.0 - spy
        g = cast(pd.DataFrame, g[g["symbol"].isin(list(ex))]).copy()
        if len(g) < 10:
            continue
        score = cast(pd.Series, g["expected_excess_return"]).astype(float)
        excess = pd.Series([ex[s] for s in g["symbol"]], index=g.index, dtype=float)
        ic = float(score.rank().corr(excess.rank()))
        top = excess[score >= score.quantile(0.8)]
        daily.append(
            {
                "as_of": as_of,
                "names": len(g),
                "ic": float(ic),
                "top_quintile_spread": float(top.mean() - excess.mean()),
            }
        )

    ics = np.asarray([d["ic"] for d in daily], float)
    complete = len(daily)
    decided = complete >= DECISION_DAYS
    use = ics[:DECISION_DAYS] if decided else ics
    mean_ic = float(use.mean()) if use.size else math.nan
    t = newey_west_t(use, NW_LAGS)
    if not decided:
        decision = f"INTERIM ({complete}/{DECISION_DAYS} days complete) — no decision"
    elif mean_ic > 0 and t >= 2.0:
        decision = "FORWARD EVIDENCE FOR THE RULE"
    elif mean_ic > 0:
        decision = "INCONCLUSIVE"
    else:
        decision = "AGAINST"
    report = {
        "preregistration": "docs/preregistration_trend_shadow.md",
        "scored_days": len(set(scores["as_of"])),
        "complete_days": complete,
        "mean_ic": mean_ic,
        "newey_west_t": t,
        "mean_top_quintile_spread": float(np.mean([d["top_quintile_spread"] for d in daily]))
        if daily
        else math.nan,
        "decision": decision,
        "daily": daily,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "daily"}, indent=2, default=str))


if __name__ == "__main__":
    main()
