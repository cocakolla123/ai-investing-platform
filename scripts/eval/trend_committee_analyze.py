"""Analyze the pre-registered trend + committee run (docs/preregistration_trend_committee.md).

Writes evals/trend_committee/results.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars

START, END = "2026-08-03", "2026-08-21"
V8 = {"v8": "replay-5181496084db", "v8-rep": "replay-0ca99da87ddd"}
BULLISH, BEARISH = {"buy", "add"}, {"avoid", "sell", "trim"}
HORIZON = 20
OUT = Path("evals/trend_committee")


def _day_boot(frame: pd.DataFrame, col: str, n: int = 10_000) -> list[float]:
    """Mean of ``col`` with a CI from resampling whole days."""
    frame = frame.dropna(subset=[col])
    if frame.empty:
        return [math.nan, math.nan, math.nan]
    days = sorted(frame["as_of"].unique())
    groups = [frame.loc[frame["as_of"] == d, col].to_numpy(float) for d in days]
    rng = np.random.default_rng(11)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        vals = np.concatenate([groups[i] for i in pick])
        means.append(vals.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(frame[col].mean()), float(lo), float(hi)]


def _diff_boot(a: pd.DataFrame, b: pd.DataFrame, col: str, n: int = 10_000) -> list[float]:
    """mean(a) - mean(b), resampling the SAME days for both arms."""
    a, b = a.dropna(subset=[col]), b.dropna(subset=[col])
    days = sorted(set(a["as_of"]) | set(b["as_of"]))
    ga = {d: a.loc[a["as_of"] == d, col].to_numpy(float) for d in days}
    gb = {d: b.loc[b["as_of"] == d, col].to_numpy(float) for d in days}
    rng = np.random.default_rng(11)
    diffs = []
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), size=len(days))]
        va = np.concatenate([ga[d] for d in pick])
        vb = np.concatenate([gb[d] for d in pick])
        if va.size and vb.size:
            diffs.append(va.mean() - vb.mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return [float(a[col].mean() - b[col].mean()), float(lo), float(hi)]


def main(code_version: str) -> None:
    engine = make_engine(Settings().database_url)
    artifact_dir = Path(Settings().artifact_dir)
    with engine.connect() as c:
        rows = c.execute(
            text("select id, config_json from replay_runs where config_json->>'code_version' = :v"),
            {"v": code_version},
        ).all()
    trend = [rid for rid, cfg in rows if cfg.get("quant_model_version") == "trend-rule-v1"]
    assert len(trend) == 1, f"trend runs found: {trend}"
    runs = {"trend": trend[0], **V8}
    name_of = {v: k for k, v in runs.items()}

    with engine.connect() as c:
        verdicts = pd.read_sql(
            text(
                "select run_id, as_of, symbol, action, conviction from replay_verdicts "
                "where run_id = any(:r)"
            ),
            c,
            params={"r": list(runs.values())},
        )
    verdicts["arm"] = verdicts["run_id"].map(lambda r: name_of.get(str(r)))
    verdicts["as_of"] = pd.to_datetime(verdicts["as_of"]).dt.date.astype(str)

    val: dict[str, list[dict[str, Any]]] = {}
    for arm, rid in runs.items():
        path = artifact_dir / "replay" / rid / "validation.jsonl"
        recs = [json.loads(line) for line in path.open()] if path.exists() else []
        val[arm] = list({r["as_of"]: r for r in recs}.values())
    shortlist = pd.DataFrame(
        [
            {"arm": arm, "as_of": r["as_of"], "symbol": sym}
            for arm, recs in val.items()
            for r in recs
            for sym in r.get("journal_hits", {})
        ]
    )

    syms = sorted({*verdicts["symbol"], *shortlist["symbol"], "SPY"})
    with engine.connect() as c:
        raw = pd.read_sql(
            text(
                "select * from bars_1d where symbol = any(:s) and ts >= '2026-07-20' "
                "and ts <= '2026-09-30'"
            ),
            c,
            params={"s": syms},
        )
    bars = canonical_bars(raw)
    bars["d"] = pd.to_datetime(bars["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    sessions = sorted(bars.loc[bars["symbol"] == "SPY", "d"].unique())
    o = bars.pivot_table(index="d", columns="symbol", values="open", aggfunc="last")
    cl = bars.pivot_table(index="d", columns="symbol", values="close", aggfunc="last")

    def excess20(as_of: str, sym: str) -> float:
        after = [s for s in sessions if s > pd.Timestamp(as_of)]
        if len(after) < HORIZON or sym not in o.columns:
            return math.nan
        d1, d20 = after[0], after[HORIZON - 1]
        po, pc = o[sym].loc[:d1].dropna(), cl[sym].loc[:d20].dropna()
        if po.empty or pc.empty or po.index[-1] != d1:
            return math.nan
        return float(
            (pc.iloc[-1] / po.iloc[-1] - 1.0) - (cl["SPY"].loc[d20] / o["SPY"].loc[d1] - 1.0)
        )

    for frame in (verdicts, shortlist):
        frame["excess20"] = [
            excess20(a, s) for a, s in zip(frame["as_of"], frame["symbol"], strict=True)
        ]
    verdicts["dir"] = verdicts["action"].map(
        lambda a: 1 if a in BULLISH else (-1 if a in BEARISH else 0)
    )
    verdicts["signed"] = np.where(
        verdicts["dir"] != 0, verdicts["dir"] * verdicts["excess20"], np.nan
    )
    verdicts["success"] = np.where(
        (verdicts["dir"] != 0) & verdicts["excess20"].notna(),
        (verdicts["dir"] * verdicts["excess20"]) > 0,
        np.nan,
    )

    per_arm: dict[str, Any] = {}
    for arm in runs:
        v = cast(pd.DataFrame, verdicts[verdicts["arm"] == arm])
        buys = cast(pd.DataFrame, v[v["action"].isin(list(BULLISH))])
        d = cast(pd.DataFrame, v[v["success"].notna()])
        s = cast(pd.DataFrame, shortlist[shortlist["arm"] == arm])
        buy_q = _day_boot(buys, "excess20")
        short_q = _day_boot(s, "excess20")
        sent = sum(r.get("dossier", [0, 0])[0] for r in val[arm])
        per_arm[arm] = {
            "verdicts": len(v),
            "buys": len(buys),
            "buy_quality": buy_q,
            "decision_quality": _day_boot(v.assign(x=v["signed"]), "x"),
            "shortlist_quality": short_q,
            "committee_lift": buy_q[0] - short_q[0],
            "brier": float(((d["conviction"] - d["success"]) ** 2).mean()) if len(d) else math.nan,
            "actions": dict(Counter(v["action"])),
            "assembled_share": (sum(r.get("assembled", 0) for r in val[arm]) / sent)
            if sent
            else math.nan,
        }

    def buys_of(arm: str) -> pd.DataFrame:
        v = cast(pd.DataFrame, verdicts[verdicts["arm"] == arm])
        return cast(pd.DataFrame, v[v["action"].isin(list(BULLISH))])

    def jaccard(a: str, b: str) -> float:
        sa = {tuple(x) for x in shortlist.loc[shortlist["arm"] == a, ["as_of", "symbol"]].values}
        sb = {tuple(x) for x in shortlist.loc[shortlist["arm"] == b, ["as_of", "symbol"]].values}
        return len(sa & sb) / len(sa | sb) if sa | sb else math.nan

    vs_v8 = _diff_boot(buys_of("trend"), buys_of("v8"), "excess20")
    vs_rep = _diff_boot(buys_of("trend"), buys_of("v8-rep"), "excess20")
    noise = abs(per_arm["v8"]["buy_quality"][0] - per_arm["v8-rep"]["buy_quality"][0])

    def excludes_zero(ci: list[float]) -> int:
        return 1 if ci[1] > 0 else (-1 if ci[2] < 0 else 0)

    same_dir = excludes_zero(vs_v8) == excludes_zero(vs_rep) != 0
    better = bool(same_dir and min(abs(vs_v8[0]), abs(vs_rep[0])) > noise)
    result = {
        "preregistration": "docs/preregistration_trend_committee.md",
        "code_version": code_version,
        "runs": runs,
        "per_arm": per_arm,
        "buy_quality_trend_minus_v8": vs_v8,
        "buy_quality_trend_minus_v8rep": vs_rep,
        "replicate_noise_abs": noise,
        "shortlist_jaccard_trend_v8": jaccard("trend", "v8"),
        "shortlist_jaccard_v8_v8rep": jaccard("v8", "v8-rep"),
        "verdict": ("trend better" if vs_v8[0] > 0 else "trend worse")
        if better
        else "no difference shown",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
