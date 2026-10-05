"""Analyze the pre-registered Phase 4 latency/cost run.

Implements docs/preregistration_eval_tracing.md: per-call and per-decision
latency, cost per decision, the wall-clock rule for the parallel-advocates
optimization, and the quality-held checks. Writes
evals/tracing/results_phase4.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars

START, END = "2026-08-03", "2026-08-07"
PHASE3_VECTOR_RUN = "replay-5181496084db"
NOISE_FLIP_RATE = 0.138  # Phase 3: vector vs vector-rep
BULLISH, BEARISH = {"buy", "add"}, {"avoid", "sell", "trim"}
HORIZON = 20
OUT = Path("evals/tracing")


def _q(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else math.nan


def _boot_mean(values: np.ndarray, n: int = 10_000) -> list[float]:
    if values.size == 0:
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(11)
    means = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(values.mean()), float(lo), float(hi)]


def main(code_version: str) -> None:
    engine = make_engine(Settings().database_url)
    artifact_dir = Path(Settings().artifact_dir)
    with engine.connect() as c:
        runs = c.execute(
            text("select id, config_json from replay_runs where config_json->>'code_version' = :v"),
            {"v": code_version},
        ).all()
    arms: dict[str, str] = {}
    for rid, cfg in runs:
        if cfg.get("start") == START and cfg.get("end") == END:
            arms["par" if cfg.get("parallel_advocates") else "seq"] = rid
    assert set(arms) == {"seq", "par"}, f"runs found: {arms}"

    with engine.connect() as c:
        ledger = pd.read_sql(
            text(
                "select request_id, role, model_id, input_tokens, output_tokens, "
                "cache_read_tokens, cost_usd, equiv_cost_usd, latency_ms, attempt, "
                "validation_ok, prompt_hash from cost_ledger "
                "where request_id like any(:pats)"
            ),
            c,
            params={"pats": [f"{rid}:%" for rid in arms.values()]},
        )
        verdicts = pd.read_sql(
            text(
                "select run_id, as_of, symbol, action, conviction from replay_verdicts "
                "where run_id = any(:r) and as_of between :s and :e"
            ),
            c,
            params={"r": [*arms.values(), PHASE3_VECTOR_RUN], "s": START, "e": END},
        )
    ledger["run_id"] = ledger["request_id"].str.split(":", n=1).str[0]
    key = ledger["request_id"].str.split(":", n=1).str[1]
    # custom_id = <as_of>-<symbol>-<stage>; as_of is 10 chars, stage is last.
    ledger["as_of"] = key.str[:10]
    ledger["symbol"] = key.str[11:].str.rsplit("-", n=1).str[0]
    name_of = {v: k for k, v in arms.items()} | {PHASE3_VECTOR_RUN: "phase3-vector"}
    ledger["arm"] = ledger["run_id"].map(name_of)
    verdicts["arm"] = verdicts["run_id"].map(name_of)
    verdicts["as_of"] = pd.to_datetime(verdicts["as_of"]).dt.date.astype(str)

    val = {}
    for arm, rid in arms.items():
        path = artifact_dir / "replay" / rid / "validation.jsonl"
        rows = [json.loads(line) for line in path.open()] if path.exists() else []
        val[arm] = list({r["as_of"]: r for r in rows}.values())

    # ---- per-call latency ----
    per_call: dict[str, Any] = {}
    for arm in arms:
        sub = ledger[ledger["arm"] == arm]
        lat = sub["latency_ms"].dropna()
        per_call[arm] = {
            "calls": len(sub),
            "retries": int((sub["attempt"] > 1).sum()),
            "validation_failures": int((sub["validation_ok"] == False).sum()),  # noqa: E712
            "p50_ms": _q(lat.tolist(), 0.5),
            "p95_ms": _q(lat.tolist(), 0.95),
            "by_role": {
                role: {
                    "n": len(g),
                    "p50_ms": _q(g["latency_ms"].dropna().tolist(), 0.5),
                    "p95_ms": _q(g["latency_ms"].dropna().tolist(), 0.95),
                }
                for role, g in sub.groupby("role")
            },
            "models": dict(Counter(sub["model_id"])),
            "prompt_hashes": dict(Counter(sub["role"] + ":" + sub["prompt_hash"].fillna(""))),
        }

    # ---- per-decision: compute + cost (assembled dives only) ----
    per_decision: dict[str, Any] = {}
    for arm in arms:
        v = verdicts[verdicts["arm"] == arm][["as_of", "symbol"]]
        sub = ledger[ledger["arm"] == arm].merge(v, on=["as_of", "symbol"])
        g = sub.groupby(["as_of", "symbol"])
        compute = g["latency_ms"].sum() / 1000.0
        per_decision[arm] = {
            "decisions": int(g.ngroups),
            "compute_s_p50": _q(compute.tolist(), 0.5),
            "compute_s_p95": _q(compute.tolist(), 0.95),
            "metered_usd_per_decision": float(g["cost_usd"].sum().mean()) if g.ngroups else 0.0,
            "equiv_usd_per_decision": float(g["equiv_cost_usd"].sum().mean())
            if g.ngroups
            else math.nan,
            "input_tokens_per_decision": float(
                (g["input_tokens"].sum() + g["cache_read_tokens"].sum()).mean()
            )
            if g.ngroups
            else math.nan,
            "output_tokens_per_decision": float(g["output_tokens"].sum().mean())
            if g.ngroups
            else math.nan,
        }

    # ---- wall clock per decision (primary) ----
    def per_day(arm: str) -> dict[str, dict[str, float]]:
        out = {}
        for r in val[arm]:
            if r.get("assembled"):
                s = r.get("seconds", {})
                out[r["as_of"]] = {
                    "wall_s_per_decision": s.get("total", math.nan) / r["assembled"],
                    "advocates_s": s.get("advocates", math.nan),
                    "total_s": s.get("total", math.nan),
                    "assembled": r["assembled"],
                }
        return out

    days = {arm: per_day(arm) for arm in arms}
    common = sorted(set(days["seq"]) & set(days["par"]))
    pct = np.array(
        [
            days["par"][d]["wall_s_per_decision"] / days["seq"][d]["wall_s_per_decision"] - 1.0
            for d in common
        ]
    )
    faster_days = int(sum(p < 0 for p in pct))
    # Headline measure (pre-registered): MEAN over days. The medians (p50) and
    # the p95 of five daily values stay as secondary rows; with only five days
    # that differ, the medians imply a larger cut than the means.
    wall = {
        arm: {
            "mean_s": float(np.mean([x["wall_s_per_decision"] for x in days[arm].values()])),
            "p50_s": _q([x["wall_s_per_decision"] for x in days[arm].values()], 0.5),
            "p95_s": _q([x["wall_s_per_decision"] for x in days[arm].values()], 0.95),
            "mean_total_s": float(np.mean([x["total_s"] for x in days[arm].values()])),
            "median_total_s": _q([x["total_s"] for x in days[arm].values()], 0.5),
            "per_day": days[arm],
        }
        for arm in arms
    }

    # ---- quality held ----
    actions = verdicts.pivot_table(
        index=["as_of", "symbol"], columns="arm", values="action", aggfunc="first"
    )

    def flips(a: str, b: str) -> dict[str, Any]:
        f = actions[[a, b]].dropna() if {a, b} <= set(actions.columns) else pd.DataFrame()
        k = int((f[a] != f[b]).sum()) if len(f) else 0
        n = len(f)
        return {
            "n": n,
            "flips": k,
            "rate": k / n if n else math.nan,
            "p_at_noise_rate": float(binom.sf(k - 1, n, NOISE_FLIP_RATE)) if n else math.nan,
        }

    def assembled_share(arm: str) -> float:
        sent = sum(r.get("dossier", [0, 0])[0] for r in val[arm])
        return sum(r.get("assembled", 0) for r in val[arm]) / sent if sent else math.nan

    par_seq = flips("par", "seq")
    fail_rate = {
        arm: per_call[arm]["validation_failures"] / per_call[arm]["calls"]
        if per_call[arm]["calls"]
        else math.nan
        for arm in arms
    }

    # descriptive outcome metrics on the same dives
    syms = sorted({*verdicts["symbol"], "SPY"})
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

    verdicts["excess20"] = [
        excess20(a, s) for a, s in zip(verdicts["as_of"], verdicts["symbol"], strict=True)
    ]
    verdicts["dir"] = verdicts["action"].map(
        lambda a: 1 if a in BULLISH else (-1 if a in BEARISH else 0)
    )
    d = verdicts[(verdicts["dir"] != 0) & verdicts["excess20"].notna()].copy()
    d["success"] = (d["dir"] * d["excess20"]) > 0
    quality = {
        arm: {
            "verdicts": int((verdicts["arm"] == arm).sum()),
            "directional": int((d["arm"] == arm).sum()),
            "brier": float(((g["conviction"] - g["success"]) ** 2).mean()) if len(g) else math.nan,
            "decision_quality": float((g["dir"] * g["excess20"]).mean()) if len(g) else math.nan,
            "actions": dict(Counter(verdicts.loc[verdicts["arm"] == arm, "action"])),
        }
        for arm in ("seq", "par", "phase3-vector")
        for g in [d[d["arm"] == arm]]
    }

    result = {
        "preregistration": "docs/preregistration_eval_tracing.md",
        "code_version": code_version,
        "runs": arms,
        "per_call": per_call,
        "per_decision": per_decision,
        "wall_clock": wall,
        "optimization": {
            "days_compared": len(common),
            "days_par_faster": faster_days,
            "mean_pct_change_wall_per_decision": _boot_mean(pct),
            # Seconds saved per night (per day's committee run), same days, mean.
            "mean_total_s_saved_per_day": float(
                np.mean([days["seq"][d]["total_s"] - days["par"][d]["total_s"] for d in common])
            )
            if common
            else math.nan,
            "works": len(common) >= 5 and faster_days >= 4,
        },
        "quality_held": {
            "par_vs_seq": par_seq,
            "seq_vs_phase3_vector": flips("seq", "phase3-vector"),
            "par_vs_phase3_vector": flips("par", "phase3-vector"),
            "assembled_share": {arm: assembled_share(arm) for arm in arms},
            "validation_failure_rate": fail_rate,
            "held": bool(
                (par_seq["n"] == 0 or par_seq["p_at_noise_rate"] >= 0.05)
                and all(assembled_share(a) >= 0.9 for a in arms)
                and all(r <= 1 / 60 for r in fail_rate.values())
            ),
        },
        "quality_descriptive": quality,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_phase4.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
