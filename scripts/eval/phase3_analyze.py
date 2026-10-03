"""Analyze the pre-registered Phase 3 retrieval ablation.

Implements docs/preregistration_eval_decisions.md: validity checks first, then
per-arm metrics, then paired comparisons against the vector vs vector-rep
noise floor. Writes evals/decisions/results_phase3.json.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import canonical_bars

CODE_VERSION = "phase3-806d954"
ARMS = {  # name -> (journal_retrieval, replicate)
    "off": ("off", 0),
    "vector": ("vector", 0),
    "hybrid": ("hybrid", 0),
    "vector-rep": ("vector", 1),
}
BULLISH = {"buy", "add"}
BEARISH = {"avoid", "sell", "trim"}
HORIZON = 20
OUT = Path("evals/decisions")


def _boot(diff: np.ndarray, n: int = 10_000) -> list[float]:
    if diff.size == 0:
        return [math.nan, math.nan, math.nan]
    rng = np.random.default_rng(11)
    means = diff[rng.integers(0, diff.size, size=(n, diff.size))].mean(axis=1)
    return [float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _ece(conv: np.ndarray, hit: np.ndarray, bins: int = 5) -> float:
    if conv.size == 0:
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conv, edges) - 1, 0, bins - 1)
    return float(
        sum(
            abs(conv[idx == b].mean() - hit[idx == b].mean()) * (idx == b).sum()
            for b in range(bins)
            if (idx == b).any()
        )
        / conv.size
    )


def main() -> None:
    engine = make_engine(Settings().database_url)
    artifact_dir = Path(Settings().artifact_dir)
    with engine.connect() as c:
        runs = c.execute(
            text("select id, config_json from replay_runs where config_json->>'code_version' = :v"),
            {"v": CODE_VERSION},
        ).all()
    run_ids: dict[str, str] = {}
    for rid, cfg in runs:
        key = (cfg.get("journal_retrieval", "vector"), int(cfg.get("replicate", 0)))
        for name, spec in ARMS.items():
            if key == spec and cfg.get("start") == "2026-08-03" and cfg.get("end") == "2026-08-21":
                run_ids[name] = rid
    missing = set(ARMS) - set(run_ids)
    assert not missing, f"runs not found for arms {missing}"

    with engine.connect() as c:
        verdicts = pd.read_sql(
            text(
                "select run_id, as_of, symbol, action, conviction, disagreement_score, "
                "payload->>'model_id' model_id, payload->'journal_context_used' hits "
                "from replay_verdicts where run_id = any(:r)"
            ),
            c,
            params={"r": list(run_ids.values())},
        )
        entries = {
            r[0]: (r[1], list(r[2] or ()))
            for r in c.execute(text("select id, created_at, symbols from journal_entries"))
        }
    name_of = {v: k for k, v in run_ids.items()}
    verdicts["arm"] = verdicts["run_id"].map(name_of)
    verdicts["as_of"] = pd.to_datetime(verdicts["as_of"]).dt.date

    # ---- validation records (structured-output accounting) ----
    val: dict[str, list[dict[str, Any]]] = {}
    for name, rid in run_ids.items():
        path = artifact_dir / "replay" / rid / "validation.jsonl"
        rows = [json.loads(line) for line in path.open()] if path.exists() else []
        # resumed days may be logged twice; keep the last record per day
        val[name] = list({r["as_of"]: r for r in rows}.values())

    # ---- validity checks ----
    shortlists = {
        name: {(r["as_of"], s) for r in rows for s in r["journal_hits"]}
        for name, rows in val.items()
    }
    ref = shortlists["vector"]
    checks: dict[str, Any] = {
        "run_ids": run_ids,
        "shortlist_identical_to_vector": {n: s == ref for n, s in shortlists.items()},
        "shortlist_sizes": {n: len(s) for n, s in shortlists.items()},
        "off_has_zero_hits": all(
            not hits for r in val["off"] for hits in r["journal_hits"].values()
        ),
        "vector_hits_only_same_symbol": all(
            sym in entries.get(eid, (None, []))[1]
            for r in val["vector"] + val["vector-rep"]
            for sym, hits in r["journal_hits"].items()
            for eid in hits
            if not eid.startswith("exp")  # prior-case lines carry experience ids
        ),
        "hybrid_hits_on_every_dive": all(
            bool(hits) for r in val["hybrid"] for hits in r["journal_hits"].values()
        ),
        "arbiter_models": dict(Counter(verdicts["model_id"])),
        "hits_dated_after_as_of": int(
            sum(
                1
                for rows in val.values()
                for r in rows
                for hits in r["journal_hits"].values()
                for eid in hits
                if eid in entries
                and entries[eid][0]
                > datetime.combine(date.fromisoformat(r["as_of"]), time(23, 59, 59), tzinfo=UTC)
            )
        ),
    }

    # ---- outcomes from bars ----
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

    def excess20(as_of: date, sym: str) -> float:
        after = [s for s in sessions if s > pd.Timestamp(as_of)]
        if len(after) < HORIZON or sym not in o.columns:
            return math.nan
        d1, d20 = after[0], after[HORIZON - 1]
        po = o[sym].loc[:d1].dropna()
        pc = cl[sym].loc[:d20].dropna()
        if po.empty or pc.empty or po.index[-1] != d1:
            return math.nan
        r = pc.iloc[-1] / po.iloc[-1] - 1.0
        m = cl["SPY"].loc[d20] / o["SPY"].loc[d1] - 1.0
        return float(r - m)

    verdicts["excess20"] = [
        excess20(a, s) for a, s in zip(verdicts["as_of"], verdicts["symbol"], strict=True)
    ]
    verdicts["direction"] = verdicts["action"].map(
        lambda a: 1 if a in BULLISH else (-1 if a in BEARISH else 0)
    )
    directional = (verdicts["direction"] != 0) & verdicts["excess20"].notna()
    verdicts["success"] = np.where(
        directional, (verdicts["direction"] * verdicts["excess20"]) > 0, np.nan
    )
    verdicts["sq_err"] = np.where(
        directional, (verdicts["conviction"] - verdicts["success"]) ** 2, np.nan
    )
    verdicts["signed_excess"] = np.where(
        directional, verdicts["direction"] * verdicts["excess20"], np.nan
    )

    # ---- per-arm metrics ----
    per_arm: dict[str, Any] = {}
    for name in ARMS:
        v = verdicts[verdicts["arm"] == name]
        d = v[v["sq_err"].notna()]
        recs = val[name]

        def stage_rate(stage: str, recs: list[dict[str, Any]] = recs) -> float:
            sent = sum(r.get(stage, [0, 0])[0] for r in recs)
            got = sum(r.get(stage, [0, 0])[1] for r in recs)
            return got / sent if sent else math.nan

        dossiers_sent = sum(r.get("dossier", [0, 0])[0] for r in recs)
        per_arm[name] = {
            "verdicts": len(v),
            "directional_scored": len(d),
            "brier": float(d["sq_err"].mean()) if len(d) else math.nan,
            "ece_5bin": _ece(d["conviction"].to_numpy(float), d["success"].to_numpy(float)),
            "decision_quality_mean_signed_excess20": float(d["signed_excess"].mean())
            if len(d)
            else math.nan,
            "hit_rate": float(d["success"].mean()) if len(d) else math.nan,
            "mean_conviction": float(v["conviction"].mean()),
            "mean_disagreement": float(v["disagreement_score"].mean()),
            "actions": dict(Counter(v["action"])),
            "structured_output": {
                "dossier_parse_rate": stage_rate("dossier"),
                "bull_parse_rate": stage_rate("bull"),
                "bear_parse_rate": stage_rate("bear"),
                "arbiter_parse_rate": stage_rate("arbiter"),
                "verdicts_assembled_per_dossier_requested": (
                    sum(r.get("assembled", 0) for r in recs) / dossiers_sent
                    if dossiers_sent
                    else math.nan
                ),
                "quarantined": sum(r.get("quarantined", 0) for r in recs),
                "key_repairs": sum(r.get("key_repairs", 0) for r in recs),
            },
        }

    # ---- paired comparisons ----
    wide = verdicts.pivot_table(
        index=["as_of", "symbol"],
        columns="arm",
        values=["sq_err", "signed_excess", "conviction", "disagreement_score"],
        aggfunc="first",
    )
    actions = verdicts.pivot_table(
        index=["as_of", "symbol"], columns="arm", values="action", aggfunc="first"
    )

    def paired(metric: str, a: str, b: str, subset: pd.Index | None = None) -> dict[str, Any]:
        cols = wide[metric]
        if a not in cols or b not in cols:
            return {"n": 0}
        frame = cols[[a, b]].dropna()
        if subset is not None:
            frame = frame[frame.index.isin(subset)]
        mean, lo, hi = _boot((frame[a] - frame[b]).to_numpy(float))
        return {"n": len(frame), "diff": mean, "ci95": [lo, hi]}

    def flip_rate(a: str, b: str, subset: pd.Index | None = None) -> dict[str, Any]:
        frame = actions[[a, b]].dropna()
        if subset is not None:
            frame = frame[frame.index.isin(subset)]
        return {"n": len(frame), "action_changed": float((frame[a] != frame[b]).mean())}

    vec = verdicts[verdicts["arm"] == "vector"]
    treated = pd.MultiIndex.from_frame(vec[vec["hits"].map(lambda h: bool(h))][["as_of", "symbol"]])
    pairs = [("hybrid", "off"), ("vector", "off"), ("hybrid", "vector"), ("vector", "vector-rep")]
    comparisons: dict[str, Any] = {}
    for a, b in pairs:
        comparisons[f"{a} - {b}"] = {
            m: paired(m, a, b)
            for m in ("sq_err", "signed_excess", "conviction", "disagreement_score")
        } | {"action_flip": flip_rate(a, b)}
    comparisons["treated: vector - off"] = {
        m: paired(m, "vector", "off", treated) for m in ("sq_err", "signed_excess", "conviction")
    } | {"action_flip": flip_rate("vector", "off", treated)}
    comparisons["treated: vector - vector-rep"] = {
        "action_flip": flip_rate("vector", "vector-rep", treated)
    }

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "preregistration": "docs/preregistration_eval_decisions.md",
        "validity": checks,
        "per_arm": per_arm,
        "paired": comparisons,
    }
    (OUT / "results_phase3.json").write_text(json.dumps(result, indent=2, default=str))
    verdicts.drop(columns=["hits"]).to_csv(OUT / "phase3_verdicts.csv", index=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
