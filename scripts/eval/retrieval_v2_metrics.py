"""Retrieval v2 metrics (docs/preregistration_eval_retrieval_v2.md).

Run only after the L2 labels are committed. Reads the committed rankings and
labels and writes evals/retrieval/results_v2.json (+ per-query rows):

- nDCG@10, Recall@5, MRR per arm: on L1 (mechanical) for S, X, all; on L2
  (independent) for X; frozen and current corpus;
- paired bootstrap 95% intervals on nDCG@10 (10,000 resamples, seed 7);
- latency p50/p95 per arm;
- coverage over the 699 campaign contexts and the 329 eligible ones;
- hand-check agreement with the L2 grader.
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine

EVAL = Path("evals/retrieval")
RUN = "replay-63220a3a5e9a"
HASH_FALLBACK = "hash-fallback"
ARMS = (
    "keyword",
    "vector",
    "hybrid",
    "production",
    "production_symbol_filter",
    "structured_fallback",
    "structured_boost",
)
PAIRS = (
    ("structured_fallback", "production_symbol_filter"),
    ("structured_boost", "production_symbol_filter"),
    ("structured_fallback", "vector"),
    ("structured_boost", "vector"),
    ("hybrid", "keyword"),
    ("hybrid", "vector"),
    ("vector", "keyword"),
    ("hybrid", "production_symbol_filter"),
)


def ndcg_at(ranking: list[str], grades: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(ranking: list[str], grades: dict[str, int], k: int = 5) -> float:
    rel = {d for d, g in grades.items() if g >= 1}
    return len(rel & set(ranking[:k])) / len(rel) if rel else 0.0


def mrr(ranking: list[str], grades: dict[str, int]) -> float:
    for i, d in enumerate(ranking):
        if grades.get(d, 0) >= 1:
            return 1.0 / (i + 1)
    return 0.0


def paired_boot(a: list[float], b: list[float], n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(7)
    diff = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [round(float(diff.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def weighted_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    """Quadratic-weighted Cohen's kappa for grades 0..k-1."""
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    n = obs.sum()
    if n == 0:
        return math.nan
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else math.nan


def _scores(rows: list[dict[str, Any]], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    per: dict[str, dict[str, list[float]]] = {
        a: {"ndcg10": [], "recall5": [], "mrr": []} for a in ARMS
    }
    for r in rows:
        g = qrels[r["query_id"]]
        for a in ARMS:
            rk = r["rankings"][a]
            per[a]["ndcg10"].append(ndcg_at(rk, g))
            per[a]["recall5"].append(recall_at(rk, g))
            per[a]["mrr"].append(mrr(rk, g))
    for a in ARMS:
        out[a] = {m: round(statistics.fmean(v), 4) for m, v in per[a].items()}
        out[a]["empty_results"] = sum(1 for r in rows if not r["rankings"][a])
    out["ndcg10_paired_diff_95ci"] = {
        f"{a}-{b}": paired_boot(per[a]["ndcg10"], per[b]["ndcg10"]) for a, b in PAIRS
    }
    return out


def _l1_facts(engine: Any) -> tuple[list[tuple], dict[str, str], Any]:
    """Exactly scripts/eval/retrieval_build_labels.py's grade-1/2 rule."""
    with engine.connect() as c:
        entries = c.execute(
            text(
                "select id, symbols, created_at, embedding_version, refs->>'experience_id' "
                "from journal_entries"
            )
        ).all()
        exps = {
            r[0]: (r[1], r[2], r[3])
            for r in c.execute(text("select id, symbol, sector, as_of from decision_experience"))
        }
        reg = c.execute(
            text(
                "select as_of, min(regime) from replay_equity where regime <> '' "
                "group by as_of order by as_of"
            )
        ).all()
        sectors = {
            r[0]: r[1] for r in c.execute(text("select symbol, sector from instruments")) if r[1]
        }
    days, vals = [r[0] for r in reg], [r[1] for r in reg]

    def regime_at(d: date) -> str | None:
        i = bisect.bisect_right(days, d) - 1
        return vals[i] if i >= 0 else None

    facts = []
    for eid, syms, created_at, version, exp_id in entries:
        if version == HASH_FALLBACK:
            continue
        if exp_id and exp_id in exps:
            sym, sector, trade_day = exps[exp_id]
            facts.append((eid, created_at, {sym}, sector, regime_at(trade_day)))
        else:
            facts.append((eid, created_at, set(syms or ()), None, None))
    return facts, sectors, regime_at


def _coverage(engine: Any, frozen: set[str]) -> dict[str, Any]:
    facts, sectors, regime_at = _l1_facts(engine)
    facts = [f for f in facts if f[0] in frozen]
    rows = [json.loads(line) for line in (EVAL / "v2_coverage_top3.jsonl").open()]
    out: dict[str, Any] = {}
    eligible = 0
    s_eligible = 0
    hit = dict.fromkeys(ARMS, 0)
    useful = dict.fromkeys(ARMS, 0)
    hit_all = dict.fromkeys(ARMS, 0)
    for r in rows:
        sym, as_of = r["symbol"], date.fromisoformat(r["as_of"])
        cutoff = datetime.combine(as_of, dtime(23, 59, 59), tzinfo=UTC)
        q_sector, q_regime = sectors.get(sym), regime_at(as_of)
        grades: dict[str, int] = {}
        for eid, created_at, e_syms, e_sector, e_regime in facts:
            if created_at > cutoff:
                continue
            if sym in e_syms:
                grades[eid] = 2
            elif q_sector and e_sector == q_sector and q_regime and e_regime == q_regime:
                grades[eid] = 1
        for a in ARMS:
            hit_all[a] += bool(r["top3"][a])
        if not grades:
            continue
        eligible += 1
        s_eligible += 2 in grades.values()
        for a in ARMS:
            top = r["top3"][a]
            hit[a] += bool(top)
            useful[a] += any(grades.get(e, 0) >= 1 for e in top)
    out["contexts"] = len(rows)
    out["eligible"] = eligible
    out["eligible_same_stock"] = s_eligible
    for a in ARMS:
        out[a] = {
            "any_hit_all_contexts": [hit_all[a], round(hit_all[a] / len(rows), 4)],
            "any_hit_eligible": [hit[a], round(hit[a] / eligible, 4) if eligible else math.nan],
            "useful_hit_eligible_L1": [
                useful[a],
                round(useful[a] / eligible, 4) if eligible else math.nan,
            ],
        }
    return out


def main() -> None:
    engine = make_engine(Settings().database_url)
    frozen = set(json.loads((EVAL / "corpus_v2.json").read_text())["ids"])
    l1: dict[str, dict[str, int]] = json.loads((EVAL / "qrels.json").read_text())
    l2: dict[str, dict[str, int]] = json.loads((EVAL / "qrels_v2_independent.json").read_text())
    results: dict[str, Any] = {
        "preregistration": "docs/preregistration_eval_retrieval_v2.md",
        "labels": {
            "L1": "evals/retrieval/qrels.json",
            "L2": "evals/retrieval/qrels_v2_independent.json",
        },
    }
    for corpus in ("frozen", "current"):
        rows = [json.loads(line) for line in (EVAL / f"v2_rankings_{corpus}.jsonl").open()]
        x = [r for r in rows if r["stratum"] == "X"]
        results[corpus] = {
            "corpus_size": rows[0]["corpus_size"],
            "L1": {
                "all": _scores(rows, l1),
                "S_same_stock": _scores([r for r in rows if r["stratum"] == "S"], l1),
                "X_cross_stock": _scores(x, l1),
            },
            "L2_X_cross_stock": _scores([r for r in x if r["query_id"] in l2], l2),
            "latency_ms": {
                a: {
                    "p50": round(float(np.percentile([r["latency_ms"][a] for r in rows], 50)), 1),
                    "p95": round(float(np.percentile([r["latency_ms"][a] for r in rows], 95)), 1),
                }
                for a in ARMS
            },
        }
    grades = [g for q in l2.values() for g in q.values()]
    results["L2_grade_distribution"] = {str(g): grades.count(g) for g in (0, 1, 2)}
    results["L2_queries"] = len(l2)
    hc = json.loads((EVAL / "v2_handcheck_grades.json").read_text())["grades"]
    mine, theirs = [], []
    for key, g in sorted(hc.items()):
        qid, eid = key.split("|", 1)
        if eid in l2.get(qid, {}):
            mine.append(int(g))
            theirs.append(int(l2[qid][eid]))
    results["handcheck_agreement"] = {
        "pairs": len(mine),
        "exact": round(sum(a == b for a, b in zip(mine, theirs, strict=True)) / len(mine), 4)
        if mine
        else math.nan,
        "within_one": round(
            sum(abs(a - b) <= 1 for a, b in zip(mine, theirs, strict=True)) / len(mine), 4
        )
        if mine
        else math.nan,
        "quadratic_weighted_kappa": round(weighted_kappa(mine, theirs), 4),
        "confusion_handcheck_vs_grader": {
            f"{a}->{b}": sum(1 for x, y in zip(mine, theirs, strict=True) if (x, y) == (a, b))
            for a in (0, 1, 2)
            for b in (0, 1, 2)
        },
    }
    results["coverage"] = _coverage(engine, frozen)
    (EVAL / "results_v2.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["coverage"], indent=2))


if __name__ == "__main__":
    main()
