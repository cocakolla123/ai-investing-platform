"""Run the pre-registered Phase 2 retrieval evaluation.

Reads evals/retrieval/{queries.jsonl,qrels.json} (committed before this script
ever ran) and compares keyword (BM25), vector (pgvector cosine, production
repository path), hybrid (RRF k=60) and the production template query, all
with as_of enforced. Writes evals/retrieval/results_phase2.json.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path

import numpy as np
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.learning.retrieval import BGE_QUERY_PREFIX, bm25_rank, rrf, tokens
from dynamo.llm.embeddings import LocalEmbedder
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.learning import JournalRepository

os.environ.setdefault("HF_HUB_OFFLINE", "1")
EVAL = Path("evals/retrieval")
DEPTH = 50
HASH_FALLBACK = "hash-fallback"


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


def paired_boot(a: list[float], b: list[float], n: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(7)
    diff = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    import sys

    # --stripped: queries with the "## Journal hits" section removed (the
    # post-hoc contamination correction; same labels, same methods).
    stripped = "--stripped" in sys.argv
    qfile = "queries_journal_stripped.jsonl" if stripped else "queries.jsonl"
    suffix = "_stripped" if stripped else ""
    queries = [json.loads(line) for line in (EVAL / qfile).open()]
    qrels: dict[str, dict[str, int]] = json.loads((EVAL / "qrels.json").read_text())
    engine = make_engine(Settings().database_url)
    repo = JournalRepository(engine)
    with engine.connect() as c:
        rows = c.execute(
            text("select id, text, created_at, embedding_version, symbols from journal_entries")
        ).all()
    entries = {r[0]: r for r in rows}
    toks = {r[0]: tokens(r[1]) for r in rows}
    embedder = LocalEmbedder()
    assert embedder.backend == "sentence-transformers", embedder.backend

    methods = ("keyword", "vector", "hybrid", "production", "production_symbol_filter")
    per_query = []
    lat: dict[str, list[float]] = {m: [] for m in methods}
    embed_ms: list[float] = []
    for q in queries:
        qid, as_of = q["query_id"], date.fromisoformat(q["as_of"])
        cutoff = datetime.combine(as_of, dtime(23, 59, 59), tzinfo=UTC)
        pool = {k for k, r in entries.items() if r[2] <= cutoff and r[3] != HASH_FALLBACK}
        grades = qrels[qid]
        ranks: dict[str, list[str]] = {}

        t = time.perf_counter()
        ranks["keyword"] = bm25_rank(q["text"], {k: toks[k] for k in pool})
        kw_ms = 1000 * (time.perf_counter() - t)
        lat["keyword"].append(kw_ms)

        t = time.perf_counter()
        qvec = embedder.embed([BGE_QUERY_PREFIX + q["text"]])[0]
        e_ms = 1000 * (time.perf_counter() - t)
        embed_ms.append(e_ms)
        t = time.perf_counter()
        hits = repo.search(embedding=qvec, top_k=DEPTH, as_of=cutoff)
        ranks["vector"] = [e.id for e, _ in hits]
        vec_ms = e_ms + 1000 * (time.perf_counter() - t)
        lat["vector"].append(vec_ms)

        t = time.perf_counter()
        ranks["hybrid"] = rrf(ranks["keyword"], ranks["vector"])
        lat["hybrid"].append(kw_ms + vec_ms + 1000 * (time.perf_counter() - t))

        template = f"{q['symbol']} investment thesis, lessons and post-trade reviews"
        t = time.perf_counter()
        pvec = embedder.embed([template])[0]
        ranks["production"] = [
            e.id for e, _ in repo.search(embedding=pvec, top_k=DEPTH, as_of=cutoff)
        ]
        lat["production"].append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        ranks["production_symbol_filter"] = [
            e.id
            for e, _ in repo.search(
                embedding=pvec, top_k=DEPTH, as_of=cutoff, symbols=[q["symbol"]]
            )
        ]
        lat["production_symbol_filter"].append(1000 * (time.perf_counter() - t))

        for m, r in ranks.items():  # the guards: nothing outside the as-of pool, ever
            assert set(r) <= pool, (qid, m, sorted(set(r) - pool))
        per_query.append(
            {
                "query_id": qid,
                "stratum": q["stratum"],
                "symbol": q["symbol"],
                "as_of": q["as_of"],
                "returned": {m: len(r) for m, r in ranks.items()},
                **{
                    f"{m}.{name}": fn(r, grades)
                    for m, r in ranks.items()
                    for name, fn in (("ndcg10", ndcg_at), ("recall5", recall_at), ("mrr", mrr))
                },
            }
        )

    def agg(rows: list[dict]) -> dict:
        out = {"n": len(rows)}
        for m in methods:
            out[m] = {
                metric: round(statistics.fmean(r[f"{m}.{metric}"] for r in rows), 4)
                for metric in ("ndcg10", "recall5", "mrr")
            }
            out[m]["empty_results"] = sum(1 for r in rows if r["returned"][m] == 0)
        pairs = [
            ("hybrid", "keyword"),
            ("hybrid", "vector"),
            ("vector", "keyword"),
            ("hybrid", "production_symbol_filter"),
        ]
        out["ndcg10_paired_diff_95ci"] = {
            f"{a}-{b}": [
                round(x, 4)
                for x in paired_boot(
                    [r[f"{a}.ndcg10"] for r in rows], [r[f"{b}.ndcg10"] for r in rows]
                )
            ]
            for a, b in pairs
        }
        return out

    def pct(xs: list[float], p: float) -> float:
        return round(float(np.percentile(xs, p)), 1)

    results = {
        "date": "2026-10-03",
        "preregistration": "docs/preregistration_eval_retrieval.md",
        "labels_commit": "5052574",
        "all": agg(per_query),
        "S_same_stock": agg([r for r in per_query if r["stratum"] == "S"]),
        "X_cross_stock": agg([r for r in per_query if r["stratum"] == "X"]),
        "latency_ms": {m: {"p50": pct(v, 50), "p95": pct(v, 95)} for m, v in lat.items()},
        "query_embedding_ms": {"p50": pct(embed_ms, 50), "p95": pct(embed_ms, 95)},
    }
    (EVAL / f"results_phase2{suffix}.json").write_text(json.dumps(results, indent=2))
    with (EVAL / f"results_phase2{suffix}_per_query.jsonl").open("w") as fh:
        for r in per_query:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
