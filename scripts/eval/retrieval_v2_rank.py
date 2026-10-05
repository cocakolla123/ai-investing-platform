"""Retrieval v2 rankings (docs/preregistration_eval_retrieval_v2.md). NO METRICS.

Runs the seven pre-registered arms and writes ranked entry ids only:

- evals/retrieval/v2_rankings_{frozen,current}.jsonl: the 41 stripped queries,
  depth 50, with per-arm latency;
- evals/retrieval/v2_coverage_top3.jsonl: the top 3 per arm for all 699
  dossier contexts of replay-63220a3a5e9a (frozen corpus).

Every returned list is asserted to be inside the query's as-of pool.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.learning.journal import JournalFilters, rank_similar
from dynamo.learning.retrieval import BGE_QUERY_PREFIX, bm25_rank, rrf, tokens
from dynamo.llm.embeddings import LocalEmbedder
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.learning import JournalRepository

os.environ.setdefault("HF_HUB_OFFLINE", "1")
EVAL = Path("evals/retrieval")
DEPTH = 50
HASH_FALLBACK = "hash-fallback"
RUN = "replay-63220a3a5e9a"
MARKER = "## Journal hits"
ARMS = (
    "keyword",
    "vector",
    "hybrid",
    "production",
    "production_symbol_filter",
    "structured_fallback",
    "structured_boost",
)


def _cutoff(as_of: date) -> datetime:
    return datetime.combine(as_of, dtime(23, 59, 59), tzinfo=UTC)


class Ranker:
    def __init__(self, corpus_ids: set[str] | None) -> None:
        engine = make_engine(Settings().database_url)
        self.repo = JournalRepository(engine)
        with engine.connect() as c:
            rows = c.execute(
                text("select id, text, created_at, embedding_version, sectors from journal_entries")
            ).all()
        self.entries = {r[0]: r for r in rows if corpus_ids is None or r[0] in corpus_ids}
        self.corpus = set(self.entries)
        self.toks = {k: tokens(r[1]) for k, r in self.entries.items()}
        self.embedder = LocalEmbedder()
        assert self.embedder.backend == "sentence-transformers", self.embedder.backend

    def pool(self, cutoff: datetime) -> set[str]:
        return {k for k, r in self.entries.items() if r[2] <= cutoff and r[3] != HASH_FALLBACK}

    def _search(self, vec: Any, cutoff: datetime, **kw: Any) -> list[tuple[Any, float]]:
        hits = self.repo.search(embedding=vec, top_k=10_000, as_of=cutoff, **kw)
        return [(e, s) for e, s in hits if e.id in self.corpus]

    def rank(
        self, q: dict[str, Any], timed: dict[str, list[float]] | None = None
    ) -> dict[str, list[str]]:
        cutoff = _cutoff(date.fromisoformat(q["as_of"]))
        pool = self.pool(cutoff)
        sym, sector, regime = q["symbol"], q.get("sector"), q.get("regime")
        r: dict[str, list[str]] = {}
        ms: dict[str, float] = {}

        def clock(name: str, fn: Callable[[], Any], extra: float = 0.0) -> Any:
            t = time.perf_counter()
            out = fn()
            ms[name] = extra + 1000 * (time.perf_counter() - t)
            return out

        r["keyword"] = clock(
            "keyword", lambda: bm25_rank(q["text"], {k: self.toks[k] for k in pool})
        )
        qvec = clock("_embed", lambda: self.embedder.embed([BGE_QUERY_PREFIX + q["text"]])[0])
        vec_hits = clock("_vsearch", lambda: self._search(qvec, cutoff))
        ms["vector"] = ms["_embed"] + ms["_vsearch"]
        r["vector"] = [e.id for e, _ in vec_hits[:DEPTH]]
        r["hybrid"] = clock(
            "hybrid", lambda: rrf(r["keyword"], r["vector"]), ms["keyword"] + ms["vector"]
        )
        template = f"{sym} investment thesis, lessons and post-trade reviews"
        pvec = clock("_pembed", lambda: self.embedder.embed([template])[0])
        r["production"] = [
            e.id
            for e, _ in clock("production", lambda: self._search(pvec, cutoff), ms["_pembed"])[
                :DEPTH
            ]
        ]
        r["production_symbol_filter"] = [
            e.id
            for e, _ in clock(
                "production_symbol_filter",
                lambda: self._search(pvec, cutoff, symbols=[sym]),
                ms["_pembed"],
            )[:DEPTH]
        ]

        def fallback() -> list[str]:
            if r["production_symbol_filter"]:
                return r["production_symbol_filter"]
            if not sector:
                return []
            return [e.id for e, _ in vec_hits if sector in (e.sectors or ())][:DEPTH]

        # Latency: production's cost, plus the vector search only when the
        # fallback actually triggers.
        r["structured_fallback"] = clock(
            "structured_fallback",
            fallback,
            ms["production_symbol_filter"]
            + (0.0 if r["production_symbol_filter"] else ms["vector"]),
        )

        def boost() -> list[str]:
            filters = JournalFilters(
                symbols=frozenset({sym}),
                sectors=frozenset({sector}) if sector else frozenset(),
                regime=regime,
            )
            return [x.entry.id for x in rank_similar(vec_hits[:DEPTH], filters, top_k=DEPTH)]

        r["structured_boost"] = clock("structured_boost", boost, ms["vector"])
        for arm, ids in r.items():  # the guard: nothing outside the as-of pool, ever
            assert set(ids) <= pool, (q.get("query_id"), arm, sorted(set(ids) - pool))
        if timed is not None:
            for arm in ARMS:
                timed.setdefault(arm, []).append(ms[arm])
        return r


def _query_runs(corpus_ids: set[str] | None, label: str) -> None:
    ranker = Ranker(corpus_ids)
    queries = [json.loads(line) for line in (EVAL / "queries_journal_stripped.jsonl").open()]
    with (EVAL / f"v2_rankings_{label}.jsonl").open("w") as fh:
        for q in queries:
            timed: dict[str, list[float]] = {}
            ranks = ranker.rank(q, timed)
            fh.write(
                json.dumps(
                    {
                        "query_id": q["query_id"],
                        "stratum": q["stratum"],
                        "symbol": q["symbol"],
                        "as_of": q["as_of"],
                        "corpus_size": len(ranker.corpus),
                        "rankings": ranks,
                        "latency_ms": {a: v[0] for a, v in timed.items()},
                    }
                )
                + "\n"
            )
    print(f"{label}: {len(queries)} queries ranked, corpus {len(ranker.corpus)}")


def _coverage_runs(corpus_ids: set[str]) -> None:
    ranker = Ranker(corpus_ids)
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        rows = c.execute(
            text(
                "select rc.symbol, rc.as_of, rc.context_text, i.sector "
                "from replay_contexts rc left join instruments i on i.symbol = rc.symbol "
                "where rc.run_id = :r and rc.stage = 'dossier' order by rc.as_of, rc.symbol"
            ),
            {"r": RUN},
        ).all()
        reg = c.execute(
            text(
                "select as_of, min(regime) from replay_equity where regime <> '' "
                "group by as_of order by as_of"
            )
        ).all()
    from dynamo.learning.journal_meta import regime_on_or_before

    history = [(r[0], str(r[1])) for r in reg]
    with (EVAL / "v2_coverage_top3.jsonl").open("w") as fh:
        for sym, as_of, ctx, sector in rows:
            cut = ctx.find(MARKER)
            q = {
                "symbol": sym,
                "as_of": as_of.isoformat(),
                "text": ctx if cut < 0 else ctx[:cut],
                "sector": sector,
                "regime": regime_on_or_before(history, as_of),
            }
            ranks = ranker.rank(q)
            fh.write(
                json.dumps(
                    {
                        "symbol": sym,
                        "as_of": q["as_of"],
                        "top3": {a: v[:3] for a, v in ranks.items()},
                    }
                )
                + "\n"
            )
    print(f"coverage: {len(rows)} contexts ranked")


def main() -> None:
    frozen = set(json.loads((EVAL / "corpus_v2.json").read_text())["ids"])
    _query_runs(frozen, "frozen")
    _query_runs(None, "current")
    _coverage_runs(frozen)


if __name__ == "__main__":
    main()
