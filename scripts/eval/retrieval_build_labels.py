"""Build the Phase 2 retrieval queries + graded relevance labels.

Implements docs/preregistration_eval_retrieval.md exactly; touches no
retriever. Outputs evals/retrieval/queries.jsonl and evals/retrieval/qrels.json.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine

RUN_ID = "replay-63220a3a5e9a"
SEED = 20261003
PER_STRATUM = 30
MAX_PER_SYMBOL = 2
HASH_FALLBACK = "hash-fallback"
OUT = Path("evals/retrieval")


def end_of_day(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=UTC)


def main() -> None:
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        entries = c.execute(
            text(
                "select id, kind, symbols, created_at, embedding_version, refs->>'experience_id' "
                "from journal_entries"
            )
        ).all()
        exps = {
            r[0]: (r[1], r[2], r[3])
            for r in c.execute(text("select id, symbol, sector, as_of from decision_experience"))
        }
        regime_rows = c.execute(
            text(
                "select as_of, min(regime) from replay_equity where regime <> '' "
                "group by as_of order by as_of"
            )
        ).all()
        sectors = {
            r[0]: r[1] for r in c.execute(text("select symbol, sector from instruments")) if r[1]
        }
        contexts = c.execute(
            text(
                "select symbol, as_of, context_text from replay_contexts "
                "where run_id = :r and stage = 'dossier' order by as_of, symbol"
            ),
            {"r": RUN_ID},
        ).all()

    regime_days = [r[0] for r in regime_rows]
    regime_vals = [r[1] for r in regime_rows]

    def regime_at(d: date) -> str | None:
        i = bisect.bisect_right(regime_days, d) - 1
        return regime_vals[i] if i >= 0 else None

    # Entry facts: from the linked experience; an unlinked thesis uses its symbols.
    facts = []
    for eid, _kind, syms, created_at, version, exp_id in entries:
        if version == HASH_FALLBACK:
            continue
        if exp_id and exp_id in exps:
            sym, sector, trade_day = exps[exp_id]
            facts.append((eid, created_at, {sym}, sector, regime_at(trade_day)))
        else:
            facts.append((eid, created_at, set(syms or ()), None, None))

    seen: set[tuple[str, date]] = set()
    eligible: dict[str, list[dict]] = {"S": [], "X": []}
    for sym, as_of, ctx in contexts:
        if (sym, as_of) in seen:
            continue
        seen.add((sym, as_of))
        cutoff = end_of_day(as_of)
        q_sector, q_regime = sectors.get(sym), regime_at(as_of)
        grades: dict[str, int] = {}
        pool = 0
        for eid, created_at, e_syms, e_sector, e_regime in facts:
            if created_at > cutoff:
                continue
            pool += 1
            if sym in e_syms:
                grades[eid] = 2
            elif (
                q_sector is not None
                and e_sector == q_sector
                and q_regime is not None
                and e_regime == q_regime
            ):
                grades[eid] = 1
        if not grades:
            continue
        stratum = "S" if 2 in grades.values() else "X"
        eligible[stratum].append(
            {
                "symbol": sym,
                "as_of": as_of.isoformat(),
                "sector": q_sector,
                "regime": q_regime,
                "stratum": stratum,
                "pool_size": pool,
                "context_sha256": hashlib.sha256(ctx.encode()).hexdigest(),
                "text": ctx,
                "grades": grades,
            }
        )

    rng = np.random.default_rng(SEED)
    queries, qrels = [], {}
    for stratum in ("S", "X"):
        rows = eligible[stratum]
        per_symbol: Counter[str] = Counter()
        for idx in rng.permutation(len(rows)):
            row = rows[int(idx)]
            if per_symbol[row["symbol"]] >= MAX_PER_SYMBOL:
                continue
            per_symbol[row["symbol"]] += 1
            qid = f"{stratum}{sum(1 for q in queries if q['stratum'] == stratum) + 1:02d}"
            grades = row.pop("grades")
            queries.append({"query_id": qid, **row})
            qrels[qid] = grades
            if per_symbol.total() >= PER_STRATUM:
                break
        print(f"stratum {stratum}: {len(rows)} eligible -> {per_symbol.total()} sampled")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queries.jsonl").open("w") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")
    (OUT / "qrels.json").write_text(json.dumps(qrels, indent=1, sort_keys=True))
    n_rel = [sum(1 for g in v.values() if g >= 1) for v in qrels.values()]
    n_g2 = [sum(1 for g in v.values() if g == 2) for v in qrels.values()]
    print(
        f"{len(queries)} queries | relevant per query: median {int(np.median(n_rel))} "
        f"(range {min(n_rel)}-{max(n_rel)}) | grade-2 per S query: median "
        f"{int(np.median([g for g, q in zip(n_g2, queries, strict=True) if q['stratum'] == 'S']))}"
    )


if __name__ == "__main__":
    main()
