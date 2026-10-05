"""Freeze the retrieval v2 corpus (docs/preregistration_eval_retrieval_v2.md).

Writes evals/retrieval/corpus_v2.json: the cutoff, the frozen entry ids, and
the regime / sector distributions of the corpus and the queries. Read-only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine

EVAL = Path("evals/retrieval")
CUTOFF = "2026-10-03T04:10:31+00:00"  # just after the v1 corpus max created_at (04:10:30.x)
V1_COUNT = 166


def main() -> None:
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        rows = c.execute(
            text(
                "select id, kind, regime_at_write, sectors, embedding_version from journal_entries "
                "where created_at <= cast(:cut as timestamptz) order by id"
            ),
            {"cut": CUTOFF},
        ).all()
        current = c.execute(text("select count(*) from journal_entries")).scalar()
        ctx_regimes = c.execute(
            text(
                "with reg as (select as_of, min(regime) r from replay_equity where regime <> '' "
                "group by as_of) "
                "select (select r from reg where reg.as_of <= rc.as_of order by reg.as_of desc "
                "limit 1) from replay_contexts rc "
                "where rc.run_id = 'replay-63220a3a5e9a' and rc.stage = 'dossier'"
            )
        ).all()
    assert len(rows) == V1_COUNT, f"frozen corpus has {len(rows)} entries, expected {V1_COUNT}"
    queries = [json.loads(line) for line in (EVAL / "queries_journal_stripped.jsonl").open()]
    in_pool = [r for r in rows if r[1] in ("lesson", "review")]  # theses post-date every query
    out = {
        "cutoff": CUTOFF,
        "frozen_count": len(rows),
        "current_count_at_freeze": int(current or 0),
        "ids": [r[0] for r in rows],
        "kinds": dict(Counter(r[1] for r in rows)),
        "embedding_versions": dict(Counter(r[4] for r in rows)),
        "corpus_regime_at_write": dict(Counter(r[2] for r in rows)),
        "query_pool_entries_regime_at_write": dict(Counter(r[2] for r in in_pool)),
        "corpus_with_sectors": sum(1 for r in rows if r[3]),
        "queries_regime_decision_date": dict(Counter(q["regime"] for q in queries)),
        "queries_by_stratum": dict(Counter(q["stratum"] for q in queries)),
        "all_699_contexts_regime": dict(Counter(r[0] for r in ctx_regimes)),
    }
    (EVAL / "corpus_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ids"}, indent=2))


if __name__ == "__main__":
    main()
