"""Retrieval v2 independent labels, L2 (docs/preregistration_eval_retrieval_v2.md).

Subcommands, run in this order:

    pool    union of each arm's top 10 per X query (frozen rankings) -> v2_pool.json
    sample  8 X queries x 5 pooled candidates for the blind hand check -> v2_handcheck_sample.json
    grade   one grader call per X query -> v2_grades_raw.jsonl + qrels_v2_independent.json

The grader sees only the stripped dossier text and each candidate's entry
text, shuffled with a fixed seed and keyed C01, C02, and so on. It never sees
the arm, rank, score, entry id or metadata. Calls go through the claude_code
backend (subscription, no API fallback) and are recorded in cost_ledger
with role "eval_grader".
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

from dynamo.config.settings import Settings
from dynamo.persistence.db import make_engine

EVAL = Path("evals/retrieval")
TOP = 10
SAMPLE_SEED = 20261005
SAMPLE_QUERIES, SAMPLE_PER_QUERY = 8, 5

SYSTEM = (
    "You grade how relevant past trading-journal entries are to a new investment "
    "decision. You are an evaluator, not a trader. Be strict and consistent."
)
INSTRUCTION = (
    "For each candidate journal entry above, answer one question: would this past "
    "entry be a useful precedent for the decision this dossier sets up?\n"
    "Grade 2 = directly useful: a similar setup, catalyst, risk or failure mode whose "
    "lesson would inform this decision.\n"
    "Grade 1 = somewhat useful: a general lesson that transfers to this decision.\n"
    "Grade 0 = not useful.\n"
    "Judge substance only. Do not give credit just because a candidate names the same "
    "sector or the same market regime; reward it only if its situation or its lesson "
    "actually bears on this decision. Grade every candidate exactly once, by its key."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["grades"],
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key", "grade"],
                "properties": {
                    "key": {"type": "string"},
                    "grade": {"type": "integer", "enum": [0, 1, 2]},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


def _queries() -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (EVAL / "queries_journal_stripped.jsonl").open()]
    return {q["query_id"]: q for q in rows}


def _texts(ids: list[str]) -> dict[str, str]:
    engine = make_engine(Settings().database_url)
    with engine.connect() as c:
        rows = c.execute(
            text("select id, text from journal_entries where id = any(:i)"), {"i": ids}
        ).all()
    return {r[0]: r[1] for r in rows}


def _shuffled(qid: str, ids: list[str]) -> list[str]:
    seed = int(hashlib.sha256(f"v2-grade:{qid}".encode()).hexdigest()[:8], 16)
    out = sorted(ids)
    random.Random(seed).shuffle(out)
    return out


def cmd_pool() -> None:
    pool: dict[str, list[str]] = {}
    for line in (EVAL / "v2_rankings_frozen.jsonl").open():
        r = json.loads(line)
        if r["stratum"] != "X":
            continue
        ids = {i for ranking in r["rankings"].values() for i in ranking[:TOP]}
        pool[r["query_id"]] = sorted(ids)
    (EVAL / "v2_pool.json").write_text(json.dumps(pool, indent=2))
    print(f"pool: {len(pool)} X queries, {sum(len(v) for v in pool.values())} pairs")


def cmd_sample() -> None:
    pool = json.loads((EVAL / "v2_pool.json").read_text())
    rng = random.Random(SAMPLE_SEED)
    qids = sorted(rng.sample(sorted(pool), SAMPLE_QUERIES))
    queries = _queries()
    picks = {q: sorted(rng.sample(pool[q], min(SAMPLE_PER_QUERY, len(pool[q])))) for q in qids}
    texts = _texts(sorted({i for v in picks.values() for i in v}))
    sample = [
        {
            "query_id": q,
            "query_text": queries[q]["text"],
            "candidates": [{"entry_id": i, "text": texts[i]} for i in picks[q]],
        }
        for q in qids
    ]
    (EVAL / "v2_handcheck_sample.json").write_text(json.dumps(sample, indent=2))
    print(f"hand-check sample: {sum(len(s['candidates']) for s in sample)} pairs")


def cmd_grade() -> None:
    from dynamo.llm.claude_code_client import ClaudeCodeResearchModel, find_claude_binary
    from dynamo.llm.cost_meter import CostMeter, SqlAlchemyLedgerWriter

    settings = Settings()
    binary = find_claude_binary(settings.claude_code_binary)
    assert binary, "the claude binary is required"
    meter = CostMeter(
        SqlAlchemyLedgerWriter(make_engine(settings.database_url)),
        settings.claude_monthly_budget_usd,
    )
    model = ClaudeCodeResearchModel(settings, meter, binary=binary, api_fallback=None)
    pool = json.loads((EVAL / "v2_pool.json").read_text())
    queries = _queries()
    texts = _texts(sorted({i for v in pool.values() for i in v}))
    raw_path = EVAL / "v2_grades_raw.jsonl"
    done = (
        {json.loads(line)["query_id"] for line in raw_path.open()} if raw_path.exists() else set()
    )
    with raw_path.open("a") as fh:
        for qid in sorted(pool):
            if qid in done:
                continue
            order = _shuffled(qid, pool[qid])
            keys = {f"C{n + 1:02d}": eid for n, eid in enumerate(order)}
            context = (
                "# Decision dossier\n" + queries[qid]["text"] + "\n\n# Candidate journal entries\n"
            )
            context += "\n".join(f"[{k}] {texts[e]}" for k, e in keys.items())
            out = model.complete_json(
                role="eval_grader",
                system=SYSTEM,
                context=context,
                instruction=INSTRUCTION,
                schema=SCHEMA,
                model_tier="analyst",
                trace={"role": "eval_grader", "request_id": f"retrieval-v2:{qid}"},
            )
            grades = {
                g["key"]: int(g["grade"]) for g in out.get("grades", []) if g.get("key") in keys
            }
            fh.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "keys": keys,
                        "grades": grades,
                        "missing": sorted(set(keys) - set(grades)),
                        "reasons": {g["key"]: g.get("reason", "") for g in out.get("grades", [])},
                    }
                )
                + "\n"
            )
            fh.flush()
            print(f"{qid}: graded {len(grades)}/{len(keys)}", flush=True)
    qrels: dict[str, dict[str, int]] = {}
    for line in raw_path.open():
        r = json.loads(line)
        qrels[r["query_id"]] = {r["keys"][k]: g for k, g in r["grades"].items()}
    (EVAL / "qrels_v2_independent.json").write_text(json.dumps(qrels, indent=2, sort_keys=True))
    print(f"labels: {len(qrels)} queries, {sum(len(v) for v in qrels.values())} graded pairs")


if __name__ == "__main__":
    {"pool": cmd_pool, "sample": cmd_sample, "grade": cmd_grade}[sys.argv[1]]()
