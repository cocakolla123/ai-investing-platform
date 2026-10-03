"""Lexical ranking and rank fusion for journal retrieval (pure, no I/O).

One implementation shared by the retrieval evaluation
(``scripts/eval/retrieval_run.py``, ``docs/eval_retrieval_2026-10-03.md``) and
the replay committee's ``journal_retrieval="hybrid"`` arm, so the method that
was measured is exactly the method that runs.

* :func:`bm25_rank` — Okapi BM25 (k1 = 1.2, b = 0.75) over lowercase
  alphanumeric tokens with scikit-learn's English stop words removed.
* :func:`rrf` — reciprocal rank fusion, ``score = sum 1 / (k + rank)``,
  k = 60 (Cormack, Clarke & Buettcher 2009). Ties break by id.

Both return ids best-first and are deterministic for identical inputs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
# bge models expect this instruction on QUERIES (not on stored passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def bm25_rank(
    query: str,
    docs: Mapping[str, Sequence[str]],
    *,
    depth: int = 50,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[str]:
    """Doc ids by BM25 score against ``query``; ``docs`` maps id -> tokens.

    Only positive-scoring documents are returned (a document sharing no term
    with the query is not a match, however short the list).
    """
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs.values()) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs.values():
        df.update(set(d))
    q = Counter(tokens(query))
    scores: dict[str, float] = {}
    for doc_id, d in docs.items():
        tf = Counter(d)
        score = 0.0
        for term, qtf in q.items():
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += qtf * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        scores[doc_id] = score
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [doc_id for doc_id, s in ranked if s > 0][:depth]


def rrf(*rankings: Sequence[str], k: int = RRF_K, depth: int = 50) -> list[str]:
    """Reciprocal rank fusion of best-first id lists."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:depth]
