# Pre-registration: retrieval evaluation v2, with structured filter arms

Written and committed before any v2 arm is run and before any v2 label
exists. v1 is `docs/preregistration_eval_retrieval.md` and
`docs/eval_retrieval_2026-10-03.md`. What is new:

- two structured arms, now possible because journal metadata was
  backfilled (docs/journal_metadata.md, 48b744e);
- an independent label set that does not use sector or regime;
- coverage per arm.

## Corpus

- **Frozen:** the 166 entries with `created_at` up to 2026-10-03 04:10:31
  UTC. The ids and distributions are in `evals/retrieval/corpus_v2.json`
  (`scripts/eval/retrieval_v2_freeze.py`). That is 79 lessons, 34 reviews
  and 53 theses, all embedded with sentence-transformers. 158 have
  sectors; `regime_at_write` is 149 risk_on and 17 neutral.
- **Current:** also 166 entries at freeze time.
  - Results are reported on both, as asked.
  - Every query has `as_of` ≤ 2026-08-13, and `as_of` is enforced, so no
    entry written after that date can appear.
  - The two runs are therefore expected to be identical.
  - The only difference from v1's corpus is the backfilled metadata.
- **Effectively searchable:** only the 113 lessons and reviews. All 53
  theses post-date every query.

## Queries

The 41 stripped queries in `evals/retrieval/queries_journal_stripped.jsonl`,
reproduced byte for byte by `scripts/eval/retrieval_strip_queries.py --check`.

- 11 same stock (S), 30 cross stock (X).
- Each text is the dossier cut at "## Journal hits", with no outcome
  context.
- `as_of` is enforced per query (`created_at` ≤ as_of 23:59:59 UTC). Every
  returned list is asserted to be a subset of that pool.

**Regime distribution.**

- Queries at their decision date: risk_on 41 of 41.
- Entries any query can see: risk_on 96, neutral 17.
- All 699 campaign contexts: risk_on 627, neutral 72.

A regime filter would therefore keep nearly everything, so no arm filters
on regime. The boost arm keeps production's regime boost as is.

## Arms (exact; none tuned on labels)

Unchanged from v1 (`scripts/eval/retrieval_run.py`, depth 50):

1. **keyword:** BM25, k1 1.2, b 0.75, over the as-of pool.
2. **vector:** BGE (`BGE_QUERY_PREFIX` + text), pgvector cosine, no filter.
3. **hybrid:** RRF, k=60, of keyword and vector.
4. **production:** the template "`{SYM}` investment thesis, lessons and
   post-trade reviews" embedded, no filter.
5. **production_symbol_filter:** production as deployed. The same template
   vector with `symbols=[SYM]`.

New:

6. **structured_fallback.** Take arm 5's ranking.
   - If it is non-empty, that is the ranking.
   - If it is empty, use the as-of pool entries whose `sectors` contains the
     query symbol's sector (today's `instruments.sector`, the queries file's
     `sector` field), ranked by arm 2's cosine similarity, depth 50.
   - No regime filter (see above). A query whose sector is NULL gets an
     empty ranking.
7. **structured_boost.** Take arm 2's top 50 candidates (no filter) and
   re-rank them with the existing `rank_similar`, using
   `JournalFilters(symbols={SYM}, sectors={query sector}, regime=query
   regime at decision date, strategy=None)`.
   - Each match adds `_FILTER_BOOST` = 0.05 to cosine. This is production's
     constant, unchanged.

## Labels

**L1, mechanical (comparability).** v1's `evals/retrieval/qrels.json`
(5052574), unchanged:

- grade 2 = same symbol;
- grade 1 = same sector AND same regime.

**Stated in advance:** on L1, arm 6 partly measures agreement with the
label rule itself. Its fallback admits only same-sector entries, and every
query is risk_on. Arm 7's sector boost does the same, to a lesser degree.
An L1 gain for these arms is weak evidence.

**L2, independent (X stratum only).**

- **Pool:** for each of the 30 X queries, the union of the top 10 from all
  seven arms, on the frozen corpus.
- **Grader:** Claude Sonnet 5 through the claude_code backend
  (subscription; every call recorded in `cost_ledger` with role
  `eval_grader`).
  - One call per X query, 30 calls in all. The user approved this on
    2026-10-05.
  - The grader sees only the query's stripped dossier text and each
    candidate's entry text. Candidates are shuffled with a fixed seed and
    labelled C01, C02 and so on.
  - It never sees the arm, rank, score, entry id or metadata columns.
- **Rubric:** "Would this past journal entry be a useful precedent for the
  decision this dossier sets up?"
  - 2 = directly useful: a similar setup, catalyst, risk or failure mode
    whose lesson would inform this decision;
  - 1 = somewhat useful: a transferable general lesson;
  - 0 = not useful.
  - The grader is told to judge substance and not to reward a matching
    sector or regime label for its own sake.
  - The exact prompt is in `scripts/eval/retrieval_v2_label.py`.
- **Commit order:** labels are committed before any metric is computed.

**Hand check.**

- **Sample:** 8 X queries × 5 pooled candidates = 40 pairs, drawn with a
  fixed seed before the grader runs.
- **Blind grading:** the session doing this evaluation grades them blind
  with the same rubric, before seeing any grader label. The grades are
  committed with L2.
- **Reported:** exact agreement, within-one agreement, and quadratic-weighted
  Cohen's kappa.
- **Disclosure:** both graders are Claude models. The 40-pair file is there
  for the user to spot-check.

## Metrics

**Retrieval quality:**

- per arm: nDCG@10 (gain 2^g − 1), Recall@5 and MRR, as in v1;
- on L1: for S, X and all;
- on L2: for X.

**Latency:** p50 and p95 per arm, including query embedding wherever the arm
embeds.

**Intervals:** paired bootstrap 95% intervals on nDCG@10 (10,000
resamples, seed 7) for:

- arms 6 and 7 against arm 5 (production as deployed);
- arms 6 and 7 against arm 2;
- the v1 pairs.

**Coverage:** over the 699 dossier contexts of `replay-63220a3a5e9a`, and
over the 329 "eligible" ones (v1's definition: the as-of pool holds at
least one L1-relevant entry).

- **Coverage:** the share where the arm's **top 3** (production keeps 3)
  holds at least one entry. For arm 5 on the eligible set this should
  reproduce 64/329 = 19%.
- **Useful coverage:** the share of eligible contexts whose top 3 holds at
  least one L1 grade-≥1 entry.

The context queries for coverage are built by the same strip rule.

## Decision rule

- On **X**, an arm is **better** than production as deployed (arm 5) only
  if its paired nDCG@10 interval **on L2** excludes zero.
- On **S** there is no independent label set, so S results are descriptive
  only.
- **Recommending a production change** (for example the structured fallback
  in the nightly paper configuration) needs both:
  - arm 6 or 7 is better than arm 5 on X under L2;
  - its S rankings are unchanged or its S nDCG@10 under L1 is not lower.
    Arm 6 is identical to arm 5 on S by construction.
- **Even then, nothing is turned on.** The report writes a recommendation.
  Any change goes through the normal paper and shadow path, with the
  user's approval.
- If neither new arm is better, that null result is the finding. No arm or
  parameter is tuned afterwards.

## Expectation, stated in advance

- **Arm 6** returns something on every X query whose sector has an entry,
  where production returns nothing. It will very likely beat production on
  X under L1, partly by construction.
- **Under L2** the outcome is open. Same-sector lessons may or may not be
  useful precedents.
- **Arm 7** differs from arm 2 only by boosts of 0.05 against cosine gaps.
  It is expected to be close to arm 2.
