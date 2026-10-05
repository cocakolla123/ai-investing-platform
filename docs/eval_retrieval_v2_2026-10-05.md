# Retrieval evaluation v2: structured filter arms, independent labels

Pre-registration: `docs/preregistration_eval_retrieval_v2.md`, commit
**fc21a4e**, committed before any v2 arm ran. Commits:

- **694799f:** rankings, pool, and the blind hand check (before any grader
  call);
- **b935972:** L2 labels (before any metric);
- **results:** this file's commit.

Numbers come from `scripts/eval/retrieval_v2_metrics.py` →
`evals/retrieval/results_v2.json`.

**Verdict, by the pre-registered rule:**

- On cross-stock queries under the independent labels (L2), both structured
  arms beat production as deployed:
  - fallback: +0.301 nDCG@10, 95% CI +0.201 to +0.409;
  - boost: +0.231, 95% CI +0.164 to +0.305.
- **That gain is coverage, not structure.**
  - Production returns nothing for every cross-stock query, so any arm that
    returns something wins.
  - Against plain vector search, neither structured arm is distinguishable
    from zero:
    - fallback: +0.109, CI −0.032 to +0.247;
    - boost: +0.038, CI −0.055 to +0.134.
  - Plain BM25 keyword scores the same as the fallback on L2 (0.303 vs
    0.301).
- The rule's conditions for recommending a production change are met for
  the structured fallback. The recommendation is below. **Nothing was turned
  on.**

## Setup (as pre-registered)

**Corpus.** The 166 entries up to 2026-10-03 04:10:31 UTC
(`evals/retrieval/corpus_v2.json`). The current corpus was also 166 at
freeze time. Rankings on the two are identical, so every number below holds
for both.

- Only the 113 lessons and reviews are reachable. All theses post-date
  every query.
- `regime_at_write` of those 113: risk_on 96, neutral 17.

**Queries.** The 41 stripped queries: 11 same-stock (S), 30 cross-stock (X).
`as_of` is enforced, and every list is asserted to sit inside its pool.

- Query regime at the decision date: risk_on 41 of 41.
- All 699 campaign contexts: risk_on 627, neutral 72.

**Arms.**

- **v1 arms, unchanged.** They reproduce v1's stripped numbers exactly,
  checked metric by metric.
- **structured_fallback:** the ticker filter; if that is empty, same-sector
  entries ranked by dossier cosine.
- **structured_boost:** `rank_similar` boosts on the vector top 50.

**Labels.**

- **L1** is v1's mechanical rule (grade 2 = same symbol; grade 1 = same
  sector and same regime). As stated in advance, the fallback is partly
  scored by agreement with that rule.
- **L2** is independent and covers X only: a blind grader (Claude Sonnet 5)
  answered "useful precedent?" for each of the 787 pooled pairs.
  - 786 were graded (one skipped).
  - Grades: 555 × 0, 177 × 1, 54 × 2.
  - The run paused at 22 of 30 queries on a plan-usage limit and resumed.
  - Cost: 30 calls, no retries, $0 metered, $2.36 API-equivalent (per the
    cost ledger, role `eval_grader`).

## Results (frozen corpus)

**nDCG@10 / Recall@5 / MRR**

| arm | L1 S (n=11) | L1 X (n=30) | **L2 X (n=30)** | L1 all (n=41) |
|---|---|---|---|---|
| keyword (BM25) | .564 / .457 / .740 | .327 / .215 / .536 | **.303** / .167 / .396 | .390 / .280 / .591 |
| vector | .635 / .508 / .909 | .127 / .055 / .197 | **.193** / .137 / .380 | .264 / .176 / .388 |
| hybrid (RRF) | .621 / .492 / .849 | .222 / .096 / .315 | **.234** / .160 / .450 | .329 / .202 / .458 |
| production template, no filter | .681 / .500 / 1.000 | .155 / .093 / .283 | **.106** / .078 / .288 | .296 / .202 / .475 |
| **production as deployed** (ticker filter) | .984 / .919 / 1.000 | .000 (empty on 30/30) | **.000** (empty) | .264 / .247 / .268 |
| structured_fallback | .984 / .919 / 1.000 | .766 / .489 / .911 | **.301** / .172 / .381 | .824 / .605 / .935 |
| structured_boost | .834 / .677 / 1.000 | .479 / .304 / .719 | **.231** / .123 / .461 | .574 / .404 / .794 |

**Paired nDCG@10 differences, 95% bootstrap CI**

| pair | L1 X | **L2 X** | L1 S |
|---|---|---|---|
| fallback − production as deployed | +.766 [.680, .846] | **+.301 [.201, .409]** | 0 (identical by construction) |
| boost − production as deployed | +.479 [.397, .558] | **+.231 [.164, .305]** | −.151 [−.204, −.092] |
| fallback − vector | +.638 [.552, .725] | **+.109 [−.032, .247]** | +.349 [.281, .420] |
| boost − vector | +.352 [.257, .442] | **+.038 [−.055, .134]** | +.198 [.110, .294] |
| hybrid − keyword | −.105 [−.153, −.057] | −.070 [−.123, −.018] | +.057 [−.063, .185] |
| vector − keyword | −.199 [−.259, −.140] | −.110 [−.182, −.035] | +.071 [−.119, .271] |

**Latency per query, p50 / p95 in ms, including query embedding**

| arm | p50 / p95 |
|---|---|
| keyword | 1.0 / 1.4 |
| vector | 19.7 / 24.2 |
| hybrid | 20.9 / 25.4 |
| production template | 13.0 / 17.3 |
| production as deployed | 8.9 / 14.0 |
| fallback | 27.6 / 31.3 |
| boost | 19.8 / 24.3 |

**Coverage, top 3 (what production keeps)**

| arm | any hit, all 699 contexts | any hit, 329 eligible | L1-useful hit, 329 eligible |
|---|---|---|---|
| production as deployed | 64 (9.2%) | **64 (19.5%)** (reproduces v1) | 64 (19.5%) |
| structured_fallback | 415 (59.4%) | 329 (100%) | 325 (98.8%) |
| structured_boost | 699 (100%) | 329 (100%) | 280 (85.1%) |
| vector / hybrid / keyword | 699 (100%) | 329 (100%) | 130 / 154 / 259 |

**Hand check (40 pairs, graded blind before the grader ran)**

- Exact agreement **80%**; within one grade **97.5%**.
- Quadratic-weighted Cohen's kappa **0.35**: fair agreement.
- The grader was more lenient. In 7 of 8 disagreements it graded higher
  than the hand check:
  - 0 → 1 five times;
  - 0 → 2 once;
  - 1 → 2 once.
- Both graders are Claude models, and the 40-pair file is available for a
  human spot-check.

## Reading it

- **The pre-registered decision.** Both new arms are "better" than production
  as deployed on X under L2. The fallback leaves S unchanged by construction.
  The production-change condition is therefore met for the fallback.
- **What it means.** The ticker filter makes cross-stock retrieval
  impossible: production returns nothing on 30 of 30 X queries. Adding any
  fallback fixes that.
- **What it doesn't mean.** The structure (sector match, metadata boosts)
  was not shown to help:
  - On L2, neither structured arm beats plain vector search.
  - BM25 does as well as the fallback.
  - The big L1 gains are largely the label rule rewarding same-sector
    entries, as stated in advance.
- **The corpus limits what any arm can show.** Most entries are bare outcome
  records ("buy at conviction 0.98 ... resolved win"). Even the lenient L2
  grader called 70% of pooled pairs "not useful". Better lessons would matter
  more than better retrieval.
- **v1 correction.** The v1 report's contamination-corrected table missed a
  result: on stripped queries, keyword beats hybrid across all queries, with
  CI −.114 to −.007, which excludes zero. v2 reproduces this exactly
  (L1 all hybrid − keyword −.062 [−.114, −.007]).

## Recommendation (no change made)

Propose, through the normal shadow and paper path and with the user's
approval, a fallback for empty ticker results in the nightly paper
committee's journal retrieval.

- **The pre-registered candidate:** `structured_fallback`. Same-sector
  entries by dossier cosine. It changes nothing for names with their own
  history.
- **Equally supported:** a plain unfiltered vector or BM25 fallback. The data
  cannot tell these apart from the structured version.
- **A cheaper first step:** shadow-log what the fallback would have retrieved
  each night, without putting it into the committee's context.
- **Larger lever:** improve what lessons say (the 70% "not useful" rate).
