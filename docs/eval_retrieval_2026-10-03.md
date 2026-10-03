# Journal retrieval evaluation — 2026-10-03 (Phases 1–2)

Pre-registration: `docs/preregistration_eval_retrieval.md`. Labels committed
before any retriever ran (`5052574`). Results: `evals/retrieval/`.

## Phase 1 — is retrieval real?

| check | result |
|---|---|
| live embedder backend | `sentence-transformers` (BAAI/bge-small-en-v1.5, cached locally) |
| journal entries | 166 (79 lessons, 34 reviews, 53 theses) |
| hash-fallback stamped, before / after | **0 / 0** — nothing to re-embed; the earlier re-embed was genuine |
| stored vector vs recomputed (all 166) | cosine min 1.000000, 0 below 0.999 |
| lessons/reviews dated before their trade's exit | 0 of 113 (min 2 days after the decision) |
| new test | default search hides fallback rows AND post-`as_of` rows together (`bc3c48b`) |

## Phase 2 — which retriever?

41 queries = real committee dossier contexts (11 same-stock "S", 30
cross-stock "X"), median 7 relevant entries each, `as_of` enforced (asserted
on every result list). nDCG@10 with paired-bootstrap 95% CIs:

| method | all (41) | S same-stock (11) | X cross-stock (30) | Recall@5 all | MRR all | p50 / p95 ms |
|---|---|---|---|---|---|---|
| keyword (BM25) | **0.388** | 0.569 | **0.322** | 0.245 | 0.594 | 1.0 / 1.7 |
| vector (pgvector) | 0.264 | 0.635 | 0.127 | 0.176 | 0.388 | 18.5 / 22.5 |
| hybrid (RRF k=60) | 0.344 | 0.674 | 0.222 | 0.204 | 0.488 | 19.4 / 25.4 |
| production template, no filter | 0.296 | 0.681 | 0.155 | 0.202 | 0.475 | 11.8 / 20.7 |
| **production as deployed** (template + ticker filter) | 0.264 | **0.984** | **0.000** — empty for all 30 | 0.247 | 0.268 | 1.8 / 3.6 |

Pre-registered comparisons (CI excludes zero = "better"):

- all: keyword > vector (−0.124 [−0.185, −0.061] for vector−keyword); hybrid > vector (+0.080 [+0.047, +0.113]); hybrid vs keyword not distinguishable (−0.044 [−0.090, +0.003]).
- S: hybrid > keyword (+0.105 [+0.015, +0.204]); hybrid vs vector not distinguishable.
- X: keyword > hybrid > vector.

Query embedding costs 13 ms p50 of the ~19 ms vector latency.

### The construct-validity catch (exploratory, NOT pre-registered)

Every cross-stock query's dossier names its sector, and 28% of the grade-1
entries name the same sector, so the "same sector" label rewards lexical
overlap. Re-running BM25 with the sector name stripped from the query drops
it on stratum X from **0.322 to 0.103**, statistically tied with vector
(0.127; diff −0.025 [−0.071, +0.020]). Keyword's cross-stock "win" was mostly
the label rule, not retrieval skill.

### Query contamination, found after the run (correction)

37 of 41 query texts contained a `## Journal hits` section of "Prior similar
case" lines from the campaign's within-run learning — summaries of earlier
trades, outcomes included, which overlap with the lessons being retrieved.
Re-run with that section stripped (`queries_journal_stripped.jsonl`, same
labels, same methods; `results_phase2_stripped.json`):

| method | all, as registered | all, stripped | S, stripped | X, stripped |
|---|---|---|---|---|
| keyword | 0.388 | 0.390 | 0.564 | 0.327 |
| vector | 0.264 | 0.264 | 0.635 | 0.127 |
| hybrid | 0.344 | 0.329 | 0.621 | 0.222 |

Vector is unchanged (bge reads only the first 512 tokens; the section sits
near the end). The only conclusion that does not survive is "hybrid > keyword
on same-stock" (stripped: +0.057 [−0.062, +0.185]). Everything else holds.

## What this means

1. **Same-stock precedent is a filter problem, already solved.** The ticker
   filter production uses is near-perfect (0.984). Vector search adds nothing.
2. **Cross-stock analogs are not found by any text method** (≈0.1 nDCG once
   the sector shortcut is removed). The useful signal there is structured
   (sector, regime, sleeve) — but `strategy` and `regime_at_write` are NULL on
   all 166 entries and `sector` on all lessons, so the journal cannot filter
   on them. Populating those columns at write time is the cheap fix.
3. **Production returns nothing for most decisions.** Only 64 of 329 eligible
   campaign contexts (19%) had any same-stock lesson; the other 81% get no
   journal precedent at all.
4. Small corpus (166 entries, ~19 symbols with lessons) — wide CIs; these are
   design signals, not final rankings.

These findings set the Phase 3 arms (see its pre-registration): "vector" =
production as deployed, "hybrid" = RRF over the dossier text with no ticker
filter, "off" = no journal hits.
