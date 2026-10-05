# Journal entry metadata: what each field means

`journal_entries` has three metadata columns that retrieval can filter or
boost on. Until 2026-10-05 they were empty on every entry (`strategy` and
`regime_at_write` on all 166, `sectors` on all but 29 reviews), so
structured retrieval could not be tested. This page fixes what each field
means. The write paths and the backfill
(`scripts/journal_backfill_metadata.py`) follow it.

## Definitions

| column | meaning | source | point in time |
|---|---|---|---|
| `strategy` | The sleeve of the trade the entry is about (`long_term`, `tactical`, `crypto`, `ballast`). | The trade's `decision_context.sleeve`, or the linked `decision_experience.sleeve`. | Only if that record existed at the entry's `created_at`. |
| `regime_at_write` | The market regime **on the day the entry was written** (`risk_on`, `neutral`, `risk_off`). | The last persisted per-session regime on or before `created_at`'s date. Nightly entries use `regime_states`; replay-derived entries use the replay's per-session regime (`replay_equity.regime`). | Never a regime dated after the write date. |
| `sectors` | The sectors of the entry's symbols. | `instruments.sector` for each symbol, distinct and sorted. Empty values are skipped. | **Today's classification, not a historical one.** `instruments` is not vintaged: DIS, for example, was "Communication Services" when some experiences were recorded and is "Consumer Discretionary" now. |

### Why regime on the write date, not on the decision date

The column is named `regime_at_write`, and production uses it that way:
`rank_similar` boosts entries whose `regime_at_write` matches the regime
**now**, when a new decision is being made. The question is "was this
lesson written in conditions like today's?". A decision-date regime would
answer a different question, so it would silently change what the column
means. The retrieval labels (`scripts/eval/retrieval_build_labels.py`) use
the regime on the trade's decision date. That is a property of the
evaluation, not of this column, and the retrieval v2 report says so.

The old seed path wrote the experience's decision-date regime into this
column. It was always NULL in practice (see below), so no stored value
changes meaning.

## Write paths

| kind | written by | strategy | regime_at_write | sectors |
|---|---|---|---|---|
| thesis | `_persist_verdict` (orchestration/assets/research.py), nightly | NULL: the sleeve is only assigned later, by the decision context | regime on the write date | from `instruments` |
| review | `post_trade_reviews` (orchestration/assets/learning_ops.py), nightly | `decision_context.sleeve` | regime on the write date | from `instruments` |
| lesson | same as review | same | same | same (new `sectors` parameter on `lesson_entries`) |
| review, lesson | `journal_seed` (replay-derived seeding) | the experience's sleeve | regime on the entry's `created_at` (the trade exit) | the experience's sector, else `instruments` |
| incident | `incident_entry` | not set | not set | not set |

Incidents describe system events, not trades. They have no sleeve, and no
code path writes them (tests only), so they are left alone.

## Why the seeded values were NULL

The seed code passed `strategy=exp.sleeve` and `regime_at_write=exp.regime`
correctly, and the repository inserts both columns. The source rows were
empty: all 928 `decision_experience` rows are replay rows, and all have a
NULL `sleeve` and a NULL `regime`.

- **sleeve:** `build_experience` takes the sleeve only from a decision
  context, and replay never passes one.
- **regime:** replay reads `gw.regime_state()`, which looks in
  `regime_states`. That table only holds nightly rows from 2026-09-11 on, so
  for every 2025 to August 2026 replay date the lookup finds nothing.

The regime replay computes for itself is stored only in
`replay_equity.regime`.

Re-embedding did not erase anything. It updates only `embedding` and
`embedding_version`. Inserts are `ON CONFLICT DO NOTHING`, so re-seeding
could never fix rows that already existed.
