# Pre-registration — tracing, latency and cost, and one optimization (Phase 4)

Written and committed before any Phase 4 replay runs. The tracing machinery (`ec9b97e`)
and the optimization are in the commit named below; the analysis script
`scripts/eval/phase4_analyze.py` is committed with this file, so the metrics
below are computed exactly as written here.

## What is measured (tracing)

Every model call now writes one `cost_ledger` row (migration 0023) carrying:

- **role**: dossier, bull, bear or arbiter;
- **request_id**: `<run_id>:<as_of>-<symbol>-<stage>`;
- **prompt_hash**: sha256 of the system prompt, i.e. its version;
- **retrieved_ids**: journal and experience ids in the call's context;
- **tokens**: input, output and cache-read;
- **latency_ms**: wall clock of the CLI call;
- **attempt**: 1 is the first try, 2 a retry;
- **validation_ok**: whether the output parsed with its required keys;
- **equiv_cost_usd**: the API-equivalent cost the Claude Code CLI reports for
  a flat-rate call; metered cost stays $0.

Each replay day's `validation.jsonl` record also carries wall-clock seconds
per stage: contexts, dossier, advocates, arbiter and total.

## The optimization

**Run bull and bear concurrently** (`--parallel-advocates`). Neither advocate
reads the other's brief: both receive the base context plus the dossier, and
only the arbiter sees both. So this changes elapsed time and nothing the
models see. In replay, the bull and bear stage-batches go out on two
threads, two CLI calls at a time instead of one. Live has the same switch
(`COMMITTEE_PARALLEL_ADVOCATES`), off by default. Turning it on in
production is the user's decision after this result.

## Arms

| arm | flag | everything else |
|---|---|---|
| **seq** (before) | none (sequential, as deployed) | shared settings |
| **par** (after) | `--parallel-advocates` | shared settings |

Shared settings are Phase 3's **vector** arm, unchanged except for the
window:

- `--mode committee --start 2026-08-03 --end 2026-08-07` (5 sessions)
- `--deep-dives-per-day 4 --universe-tier 1 --discovery off`
- `--quant-model-version campaign-v8 --no-require-shipping-bundle`
- `--no-macro-digest --no-within-run-learning --backend claude_code`
- `--max-usd 250`
- `--code-version phase4-ec9b97e`
- `CLAUDE_CODE_ARBITER_FALLBACK=""`

**Order and isolation.** **seq** runs to completion first, then **par**.
They never overlap, and no other Claude Code process runs during either.
The nightly committee schedule only fires on market days, and these runs
happen on a weekend. Latency under contention would measure the contention,
not the change.

**Volume.** 2 arms × 5 sessions × 4 dives × 4 stages = **160 model calls**,
plus retries, on the subscription ($0 metered).

## Metrics

Computed by `scripts/eval/phase4_analyze.py`.

**Latency per call.** p50 and p95 of `latency_ms` per role and overall, per
arm. Also: the retry count (rows with attempt > 1) and validation failures
(validation_ok false).

**Latency per decision.** Two numbers, because they answer different
questions:

1. **Wall-clock per decision**: day `seconds.total` / verdicts assembled,
   per day. This is the elapsed time a decision costs and what the
   optimization targets. It is the **primary** metric.
2. **Compute per decision**: the sum of `latency_ms` over the dive's calls
   (all stages, retries included). Parallelism should not reduce this.
   Reporting it shows whether concurrency slowed individual calls.

p50 and p95 are taken across decisions for (2), and across days for (1).

**Cost per decision.** Metered USD (expected $0); input and output tokens;
API-equivalent USD (the CLI's own figure), per assembled decision.

## Decision rules

**The optimization works** if **par**'s wall-clock per decision is lower
than **seq**'s on at least 4 of the 5 days. The reduction is reported as
the mean per-day % change with a bootstrap 95% CI over days (10,000
resamples).

**Quality held**, judged against the Phase 3 noise floor: an identical
re-run flips 13.8% of actions. **par** and **seq** see identical inputs, so
their action differences are pure sampling noise unless something broke. On
~20 paired dives, 6 or more flips has probability 0.048 at a 13.8% flip
rate. So:

- **held** if par-vs-seq flips ≤ 5 of 20 (scaled to the actual pair count:
  the one-sided binomial p at 13.8% must be ≥ 0.05);
- AND both arms assemble verdicts on at least 90% of requested dossiers;
- AND neither arm has a validation-failure rate above Phase 3's worst stage
  (1 failure in 60).

Brier score and decision quality (mean signed 20-session excess vs SPY over
directional verdicts) are reported for **seq**, **par** and Phase 3's
**vector** run on the same 20 dives. At this size they are descriptive only;
no difference in them is called real.

## Expectation, stated in advance

The two advocate calls take about the same time, so overlapping them should
cut the advocate stage roughly in half. Dossier and arbiter are unchanged,
which puts the overall wall-clock reduction at about 20–35% per decision.
Compute per decision should be unchanged or slightly higher.
