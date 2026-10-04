# Phase 4 result — tracing, latency, cost, and running bull and bear in parallel

Pre-registration: `docs/preregistration_eval_tracing.md` (2615995), committed
before the runs. Machinery: ec9b97e. Analysis:
`scripts/eval/phase4_analyze.py` → `evals/tracing/results_phase4.json`.

| arm | run id | calls | decisions |
|---|---|---|---|
| seq (bull then bear, as deployed) | replay-44b04a0f2c2e | 80 | 20 |
| par (`--parallel-advocates`) | replay-b5888959bef6 | 82 | 20 |

The arms ran one after the other on a Saturday, with no other Claude Code
process running. Metered cost was $0.

**Verdict:**

- The optimization **works**, but the gain is smaller than predicted.
- **Quality is not shown to hold** under the pre-registered rule: it misses
  by one flip and by one recovered retry.
- **Recommendation: leave `COMMITTEE_PARALLEL_ADVOCATES` off in live.**

## Latency per call (both arms)

| role | model | p50 | p95 |
|---|---|---|---|
| dossier | Sonnet 5 | 59–64 s | 77–79 s |
| bull | Sonnet 5 | 41–42 s | 50–54 s |
| bear | Sonnet 5 | 48–51 s | 57–64 s |
| arbiter | Fable 5 | 87–90 s | 106–109 s |
| **all** | | **53 s** | **97 s** |

Per-call latency was the same in both arms. Running two at a time did not
slow either call.

## Per decision

| | seq | par |
|---|---|---|
| **wall clock per decision**, p50 / p95 (primary) | 270 s / 273 s | **232 s / 264 s** |
| compute per decision (sum of call latencies), p50 / p95 | 244 s / 276 s | 246 s / 288 s |
| advocate stage per day (4 dives) | 338–376 s | 185–244 s |
| metered cost | $0.00 | $0.00 |
| API-equivalent cost (the CLI's own figure) | $0.89 | $0.93 |
| tokens in / out | 43.7k / 20.7k | 44.9k / 21.6k |
| retries, schema failures | 0, 0 | 2, 2 (both recovered on retry) |

The API-equivalent cost is what these calls would cost on the metered API.
It shows what the subscription is saving: about $0.90 per decision, or
roughly $3.60 a night at 4 dives.

## Optimization rule: **met**

Parallel was faster on **5 of 5 days**. Mean change in wall clock per
decision: **−10.9%** (95% CI −16.1% to −5.2%).

The advocate stage itself shrank by about 43%. The pre-registered
expectation of 20–35% overall was too high: the arbiter (Fable, ~90 s ×
4 dives) and the dossiers run one at a time and dominate the day. That is
where any further speed-up has to come from.

## Quality rule: **not met**

The pre-registered rule had three parts:

1. **par vs seq action flips: 6 of 20** (30%). At Phase 3's 13.8% noise
   rate, 6 or more flips has p = 0.048, just under the 0.05 line.
   For comparison:
   - seq vs Phase 3's identical vector run flipped 3 of 20;
   - par vs that run flipped 5 of 20 (p = 0.13).
2. **Schema-failure rate: par 2/82 (2.4%)**, above the 1-in-60 (1.7%) bar.
   - One failure was an **arbiter** call. The arbiter runs after both
     advocates whatever the setting, so this change cannot have caused it.
   - The other was a bear call.
   - Both succeeded on retry. All 20 decisions were assembled in both arms.
3. Assembly: 100% in both arms. **Passed.**

The flips are most likely sampling noise: the change affects no input the
models see, and seq vs Phase 3 shows the same kind of drift. But the rule
was written in advance so that "most likely noise" would not get to
override it, and it failed narrowly on two counts.

Descriptive only, on the same 20 dives:

| run | Brier | decision quality (mean signed excess20) |
|---|---|---|
| seq | 0.198 | +14.6% |
| par | 0.203 | +13.4% |
| Phase 3 vector | 0.182 | +16.6% |

None of these differences is meaningful at this size.

## Recommendation

**Keep `COMMITTEE_PARALLEL_ADVOCATES=false`.** Live gains about 11% of
wall clock, roughly 2½ minutes on an ~18-minute nightly committee. That
isn't worth shipping a change whose quality check failed, even narrowly.

If speed ever matters, the bigger levers are:

- the arbiter: about 35% of the time, on the slowest model;
- running the four dossiers concurrently.

Either needs its own pre-registered test.

## What tracing now gives every run

Every model call writes one `cost_ledger` row (migration 0023) with:

- role and request id;
- prompt version hash;
- the journal and experience ids retrieved into its context;
- latency;
- attempt number;
- whether its output validated;
- the API-equivalent cost.

Each replay day also records wall-clock seconds per stage in
`validation.jsonl`. Live committee calls fill the same columns, except for
the request id and the retrieved ids: only the replay path passes those through today.
