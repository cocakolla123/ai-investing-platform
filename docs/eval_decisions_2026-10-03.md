# Phase 3 result — does journal retrieval change committee decisions?

Pre-registration: `docs/preregistration_eval_decisions.md` (a07766e), written
before any run. Code: 806d954. Analysis: `scripts/eval/phase3_analyze.py` →
`evals/decisions/results_phase3.json`, per-verdict rows in
`evals/decisions/phase3_verdicts.csv`.

**Verdict: no difference distinguishable from noise.** Turning retrieval off,
on (vector), or hybrid did not measurably change calibration or decision
quality over 2026-08-03 → 08-21. This is the outcome the pre-registration
named as most likely. It does **not** show retrieval is useless; it shows
that at this sample size (≈60 dives per arm, ≈45 directional) any effect is
smaller than the run-to-run noise of the committee itself.

## Runs

| arm | run id | verdicts | directional scored |
|---|---|---|---|
| off | replay-3bbdbc9cf162 | 57 | 43 |
| vector | replay-5181496084db | 59 | 48 |
| hybrid | replay-8533c2adb458 | 59 | 46 |
| vector-rep | replay-0ca99da87ddd | 59 | 48 |

15 sessions each, `claude_code` backend (metered cost $0; plan usage only),
arbiter `claude-fable-5` on all 234 verdicts (no fallback), analyst
`claude-sonnet-5`.

## Validity checks (all pass)

- Shortlists identical across arms (60 day/symbol pairs each).
- **off** had zero journal hits; **vector** hits were all same-symbol;
  **hybrid** had hits on every dive.
- Zero hits dated after their dive's `as_of`.
- Every verdict's arbiter model was the pinned one.

## Per-arm metrics

| arm | Brier ↓ | ECE (5-bin) ↓ | decision quality (mean signed excess20) | hit rate | mean conviction | disagreement |
|---|---|---|---|---|---|---|
| off | 0.226 | 0.127 | +11.4% | 69.8% | 0.589 | 0.506 |
| vector | 0.227 | 0.103 | +10.3% | 68.8% | 0.588 | 0.505 |
| hybrid | 0.229 | 0.118 | +10.3% | 67.4% | 0.590 | 0.512 |
| vector-rep | 0.226 | 0.105 | +10.3% | 68.8% | 0.588 | 0.510 |

Structured output: dossier and bull parse 100% in every arm; bear and arbiter
98–100%; assembled verdicts per requested dossier 0.95–0.98; quarantines
0–1 per arm; 12–20 key repairs per arm (extra keys stripped, then accepted).

## Paired comparisons (bootstrap 95% CI, 10,000 resamples)

| comparison | Brier diff (n) | signed-excess diff | action flip rate |
|---|---|---|---|
| hybrid − off | −0.007 [−0.020, +0.002] (37) | +0.009 [0.000, +0.027] | 25.0% |
| vector − off | −0.005 [−0.016, +0.004] (41) | +0.008 [0.000, +0.024] | 19.6% |
| hybrid − vector | −0.000 [−0.005, +0.004] (43) | 0.000 | 15.5% |
| **vector − vector-rep (noise floor)** | +0.000 [−0.004, +0.004] (44) | 0.000 | **13.8%** |
| treated subset: vector − off | −0.013 [−0.036, +0.002] (19) | +0.017 [0.000, +0.052] | 14.3% (n 21) |

No Brier CI excludes zero. The signed-excess CIs touch zero at their lower
bound; that comparison is nearly degenerate. Both arms usually say "buy" on
the same name, so the paired difference is exactly zero except on the rare
dives where one arm is bearish. Under the pre-registered rule, nothing is
called real.

The most informative number is the **noise floor**. Re-running the identical
vector arm changed 13.8% of actions. Retrieval vs off changed 19.6% (vector)
and 25% (hybrid). That is somewhat above the floor, but with about 56 pairs
the gap is a handful of dives. Of the 11 vector-vs-off flips, the
retrieval-on side was right about as often as it was wrong (table in the
analysis output).

## Exploratory (not pre-registered; hypotheses only)

- Within each arm, the committee's **buys** averaged +10–12% excess over the
  next 20 sessions, against roughly 0% for the names it passed on. The whole
  shortlist averaged +8.2% (median +1.7%; a few large winners such as META
  +34% and TSLA +17% dominate). This favors the committee separating names
  within the shortlist, but:
  - only 11–15 names were passed on per arm;
  - the 15 overlapping 20-day windows share most of their days;
  - this is one three-week window.

  Treat it as a hypothesis for a longer, pre-registered window, not as
  evidence.
- The committee says "buy" to 70–80% of shortlisted names. Most of the
  decision-quality number is therefore whatever the shortlist was worth in
  August, not the committee's choice.

## Deviations

None from the protocol. No arm hit a plan-usage limit; each ran its 15 days
in one pass. Of 60 shortlisted dives, **off** produced 57 verdicts and the
other arms 59. The gaps are stage failures (two logged schema errors: a
bear rebuttal in **off**, an advocate brief in **vector-rep**; the rest are
the parse-rate shortfalls above), handled as designed. Paired comparisons
use only dives present in both arms.
