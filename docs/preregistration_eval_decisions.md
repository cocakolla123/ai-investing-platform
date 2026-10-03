# Pre-registration — does journal retrieval change committee decisions? (Phase 3)

Written and committed before any Phase 3 replay runs. Machinery: `806d954`.
Retrieval findings that shaped the arms: `docs/eval_retrieval_2026-10-03.md`.

## Arms (identical in everything but the journal section of the dossier)

| arm | `--journal-retrieval` | `--replicate` | what the committee sees |
|---|---|---|---|
| **off** | `off` | 0 | no journal hits, no prior-case lines |
| **vector** | `vector` | 0 | production as deployed: template query + ticker filter (hits only for names with past lessons) |
| **hybrid** | `hybrid` | 0 | BM25 + vector RRF over the dossier text, no ticker filter (hits for every name) |
| **vector-rep** | `vector` | 1 | an exact repeat of **vector** — measures run-to-run LLM noise |

Shared settings: `--mode committee --start 2026-08-03 --end 2026-08-21` (15
sessions) `--deep-dives-per-day 4 --universe-tier 1 --discovery off
--quant-model-version campaign-v8 --no-require-shipping-bundle
--no-macro-digest --no-within-run-learning --backend claude_code
--code-version phase3-806d954`, with `CLAUDE_CODE_ARBITER_FALLBACK=""` so no
arm can switch models mid-run (a plan-window exhaustion pauses the run; it
resumes without re-spending). Within-run learning and per-arm macro digests
are off so the arms cannot diverge through their own feedback or through
independently sampled digests. Models are recorded in each run's
`config_json` and checked afterwards.

**Volume:** 4 arms × 15 sessions × 4 deep dives × 4 stages = **960 model
calls**, on the Claude subscription ($0 metered; `--max-usd 250` is an inert
safety cap). The four arms run concurrently so a plan-limit pause hits all of
them at the same time.

## Outcome (computed after the fact from bars, not by the replay)

For a verdict on day *t*: `excess20 = close(t+20)/open(t+1) − SPY close(t+20)/SPY open(t+1)`.
Bullish calls (buy, add) succeed when `excess20 > 0`; bearish calls (avoid,
sell, trim) succeed when `excess20 < 0`; hold and no_action have no direction
and are excluded from the scored metrics (reported as counts).

## Metrics

Primary: **Brier score** of `(conviction, success)` over directional verdicts.

Secondary: expected calibration error (5 equal-width bins); **decision
quality** = mean signed `excess20` over directional verdicts (+ for bullish,
− for bearish); **structured-output pass rate** = parsed / requested per
stage and assembled / dossiers requested, plus key repairs per day (the
per-day validation records); **bull–bear disagreement** = mean
`disagreement_score`; action mix; mean conviction.

## Analysis and decision rule

Comparisons are **paired** on deep dives present in both arms (same day, same
symbol); paired bootstrap, 10,000 resamples, 95% CI. A difference between two
retrieval arms is called **real only if** its CI excludes zero **and** it is
larger than the vector vs vector-rep difference on the same metric (the noise
floor). Secondary analysis, also pre-registered: the **treated subset** —
deep dives where the arm's dossier actually differed from **off** (for
**vector**, names with same-stock lessons).

Validity checks, reported before any result: shortlists identical across
arms; **off** has zero hits; **vector** hits only same-symbol entries;
**hybrid** has hits on every dive; every verdict's arbiter model is the
pinned one; no hit is dated after its dive's `as_of`.

## Expectation, stated in advance

About 60 deep dives per arm, roughly half directional. That detects only
large effects; the most likely result is **no difference distinguishable
from noise**, which will be reported as such.
