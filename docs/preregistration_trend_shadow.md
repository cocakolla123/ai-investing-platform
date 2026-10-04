# Pre-registration — forward shadow of the trend rule

Written and committed before the first shadow score exists. The user
approved this on 2026-10-03, after the trend-rule replay came back
INCONCLUSIVE (`docs/trend_rule_2026-10-03.md`). That replay overlapped the
period the rule was chosen from, so this is the clean test: scores made
before the outcomes exist.

## What runs

Every market night, the `scores_job` (23:00 ET) gains `shadow_quant_scores`.
It scores bundle `trend-rule-v1` on the live tier 1 + 3 universe, using the
same feature path as production, and writes `shadow_quant_scores` (migration
0024).

**Never traded.** Nothing that trades reads that table, and production keeps
its own model (baseline-v0 abstains today). A bundle failure is logged, never
raised, so the shadow can't break the job production depends on. Switch:
`SHADOW_QUANT_BUNDLES` (default `trend-rule-v1`; empty disables).

## Outcome per score

A row's `as_of` is the UTC date of the run, which is the next calendar day
after the evening it was scored.

- **Entry:** the open of the first trading session on or after `as_of`, the
  first moment the score could be acted on.
- **Exit:** the 20th session's close.
- **excess20:** the name's return minus SPY's over the same span, on
  canonical bars (adjusted scale, 1c6c04d).
- **Equities only.** Crypto pairs are excluded: different calendar, not
  benchmarked to SPY.

## Metrics, per scored day

- **Rank IC:** Spearman correlation of `expected_excess_return` with excess20
  across that day's scored names. This is the primary metric.
- **Top-quintile spread:** mean excess20 of the top 20% by score, minus the
  mean of all scored names.

## Decision rule (once, at the decision point)

**Decision point: the first 60 scored days with complete 20-session
outcomes.** Starting 2026-10-05, that is about early January 2027 for
scoring, and outcomes complete about early February 2027.

- **Forward evidence for the rule:** mean daily IC > 0 AND its Newey–West
  t-stat ≥ 2.0. The lag is 19, because consecutive days' 20-session outcomes
  overlap.
- **Inconclusive:** mean IC > 0 and t < 2.0.
- **Against:** mean IC ≤ 0.

`scripts/eval/trend_shadow_report.py` can be run at any time. Before the
decision point its output is labeled **INTERIM**, and no decision is drawn
from it: looking early is allowed, deciding early is not.

Whatever the result, promoting the rule to production, which means trading
it, needs the user's explicit approval through the promotion pipeline. The
pre-registered committee comparison (`docs/preregistration_trend_committee.md`)
is separate evidence and is reported separately.

## Expectation, stated in advance

About 105 equity names per day. With a true IC near Phase 5's 0.04, the daily
IC's standard error is about 0.1. Even with Newey–West, 60 overlapping days
give roughly 3 independent 20-day blocks. **Inconclusive is the most likely
outcome.** A clear pass would need an IC well above what history showed.
