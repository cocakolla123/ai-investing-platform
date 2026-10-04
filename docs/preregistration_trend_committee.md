# Pre-registration — trend rule + committee ("quant mixed with committee")

Written and committed before the run. Analysis script
`scripts/eval/trend_committee_analyze.py` is committed with this file.

## Question

The user wants the live system to be "quant mixed with committee": the quant
leg picks the shortlist, the LLM committee decides. The trend rule replay
(`docs/trend_rule_2026-10-03.md`) was inconclusive on its own. This asks
whether **the committee makes better decisions on a trend-rule shortlist
than on the v8 shortlist it uses today**, and how much of any difference
comes from the shortlist versus the committee's choices within it.

## Arms

| arm | quant leg | source |
|---|---|---|
| **trend** | `trend-rule-v1` | NEW run, this pre-registration |
| **v8** | `campaign-v8` | Phase 3's **vector** run `replay-5181496084db` |
| **v8-rep** | `campaign-v8` | Phase 3's **vector-rep** run `replay-0ca99da87ddd` |

The trend arm copies Phase 3's settings exactly except for the model and the
code version:

- `--mode committee --start 2026-08-03 --end 2026-08-21` (15 sessions)
- `--deep-dives-per-day 4 --universe-tier 1 --discovery off`
- `--quant-model-version trend-rule-v1 --no-require-shipping-bundle`
- `--no-macro-digest --no-within-run-learning --backend claude_code`
- `--max-usd 250`
- `--code-version trend-committee-6f6d039`
- `CLAUDE_CODE_ARBITER_FALLBACK=""` (vector retrieval, sequential advocates,
  as deployed)

Two v8 replicates already exist, so the comparison carries its own noise
floor. **Volume:** 15 × 4 × 4 = **240 model calls** on the subscription, $0
metered. The user approved this on 2026-10-03.

**Known difference.** The v8 runs predate the bar-scale fix (1c6c04d), so
their shortlists were built on the old Tiingo open/high/low. Re-running v8
would double the cost. Outcomes for every arm are measured on the fixed
bars. The shortlist overlap between arms is reported.

## Metrics

All use excess20: return from the next open to the 20th session close, minus
SPY over the same span, on the fixed canonical bars.

1. **Buy quality (primary):** mean excess20 of BUY/ADD verdicts. This is
   what the live system would actually buy.
2. **Decision quality:** mean signed excess20 over directional verdicts
   (Phase 3's metric).
3. **Shortlist quality:** mean excess20 of every shortlisted name,
   whatever the committee said. This is the quant leg alone.
4. **Committee lift:** buy quality minus shortlist quality, per arm.
5. Brier score, action mix, parse/assembly rates, and shortlist overlap
   (Jaccard) between trend and v8.

The arms buy **different names**, so comparisons are unpaired. CIs come from
a bootstrap that resamples whole **days** (10,000 resamples), because
verdicts on the same day share a market move.

## Decision rule

- **Trend + committee is better** only if the day-clustered 95% CI of
  (trend buy quality − v8 buy quality) excludes zero, in the same direction
  against **both** v8 and v8-rep, AND the difference exceeds |v8 − v8-rep|,
  the replicate noise.
- Otherwise: **no difference shown.**
- Either way, this is one 3-week window with overlapping 20-day outcomes.
  A "better" here would justify the forward shadow carrying more weight, not
  a promotion. Promotion needs the user's approval through the promotion
  pipeline.

## Expectation, stated in advance

About 45 directional verdicts per arm, about 15 days of clustering, and one
window. **The most likely result is no difference shown.**
