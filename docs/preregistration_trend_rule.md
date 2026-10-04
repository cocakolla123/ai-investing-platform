# Pre-registration — the trend rule as the quant leg

Written and committed before any trend-rule replay runs.

## Why this test exists

Phase 5 (`docs/ml_baselines_2026-10-03.md`) found that one published
feature, 12-1 momentum (`mom_12_1`: the 12-month return skipping the last
month), ranked stocks at least as well as the 79-feature LightGBM model:

- pooled rank IC +0.0399 against LightGBM's +0.0295;
- fold t 3.14 against 1.58;
- winning-fold share 0.79.

The user wants the live system to be "quant mixed with committee", and the
trend rule is the strongest quant candidate. This test asks the question a
rank IC can't answer: **does trading it beat SPY after costs and after
adjusting for market exposure?** The campaign correction (44d266f) showed
these can disagree.

## Honest limits, stated before the result

- **Not fully out of sample.** Phase 5's folds cover 2023–2026, and these
  replay windows sit inside that span. The rule has no fitted parameters,
  but choosing it was informed by that period. The clean test is forward
  paper trading; this one checks whether the ranking turns into a
  portfolio edge on the stocks the system actually trades.
- **Different universe.** Phase 5 ranked 1,315 names. The live system trades
  tier 1: 55 large names plus 5 crypto pairs. Momentum among 55 mega-caps
  is a narrower bet.
- **Survivorship.** Today's tier-1 list is applied back to 2024-12-01, its
  membership start. That flatters any long-only rule.
- **Power.** The primary window is about 1.6 years. Clearing alpha t ≥ 1.96
  over 1.6 years needs an information ratio of roughly 1.5, which is rare.
  **The most likely outcome is "positive but not significant"**, and it will
  be reported as that.

## The rule (bundle `trend-rule-v1`, written by `scripts/eval/trend_rule_bundle.py`)

**Ranker: `TrendRule`** (`src/dynamo/research/quant/models.py`). The order
is plain `mom_12_1`. The value is IC × σ × z, with IC 0.04 and σ 0.08 fixed
here, where z is the normal score of the name's rank within that session's
universe. This puts scores on the 20-session expected-excess-return scale
that sizing and the buy gate (`expected_excess_return > 0`) read. It does
not change the order.

**Downside and vol: abstain** (`AbstainModel`, NaN). This has consequences:

- no downside-based AVOIDs (v8 had them);
- position sizing falls back to its no-forecast volatility path.

**Missing momentum: the name abstains.**

## Runs (quant-only replay: $0, zero model calls)

Shared settings:

- `--mode quant --universe-tier 1 --universe-tier 3 --discovery off`
- `--initial-equity 100000 --ballast SPY --no-require-shipping-bundle`
- `--code-version trend-5970f24`

These are the settings of the earlier v8 control, `replay-ad0c913a7f4c`.
Tier 3 has no members before 2026-09-16, so it adds nothing inside these
windows.

| run | model | window | role |
|---|---|---|---|
| **trend-long** | trend-rule-v1 | 2025-01-02 → 2026-08-14 (~405 sessions) | **primary** |
| **trend-camp** | trend-rule-v1 | 2026-03-03 → 2026-08-14 (115 sessions) | head-to-head with v8 |
| **v8-camp** | campaign-v8 | 2026-03-03 → 2026-08-14 | control, re-run on the same code |

v8 trained through 2026-02-28, so only the campaign window is out of sample
for it. That is why the head-to-head uses that window. The old control's
+14.57% is quoted for reference only; the code has changed since, including
the revenue-tag fix.

## Metrics

From `dynamo.cli.replay report` per run:

- total return and excess vs SPY;
- Sharpe and deflated Sharpe;
- max drawdown;
- turnover and cost drag;
- beta, annual alpha, tracking error, information ratio, alpha t-stat, and
  years to significance.

Head-to-head: the paired daily return difference, trend-camp minus v8-camp,
and its t-stat.

## Decision rule

- **GO** (for the next steps, each still needing the user's approval: a
  pre-registered committee arm with the trend shortlist, then forward paper
  shadowing): **trend-long** has annual alpha vs SPY > 0, alpha t ≥ 1.96,
  net of costs, AND max drawdown no worse than −15% (promotion stage 1).
- **Inconclusive:** alpha > 0 but t < 1.96. The rule is not shown to work.
  Forward paper shadowing is the honest way to get more evidence; no
  promotion.
- **NO-GO:** alpha ≤ 0 or drawdown worse than −15%.
- **Head-to-head:** the trend rule "beats v8" only if the paired daily t ≥ 2.0.
- **No rescue.** No re-run with another IC, σ, feature, window, universe or
  threshold. Promotion to production always needs the user's explicit
  approval; nothing is promoted automatically.

## Validity checks, reported first

- No stored `quant_scores` rows fall inside either window (checked
  2026-10-03: the only stored scores are baseline-v0, from 2026-09-16 on).
  Replay prefers stored scores over the bundle, so any such row would
  contaminate the arm.
- Zero model calls: no `cost_ledger` rows reference these run ids.
- The bundle loads with the sha256s its manifest records.

---

## Amendment (2026-10-03, after the first runs, before any re-run)

**The first runs (`--code-version trend-5970f24`) are invalid because of an
engine bug they exposed.**

- **What happened.** **trend-long** showed a −74% drawdown, beta 1.84 and
  "alpha" of +329%/yr.
- **Cause.** On 2025-02-04, 03-04 and 09-03 the replay bought NFLX at
  Tiingo's *raw* open (971.61) while valuing it at the split-adjusted close
  (97.26; 10:1 split on 2025-11-17). A ~$8k target became an $80k
  position, the book showed a fake −72% day, and the drawdown ladder then
  liquidated it.
- **Fix: 1c6c04d.** `canonical_bars` now moves open, high and low onto the
  adjusted-close scale. The panel-cache fingerprint covers `canonical_bars`.
- **Scope of the bug.** It hits every Tiingo-priced name at about the 1%
  scale through dividends, and pre-split dates hard. The camp runs were
  affected only mildly, but all three runs are redone for consistency.

**Re-run plan.** The same three runs, the same settings and the same
decision rule, with `--code-version trend-1c6c04d`. Nothing else changes:
no IC, σ, feature, window, universe or threshold. The invalid first results
are reported beside the valid ones.

**Preflight deviation, for trend-long only.**

- Preflight requires HY OAS (`BAMLH0A0HYM2`) back to start − 900 days.
- FRED serves only about 3 years of that ICE index, so it starts on
  2023-08-15 and can't be backfilled.
- The regime model reads OAS only as its current level and its 20-session
  change. The 900 days exist for the SPY volatility percentile, which is
  complete.
- Every session in the window (2025-01-02 onward) therefore has real OAS
  data, and the run uses `--skip-preflight`.
- Every other check passed on the first attempt (logged in
  `artifacts/logs/trend_long.log`).
- T10Y3M was backfilled to 2022-01-03.
