# Trend rule result — 12-1 momentum as the quant leg

Pre-registration: `docs/preregistration_trend_rule.md` (5de7d37), amended
before the re-run (c89483b). Analysis: `scripts/eval/trend_rule_analyze.py`
→ `evals/trend_rule/results.json`. Quant-only replays: zero model calls,
$0.

**Verdict: INCONCLUSIVE.**

- Over the 1.6-year primary window, the trend rule made money but lagged
  SPY by 11 points. Its beta-adjusted alpha was a statistically empty
  +0.9%/yr (t 0.20).
- On the campaign window it beat v8, but not significantly.
- It is **not** shown to work. Nothing is promoted.

## Validity

- No stored quant scores inside either window.
- Zero model calls.
- Equity curves are clean after the engine fix (below). In trend-long the
  largest daily move is +3.7% (2025-04-09, the real tariff-pause rally),
  cash never goes negative, and gross exposure stays ≤ 0.96.

**The first runs were invalid.** They exposed a real engine bug:

- Tiingo open/high/low stayed on the raw price scale while the close was
  split-adjusted.
- Pre-split NFLX filled at 971.61 against a mark of 97.26. The book showed
  a fake −72% day and was liquidated three times.
- Fixed in 1c6c04d; see the amendment in the pre-registration.
- The invalid results are kept in
  `evals/trend_rule/results_INVALID_trend-5970f24.json`.

## Results (`--code-version trend-1c6c04d`)

| run | window | return | vs SPY | Sharpe (deflated) | max DD | beta | alpha/yr | alpha t | turnover | costs |
|---|---|---|---|---|---|---|---|---|---|---|
| **trend-long** (primary) | 2025-01-02 → 2026-08-14 (405 sessions) | +23.71% | −11.35% | 1.10 (0.92) | −13.93% | 0.65 | +0.95% | **+0.20** | 9.6× | $774 |
| trend-camp | 2026-03-03 → 2026-08-14 (115) | +15.11% | +0.39% | 2.36 (0.94) | −7.08% | 0.91 | +3.49% | +0.67 | 3.4× | $222 |
| v8-camp | same | +11.17% | −3.55% | 1.90 (0.90) | −7.33% | 0.85 | −2.48% | −0.49 | 2.8× | $230 |

**Head-to-head, trend-camp minus v8-camp:** +7.8%/yr annualized, paired
daily t **1.00**. The bar was 2.0, so **the trend rule does not beat v8.**

## Reading it

- **Decision: INCONCLUSIVE.** trend-long's alpha is above zero and its
  drawdown is inside −15%, but alpha t is 0.20 against a bar of 1.96. At
  this information ratio (0.16) it would take about 150 years of data to
  prove. That is the honest way of saying there's essentially no measurable
  edge over the primary window.
- **Why it lagged SPY by 11 points while showing positive alpha.** Average
  market exposure was low (beta 0.65). In 2025 the regime filter cut gross
  exposure to ~37% around the April tariff selloff, so the book missed part
  of the rebound. The lag is mostly that defensive positioning, not stock
  selection.
- **Momentum among 55 mega-caps is a narrow bet.** Phase 5's IC was
  measured across 1,315 names, where the cross-section is wider. This test
  ran on the universe the system actually trades, and the edge did not
  carry over in a measurable way.
- **Against v8**, the rule did better on every metric in the one window
  where v8 is out of sample: +3.9 points of return, higher Sharpe, and
  alpha +3.5%/yr against −2.5%/yr. It is also simpler, with nothing fitted.
  None of this is significant (t 1.00).

## What the pre-registration allows next

- **No promotion** and **no rescue re-runs** with another IC, σ, window or
  universe.
- The rule permits **forward paper shadowing** as the way to collect
  honest evidence: score the trend rule nightly beside the live system
  without trading it. This needs the user's approval and a registry entry,
  but no promotion.
- A **committee arm** (trend shortlist plus the LLM committee) would test
  the "quant mixed with committee" setup directly. Estimate: about 240 calls
  on the subscription, the same 15-day window as Phase 3, compared against
  Phase 3's v8-shortlist vector arm. It needs its own pre-registration and
  the user's go.

## Side effect worth knowing

The bar-scale fix (1c6c04d) changes every high/low/open-based feature on
Tiingo-priced history (the 55 tier-1 names), and every replay fill price on
those names. Results computed before it are slightly off:

- v10;
- Phase 5;
- the wide campaign's quant arms.

The effect is about 1% from dividends, and large only around splits, for 55
of 1,315 names, so a flip is unlikely, but those results have not been
re-run to check. Any future retrain will see different features. That's intended, and the panel cache now rebuilds
for it.
