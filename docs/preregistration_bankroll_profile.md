# Pre-registration — bankroll risk profile v1 vs flat sizing

Written and committed before any profile run. Machinery: 8f59aba
(`src/dynamo/portfolio/bankroll.py`). The user asked for this on 2026-10-04:
"more money means u can do little more risky or keep some percentage of it
for riskiness".

## What is compared

The committee's decisions (WHAT to buy) stay fixed. Only sizing (HOW MUCH)
changes. Both arms replay the same saved verdicts from the wide committee
campaign run `replay-63220a3a5e9a`.

| tier (by starting equity) | max positions | per-position cap | risk sleeve (crypto + non-core) | cash floor |
|---|---|---|---|---|
| small (< $5k) | 8 | 15% | 0% | 2% |
| medium ($5k–$50k) | 15 | 10% | 10% | 5% |
| large (≥ $50k) | 25 | 8% (today) | 15% | 5% |

**flat** is today's sizing: 25 names, 8% cap, crypto ≤ 10%, no non-core cap.

These defaults are "common practice", chosen before any run, not fitted.

## Runs (portfolio mode: $0, zero model calls)

The same flags as the campaign's portfolio passes:

- `--mode portfolio --source-run-id replay-63220a3a5e9a`
- `--universe-tier 1 --universe-tier 3 --start 2026-03-03 --end 2026-08-14`
- `--ballast SPY --min-feature-coverage 0.35`
- fractional shares, $0 commission
- `--code-version bankroll-8f59aba`

| bankroll | flat | profile (`--bankroll-profile v1`) | tier |
|---|---|---|---|
| $1,000 | ✓ | ✓ | small |
| $10,000 | ✓ | ✓ | medium |
| $100,000 | ✓ | ✓ | large |

The code includes the replay fixes in 8cc854a: crypto is no longer
liquidity-vetoed, and names the committee debated are tradable.

## Metrics, per bankroll

- Total return and excess vs SPY.
- Sharpe and max drawdown.
- Beta, annual alpha and alpha t-stat.
- Fills, costs, and average number of positions held.
- **Primary:** the paired daily return difference, profile − flat, and its
  t-stat.

## Decision rule

- At each bankroll, the profile is **better** (or **worse**) only if the
  paired daily t ≥ 2.0 (or ≤ −2.0) AND its max drawdown stays within −15%.
  Otherwise: **no difference shown**.
- A **better** at any bankroll would justify proposing the profile for live
  paper. Wiring it into live needs the user's approval. Nothing changes in
  live from this test alone.
- **No re-tuning:** the tier numbers will not be changed after seeing results.

## Known limits, stated in advance

- **Frictions.** Fractional shares and $0 commission, as in the campaign.
  The small-account rationale (fees and minimums) therefore doesn't bite
  here. Only the risk-appetite part is tested.
- **One window, few decisions.** 115 sessions and ~72 committee buys, with
  most of the book in SPY ballast. Each arm runs once.
- **Crypto drives the outcome.** The small tier excludes crypto and
  discovery-added names entirely, so its result depends largely on how
  crypto and those names did in March–August 2026.
- **Expected outcome:** no difference shown at any bankroll.

---

## Amendment (2026-10-04, before any result was looked at)

The first launch (`--code-version bankroll-8f59aba`) was stopped after 2
completed runs and 1 partial run. While diagnosing why so few committee
buys trade, two more replay bugs were found and fixed in 5dd26dd:

- SPY ballast trades were spending the monthly turnover budget.
- Debated names couldn't fill at the next open.

Arms launched before and after the fix would have run on different code.
The stopped runs (`replay-9990807c7961`, `replay-9a15ebc94cce`,
`replay-ce12d4567c20`) are discarded **unread**: no metric from them was
computed or viewed.

The re-launch uses `--code-version bankroll-5dd26dd`. Arms, settings, tier
numbers and the decision rule are unchanged.
