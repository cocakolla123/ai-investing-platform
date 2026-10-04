# Trend rule + committee result — "quant mixed with committee"

Pre-registration: `docs/preregistration_trend_committee.md` (a58fb0a).
Analysis: `scripts/eval/trend_committee_analyze.py` →
`evals/trend_committee/results.json`.

The trend run is `replay-c958db4a4a55`: 15/15 sessions, 59 verdicts, $0
metered. It was resumed three times: once after a 5-hour plan-window limit,
and twice after the Mac logged out and Docker quit. Days already saved are
never re-run (deterministic run id), so the resumes don't change results.

**Pre-registered verdict: TREND WORSE.** On this window, the committee made
clearly worse buys from the trend-rule shortlist than from the v8 shortlist.
Most of that gap came from **crypto**, which the trend rule never
shortlisted. That is reported below as an exploratory breakdown.

## Results (excess vs SPY, next open → 20th session close; day-clustered 95% CI)

| arm | buys | **buy quality** (primary) | shortlist quality (quant leg alone) | committee lift | Brier |
|---|---|---|---|---|---|
| **trend** | 46 | **−0.1%** [−1.4%, +1.2%] | +0.0% [−1.8%, +1.5%] | −0.1 pts | 0.273 |
| v8 | 48 | **+10.3%** [+7.1%, +13.5%] | +8.2% [+5.1%, +11.3%] | +2.1 pts | 0.227 |
| v8-rep | 48 | +10.3% [+6.9%, +13.6%] | +8.2% | +2.1 pts | 0.226 |

- **trend − v8:** −10.4 pts [−14.1, −6.9]
- **trend − v8-rep:** −10.4 pts [−14.2, −6.7]
- **Replicate noise** |v8 − v8-rep|: 0.0 pts. Both v8 runs saw the identical
  shortlist and made nearly identical buys.
- **Shortlist overlap** (Jaccard) trend/v8: **0.06**. The two quant legs
  sent the committee almost entirely different names.

Both CIs exclude zero in the same direction, and the gap exceeds the
replicate noise. By the pre-registered rule: **trend worse**.

**Where the gap comes from:** the shortlist, not the committee.

- The trend rule's shortlist was worth about 0% against SPY. The committee
  bought most of it and added nothing (−0.1 pts).
- The v8 shortlist was worth +8.2%. The committee added a little on top
  (+2.1 pts).

## Exploratory breakdown (not pre-registered; for interpretation only)

| arm | stock buys | mean excess | crypto buys | mean excess |
|---|---|---|---|---|
| trend | 46 | −0.1% | 0 | — |
| v8 | 35 | **+4.6%** | 13 | **+25.6%** |

- **Crypto drove most of v8's lead.** BTC, SOL and LINK rallied 26–32% against
  SPY in this window. v8 used its reserved crypto shortlist slot; the trend
  rule's momentum scores never made a crypto pair a candidate.
- **On stocks alone, v8 still led**: +4.6% against −0.1%, a gap of about 4.7
  points rather than 10. That comparison was not pre-registered and has no
  CI here.
- The trend rule's best buys were MRK (+14%), AMD, AAPL and ABBV. v8's best
  stock buy was CRM (+35%).

## Reading it

- **One three-week window.** The 15 overlapping 20-day outcomes span about
  7 weeks of market. A crypto rally in that window decided most of the
  result. This says the trend leg did worse in August 2026, not that it
  always will.
- **Consistent with the earlier trend-only replay.** That replay was
  INCONCLUSIVE, with no measurable stock-selection edge among the 55 large
  names. Here the trend shortlist's stocks also came out flat against SPY.
- **Known difference:** the v8 runs predate the bar-scale fix (1c6c04d). Its
  shortlist was built on slightly different high/low/open features. Outcomes
  for both arms use the fixed bars.

## What this means

- **Do not switch the live quant leg to the trend rule.** The evidence so far
  points the other way.
- **The forward shadow keeps running** (`docs/preregistration_trend_shadow.md`).
  It costs nothing, needs no model calls, and is the clean out-of-sample
  test, decided around February 2027. This result doesn't change its
  pre-registered rule.
- **Open question worth its own pre-registered test:** how much of the
  committee's historical edge comes from the reserved crypto slot riding
  crypto beta? Today's replays measure excess against SPY for crypto too,
  and that flatters crypto in a rally and punishes it in a selloff.
