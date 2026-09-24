# Pre-registration — campaign-v10

Written and committed **before** training starts. The git timestamp of this
file is the evidence that the settings and the decision rule were fixed
before any result was seen. Nothing below may change after the run starts.

## What is being tested

Whether the quant ranker, given the ten published-anomaly features added in
`c2b25a1` (from paperswithbacktest/awesome-systematic-trading) and all the
price history the database holds, produces an out-of-sample cross-sectional
edge that clears the existing ship gate.

## Fixed settings (identical to campaign-v8 except where marked)

| setting | value | vs v8 |
|---|---|---|
| label | `campaign-v10` | — |
| symbols | `config/training_universe_wide.txt` (1,315 names) | same |
| features | full catalog, 79 (all groups) | **+10 anomaly features** |
| window | `--train-start 2021-03-01 --train-end 2026-09-23` | **wider** (v8: 2022-01-03 → 2026-02-28) |
| session stride | 2 | same |
| LightGBM | `max_depth 3, lambda_l2 10, min_data_in_leaf 120` | same |
| label kind | default (`sector_excess`), 20-session horizon | same |
| walk-forward | `walkforward_for` unchanged: train 2y, test 2mo, purge = horizon | same |

**Window rationale (data availability, not results).** 2021-03-01 is the
earliest date whose 420-day feature lookback lands inside the ~1,200-symbol
coverage that starts in 2020 (2019 covers only ~460 names). 2026-09-23 is the
last session with bars; the final 20 sessions serve only as label horizon.
Fold count is an output of `walkforward_for`, not a choice.

## Pass bar (unchanged, `research/quant/training.py`)

Rank task ships iff ALL of: beats the historic-average baseline; pooled rank
IC > 0; ≥ 4 folds; rank-IC t-stat ≥ **2.0**; winning-fold share ≥ 0.60.

## Decision rule

- **Pass:** report the numbers to the user. The bundle stays `candidate`.
  Promotion to production requires the user's explicit approval (standing
  rule: the promotion pipeline plus human approval is mandatory). Nothing is
  promoted automatically.
- **Fail:** report the numbers as a NO-GO. **No re-run with different
  hyperparameters, windows, strides, feature subsets or seeds** to rescue it.
  The next honest lever is more data (2018–2019 for the full panel), not
  more knobs.

## Known caveats, stated in advance

- Survivorship: today's constituent list applied historically (optimistic).
- From 2026-09-12 the 55 tier-1 names' canonical close is Alpaca
  (split-adjusted) instead of Tiingo (split + dividend adjusted); this
  touches only the last labels, by at most a dividend.
- IEX-only volume for most names (volume features are relative, so ranks are
  comparable; absolute levels are not).
- A pass is evidence the ranker has signal, not that any portfolio built on
  it will beat SPY after beta — see the campaign correction (44d266f).

---

## RESULT (appended 2026-09-24, after the single pre-registered run)

**NO-GO.** 680 sessions, 1,315 symbols, 79 features, 894,200 rows, 19 folds.

| | campaign-v8 | campaign-v10 |
|---|---|---|
| folds | 11 | 19 |
| pooled rank IC | +3.37% | +2.95% |
| rank-IC t-stat (gate 2.0) | 1.72 | **1.58** |
| winning-fold share (gate 0.60) | 0.45 | **0.53** |
| fold IC dispersion | 6.5% | 8.1% |
| downside AUC | 0.605 | 0.617 (Brier worse than base rate) |

The ten anomaly features plus fourteen more months of history did not
produce a pass; the edge is a small, regime-unstable ~3% IC (folds range
−15% to +18%). Per the decision rule: reported as NO-GO, bundle left as
`candidate`, no rescue re-run.

**Found after the run (not a rescue lever, a data defect):** since 2024,
605 of 1,218 CIKs report revenue ONLY under the ASC 606 tag
`RevenueFromContractWithCustomerExcludingAssessedTax`; the catalog and the
quality score read only `Revenues`. Every revenue-based input — ps_ttm,
ps_sector_pctile, rev_growth_yoy, rev_cagr_3y, and three quality sub-scores
— was NaN for half the panel in both v8 and v10. Fixing it is a data fix; a
model trained afterwards would be a NEW test with its own pre-registration,
and should be counted as a second attempt (multiple testing), not as v10.
