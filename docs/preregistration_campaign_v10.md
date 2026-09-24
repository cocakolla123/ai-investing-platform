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
