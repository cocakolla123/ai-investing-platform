# Pre-registration — quant baselines and feature-group ablations (Phase 5)

Written and committed before any Phase 5 model is fitted. Data and folds are
campaign-v10's exactly (`docs/preregistration_campaign_v10.md`), read from
its cached feature panel, so every number here is comparable to v10's.

## Data and folds (fixed by v10, not chosen here)

1,315 symbols (`config/training_universe_wide.txt`), 2021-03-01 → 2026-09-23,
session stride 2 (680 panel sessions), 79 features, 20-session
sector-excess rank target (`excess_return_20d`), `walkforward_for` geometry
(train 2y, test 2mo, purge = horizon) → v10's 19 folds. Revenue-tag fix NOT
applied (deferred to after this phase, per the user's decision), so this
measures the v10 world.

## Models (rank task, identical folds)

| name | definition |
|---|---|
| lightgbm | v10's: `max_depth 3, min_data_in_leaf 120, lambda_l2 10` |
| baseline-historic-average | v10's bar-to-beat (StatsforecastBaseline) |
| baseline-momentum | score = `mom_12_1` (missing → 0); no fitting |
| baseline-linear | ridge regression, alpha = 1.0, on standardized features (missing → 0) |

## Ablations

LightGBM refit with ONE catalog feature group removed, for every group (17),
same folds and parameters.

## Metrics and decision rules

Per-fold rank IC (mean Spearman per test date), pooled IC, fold t-stat.
Comparisons are **paired by fold**: t-stat of the 19 per-fold differences.

- LightGBM "beats" a baseline only if the paired fold t ≥ 2.0.
- A feature group "matters" only if removing it changes the paired fold IC
  with |t| ≥ 2.0; with 17 groups tested, about one such result is expected
  by chance, so the Bonferroni line (two-sided 0.05/17, t ≈ 3.4 at 18 df)
  is reported beside it and a group clearing only 2.0 is called "suggestive".
- No improvement percentage is quoted unless it clears these lines.

## Tracking

MLflow, local file store `artifacts/mlruns`, one run per model/ablation:
parameters, dataset hash (sha256 over the panel's feature matrix and labels),
feature list, per-fold metrics, and the results JSON as an artifact.
