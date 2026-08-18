# Tuning log — campaign-v3 quant bundle retune (2026-08-18, overnight session)

Window: 2020-01-01 → 2025-12-31 (1488 feature sessions, 49 usable symbols, 81,840 panel
rows). Walk-forward: doc-08 defaults (train 756 / test 126 / purge 20 / embargo 10 /
roll 126) → 5 folds, identical geometry to campaign-v2, so per-fold numbers are directly
comparable. All runs trained via `dynamo.cli.quant_train` on the on-disk panel cache
(`artifacts/panel_cache`, populated once by tune-a at ~8 s/session, then served
1488/1488 for every later run — cached runs take ~70 s end-to-end).

Universe note: the active tier-1/3 universe at 2025-12-31 now selects 55 symbols
(campaign-v2 trained on 49), but only 49 carry a usable label+feature row, so the
effective panel matches v2's.

## Ship gates (doc-04 rank task)

- pooled rank-IC t-stat over folds >= 2.0
- win share >= 0.60 — fraction of folds where the model's fold IC beats the BEST
  baseline's fold IC (baseline suite here: `historic_average`)
- pooled IC must beat the pooled best-baseline IC

## The runs

Label ordering is chronological. `depth` = `--lgbm-max-depth`, `min` =
`--lgbm-min-data-in-leaf` (LightGBM `min_child_samples`), `l2` = `--lgbm-lambda-l2`
(`reg_lambda`). Feature-group subset runs are walk-forward report cards ONLY (the
bundle loader refuses subset bundles as feature drift — they can never be replayed).

| run | depth | min | l2 | features | pooled IC | t-stat | win share | fold ICs (0..4) | dispersion | downside AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| campaign-v2 (baseline, lib defaults) | — | 20 | 0 | 51 (all) | +0.0229 | 1.18 | 0.20 | −0.011, +0.019, +0.015, +0.097, −0.006 | 0.0432 | 0.485 |
| tune-a | 3 | 120 | 10 | 51 (all) | **+0.0486** | **3.26** | 0.20 | −0.001, +0.034, +0.067, +0.086, +0.056 | **0.0334** | 0.519 |
| tune-b | 2 | 240 | 50 | 51 (all) | +0.0634 | 3.00 | 0.20 | −0.008, +0.063, +0.081, +0.122, +0.059 | 0.0473 | 0.538 |
| tune-c | 4 | 60 | 1 | 51 (all) | +0.0349 | 1.68 | 0.20 | −0.048, +0.055, +0.063, +0.049, +0.054 | 0.0465 | 0.513 |
| tune-d | 1 | 240 | 50 | 51 (all) | +0.0583 | 2.15 | 0.20 | −0.004, +0.064, +0.107, +0.127, −0.003 | 0.0607 | 0.580 |
| tune-e | 2 | 240 | 200 | 51 (all) | +0.0688 | 2.52 | 0.20 | −0.019, +0.040, +0.119, +0.132, +0.073 | 0.0612 | 0.554 |
| tune-f | 2 | 240 | 50 | 19 (momentum+volatility+volume) | +0.0205 | 1.71 | 0.00 | +0.022, +0.034, −0.015, +0.006, +0.056 | 0.0267 | 0.593 |
| tune-g | 2 | 240 | 50 | 37 (f + meanrev+value+quality) | +0.0470 | 2.23 | 0.20 | −0.031, +0.063, +0.088, +0.076, +0.040 | 0.0471 | 0.623 |
| tune-h | 3 | 240 | 50 | 51 (all) | +0.0484 | 2.14 | 0.20 | −0.040, +0.053, +0.074, +0.075, +0.081 | 0.0507 | 0.543 |
| **campaign-v3** (= tune-a config) | 3 | 120 | 10 | 51 (all) | +0.0486 | 3.26 | 0.20 | −0.001, +0.034, +0.067, +0.086, +0.056 | 0.0334 | 0.519 |

Fold test windows: 0 = 2023-02→2023-08, 1 = 2023-08→2024-02, 2 = 2024-02→2024-08,
3 = 2024-08→2025-02, 4 = 2025-02→2025-08.

## What the grid says

1. **Regularization is the whole game.** Library defaults (v2) → IC 0.023, t 1.18.
   Every regularized config at depth ≤ 3 with min ≥ 120 more than doubles pooled IC and
   clears the t-stat gate. The light config (tune-c) confirms the monotone direction.
2. **Depth 2–3 is the sweet spot.** Stumps (tune-d) lose fold-4 entirely; depth 4
   (tune-c) collapses back toward the overfit v2 profile.
3. **Full features beat subsets.** The momentum+volatility+volume card (tune-f) is weak
   alone (IC 0.021); adding meanrev/value/quality (tune-g) recovers most but not all of
   the full model (0.047 vs 0.049–0.069 for full-feature configs). Macro/calendar/
   insider/estimates carry real incremental rank signal. Note: the SUBSET runs are the
   only place the downside head looks decent (AUC 0.59–0.62 on 19–37 features) — with
   all 51 features the downside classifier sits at 0.52–0.58. The downside gate is
   informational; rank gates are what block.
4. **Win share is stuck at 0.20 for every config — and that is the honest headline.**
   The `historic_average` baseline's fold ICs on this window are
   [+0.078, +0.111, +0.127, +0.037, +0.112] (pooled +0.093, t 5.7): 2020-2025 was a
   persistent-winner regime, so ranking symbols by their long-run mean excess return is
   an extremely strong naive strategy. Every LGBM config beats it only in fold 3
   (2024-08→2025-02, the one regime break). No amount of hyperparameter tuning turned
   51 point-in-time features into a model that out-ranks that baseline 3 folds out of 5.

## Final selection — campaign-v3

`--lgbm-max-depth 3 --lgbm-min-data-in-leaf 120 --lgbm-lambda-l2 10`, all 51 features.

Chosen for **per-fold consistency**, per the selection rule (a config with positive IC
across all folds beats a bigger pooled IC concentrated in fewer folds): highest t-stat
(3.26), lowest dispersion (0.0334), and a worst fold of −0.0007 (effectively zero — no
materially negative fold anywhere in the window). tune-b/tune-e have higher pooled ICs
but strictly worse dispersion and worse worst-folds. Training is deterministic
(fixed seed, `deterministic=true`): campaign-v3 reproduces tune-a's metrics bit-for-bit.

Registered as CANDIDATE `campaign-v3` (model_type `quant_bundle`), bundle manifest
`artifacts/models/quant/campaign-v3.bundle.json`.

## Verdict vs ship gates

**NO-GO.** t-stat 3.26 >= 2.0 passes; win share 0.20 < 0.60 fails (and pooled IC
+0.0486 < baseline +0.0929). campaign-v3 is a large, real improvement over v2
(IC 2.1×, t-stat 1.18 → 3.26, regime concentration fixed, downside AUC back above 0.5)
— but it does not beat the historic-average baseline on this window, so the ship gate
stands un-met. Per the runbook, the paid committee step remains prohibited. The free
quant campaign proceeds as **mechanics validation only**, clearly labeled as such in
every report.

Tuning-run registry hygiene: tune-a..h remain registered candidates (labels are
single-use and never deleted, per runbook); none of them may be promoted.
