# Phase 5 result — quant baselines and feature-group ablations on v10's folds

Pre-registration: `docs/preregistration_ml_baselines.md` (e7a5af7), committed
before any model was fitted. Script: `scripts/eval/ml_baselines.py`.
Results: `evals/ml/results_phase5.json`. MLflow experiment
`quant-phase5-v10-folds` logs one main run (all four models on the same 19
walk-forward folds), 17 drop-one-group ablation runs, and one summary run.
These are not 19 independent experiments: the ablations re-fit LightGBM on
the same folds with one feature group removed.

The re-run on the fixed bars (2026-10-04) is a second experiment,
`quant-phase5-v10-folds-barfix`, with the same structure (one main run, 17
ablations, one summary); see `docs/rerun_after_fixes_2026-10-04.md`.

## Reproduction check

The panel is 894,200 rows × 79 features with 19 walk-forward folds (dataset
sha256 `cd87b2744146…`). LightGBM scores pooled rank IC **+0.0295**, fold t
**1.58**, matching campaign-v10 exactly. These numbers describe v10's world.

## Models (rank IC on the 20-session sector-excess target)

| model | pooled IC | fold t | LightGBM − model (paired fold t) |
|---|---|---|---|
| lightgbm (v10) | +0.0295 | 1.58 | — |
| baseline-momentum (`mom_12_1` alone, no fitting) | **+0.0399** | **3.14** | −0.0105 (t −0.49) |
| baseline-historic-average | +0.0401 | 2.25 | −0.0106 (t −0.57) |
| baseline-linear (ridge) | −0.0046 | −0.53 | +0.0341 (t +1.57) |

**LightGBM beats no baseline** (pre-registered bar: paired t ≥ 2.0). One
published feature, 12-1 momentum, ranks stocks at least as well as the
79-feature model, and its fold t clears 3. LightGBM is numerically below
momentum, but that gap is not significant either (t −0.49).

## Ablations (LightGBM with one group removed; paired vs full model)

| group removed | features | pooled IC | diff vs full | paired t |
|---|---|---|---|---|
| volume | 5 | +0.0198 | −0.0097 | −1.75 |
| anomaly_price | 6 | +0.0277 | −0.0018 | −0.41 |
| insider | 3 | +0.0296 | +0.0001 | +0.06 |
| macro | 5 | +0.0307 | +0.0013 | +0.12 |
| meanrev | 5 | +0.0299 | +0.0004 | +0.13 |
| momentum | 8 | +0.0298 | +0.0004 | +0.20 |
| momentum_quality | 3 | +0.0302 | +0.0008 | +0.31 |
| quality | 7 | +0.0305 | +0.0011 | +0.40 |
| anomaly_fundamental | 4 | +0.0308 | +0.0014 | +0.43 |
| value | 6 | +0.0337 | +0.0042 | +0.51 |
| volatility | 6 | +0.0319 | +0.0024 | +0.78 |
| shortinterest | 3 | +0.0316 | +0.0021 | +0.89 |
| calendar | 3 | +0.0328 | +0.0033 | +0.92 |
| attention | 1 | +0.0295 | +0.0001 | +1.00 |
| weekly | 7 | +0.0330 | +0.0036 | +1.42 |
| accumulation | 4 | +0.0334 | +0.0039 | +1.79 |
| estimates | 3 | +0.0295 | 0.0000 | n/a (no variation: the group carries no usable data in this panel) |

No group clears |t| ≥ 2.0, let alone the Bonferroni line (3.4). **No
feature group is shown to matter.** By the pre-registered rule, no
improvement percentage is quoted.

## What this suggests (hypotheses for a future pre-registered test)

- Removing 14 of 17 groups nudges IC *up*. The model is likely fitting noise
  in most groups, so a much smaller model, or plain momentum, is the
  candidate to test. Volume is the one group whose removal hurts, and it
  doesn't reach significance either.
- A momentum-only quant leg is the obvious next candidate. Any such change
  must go through a new pre-registration and the existing campaign gate
  (net of costs, beta-adjusted, vs SPY). Rank IC on these folds is not that
  gate.
- The ASC606 revenue-tag fix (next task) changes the quality and value
  inputs. Any retrain after it is a new pre-registered test, not a re-read
  of this table.

## Deviation

The pre-registration named an MLflow "local file store". MLflow 3 put the
plain-directory store into maintenance mode, so the runs were logged to a
local SQLite file in the same directory (`artifacts/mlruns/mlflow.db`). The
logged content is unchanged.
