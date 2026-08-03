---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 04 — Quant Models

Quantitative models give the platform numbers the committee cannot: expected return, downside
probability, volatility forecasts, confidence, and an opportunity ranking over the whole
scored universe. They are **advisory** like everything except the risk engine (`06`): the
arbiter sees quant scores (`03 §1`), sizing uses them as a Kelly ceiling input (`06 §4`), and
the deep-dive selector uses rank moves — but no model output ever bypasses the veto contract.

## 1. Scope and model roster

All models sit behind the `ForecastModel` port (`01 §2`). Per ADR-0003, **no deep learning
initially** (no LSTM/transformers — insufficient data scale at one-portfolio size); the port
is the door through which DL can enter later if its revisit criteria are met. Nothing outside
`research/quant/` may import a model library.

```python
class ForecastModel(Protocol):
    def fit(self, X: FeatureFrame, y: LabelFrame, cfg: TrainConfig) -> FitResult: ...
    def predict(self, X: FeatureFrame) -> PredictionFrame: ...
    def feature_importances(self) -> dict[str, float]: ...
```

| Role | Implementation | Purpose |
|---|---|---|
| **Primary** | LightGBM (regression + binary objectives) | Best accuracy/cost on tabular features at this data scale; fast monthly retrains on the single host |
| **Challengers** | CatBoost, RandomForest (scikit-learn) | Same features, same folds; keep the primary honest, catch library-specific pathologies |
| **Classical baselines** | Nixtla `statsforecast`: `AutoETS`, `AutoARIMA`, `HistoricAverage`, `SeasonalNaive` | The bar to beat. Cheap, no features, hard to fool |

**Baseline rule:** any ML model must beat the best statsforecast baseline out-of-sample on the
primary metric (§6) across the walk-forward folds. A model that does not beat the baselines
OOS **does not ship** — it never reaches `candidate` status in the registry (§7). This is the
quant analogue of "evidence over eloquence" (`00 §8`).

Three model families, one per target (§4): the excess-return ranker, the downside classifier,
the vol forecaster. Each family trains primary + challengers + baselines on identical folds.

## 2. Feature catalog

Target size ~40–60 features total — deliberately small. Every feature is a pure function of
data available at `as_of` (§2.1), computed from **adjusted** series (`02 §5`), registered in a
single `features/catalog.py` with name, group, lookback, source tables, and owner formula.
Features are built nightly at 23:00 ET (`02 §4`) for Tier 1 + promoted names and exported to
the Parquet sidecar for research reads.

### 2.1 The look-ahead audit rule (applies to every group)

> A feature value at `as_of` may use only records whose `filed_at`/`fetched_at`/`published_at`
> ≤ `as_of` data-cutoff. Fundamentals come exclusively from `xbrl_facts`, which is **keyed by
> `filed_at` and never restated backwards precisely for this reason** (`02 §3`). Macro uses
> FRED `vintage_ts`, never revised values. The feature builder takes `as_of` as an explicit
> argument; there is no "latest" accessor in the feature code path.

Violations are caught by the automated shift test in §9 — the rule is tested, not trusted.

### 2.2 Groups

| # | Group | ~Count | Definition approach | Lookbacks | Look-ahead note |
|---|---|---|---|---|---|
| 1 | Momentum / trend | 8 | Cumulative adjusted returns, skip-month momentum (12-1), distance to 50/200d MA, MA slope, RS vs sector ETF | 20d, 60d, 120d, 252d (skip 21d) | Prices only; `fetched_at` ≤ as_of trivially holds; adjustment version pinned per run |
| 2 | Mean-reversion | 5 | Short-horizon return z-scores, RSI(14), distance from 20d mean in σ units, max drawdown from 20d high | 5d, 10d, 14d, 20d | Same as momentum |
| 3 | Volatility | 6 | Realized vol (close-close + Parkinson), EWMA(60d) vol, vol-of-vol, vol percentile vs own 2y history, ATR(14)/price | 20d, 60d, 252d, 2y percentile window | Same; EWMA definition shared with risk engine (`06 §4`) for consistency |
| 4 | Volume / liquidity | 5 | Dollar-volume level and z-score, volume trend (20d/60d ratio), Amihud illiquidity, days-to-exit estimate | 20d, 60d | Same |
| 5 | Fundamental value | 6 | EV/EBIT, FCF yield, P/S vs sector percentile, yield vs own 3y history | Trailing-twelve-month from XBRL tags | **Only `xbrl_facts` rows with `filed_at` ≤ as_of.** TTM assembled from the four most recent *filed* quarters, not calendar quarters |
| 6 | Fundamental quality / growth | 7 | Gross margin level + stability (12q σ), ROIC, ROIC persistence, revenue growth (YoY, 3y CAGR), accruals ratio, FCF conversion | 4–12 filed quarters | Same `filed_at` rule; growth compares filings, never restated series |
| 7 | Estimate revisions | 3 | EPS estimate revision breadth and magnitude (Finnhub), forward-estimate momentum | 30d, 90d | Keyed on `fetched_at` of the estimate snapshot; snapshots stored append-only |
| 8 | Insider activity | 3 | Net insider buying dollars (90d), buyer count, cluster-buy indicator (mirrors `insider_cluster` detector, `02 §2`) | 90d, 180d | `insider_tx.filed_at` ≤ as_of — Form 4s count when filed, not when traded |
| 9 | Macro / regime context | 5 | RegimeState one-hot (`06 §8`), HY OAS level + 60d change, yield-curve slope, breadth (% Tier 1 > 200d MA) | Point-in-time daily | FRED vintages via `vintage_ts`; regime computed from same-day close data only |
| 10 | Calendar | 3 | Days to next earnings (Finnhub calendar), days since last earnings, month-of-year | Event distance | Calendar snapshot `fetched_at` ≤ as_of; unknown next-earnings ⇒ explicit sentinel, never imputed |

Total: ~51 features. Additions go through a PR touching `features/catalog.py` + a golden
snapshot update (§9); feature count creeping past ~60 is a review flag, not an achievement.

Missing data policy: LightGBM/CatBoost consume NaN natively; RandomForest gets median-impute
fitted **on the training fold only**. A symbol missing > 30% of features on a night is scored
with `confidence` clamped low (§8) and flagged.

## 3. Cross-sectional normalization

Features in groups 1–8 are converted to **cross-sectional ranks/z-scores within the scoring
universe per night** before training and inference (winsorized at 1st/99th percentile). This
makes the ranker learn *relative* attractiveness — matching how the output is consumed (a
ranked watchlist, `05`) — and immunizes against level shifts across regimes. Groups 9–10 stay
raw (they are context, shared across names).

## 4. Labels and targets

| Target | Definition | Type | Consumer |
|---|---|---|---|
| `excess_return_20d` | Forward 20-trading-day adjusted total return minus the same-period return of the symbol's **sector ETF** (SPDR map in config) | Regression — **primary ranker** | Watchlist rank (`05`), Kelly input (`06 §4`) |
| `downside_20d` | Indicator: minimum cumulative return over the next 20 trading days < −10% | Binary classifier → calibrated probability | `downside_prob` in QuantScore; arbiter context; sizing sanity |
| `realized_vol_20d` | Annualized realized vol over the next 20 trading days | Regression (log-vol) | Cross-check on EWMA σ used in sizing (`06 §4`); large disagreement is a flag |

**Why 20 days:** it matches the tactical sleeve's days–weeks horizon (`00 §3`), is long enough
to clear microstructure noise on EOD data, short enough to give ~12 non-overlapping periods
per symbol-year, and aligns the purge/embargo arithmetic (§5) with a single constant. The
long-term sleeve is served by quality scores (`05`), not by stretching this horizon; a 60d
auxiliary label may be added later through the same registry path.

**Why excess-vs-sector, not raw:** raw 20d returns are dominated by market and sector beta —
a model predicting them mostly relearns beta, which the platform neither wants (regime and
exposure are the risk engine's job, `06 §8–9`) nor can act on better than an ETF. Sector
excess isolates the cross-sectional selection signal, the only edge this system claims. It
also stabilizes training across regimes: the label distribution is roughly market-neutral by
construction.

Labels use adjusted closes with the same corporate-action versioning as features; a label
window crossing an adjustment recompute invalidates and rebuilds downstream (Dagster handles
this via asset deps, `02 §5`).

## 5. Training protocol — purged, embargoed walk-forward

López de Prado-style combinatorially honest evaluation, simplified to rolling folds:

```
|——— train 3y ———|purge 20d|— validate 6m —|embargo 10d|
        →  roll forward 6 months, repeat
```

> The canonical walk-forward protocol is owned by `08 §4` (`backtest/walkforward.py`) — one
> shared implementation for model training and strategy validation. The parameters below match
> it; any change lands there first.

| Parameter | Default | Rationale |
|---|---|---|
| Train window | 3 years (rolling, not expanding) | Bounds regime staleness; keeps monthly retrains < 10 min on the host |
| Validation window | 6 months | Enough non-overlapping 20d periods for a usable IC estimate |
| Purge | 20 trading days between train end and validation start | Exactly one label horizon — no label in train overlaps any validation feature date |
| Embargo | 10 trading days after validation before data re-enters any later train set (canonical value, `08 §4`) | Blocks leakage through serial correlation of features; purging already removes all label-window overlap |
| Folds retained | All rolling folds over ≥ 10y of history | OOS metrics are reported **per fold and pooled**; a model must win pooled *and* in ≥ 60% of folds |

Rules:

1. **Universe = point-in-time Tier 1.** Each fold's training and validation rows include only
   symbols with an active `universe_membership` tier-1 row on that date (`02 §2`), drawing
   prices from Tiingo's delisted-inclusive history. This is the survivorship defense: a name
   that later died or was demoted is in the folds where it was actually in the universe.
2. **No hyperparameter search on OOS.** Hyperparameters are tuned once via nested
   time-series CV *inside training folds only*, frozen into `TrainConfig`, and recorded in the
   registry. The walk-forward validation windows are touched exactly once per candidate — they
   are the report card, not the tuning set. Re-tuning requires a new model version.
3. **Retrain cadence: monthly** (first weekend of the month, Dagster job), producing a new
   `model_versions` row as `candidate`. Retrained models do **not** auto-deploy — promotion
   goes through the pipeline in `08` like everything else.
4. Determinism: fixed seeds, single-threaded tree building for the record-of-truth fit,
   `features_hash` (§7) pinning the exact feature set. Same inputs ⇒ bit-identical artifact.

## 6. Calibration and skill metrics

| Metric | Applies to | Ship bar |
|---|---|---|
| **Rank IC** (Spearman, feature date → 20d excess return), reported as mean and t-stat over non-overlapping validation periods | Excess-return ranker — **primary skill metric** | Pooled OOS rank IC > best statsforecast baseline, positive with t-stat ≥ 2, wins ≥ 60% of folds |
| Brier score + reliability curve (10 bins) | Downside classifier | Beats base-rate Brier; calibration curve within ±10pp per bin after calibration |
| RMSE on log-vol; QLIKE | Vol forecaster | Beats `AutoETS`-on-realized-vol baseline |
| IC decay profile (IC at 5/10/20/40d) | Ranker | Reported, not gated — feeds horizon sanity checks |

**Downside classifier calibration:** raw scores are calibrated with **isotonic regression**
fitted on validation folds (Platt scaling as fallback when a fold has < 500 positive labels —
isotonic overfits small samples). The stored `downside_prob` is always the calibrated value;
the calibrator is part of the model artifact, versioned with it. Reliability curves are
rendered on the dashboard next to the committee's calibration curves (`00 §4`) — the same
"is this trustworthy" standard applies to both humans-of-silicon.

Confidence (§8) is derived per prediction from: distance of the input row from the training
feature distribution (per-feature percentile coverage), fold-to-fold dispersion of the metric,
and missing-feature fraction. It is a 0–1 heuristic, documented as such — not a posterior.

## 7. Model registry

The registry is the only bridge between "a model exists" and "production uses it" (`01 §5`).

```sql
CREATE TABLE model_versions (
    id              text PRIMARY KEY,        -- e.g. ranker-lgbm-2026-08-01-a
    model_type      text NOT NULL,           -- ranker | downside | vol
    algo            text NOT NULL,           -- lightgbm | catboost | rf | statsforecast-<name>
    features_hash   text NOT NULL,           -- sha256 of ordered feature list + catalog version
    train_start     date NOT NULL,
    train_end       date NOT NULL,
    hyperparams     jsonb NOT NULL,
    metrics         jsonb NOT NULL,          -- per-fold + pooled OOS: rank_ic, brier, rmse, baselines beaten
    artifact_path   text NOT NULL,           -- object storage / backed-up volume path
    status          text NOT NULL,           -- candidate | paper | production | retired
    created_at      timestamptz NOT NULL,
    promoted_at     timestamptz,
    promoted_by     text                     -- human, for candidate→paper and paper→production
);
```

- `status` transitions are written **only by the promotion tool** (`08`) — the training job
  can insert `candidate` rows and nothing else; DB grants enforce this (`01 §7`), the app
  layer double-checks. `production` configs may reference only registry ids (`01 §5`).
- Exactly one `production` version per `model_type`; the previous holder flips to `retired`
  in the same transaction. Challengers can sit in `paper` indefinitely, scored nightly in
  shadow for comparison (`10`).
- Artifacts (model file + calibrator + feature list + `TrainConfig`) are content-addressed;
  the nightly scorer refuses to load an artifact whose hash disagrees with the registry row.

## 8. Inference — nightly scoring

Runs in the 23:00 ET feature/scoring job (`02 §4`) over **Tier 1 + currently promoted Tier-3
names**, using the `production` version of each model family:

```sql
CREATE TABLE quant_scores (
    symbol                  text NOT NULL,
    as_of                   date NOT NULL,
    model_version           text NOT NULL REFERENCES model_versions(id),
    expected_excess_return  double precision,   -- 20d, vs sector ETF
    downside_prob           double precision,   -- calibrated P(< -10% within 20d)
    vol_forecast            double precision,   -- annualized 20d
    rank                    integer,            -- 1 = best, cross-sectional on the night
    confidence              double precision,   -- 0-1, §6
    feature_coverage        double precision,   -- fraction of features non-missing
    PRIMARY KEY (symbol, as_of, model_version)
);
```

These are the `QuantScore` rows consumed by the arbiter (`03 §1`), the watchlist blend
(`05`), sizing (`06 §4`), and deep-dive selection (quant-rank jumps, `03 §1`). Failure
behavior mirrors the platform norm: a symbol whose features fail quality checks is skipped
with a logged suppression (`02 §6`); a scoring job that cannot load the production model
raises an alert and produces **no rows** — stale scores are never silently reused, and the
risk engine treats missing scores as "no new risk" fuel (`06 §1`).

Runtime budget: ~650 symbols × 3 models ≈ seconds on the host; cost is $0 — this is the
code-only counterweight to the metered Claude spend (`03 §5`).

## 9. Drift monitoring hooks

Two hooks emit nightly to Prometheus + the learning-loop tables; **thresholds, alert routing,
and the degraded-model response live in `10-learning-loop.md`** — this doc only defines the
emission points:

- **Feature distribution shift:** PSI per feature between the current night's cross-section
  and the training-window distribution stored in the model artifact.
- **Rolling IC decay:** realized rank IC of `production` predictions computed as each 20d
  window matures, tracked as a rolling series per model version — the live continuation of
  the §6 report card, and the primary retirement signal.

## 10. Testing

| Test class | What it asserts |
|---|---|
| **Property tests** (hypothesis) | No NaN/inf escapes the feature builder into training frames undeclared; scoring is deterministic given seed + artifact (bit-identical rows on rerun); normalization is scale-invariant; every feature respects declared lookback (window start never precedes `as_of − lookback`) |
| **Golden feature snapshots** | Frozen raw-data fixtures for 5 symbols × 60 days → exact expected feature values checked into the repo; any formula change shows up as a reviewed golden diff, mirroring the agent-output golden tests (`03 §7`) |
| **Look-ahead audit test** (automated, CI) | Shift **all** features forward +1 day against unshifted labels and retrain on one reference fold: OOS rank IC must collapse toward 0 (assert `abs(IC_shifted) < 0.25 × IC_unshifted` and t-stat < 1). A feature set that still "predicts" after the shift contains leakage; the build fails. Run on every change to `features/` and monthly against the current catalog |
| **Survivorship test** | Reference fold trained with point-in-time membership vs current-membership must show the expected IC gap on a fixture; guards against a regression that silently drops the membership join |
| **Registry integrity** | Attempting a `status` write outside the promotion tool fails (grant test); loading an artifact with a tampered hash fails |
| **Baseline harness** | statsforecast baselines run in the same walk-forward harness as ML models — asserted structurally in CI so the "bar to beat" can never quietly diverge from the candidates' evaluation |

## 11. Open questions (carried to 14-design-review)

1. Should the ranker get a 60d sibling label for the long-term sleeve, or does quality + committee
   conviction (`05`) suffice? → decide with M3 evidence.
2. Cross-sectional normalization universe during Tier-3 promotion surges (promoted names widen
   the cross-section) — pin to Tier 1 only, or include promoted? Default: include, revisit if
   rank churn spikes.
