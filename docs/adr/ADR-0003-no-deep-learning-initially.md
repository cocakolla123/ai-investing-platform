# ADR-0003 — No LSTM/transformer forecasting initially; modular `ForecastModel` port

**Status:** accepted (user-approved) · 2026-08-03

## Context
The user's original spec listed LSTM and transformer time-series models alongside gradient
boosting. Dynamo's training data is daily bars and fundamentals for ~500–600 names over
~10–20 usable years — small and extremely noisy by deep-learning standards. Deep sequence
models at this scale overwhelmingly memorize noise and produce impressive in-sample results
that fail out-of-sample; gradient boosting plus classical statistical forecasting is the
documented frontier for tabular/cross-sectional financial prediction at this data size.

## Decision
Start with LightGBM (primary), CatBoost/RandomForest (challengers), and Nixtla statsforecast
classical baselines. All forecasting sits behind the `ForecastModel` port so any model family
can be added without touching consumers.

## Revisit criteria (all required)
1. GBM ranker shows a **stable, positive out-of-sample rank IC** across ≥ 4 consecutive
   walk-forward folds (there is real signal to chase), **and**
2. Evidence of a capacity ceiling: added informative features no longer improve OOS IC, **and**
3. A candidate deep model beats the production GBM on the identical purged walk-forward
   protocol, then survives the full promotion pipeline (doc 08) including paper trading.

## Consequences
Lower cost, faster iteration, explainable feature importances; a documented, criteria-gated
door back to deep learning rather than a prohibition.
