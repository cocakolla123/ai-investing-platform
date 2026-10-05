# Dynamo — Personal AI Investing Platform

A production-quality, single-user AI investing platform:

- **Claude research committee** — adversarial multi-agent analysis (dossier analyst → independent bull & bear advocates → Opus arbiter) with structured, evidence-referenced output. Claude is the only hosted LLM provider.
- **Quant models** — LightGBM/CatBoost + classical statistical forecasting for expected return, downside probability, volatility, and ranking.
- **Three-tier universe** — ~500–600 core names get full analysis; the rest of the US market is surveilled by near-free code-only anomaly detectors; interesting names are dynamically promoted.
- **Risk engine first** — hard-constraint veto over all advisory signals: exposure limits, VaR/CVaR, drawdown ladder, volatility targeting, fractional Kelly, regime detection. *Not investing* is a first-class output.
- **Continuous learning** — immutable decision context on every trade, post-trade reviews, confidence calibration, searchable journal, drift detection. Production logic changes only via backtest → OOS → walk-forward → stress → paper → shadow → human approval.
- **Paper trading first** (Alpaca), identical code path for live (real money) behind an explicit human approval gate.

## Status

**M1–M7 implemented; paper-validation phase.** All modules are built and tested: 2,701 unit
tests passing and 56 skipped, plus 60 integration tests against Postgres/Timescale
(counted 2026-10-05). The complete technical
design lives in [`docs/design/`](docs/design/); decisions in [`docs/adr/`](docs/adr/).
Remaining before real money: accumulate clean unattended paper runs, tune from results, then
M8 hardening + the explicit human approval gate for live (real money) (`docs/design/00` §5).

## Results so far (all paper or replay; no real money)

Every result below was pre-registered before it ran, and null results are
reported as null.

- **Quant model:** LightGBM (campaign v10) beat no registered baseline. Plain
  12-1 momentum ranked stocks at least as well. ([report](docs/ml_baselines_2026-10-03.md),
  [re-run on fixed prices](docs/rerun_after_fixes_2026-10-04.md))
- **Retrieval, v1:** BM25 beat vector and hybrid on cross-stock queries.
  Production's ticker filter returns nothing for any stock without its own
  history (19.5% of eligible decisions got a precedent). ([report](docs/eval_retrieval_2026-10-03.md))
- **Retrieval, v2:** after filling journal metadata, a sector fallback beats
  production only by returning something. Against plain vector or BM25 it is
  not distinguishable on independent labels. ([report](docs/eval_retrieval_v2_2026-10-05.md))
- **Retrieval on decisions:** turning journal retrieval off, on or hybrid made
  no difference beyond run-to-run noise in committee decisions. ([report](docs/eval_decisions_2026-10-03.md))
- **Parallel advocates:** cut committee wall clock by 10.9% (CI 5.2% to 16.1%),
  about 2 minutes a night. It was declined because its pre-registered quality
  rule narrowly failed. ([report](docs/eval_tracing_2026-10-03.md))
- **Trading:** no strategy shows alpha distinguishable from zero. The committee
  campaign's +11.1% (SPY about +14.7%) was mostly SPY ballast, averaging 76.7%
  of equity, with about 2.6 committee positions at a time. ([report](docs/rerun_after_fixes_2026-10-04.md))
- **Everything is paper or replay.** The nightly pipeline trades an Alpaca
  paper account; no real-money order has ever been placed (see the glossary).

## Quickstart

1. `cp .env.example .env` and fill in keys. Minimum for a research-only run:
   `ANTHROPIC_API_KEY` (fund console.anthropic.com credits), `POSTGRES_PASSWORD`.
   Add `TIINGO_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`, `ALPACA_API_KEY/SECRET`
   (paper) as you enable more of the pipeline — every adapter degrades gracefully
   when its key is absent.
2. `docker compose up -d postgres && make migrate`
3. `make dagster` → http://127.0.0.1:3000 (pipeline UI; schedules run the nightly loop)
4. `make api` → http://127.0.0.1:8000 · `make dash` → http://127.0.0.1:8501
5. `make pipeline-dry` — wiring check with no keys or DB needed.

macOS note: LightGBM needs libomp (`brew install libomp`). Linux containers install
libgomp1 via `infra/docker/Dockerfile.app`.

## Development

```
make dev        # uv sync
make check      # lint + typecheck + unit tests
make up         # full docker compose stack (dagster, api, dashboard, prometheus, grafana)
```

Environments: `DEV | TEST | PAPER | LIVE`, selected by `DYNAMO_ENV` (default `DEV`).
Settings come from the process environment, then a single `.env` file in the working
directory; see `.env.example` for the keys.

## Glossary: paper, replay, shadow, LIVE

- **Paper:** the nightly production pipeline (Dagster) trading an Alpaca **paper**
  account: simulated fills, no real money. Docs call it "the nightly paper
  configuration" or "production (paper)". This is what runs today.
- **Replay:** offline re-runs of past dates through the same code (`dynamo.cli.replay`),
  writing only to replay tables. Used for every backtest and evaluation.
- **Shadow:** a candidate scored nightly beside production and never traded
  (for example `shadow_quant_scores`, docs/preregistration_trend_shadow.md).
- **LIVE:** real money. It requires `DYNAMO_ENV=LIVE`, a signed approval artifact
  in `approvals/`, no kill-switch file, and the readiness criteria, all checked by
  `live_orders_permitted()` in `src/dynamo/execution/gate.py` (see
  [`approvals/README.md`](approvals/README.md)). The broker also refuses to build a
  real-money client unless `DYNAMO_ENV=LIVE`. LIVE has never been enabled.

In code, `live` often just means "not replay" (for example `source='live'`,
"live parity"); those identifiers keep their names.

## Layout

- `src/dynamo/domain` — pure typed entities & ports (no internal imports; enforced by import-linter)
- `src/dynamo/{research,risk,portfolio,backtest,learning}` — core logic (imports domain only)
- `src/dynamo/{data,llm,persistence,execution}` — adapters implementing domain ports
- `src/dynamo/{api,observability,config}` + `orchestration/` — wiring, FastAPI, Dagster
- `dashboard/` — Streamlit (M1–M5), React (M6)
- `docs/design/` — the technical design document (one file per section)
