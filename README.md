# Dynamo — Personal AI Investing Platform

A production-quality, single-user AI investing platform:

- **Claude research committee** — adversarial multi-agent analysis (dossier analyst → independent bull & bear advocates → Opus arbiter) with structured, evidence-referenced output. Claude is the only hosted LLM provider.
- **Quant models** — LightGBM/CatBoost + classical statistical forecasting for expected return, downside probability, volatility, and ranking.
- **Three-tier universe** — ~500–600 core names get full analysis; the rest of the US market is surveilled by near-free code-only anomaly detectors; interesting names are dynamically promoted.
- **Risk engine first** — hard-constraint veto over all advisory signals: exposure limits, VaR/CVaR, drawdown ladder, volatility targeting, fractional Kelly, regime detection. *Not investing* is a first-class output.
- **Continuous learning** — immutable decision context on every trade, post-trade reviews, confidence calibration, searchable journal, drift detection. Production logic changes only via backtest → OOS → walk-forward → stress → paper → shadow → human approval.
- **Paper trading first** (Alpaca), identical code path for live behind an explicit human gate.

## Status

**M1–M7 implemented; paper-validation phase.** All modules are built and tested (695 unit
tests, integration tests against Timescale, Dagster dry-run green). The complete technical
design lives in [`docs/design/`](docs/design/); decisions in [`docs/adr/`](docs/adr/).
Remaining before real money: accumulate clean unattended paper runs, tune from results, then
M8 hardening + the explicit human live-approval gate (`docs/design/00` §5).

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

Environments: `DEV | TEST | PAPER | LIVE`, one `.env.<env>` each — see `.env.example`.

## Layout

- `src/dynamo/domain` — pure typed entities & ports (no internal imports; enforced by import-linter)
- `src/dynamo/{research,risk,portfolio,backtest,learning}` — core logic (imports domain only)
- `src/dynamo/{data,llm,persistence,execution}` — adapters implementing domain ports
- `src/dynamo/{api,observability,config}` + `orchestration/` — wiring, FastAPI, Dagster
- `dashboard/` — Streamlit (M1–M5), React (M6)
- `docs/design/` — the technical design document (one file per section)
