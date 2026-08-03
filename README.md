# Dynamo — Personal AI Investing Platform

A production-quality, single-user AI investing platform:

- **Claude research committee** — adversarial multi-agent analysis (dossier analyst → independent bull & bear advocates → Opus arbiter) with structured, evidence-referenced output. Claude is the only hosted LLM provider.
- **Quant models** — LightGBM/CatBoost + classical statistical forecasting for expected return, downside probability, volatility, and ranking.
- **Three-tier universe** — ~500–600 core names get full analysis; the rest of the US market is surveilled by near-free code-only anomaly detectors; interesting names are dynamically promoted.
- **Risk engine first** — hard-constraint veto over all advisory signals: exposure limits, VaR/CVaR, drawdown ladder, volatility targeting, fractional Kelly, regime detection. *Not investing* is a first-class output.
- **Continuous learning** — immutable decision context on every trade, post-trade reviews, confidence calibration, searchable journal, drift detection. Production logic changes only via backtest → OOS → walk-forward → stress → paper → shadow → human approval.
- **Paper trading first** (Alpaca), identical code path for live behind an explicit human gate.

## Status

**Milestone 0 — design.** No functional code yet. The complete technical design lives in
[`docs/design/`](docs/design/) (start at `00-goals-and-non-goals.md`); decisions are recorded in
[`docs/adr/`](docs/adr/). Implementation begins at M1 (research pipeline) after design sign-off.

## Development

```
make dev        # uv sync
make check      # lint + typecheck + unit tests
make up         # docker compose stack
```

Environments: `DEV | TEST | PAPER | LIVE`, one `.env.<env>` each — see `.env.example`.

## Layout

- `src/dynamo/domain` — pure typed entities & ports (no internal imports; enforced by import-linter)
- `src/dynamo/{research,risk,portfolio,backtest,learning}` — core logic (imports domain only)
- `src/dynamo/{data,llm,persistence,execution}` — adapters implementing domain ports
- `src/dynamo/{api,observability,config}` + `orchestration/` — wiring, FastAPI, Dagster
- `dashboard/` — Streamlit (M1–M5), React (M6)
- `docs/design/` — the technical design document (one file per section)
