---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 08 — Backtesting & Validation

Backtesting exists to answer one question honestly: *would this strategy or model have earned
its risk?* Most backtests are optimistic fiction; this doc's job is the machinery that keeps
ours from being. It also owns the **promotion pipeline** — the only path by which any model,
strategy, or production parameter change reaches paper or live, ending in human approval
(auto-modifying production logic is a hard non-goal, `00-goals-and-non-goals.md` §7).

## 1. Two-engine design

| Engine | Purpose | Fidelity | Speed |
|---|---|---|---|
| **vectorbt** (research engine) | Fast signal iteration: parameter sweeps, feature research, quick kill of bad ideas | Vectorized, coarse costs (flat bps), no risk engine | Thousands of variants/hour |
| **Event-driven validator** (`backtest/validator.py`) | Final validation. A thin daily event loop that **imports the actual production `risk/` and `portfolio/` modules** — the same `RiskVerdict`, sizing cascade, drawdown ladder, and `TargetPortfolio` construction that run in paper/live | Full: constraints, ladder, turnover budget, cost model (§3) | Minutes per run |

The research engine may reject ideas; it may never approve them. **Every promotion-pipeline
result (§7) comes from the event-driven validator** — importing the production modules is the
only mechanism by which backtest ≈ paper ≈ live holds (guiding principle 3, doc 00 §8). The
validator is deliberately thin: it owns the event loop, fills, and accounting; all decision
logic lives in `risk/` and `portfolio/` and is reused, not reimplemented.

Per simulated session the validator runs the same sequence the nightly pipeline runs:

1. Advance the clock one trading day (exchange calendar, doc 01 §6); load point-in-time data
   with cutoff = that session's close.
2. Compute signals/scores for the strategy under test (research inputs are replayed or
   simulated per `StrategySpec`; committee verdicts are backtested only as recorded history —
   no retroactive Claude calls, ever).
3. Call the production risk engine → `RiskVerdict` (drawdown ladder, regime, caps — all live,
   doc 06).
4. Call production portfolio construction → `TargetPortfolio` (sleeve split, turnover budget,
   noise floor — all live, doc 07).
5. Diff → orders → fills at next open with the §3 cost model; update equity, ladder state,
   high-water mark.

Because steps 3–4 are imports, a bug fixed in `risk/` is fixed in the backtest the same
commit, and a constraint added in doc 07 is automatically binding historically.

Data comes from the **DuckDB/Parquet sidecar** (`02-data-platform.md` §3): columnar
full-history scans without touching the OLTP Postgres. The sidecar is rebuildable from
Postgres, and each backtest pins the partition manifest version it read (§8).

```python
class BacktestEngine(Protocol):
    def run(self, strategy: StrategySpec, window: DateRange,
            config: BacktestConfig) -> BacktestReport: ...
```

### Run cadence

| Run | Engine | When |
|---|---|---|
| Research iteration | vectorbt | On demand (dev), off the sidecar |
| Promotion stages 1–4 | Event-driven validator | On demand via `promotion.py`, promotion-grade config (§8) |
| Reproducibility regression | Event-driven validator | Weekly CI: re-run one archived promotion run, diff byte-for-byte |
| Production-strategy health re-run | Event-driven validator | Monthly: re-run each production strategy over the trailing 2y; degradation feeds `10-learning-loop.md` decay monitoring |

## 2. Survivorship and look-ahead defenses (data layer)

- **Point-in-time universe:** every backtest joins `universe_membership` on
  `valid_from/valid_to` (doc 02 §2) — a symbol is only tradable on dates it was actually in
  tier. No backtest may take a symbol list "as of today" into the past; the validator has no
  API for it.
- **Delisted names:** Tiingo's delisted coverage (doc 02 §1) means dead companies stay in the
  history with their terminal returns — the graveyard is priced in.
- **Corporate actions:** signals and labels computed on adjusted series, fills on raw prices
  (doc 02 §5), with the `adjustment_version` recorded.
- **Fundamentals:** `xbrl_facts` keyed by `filed_at`, never restated backwards (doc 02 §3) —
  a backtest on date *t* sees only facts filed by *t*. Same discipline for FRED macro via
  `vintage_ts`.
- **Cutoff discipline:** every feature row carries its data-cutoff timestamp (doc 01 §6); the
  validator asserts `cutoff ≤ decision date` on every input it consumes and aborts on
  violation rather than warning.

## 3. Cost and slippage model

Commission $0 (Alpaca). Slippage is deliberately conservative — a strategy that only works at
optimistic costs does not work:

```
cost_per_side = half_spread(liquidity_decile) + impact
impact        = k · σ_daily · sqrt(order_value / ADV_dollar)     # square-root model, k = 1.0
```

| Liquidity decile (20d median dollar volume) | Half-spread assumption |
|---|---|
| 1–2 (mega/large, > $100M ADV) | 2 bps |
| 3–5 ($20–100M) | 5 bps |
| 6–8 ($5–20M) | 12 bps |
| 9–10 (< $5M) | 25 bps |

Defaults are intentionally on the expensive side of the literature. From M5, realized paper
fills are tracked against this model (`10-learning-loop.md` slippage tracking); the model is
recalibrated from actual fill data quarterly, and calibration changes are versioned in
`BacktestConfig` — a cost-model change invalidates cached reports (§8). Fills execute at next
session's open after a signal date (no same-bar fills, ever).

## 4. Walk-forward protocol (`backtest/walkforward.py`)

One shared implementation used by **both** quant model training (`04-quant-models.md`) and
strategy validation — two consumers, one splitter, zero chance of the strategy path quietly
using a leakier scheme.

- **Rolling scheme (defaults):** train 3y → embargo 10 trading days → test 6 months → roll
  forward 6 months. Minimum 6 folds for any promotion-grade result.
- **Purging:** samples whose label window overlaps the test period are dropped from training
  (labels look forward, so unpurged adjacency leaks).
- **Embargo:** 10 trading days between train end and test start, covering feature lookbacks
  and serial correlation.
- **Frozen config:** a strategy/model's full configuration (features, parameters, thresholds)
  is hashed and **frozen before OOS evaluation begins**. Any config change after seeing OOS
  results restarts the protocol from stage 1 with a new trial logged (§6). The walkforward
  runner enforces this: it refuses to evaluate a config hash that has already seen the
  requested OOS window.
- Results reported **per-fold and aggregated**; a strategy that lives on one heroic fold fails
  review regardless of the aggregate.

## 5. Metrics

Every `BacktestReport` includes, over the full window and per-fold:

| Group | Metrics |
|---|---|
| Return | CAGR, total return, alpha/beta vs SPY |
| Risk-adjusted | Sharpe, Sortino, Calmar, **deflated Sharpe** (§6) |
| Drawdown | Max drawdown, depth/duration table, drawdown-ladder trigger counts (doc 06 §3) |
| Trade quality | Profit factor, win rate, avg win/loss, **worst-5 trades** (with dates and context) |
| Behavior | Exposure (avg/max gross), cash fraction over time, one-way turnover, position count distribution |
| Regime | All of the above broken down per `RegimeState` (`risk_on`/`neutral`/`risk_off`/`crisis`), from the doc 06 §8 backfill — a strategy that only earns in `risk_on` is labeled as such, not rejected silently |

Benchmarks in every report: SPY total return, and for forecasting models the §6 naive
baselines. No single-number verdicts — the report is a document, not a scalar.

```python
class BacktestReport(BaseModel):
    run_id: str; engine: Literal["vectorbt", "event_driven"]
    strategy_ref: str; window: DateRange
    provenance: RunProvenance            # git commit, config hash, data snapshot, seed, trial no. (§8)
    metrics: MetricsBlock                # full window
    folds: list[MetricsBlock]            # per walk-forward fold (§4)
    per_regime: dict[str, MetricsBlock]  # regime states from doc 06 §8 backfill
    worst_trades: list[TradeRecord]      # worst-5 with dates and entry context
    stress: list[StressResult]           # 2008 / 2020 / 2022 + synthetic shocks
    audits: AuditBlock                   # look-ahead shift result, baseline comparisons,
                                         # deflated Sharpe, constraint-violation count (must be 0)
    equity_curve_ref: str                # Parquet artifact path
```

Only `engine="event_driven"` reports are admissible as promotion evidence (§7); the registry
rejects vectorbt reports attached to a stage transition.

## 6. Overfitting defenses

1. **No hyperparameter search on OOS.** Tuning happens inside training folds only. The
   walkforward runner's config-freeze (§4) makes this mechanical, not aspirational.
2. **Trial accounting.** Every backtest run of a strategy family increments a persistent trial
   counter (keyed by strategy family, stored in the model registry). Reports state the trial
   count and report the **deflated Sharpe ratio** (Bailey & López de Prado) computed against
   it. Running 200 variants and reporting the best is only cheating if you hide the 200.
3. **Mandatory naive baselines.** Quant forecasting models must beat `statsforecast` naive,
   seasonal-naive, and historical-mean baselines on the same folds; strategies must beat
   buy-and-hold SPY and a random-entry/same-exit-rules bootstrap on risk-adjusted terms.
   A model that cannot beat naive is retired regardless of its absolute Sharpe.
4. **Look-ahead audit (automated).** Every promotion-grade run is re-executed with all features
   shifted +1 day. If performance does not degrade materially (default: Sharpe drop ≥ 50%),
   the pipeline **fails the run** — surviving the shift means the edge was leakage. This is a
   required stage gate, not an optional diagnostic.
5. **Stress windows mandatory.** Every report includes 2008 (GFC), 2020 (COVID crash), and
   2022 (rate shock) windows explicitly, with ladder behavior shown. Absent history for those
   windows (young data) is stated on the report, never silently skipped.
6. **Regime honesty.** Per-regime breakdown (§5) is rendered in every report so regime-luck is
   visible at review time.

## 7. The promotion pipeline

This doc owns the pipeline. It applies to quant models (doc 04), strategies (doc 05), and
production parameter changes flagged as promotion-scoped (risk config, doc 06 §11;
construction config, doc 07 §8).

### Stages and pass criteria

| # | Stage | Pass criteria (defaults; per-artifact overrides recorded in registry) |
|---|---|---|
| 1 | Backtest (event-driven validator, full history) | Beats naive baselines & SPY risk-adjusted; deflated Sharpe > 0; max DD within bucket ladder tolerances; look-ahead audit passed |
| 2 | Out-of-sample | Held-out period (most recent 12m, never touched in development): Sharpe ≥ 50% of in-sample; same sign of alpha |
| 3 | Walk-forward | ≥ 6 folds per §4; positive median fold Sharpe; no single fold contributing > 50% of total return |
| 4 | Stress | 2008/2020/2022 replays + doc 06 §6 synthetic shocks: no ladder −15% breach; behavior explainable at review |
| 5 | Paper | **≥ 30 trading days** nightly-pipeline paper execution; realized vs backtest tracking error within band; slippage within model (§3); zero risk-constraint violations |
| 6 | Shadow | Runs alongside current production on identical inputs; decisions **logged, not executed**; ≥ 20 trading days; divergence report reviewed |
| 7 | Human approval | Owner reviews the evidence bundle and signs; without the artifact, the transition API refuses |

Every stage emits a report artifact; the accumulated set is the **evidence bundle** attached to
the final approval.

### State machine (model registry)

Statuses on `model_versions` / strategy registry rows:

```
candidate → validating → paper → shadow → production → retired
                │            │        │          │
                └────────────┴────────┴──────────┴──→ rejected/retired (any stage may fail back)
```

- **Only the promotion tool writes transitions.** `backtest/promotion.py` is a CLI; no other
  code path (and no Dagster job) has write access to the status column — enforced by a
  dedicated DB role grant, not convention.
- **Production config can only reference production-status registry entries.** Config loading
  resolves model/strategy references against the registry and refuses anything not in
  `production` status (doc 01 §5). A typo'd or unpromoted reference fails startup loudly.
- Every transition writes a `PromotionRecord`: who/when, from→to, evidence-bundle ref, trial
  count at promotion, git commit, config hash.
- **Human approval is a required stage, never bypassed.** The approval is a signed artifact
  checked into the repo (`approvals/promotion-<name>-<version>.yaml`), same mechanism as the
  live gate (doc 01 §5); `promotion.py` verifies it before writing `shadow → production`.
  There is no `--force`. Demotion (`production → retired`) requires the CLI but not the
  30-day pipeline — pulling something out is always fast.

## 8. Reproducibility

Every backtest run records, in the report header and the registry:

| Field | Source |
|---|---|
| `git_commit` | Repo HEAD (dirty tree → refuse promotion-grade runs) |
| `config_hash` | SHA-256 of the frozen `StrategySpec` + `BacktestConfig` |
| `data_snapshot` | Parquet partition manifest version from the sidecar export (doc 02 §3) |
| `seed` | For any stochastic component (Monte Carlo, bootstrap baselines) |
| `trial_number` | Per strategy family (§6) |

Reports (HTML + JSON) are stored as artifacts under `artifacts/backtests/<run_id>/`, backed up
with the database (`12-operations.md`). Re-running with identical fields must reproduce the
report byte-for-byte (excluding timestamps); a scheduled weekly CI job re-runs one archived
promotion-grade backtest and diffs it as a regression check.

## 9. Testing the test harness

The backtester is production code that can lie; it gets its own suite:

| Test | Contract |
|---|---|
| Engine unit tests | Hand-computed toy cases: 3 symbols, 10 bars, known fills/costs → exact equity curve, drawdown, and per-trade P&L asserted to the cent |
| **Parity test** | Event-driven validator vs vectorbt on a simple MA-cross strategy with matched (flat) costs and no risk constraints: terminal equity within 0.5%, trade lists identical |
| **Look-ahead canary** | A deliberately look-ahead-biased strategy (uses close of *t* to trade at open of *t*) is kept in the test suite; the §6 audit **must** flag it — if the canary passes validation, CI fails |
| Survivorship canary | A strategy that buys names the day before delisting must show the losses; a run configured without the `universe_membership` join must be impossible to construct (API-level, asserted) |
| Cost-model sanity | Zero-cost config on the validator matches vectorbt exactly; monotonicity: larger orders never model lower impact |
| Promotion-tool tests | State machine rejects illegal transitions, refuses missing approval artifacts, refuses dirty-tree promotion runs; registry role-grant checked in integration tests |

## 10. Open questions (carried to 14-design-review)

1. Shadow stage for *parameter* changes (vs whole strategies): is 20 days of divergence
   logging overkill for a single cap tweak? → propose a scoped "minor change" lane in M6 with
   explicit criteria, still ending in human approval.
2. Should the paper stage require a minimum trade count (not just 30 days) for low-frequency
   long-term strategies? → decide once first long-term candidate reaches stage 5.
