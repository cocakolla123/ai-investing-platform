---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 00 — Goals and Non-Goals

## 1. Product vision

Dynamo is a **single-user, production-quality AI investing platform** whose objective is to
**maximize long-term, risk-adjusted growth of one personal portfolio while protecting capital
through disciplined, auditable risk management.**

It is not a trading bot and not a stock-picking chatbot. It is a research-and-portfolio
intelligence system in which:

- **Claude** functions as an institutional-style research team (adversarial committee, structured
  evidence-referenced output) — never as the sole decision-maker.
- **Quantitative models** provide numerical estimates (expected return, downside probability,
  volatility, ranking) that are validated out-of-sample before they are trusted.
- **The risk engine holds final authority.** Every other component is advisory. "Do nothing" is a
  frequent, correct, and celebrated output.
- **The system learns**, but never modifies its own production logic: every improvement passes a
  fixed promotion pipeline ending in human approval.

## 2. Primary objective function

Long-horizon risk-adjusted growth, operationalized as:

1. **Capital preservation constraints first** — max-drawdown ladder and exposure limits are hard
   constraints (see `06-risk-engine.md`).
2. **Risk-adjusted return second** — Sortino/Sharpe over rolling 1y/3y windows against a passive
   benchmark (SPY total return) once live capital exists.
3. **Process quality always** — because outcome metrics on a single personal portfolio are
   statistically weak for years, the platform is primarily steered by *process* metrics (§4).

## 3. Users and scope

- **One user** (the owner). No multi-tenancy, no auth beyond a single dashboard credential.
- **Assets:** US equities + ETFs (tradable), Kalshi prediction markets (small gated sleeve,
  demo-first — ADR-0009), Polymarket (research-only).
- **Horizons:** long-term quality sleeve (months–years) + tactical sleeve (days–weeks),
  allocation configurable.
- **Execution:** Alpaca **paper** trading. Live trading is a future *decision* gated by
  §5 live-readiness criteria plus explicit human approval — never a default milestone.

## 4. Success metrics

Outcome metrics take years to be significant for one portfolio; process metrics are measurable
within weeks. Both are tracked from M1.

### Process metrics (primary steering signals)

| Metric | Target | Why |
|---|---|---|
| Nightly pipeline completion rate | ≥ 99% unattended | Reliability is a feature |
| Agent output schema validity | ≥ 99% | Structured output is the platform's contract |
| Confidence calibration (Brier score / reliability curves per agent & committee) | Monotonic improvement; committee better than any single agent | The core "is the AI trustworthy" measure |
| Recommendation traceability | 100% of claims carry evidence refs | Explainability requirement |
| Abstention rate (`no_action`/`avoid`) | Reported prominently; no target — but never near-zero | A near-zero abstention rate means the committee is a yes-machine |
| Claude spend vs budget | ≤ configured cap, always | Hard constraint |
| Data quality incidents caught before use | 100% of injected synthetic faults | Trust in inputs |
| Decision-context completeness | 100% of orders have immutable snapshots | Learning loop depends on it |

### Outcome metrics (reported, not chased short-term)

Paper/live portfolio: total return, Sharpe, Sortino, max drawdown, profit factor, win rate,
alpha/beta vs SPY, per-sleeve and per-strategy attribution, turnover, realized slippage vs model.

**Anti-goal:** "beat SPY in 3 months." Any strategy change justified by short-window
outperformance without process evidence is treated as a red flag, not a success.

## 5. Live-readiness gate (defined now, evaluated much later)

Live capital requires **all** of:
1. ≥ 3 consecutive months of clean paper operation (no unreconciled positions, no missed risk
   checks, pipeline completion ≥ 99%).
2. Committee calibration curves demonstrably informative (conviction correlates with outcomes).
3. Risk engine property-test suite green; drawdown ladder exercised in simulation.
4. Backup/restore drill passed within the previous quarter.
5. A written human approval artifact (signed file checked into the repo) — the code refuses live
   orders without it (`09-execution.md`).

## 6. Budget contract

**Hard ceiling: $150/month total** (target ~$70–120):

| Item | Budget |
|---|---|
| Claude API | $40–90 (kill-switch at configured cap, alert at 70%) |
| Market data (Tiingo) | ~$10 |
| Host (single VPS or home server) | $0–15 |
| Backups/object storage | ~$5 |

Design consequences: Message Batches API for all research (50% discount), prompt caching, model
tiering (Haiku screen → Sonnet analysis → Opus arbitration), deep-dive caps per night, three-tier
universe with code-only surveillance for the broad market, free-tier data providers with one paid
validation source.

## 7. Non-goals (explicit)

| Non-goal | Rationale / revisit condition |
|---|---|
| Intraday / high-frequency trading | No edge at personal scale; free data isn't tick-grade. Not revisited. |
| Options, futures, crypto trading | Complexity + risk model rewrite. Revisit only after live equities run ≥ 1y cleanly. |
| Beating prediction markets systematically | Still a non-assumption. Kalshi is now *tradable* behind the same gates as equities (demo-first, ≤ 5% sleeve, evidence-gated — ADR-0009); Polymarket remains research-only with no execution hooks. The sleeve tests whether committee estimates carry edge; it does not presume they do. |
| Multi-user / SaaS | Single-user by design; removes auth, tenancy, compliance surface. |
| Kubernetes | Compose on one host; images stay 12-factor/K8s-compatible. Revisit only on real scale need. |
| Deep learning forecasting (LSTM/transformers) | ADR-0003: insufficient data scale; modular `ForecastModel` port keeps the door open with defined revisit criteria. |
| Full-market deep research | Tiers: ~500–600 core names get deep analysis; the rest get code-only surveillance + dynamic promotion (`02-data-platform.md`). |
| Auto-modifying production logic | Forbidden. Promotion pipeline + human approval only (`08`, `10`). |
| Sub-second latency anywhere | Nightly batch cadence + 15-min intraday risk checks are sufficient for the strategies in scope. |

## 8. Guiding principles (tie-breakers for design disputes)

1. **Risk engine wins.** Any conflict between opportunity and a risk constraint resolves to the
   constraint.
2. **Evidence over eloquence.** An unreferenced claim is discarded, however plausible.
3. **Same code path** for backtest final-validation, paper, and live (adapter swaps only).
4. **Boring reliability beats clever fragility.** Fewer components, one database, one host.
5. **Every dollar of spend is metered** and visible; the system degrades gracefully (smaller
   batches) rather than surprising on cost.
6. **Human-in-the-loop at every irreversible boundary**: universe changes, model promotion,
   live trading.
