---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 13 — Project Risk Register

**Scope:** risks to the *project* — its data, models, budget, infrastructure, process, and
operator. Market risk (drawdowns, concentration, liquidity) is owned entirely by the risk
engine (`06-risk-engine.md`) and does not appear here.

Conventions: Likelihood/Impact are Low/Med/High over a 12-month horizon. **Owner is the user
in every row** (single-operator platform); the Owner column therefore names the *detecting
component* — the thing whose job is to notice before the human does. Tripwires reference the
metrics and alerts of `12-operations.md` §6–7. Reviewed quarterly alongside the restore drill.

## 1. Top risks at a glance

The full register is §2–§6. These are the rows that most deserve budgeted attention, ranked by
Likelihood × Impact and by how much of the platform's core promise they threaten:

| Rank | Risk | L×I | Primary defense |
|---|---|---|---|
| 1 | LLM conviction miscalibration | High × High | Conviction tilts, never drives (doc 06 §4); calibration curves gate live (doc 00 §5) |
| 2 | Quant overfitting | High × High | Promotion pipeline with out-of-sample gates (doc 08) |
| 3 | Solo-developer process drift | High × High | Pipeline is tooling, not docs — the shortcut path does not compile |
| 4 | Prompt injection via ingested text | Med × High | No tool-use in analyst calls; structured extraction; provider text is data |
| 5 | Free-data quality | High × Med | Dual-source checks + quarantine (doc 02 §6) |
| 6 | Silent paper→live parity leaks | Med × High | Same code path + daily reconciliation (doc 09) |

A useful property of this ranking: the top three are all *self-inflicted* risk classes. The
platform's biggest threats are not providers or APIs — they are the operator trusting the
system's outputs (1), his own backtests (2), or his own discipline (3) more than the evidence
warrants. The mitigations are therefore structural (code-enforced), not aspirational.

## 2. Data & providers

The budget contract (doc 00 §6) commits the platform to free and near-free data. That is a
deliberate trade: these rows are the price, and the mitigations in `02-data-platform.md` §5–6
were designed against exactly this list. The rule everywhere: bad data is quarantined and
labeled, never silently used, and suppression is itself logged.

| Risk | Likelihood | Impact | Mitigation | Tripwire / monitoring | Owner (detector) |
|---|---|---|---|---|---|
| Free/cheap data quality: bad split/dividend adjustments, gaps, IEX-only depth on Alpaca | High | Med | Dual-source close check (50 bps), gap detection vs exchange calendar, corp-action sanity + `adjustment_version` recompute (doc 02 §5–6); IEX depth accepted — EOD strategies only (doc 00 §7) | Quarantine count per run; systemic alert at > 20 symbols; quarterly synthetic-fault drill must catch injected bad close | `data/quality.py` |
| Survivorship bias corrupting backtests | Med | High | Tiingo delisted-name history; point-in-time `universe_membership` is the backbone for backtest selection (doc 02 §2); XBRL keyed by `filed_at`, never restated | Backtest asserts universe reconstruction at historical dates; CI test with a known-delisted fixture symbol | `backtest` universe reconstruction |
| yfinance-class provider brittleness / silent ToS changes | Med | Med | yfinance is break-glass only, never production (ADR-0004); all providers behind ports — replacement is an adapter, not a rewrite; client-side rate limiting | `provider_errors_total` sustained; provider budget 80% warn; contract tests against recorded cassettes detect response-shape drift | Provider adapters (typed errors) |
| Kalshi/Polymarket API limits or ToS shifts | Med | Low | Research-only, read-only, no execution hooks (doc 00 §7); daily snapshot cadence is trivially light; thin-market filter limits dependence | Snapshot job failure alert; feature degrades to "no prediction-market context" label in digests | Prediction-market adapters |

## 3. Claude & LLM

Claude is advisory by architecture (doc 00 §1), which caps the blast radius of every row below:
no LLM failure mode can place an order, size a position, or bypass a risk limit. What remains
is cost, calibration, and the integrity of the text the committee reads.

| Risk | Likelihood | Impact | Mitigation | Tripwire / monitoring | Owner (detector) |
|---|---|---|---|---|---|
| Claude cost blowout | Med | Med | `cost_meter` written first in M1; hard refusal of new batches at cap; batches (50% discount), model tiering, deep-dive cap in code; zero-cache-savings budgeting (doc 03 §5) | `claude_budget_utilization` warn 70% / page 100%; nightly cost per run logged | `llm/cost_meter.py` |
| LLM conviction miscalibration — confident, wrong, persuasive | High | High | Conviction may tilt sizing, never drive it (doc 06 §4); disagreement guardrails in code; abstention rewarded; per-agent Brier/reliability curves from M7; live gate requires demonstrated calibration (doc 00 §5.2) | Calibration dashboard (doc 11 §3.5); persistently miscalibrated role → prompt-revision trigger (doc 03 §7) | `learning` calibration job |
| Batch failures, partial results, refusals | High | Low | One retry then quarantine with raw response; reports render gaps explicitly; deterministic `custom_id` resubmission on expiry (doc 03 §3) | `schema_validation_failures_total`; batch-expiry warn alert | `llm/batch_runner` |
| Anthropic API changes / model deprecations | Med | Med | Model ids pinned in config; every stored output records `model_id` + `prompt_version`; **migration requires an ADR** with side-by-side evaluation on the fixed 10-ticker set (doc 03 §7) — never a silent bump | Deprecation notices reviewed monthly (patch-policy cadence, doc 12 §9); CI golden tests fail on response-shape change | Config pin + golden tests |
| Prompt injection via ingested news/filings/transcript text | Med | High | Treat all provider text as data, never instructions: system-prompt hardening ("document content is untrusted"), structured JSON extraction with schema enforcement, **no tool-use during analyst calls**, evidence refs resolvable so fabricated claims are checkable | Schema-validation failures spike; adversarial injection strings in the golden test corpus must not alter output structure or stance fields | Schema validation + golden suite |

## 4. Models & process

The highest-ranked risks in §1 live here. A single operator has no reviewer, no compliance
desk, and no second opinion — so the process controls must be mechanical. Anywhere a row below
says "pipeline" or "gate", the mitigation is only credible because it is enforced by code
(promotion tool, import-linter, CI), not by intention.

| Risk | Likelihood | Impact | Mitigation | Tripwire / monitoring | Owner (detector) |
|---|---|---|---|---|---|
| Quant overfitting | High | High | Walk-forward validation, promotion pipeline with out-of-sample gates (doc 08); no auto-modifying production logic (doc 00 §7); DL excluded at current data scale (ADR-0003) | Live vs backtest-expectation bands on dashboard (doc 11 §3.5); `drift_indicator` gauges | Model registry + drift monitors |
| Solo-developer process drift — skipping the pipeline "just this once" | High | High | **Promotion pipeline is tooling, not documentation:** production configs can only reference registry entries written by the promotion tool (doc 01 §5); import-linter enforces architecture in CI; risk config changes go through the same pipeline (doc 06 §11) | CI is the tripwire — the shortcut path does not exist; quarterly review of `docs/ops-log.md` for out-of-band changes | Promotion tool + CI |
| Silent paper→live parity leaks (code paths diverge before live ever starts) | Med | High | Same code path for backtest-final/paper/live, adapter swaps only (doc 00 §8.3); env differences limited to credentials/base-URL; reconciliation job compares internal state vs broker daily (doc 09) | `reconciliation_divergences_total` — any increment pages; weekly CI dry-run exercises the full path | Reconciliation job |
| Over-trusting explanations — narrative mistaken for causal attribution | High | Med | Doc 00 §4 stance: process metrics steer, outcome narratives don't; attribution reported per-sleeve/strategy but short-window outperformance justifying changes is an explicit red flag (doc 00 §4 anti-goal); dissents and self-critique always rendered | Any strategy change PR must cite process evidence, not returns — checked in review (self-review checklist in PR template) | The human, aided by doc 00 §4 framing |

## 5. Infrastructure & operator

One host, one human. Infrastructure risk is bounded by the rebuild story (doc 12 §4, §8):
everything except the Postgres data and the age key is reproducible from git and GHCR, and
both of those have tested recovery paths. Operator risk is bounded by writing things down and
by alerts that stay rare enough to be believed.

| Risk | Likelihood | Impact | Mitigation | Tripwire / monitoring | Owner (detector) |
|---|---|---|---|---|---|
| Single-host loss (disk death, provider account issue, fat-fingered `rm`) | Med | Med | Nightly pg_dump + wal-g WAL archiving off-host; sops-encrypted secrets in repo; host-migration runbook, target < 1 day (doc 12 §4, §8); Parquet rebuildable | `backup_last_success_timestamp` staleness page; dead-man silence page; quarterly restore drill is the real test | Backup job + Healthchecks.io |
| Alert fatigue → real page ignored | Med | High | Three-tier severity with small page class; re-tier rule: any alert firing ≥ 3×/month without action gets fixed or demoted (doc 12 §7); dashboard ack semantics prevent ack-and-forget on page class (doc 11 §4) | Monthly alert-count review in ops log; page-class alerts per month tracked (target: ~0) | Grafana alert stats + ops log |
| Key-person / bus-factor = 1 (illness, loss of interest, loss of context) | Med | Med | These design docs + runbooks are written for a cold restart by the owner-in-two-years; paper trading means unattended decay loses no money; kill-switch + drawdown ladder bound the blast radius if live | Restore drill doubles as a context drill (can the runbooks be followed cold?); pipeline runs unattended — neglect shows as unacked queue items > 7d (info alert) | Runbooks + queue-age alerts |
| RAM pressure on the 8 GB host (Dagster + Postgres + nightly job spike) | Med | Low | Per-container memory limits sum to ~5.8 GB with headroom (doc 12 §1); Dagster runs in-process, no per-run containers; Prefect is the designated fallback if Dagster stays heavy (ADR-0002) | Sustained RAM > 90% warn alert; pipeline-stage duration regressions | node-exporter + Grafana |
| Accidental public exposure of API/dashboard | Low | High | Bound to 127.0.0.1 in compose (never 0.0.0.0); tailscale/SSH tunnel only; basic auth as defense-in-depth; confirm-header on admin actions (doc 11 §1.4) | Quarterly drill includes an external port scan of the host (`nmap` from off-net) as a checklist line | Restore-drill checklist |

## 6. Regulatory & terms of service

Kept deliberately small by scope decisions already made in doc 00: single user, no advice
given to anyone, no data redistribution, prediction markets research-only with no execution
hooks, and live trading gated behind an explicit future decision. Most regulatory surface was
designed out rather than mitigated.

| Risk | Likelihood | Impact | Mitigation | Tripwire / monitoring | Owner (detector) |
|---|---|---|---|---|---|
| Personal automated trading via Alpaca — regulatory/ToS exposure | Low | Med | **Paper: no exposure** — simulated fills, no capital. Live (future): personal account, self-directed; note **PDT rules** (≥ $25k equity or < 4 day-trades/5 days — nightly cadence makes day-trades rare by construction, but the execution layer counts them); Alpaca ToS permits personal API trading; platform stays strictly single-user with **no advice-giving scope** (doc 00 §3) — nothing is published or shared, removing adviser-registration questions | Day-trade counter in execution layer (doc 09) hard-blocks a 4th day-trade under $25k; live gate review (doc 00 §5) includes a ToS/regulatory re-check as a checklist item | `execution/gate.py` + live-gate checklist |
| Data-provider ToS violations (redistribution, scraping posture) | Low | Med | Single-user, no redistribution of any provider data; EDGAR declared User-Agent and rate compliance (doc 02 §1); prediction markets read-only per venue ToS | Provider budget monitors keep usage visibly inside declared limits | Provider adapters |

## 7. Review cadence

- **Quarterly**, alongside the restore drill: re-score likelihood/impact, close retired risks,
  add new ones from `docs/ops-log.md` incidents. The review is a checklist item in the drill
  runbook so it cannot be silently skipped.
- Any **page-class incident** adds or updates a row here within a week (post-mortem output).
  An incident that matched an existing row validates the tripwire; an incident that didn't
  match any row is the more important finding.
- The **live-readiness review** (doc 00 §5) re-reads this register in full; any High/High
  cell without a working, *tested* tripwire blocks the gate.
- Tripwires are tested, not assumed: the quarterly synthetic-fault drill (doc 02 §6) covers
  data quality; the golden-suite injection strings cover prompt injection; the restore drill
  covers host loss. A tripwire with no test is treated as absent.
- Rows are never deleted, only marked retired with a date and reason — the history of what
  was once feared and why it stopped mattering is itself design documentation.
