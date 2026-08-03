---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 14 — Principal-Engineer Design Review

Adversarial self-review of docs 00–13 before any implementation. Method: full read of every
section against (a) internal consistency, (b) the user's original requirements, (c) failure
modes (look-ahead, cache/adjustment invalidation, provider death, batch expiry, silent money
leaks), and (d) a simplification pass. Every finding is **resolved** (doc edited) or
**accepted** (rationale recorded). Open questions raised inside individual docs (§ "Open
questions") are tracked there with their decision milestones and are not duplicated here.

## 1. Findings

| # | Severity | Finding | Resolution |
|---|---|---|---|
| F1 | High | **Walk-forward defaults conflicted**: doc 04 specified purge 20d + embargo 20d; doc 08 (owner of the shared `walkforward.py`) specified embargo 10d. One implementation cannot have two specs — the exact drift the shared-splitter rule exists to prevent | **Resolved.** Canonical protocol: purge 20 trading days (one label horizon) + embargo 10 trading days, owned by doc 08 §4. Doc 04 §5 edited to match and now explicitly defers to doc 08 |
| F2 | High | **Sleeve rules contradicted each other**: doc 05 §5 deliberately allows the same symbol in both sleeves (separate lots, per-sleeve exits, combined caps); doc 07 step 2 said "held once, attributed to long-term" | **Resolved.** Doc 07 edited to defer to doc 05 §5: both-sleeve positions allowed as separate lots; combined exposure counts against per-position/sector/cluster caps; two lots count as two positions against count caps |
| F3 | High | **Budget-exhaustion alert conflated two halt mechanisms**: doc 12's Claude-100% row said cost_meter "engages the kill-switch" — but the global kill-switch **cancels open orders** (doc 09 §7). Research-budget exhaustion must never touch orders or protective stops | **Resolved.** Doc 12 §7 edited: cost_meter refuses new batch submissions only; the global kill-switch is a separate mechanism reserved for trading halts. Degraded (quant-only) mode already specified in doc 12 §8 |
| F4 | Med | **Earnings-call transcript source unspecified**: doc 03 has an earnings-call reader and `earnings_events.call_transcript_ref` exists (doc 02), but no provider in the matrix supplies transcripts (Finnhub free tier does not) | **Accepted as a known gap.** M1 ships without transcript ingestion: the earnings-call reader activates only when a transcript is available (company IR PDFs / 8-K exhibits ingested via EDGAR where present). A budget-compatible transcript source is a tracked data-roadmap item; `data_gaps` in dossiers (doc 03 §2) will quantify how often it is missed. Revisit at M2 |
| F5 | Med | **Shared-dossier bias**: committee v2 funnels bull/bear/arbiter through one dossier — a systematic dossier omission biases every downstream judgment (single-extractor risk noted in ADR-0005) | **Accepted with mitigations already designed**: advocates receive selected raw excerpts alongside the dossier (ADR-0005), `data_gaps` is a mandatory dossier field, post-trade reviews classify `missing_information` against what was knowable (doc 10 §3), and per-role calibration isolates a chronically blind dossier stage (doc 10 §5). No change |
| F6 | Med | **Tiingo single-point-of-failure for survivorship-clean history**: the backtest's delisted coverage depends on one $10/mo provider | **Accepted.** Bars are dual-sourced going forward (Alpaca+Tiingo) and archived in Postgres + backups — history already ingested survives a Tiingo exit; only *future* delisted coverage would need a replacement adapter (port swap, ADR-0004). Tripwire: provider-outage alert + doc 13 §2 row |
| F7 | Low | Filename drift in cross-references (doc 07 referenced `05-strategies.md`, `11-dashboard.md`) | **Resolved.** References corrected to actual filenames. CI link-check for docs added to the M1 backlog |
| F8 | Low | Doc 04 measures purge/embargo in trading days while some risk windows (doc 06) use calendar defaults — a future implementer could conflate them | **Accepted.** Convention recorded here: all walk-forward and label arithmetic is **trading days** (exchange calendar, doc 01 §6); risk lookbacks state their unit inline. Property tests assert calendar-aware window math (doc 04 §10) |
| F9 | Low | Prompt-injection defense relies partly on prompt hardening, which is not a guarantee | **Accepted with layered controls**: the binding defenses are structural — no tool-use in analyst calls, JSON-schema-enforced output, no LLM in any risk/execution path, adversarial injection strings in the golden corpus (doc 13 §3). Prompt hardening is defense-in-depth, not the wall |

## 2. Requirements traceability (user spec → design)

| Requirement (from the user's brief) | Where designed |
|---|---|
| Claude-only LLM, institutional research-team structure | 03 (committee v2), ADR-0005 |
| All listed analyst roles (fundamental, filings, earnings, macro, news, risk, thesis, bear/devil's advocate, audit, post-trade, self-critique) | 03 §1 roster: roles map to calls or folded arbiter/review responsibilities; event specialists for earnings/filings |
| Continuous analysis of filings/earnings/news/insider/institutional/macro/rates | 02 §1 provider matrix + §4 cadences (13F institutional via EDGAR) |
| Structured JSON outputs, explanations, bull/bear, confidence, upside/downside, size, exits | 03 §2 schemas (`AgentOpinion`, `CommitteeVerdict`); traceable Evidence refs |
| Claude never sole decision-maker; AI committee; disagreement surfaced | 03 §1, §4; 06 §1 veto contract; 05 §4.2 conflict handling |
| Quant models: GBM family, RF, statistical forecasting; return/downside/vol/confidence/ranking | 04 (LightGBM/CatBoost/RF + statsforecast), ADR-0003 (LSTM/transformers deferred with revisit criteria, modular `ForecastModel` port) |
| US stocks + ETFs; multi-asset modular; prediction markets (Kalshi/Polymarket) with estimated-vs-implied probability, uncertainty, why-wrong | 00 §3; 02 §7; 03 §6 (`PredictionMarketView`); ports make new venues adapters |
| Three-tier universe: 500–600 core / broad surveillance / dynamic promotion with human-approved evolution | 02 §2, ADR-0007 |
| Long-term quality investing (moat, growth, profitability, FCF, valuation, balance sheet, management) continuously updated | 05 §2 quality score |
| Tactical strategies (momentum, breakouts, RS, mean reversion, volume, catalysts, earnings reactions); configurable allocation | 05 §1, §3 |
| Portfolio-level optimization; sizing, cash, diversification, sector exposure; Low/Med/High buckets as guidelines | 07; 06 §2 presets |
| Risk engine highest priority: sizing, limits, stops, trailing, drawdown protection, vol targeting, conservative Kelly, VaR/CVaR, stress, Monte Carlo, correlation, liquidity, regime | 06 (all sections); "not investing" first-class: 06 §9, 07 §4, 00 §4 abstention metric |
| Record complete decision context; post-investment review; failure classification; calibration; drift/decay detection | 09 §6 + 10 §2 (immutable, hash-chained), 10 §3–§7 |
| Searchable journal consulted before new recommendations | 10 §6; injected into arbiter context (03 §1) |
| Never auto-modify production logic; backtest→OOS→walk-forward→stress→paper→shadow→human approval; separate dev/test/paper/live | 08 §7 pipeline (tooling-enforced), 01 §5 environments |
| Explainability: every claim traceable to data | 03 §2 Evidence; 05 §4.4 watchlist traceability; 06 §1 logged vetoes; 00 §4 traceability metric |
| Dashboard (performance, allocation, positions, risk, confidence, strategy perf, alerts, regime, attribution) | 11 §3 screen inventory |
| Engineering standards: Clean Architecture, SOLID, DI, typed Python, tests, Docker, K8s-ready, CI/CD, logging, monitoring, secrets, docs | 01 (ADR-0006 lite variant), 12; pyproject enforces the dependency rule |
| Paper trading first; live behind explicit human approval | 09 §7 gate; 00 §5 readiness criteria |
| Budget $50–150/mo | 00 §6 contract; 03 §5 cost model + kill-switch; ADR-0004 |

No requirement from the brief is unaddressed. Deviations from the brief (deep learning
deferred, universe tiering, committee consolidation, prediction markets as research-only,
Clean-Architecture-lite) were each explicitly approved by the user and are recorded as ADRs
0003, 0005, 0006, 0007.

## 3. Simplification pass (standing instruction: recommend simplifications before finalizing)

Simplifications already embedded in this design, confirmed as final:

1. **One database** (Postgres+Timescale+pgvector) + disposable Parquet sidecar — no ClickHouse,
   no vector DB, no Redis (Dagster queues via Postgres; nothing else needs a broker).
2. **Rule-based portfolio construction** before convex optimization, with objective entry
   criteria for the upgrade (07 §2) — avoids error-maximizing optimizers at 10–30 positions.
3. **Rule-based regime detection** instead of HMMs (06 §8) — explainable, testable, sufficient.
4. **4-call committee** instead of 10+ parallel roles (ADR-0005) — ~45% cheaper at equal or
   better judgment quality.
5. **Streamlit-then-React** — no frontend investment before the data earns it (ADR-0008).
6. **No Vault, no Loki, no K8s, no WebSockets, no API versioning** — each rejected in place
   with a revisit condition (ADR-0008, 12 §5, 00 §7, 11 §1.3, §2).
7. **Local embeddings** (bge-small) — zero API cost, no second LLM provider.

Considered and *not* adopted (would cut scope the user explicitly wants): dropping minute bars
(needed for 15-min intraday risk checks); dropping prediction markets (explicit requirement,
already scoped to research-only); merging the two backtest engines (the research/validator
split is what keeps promotion evidence honest, 08 §1).

## 4. Verdict

With F1–F3 fixed and F4–F9 accepted with rationale, the design is internally consistent,
budget-feasible, and implements every requirement of the brief or records a user-approved
deviation. **Design phase complete; recommended next step: user sign-off, then M1 (research
pipeline) implementation per the roadmap.**

Residual sharp edges an implementer should keep in view: the Message-Batches sensor logic
(expiry/resubmission, 03 §3), corporate-action invalidation cascades (02 §5), and the
promotion tool's DB grants (08 §7) — all three are where quiet correctness bugs would be most
expensive, and all three have explicit test requirements in their docs.
