# ADR-0007 — Three-tier universe with dynamic promotion

**Status:** accepted (user-specified) · 2026-08-03

## Context
Deep research on all ~6k US listed names multiplies data cost, Claude cost, and noise; a fixed
small universe goes stale and misses emerging opportunities.

## Decision
- **Tier 1 (~500–600 liquid names + curated ETFs):** full pipeline (data depth, quant scores,
  committee eligibility, portfolio membership).
- **Tier 2 (rest of US market):** free-data surveillance with code-only anomaly detectors
  (price/volume z-scores, relative strength, filings, insider clusters, analyst actions, news
  bursts); flagged names get a cheap Haiku triage call.
- **Tier 3:** triggered names are temporarily promoted to full treatment (default 30 days);
  sustained interest → human-approved permanent addition; decayed Tier-1 names → demotion
  recommendations. All transitions recorded in the point-in-time `universe_membership` table
  with reasons.

## Consequences
Broad-market awareness at near-zero marginal cost; compute concentrated where positions can
actually be taken; the membership table doubles as the survivorship-bias defense for backtests;
the core universe evolves under human control with an auditable history.
