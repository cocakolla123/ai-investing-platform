# ADR-0009 — Kalshi prediction-market trading sleeve (demo-first); Polymarket stays read-only

**Status:** accepted · 2026-08-04

## Context
The platform has treated prediction markets as research inputs only (doc 00 §7: "no execution
hooks"). The owner has now explicitly requested Kalshi trading. The research pipeline already
produces `PredictionMarketView` rows (estimated vs implied probability, edge, confidence,
uncertainties — doc 03 §6) and the execution layer already has the discipline the sleeve needs:
the live gate + kill-switch (doc 09 §7), the order state machine with append-only events
(doc 09 §2), hash-chained `DecisionContext` (doc 10 §2), and sticky reconciliation halts.

## Decision
Enable a small, evidence-gated **Kalshi** trading sleeve that reuses the equity path's
machinery rather than inventing a parallel one:

- **Demo first.** `DYNAMO_ENV != LIVE` talks exclusively to the Kalshi demo exchange
  (`kalshi_demo_base_url`). The production API is reachable only in LIVE, and every submission
  still passes the SAME `live_orders_permitted` gate: kill-switch absent (checked in every
  environment), signed unexpired approval covering `KALSHI:<market_id>` (or `all`), and
  readiness met. The kill-switch halts PM orders through the same single pathway as equities.
- **Risk engine has final authority.** `risk/pm_sizing.py` is pure, fail-closed code: trusted
  quotes only, `|edge| >= pm_min_edge` (0.08) against the *current* price,
  `confidence >= pm_min_confidence` (0.6), resolution within `pm_max_days_to_resolution` (90),
  fractional Kelly capped at `pm_kelly_fraction_cap` (0.25), per-market cap
  `pm_per_market_cap` (1% of equity), hard sleeve cap `pm_sleeve_cap` (5% of equity),
  at most `pm_max_open_markets` (10) distinct markets. Any missing/NaN input skips the market
  with a recorded reason.
- **Same execution discipline.** `PMOrder` reuses `OrderState` + `ORDER_TRANSITIONS`; every
  transition writes one `pm_order_events` row; no submission without a hash-chained
  `DecisionContext` (`symbol=KALSHI:<market_id>`, payload `kind='prediction_market'`); rejects
  are terminal; duplicates mean fetch-state; 15-minute reconciliation vs Kalshi
  positions/balance engages a sticky halt cleared only by human acknowledgment.

## Why Polymarket stays read-only
- **No demo environment** — there is nowhere to run the mandatory paper phase.
- **Wallet custody** — trading requires holding crypto keys/funds, an asset class and custody
  surface doc 00 §7 explicitly excludes.
- **US-person legal verification pending** — Polymarket's history with US regulators (CFTC
  settlement, access restrictions) makes ToS/legal compliance unverified for this operator.
No Polymarket order code exists anywhere in the platform, by design.

## Non-goal preserved
"Beating prediction markets systematically" remains a **non-assumption** (doc 00 §7). The
sleeve is small (≤ 5% of equity), demo-first, and evidence-gated: it exists to test whether the
committee's probability estimates carry real edge, with calibration
(`scope_type='prediction_market'`) scoring every resolved market.

## Revisit
- **Kalshi live:** only after the standard live-gate readiness (≥ 3 clean paper months on the
  demo exchange, calibration OK, risk suite green) AND prediction-market Brier consistently
  better than the market's own implied probabilities.
- **Polymarket execution:** only if all three blockers change — a sanctioned demo/sandbox, a
  custody solution the owner accepts, and written confirmation of US-person eligibility.
- **Sleeve growth:** any cap increase requires a promotion-style evidence bundle, never a
  config-only edit.

## Addendum (2026-08-09) — OddsPipe cross-venue enrichment
OddsPipe (https://oddspipe.com, free tier, `X-API-Key` auth) is wired in as an **enrichment
source only** — the direct Kalshi and Polymarket adapters stay primary and nothing depends on
OddsPipe being available:

- **What it adds.** `/v1/spreads` supplies matched cross-venue pairs (the same question priced
  on both Kalshi and Polymarket, with both YES prices and the spread), persisted to
  `pm_cross_venue_spreads` (migration 0005) by the `oddspipe_spreads` asset in the 20:00 ET
  snapshot chain. `/v1/markets/{id}/candlesticks` (1m/5m/1h/1d) is available for optional
  historical backfill. The macro/PM research context gains a short "Cross-venue divergences"
  section (top 5 by |YES spread|), and the Prediction Markets dashboard page shows the top 10.
- **Corroboration gate** (`risk/pm_sizing.py`, pure and additive): `select_and_size_pm_trades`
  accepts an optional `cross_venue` mapping (Kalshi market_id → matched Polymarket YES price),
  passed IN by orchestration — the risk module never fetches. When a match exists for a
  candidate: if the Polymarket price agrees with our estimate's direction versus the Kalshi
  price, the trade is unchanged and its rationale notes the corroboration; if it disagrees AND
  `|polymarket − estimated| > pm_cross_venue_veto_gap` (default 0.10), the trade is skipped
  with reason `cross_venue_disagreement` — both venues pricing against our estimate means
  stand down. Missing, unmatched, or non-finite entries change nothing (NaN-safe,
  deterministic; all existing caps and floors are untouched).
- **Graceful degradation.** No `ODDSPIPE_API_KEY`, API down, or malformed payloads each
  produce one log line and an empty result: the snapshot chain, the sleeve, and the research
  context behave exactly as before OddsPipe existed.
