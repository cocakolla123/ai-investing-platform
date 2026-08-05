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
