# ADR-0011: Polymarket shadow trading (simulated fills, internal ledger)

Status: accepted (2026-08-10); superseded in part by
[ADR-0012](ADR-0012-polymarket-live-execution.md) — a real CLOB execution
path now exists behind the doc 09 §7 live gate, but the simulated sleeve
described here remains the default in every non-LIVE environment.

## Context

The Kalshi sleeve (ADR-0009) validates the prediction-market pipeline against
a real demo exchange. Polymarket frequently prices the same questions
differently, but it has **no demo mode**: the only real endpoints are the
production CLOB, and we will not point research-stage code at real money.
We still want the platform's Polymarket views (already produced nightly by
the macro digest) to accumulate a track record — entries, exits, P&L,
calibration — before any live gate discussion.

## Decision

Shadow-trade Polymarket by **simulating the broker ourselves**:

- Decisions run through the exact pipeline the Kalshi sleeve uses —
  `risk.pm_sizing` with the same gates (≥8pt edge, ≥0.6 confidence,
  quarter-Kelly, 1% per-market and 5% sleeve caps, OddsPipe cross-venue
  veto), the same `PMOrderManager` state machine, deterministic client order
  ids, and the mandatory hash-chained DecisionContext per order. A new
  `PMSizingParams.venue` field selects the one venue a sizing pass may
  trade, and all caps bind against the **sim bankroll**
  (`polymarket_sim_bankroll_usd`, default $1,000), never portfolio equity.
- Execution is `PolymarketSimBroker` (`dynamo/execution/polymarket_sim.py`):
  an implementation of the `PMBroker` port that fills every order
  immediately and in full at the fresh Gamma YES price plus
  `polymarket_sim_slippage_cents` **against us**, and rejects fail-closed on
  missing/stale/degenerate quotes, closed markets, prices at/above the 100c
  payout, and fills that would overdraw the bankroll. Gamma quotes one price
  per outcome (no book), so outcome price + slippage stands in for crossing
  the spread — the assumption is documented on the broker.
- The broker is **structurally incapable of real execution**: it holds no
  HTTP client, imports no transport library, and knows no endpoint or
  credential. `POLYMARKET_API_KEY` is read nowhere. Real execution stays
  deliberately unwired until an explicit live-gate decision, which would be
  its own ADR.
- Persistence reuses the existing `pm_orders` / `pm_order_events` /
  `pm_positions` / `pm_trades` tables with `venue='polymarket'` rather than
  forking parallel `pm_sim_*` order tables — the tables already key on
  venue, and one order path means one state machine to audit. The venue
  string **is** the SIMULATED marker (`is_simulated_venue` in the domain):
  no real Polymarket path exists, so no flag can disagree with reality.
  Kalshi call sites (sizing, reconciliation, resolution) now filter
  `venue='kalshi'` so simulated rows can never leak into real-sleeve math.
  The one new table is `pm_sim_ledger` (migration 0007): seed deposit, one
  debit per simulated fill, one credit per settlement, idempotent on
  `(venue, kind, ref)` — settling twice or re-syncing fills is a no-op, and
  the bankroll is the signed sum of the ledger.
- Settlement (`pm_sim_resolution`, in the existing 20:45 ET resolution job):
  when Gamma reports a position's market closed at an extreme price, the
  position settles — win credits `contracts × $1` to the bankroll, loss
  credits nothing — with realized P&L recorded as a `settlement` trade.
  Exits are hold-to-resolution, matching the Kalshi sleeve's v1 semantics.
  No schedules were enabled by this change.
- Every user-facing rendering (daily report, Prediction Markets dashboard)
  labels the sleeve **"SIMULATED — tracked internally, no real orders
  sent."**

## Consequences

- Polymarket edge estimates accrue an auditable simulated track record with
  zero counterparty or capital risk, using the same discipline (gates,
  DecisionContexts, order events) the real sleeves obey.
- Simulated fills are optimistic about liquidity (always filled, fixed
  slippage); results are an upper bound on realizable performance, which is
  acceptable for a validation sleeve and stated on the dashboard.
- Calibration observations for Polymarket views are not yet fed to the
  learning loop (Kalshi-only today); a follow-up can extend
  `pm_sim_resolution` the same way `pm_resolution` does it.
- If a live Polymarket integration is ever approved, it must arrive as a new
  broker behind the doc 09 §7 gate plus its own ADR; nothing in this change
  needs to be undone for that.
