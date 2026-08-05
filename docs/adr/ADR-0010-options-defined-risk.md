# ADR-0010 — Defined-risk options sleeve + same-day tactical entries

**Status:** accepted · 2026-08-04

## Context
Doc 00 §7 listed options trading as a non-goal ("complexity + risk model rewrite"). The owner
has now explicitly requested an options module and same-day tactical entries. The platform
already has every discipline the sleeve needs — the live gate + kill-switch (doc 09 §7), the
order state machine with append-only events (doc 09 §2), hash-chained `DecisionContext`
(doc 10 §2), sticky reconciliation halts, and a pure fail-closed sizing pattern proven by the
Kalshi sleeve (ADR-0009). Alpaca supports options in PAPER accounts, so the mandatory paper
phase runs on the existing broker.

## Decision
Enable a small, evidence-gated **defined-risk options sleeve** plus a capped **same-day entry
path**, reusing the equity machinery rather than inventing a parallel one.

### Hard lines (enforced in code, not convention)
- **Defined-risk only, unrepresentable otherwise.** The only structures that can exist are
  long calls, long puts, covered calls (short call fully covered by ≥ 100 shares/contract),
  and cash-secured puts (full assignment cash reserved). `OptionStructure` has exactly four
  members and every order/position/decision model validates its `(side, right)` leg against
  the structure — constructing a naked short raises at model construction. Collateral is
  validated **at proposal** (`OptionTradeDecision` validators) **and at submission**
  (`OptionsOrderManager` share/cash lookups). Max loss is finite for every representable
  structure (`max_loss_dollars`).
- **Hard caps** (Settings, add-only): `options_premium_at_risk_cap` 5% of equity (total long
  premium + CSP assignment obligations), `options_per_position_cap` 1%, DTE window
  `[21, 90]`, `options_min_open_interest` 100, `options_max_spread_pct` 10%,
  `options_min_conviction` 0.7, `options_iv_rank_max` 0.8 (never buy premium at extreme IV
  rank), `options_max_open_positions` 8.
- **Risk engine has final authority.** `risk/options_sizing.py` is pure, fail-closed code:
  any missing/NaN spot, chain, IV rank, quote, OI, or greek skips the underlying with a
  recorded reason. Covered calls contribute zero to the premium cap (share-covered; the
  share downside is owned by the equity risk engine); CSPs count their full assignment
  obligation.
- **Contract choice** is deterministic: nearest monthly expiry (third Friday) inside the DTE
  window; strike by delta target (~0.6 long legs, ~0.3 short legs) from chain greeks when
  available, else a documented moneyness heuristic (long ~5% ITM, short ~5% OTM). Limit
  orders only — the broker port has no market-order surface.
- **Same execution discipline.** `OptionOrder` reuses `OrderState` + `ORDER_TRANSITIONS`;
  every transition writes one `options_order_events` row; no submission without a
  hash-chained `DecisionContext` (`symbol=OPT:<occ>`); the gate is consulted on every
  submission; rejects are terminal; duplicates mean fetch-state; 15-minute reconciliation vs
  broker option positions engages a sticky halt cleared only by human acknowledgment.
- **IV rank proxy** is computed from our own stored ATM-IV history
  (`options_iv_history`, min 10 observations — fail-closed until history accrues). Chain
  snapshots are persisted minimally (spot ±20%, target expiries only).
- **Degrade, don't guess.** If the installed alpaca-py lacks the options surface, the chain
  adapter reports unavailable and the sleeve skips with a logged reason.

### Same-day tactical entries
The 15-minute intraday loop (`intraday_entries_job`) evaluates (a) newly promoted Tier-3
names with a material triage verdict today and (b) breakout-strategy triggers vs live quotes
for watchlist tacticals. Triggered names run the **same** sizing cascade as the nightly path
and propose marketable-limit equity orders through the standard OrderManager (decision
context, gate). Caps: `intraday_entry_cap_per_day` (3). **PDT guard** (`risk/intraday.py`):
a name entered same-day is never sold same-day automatically; stops are exempted, but under
$25k equity an exit that would be the 4th day trade in 5 business days is logged + alerted
instead of executed — UNLESS it is a disaster stop, which always runs.

## Why these four structures and no others
Every excluded structure either has unbounded loss (naked calls), practically-unbounded loss
(naked puts without reserved cash), or requires multi-leg margin/assignment modeling the risk
engine does not yet have (spreads, straddles, calendars). The four in scope have loss bounded
by premium paid or by collateral already held, so the existing equity-first risk engine stays
sound without a rewrite.

## Revisit criteria
- **Defined-risk spreads / multi-leg** (verticals, collars): only after ≥ 6 months of clean
  paper options history, per-structure calibration (fill quality, realized vs modeled max
  loss), and an ADR with multi-leg margin modeling in the risk engine.
- **Live options:** the standard doc 09 §7 gate plus a dedicated options readiness review
  (assignment handling drill, expiry-week runbook).
- **Cap increases:** promotion-style evidence bundle, never a config-only edit.
- **Assignment/exercise automation:** currently expiries are closed out with an alert for
  human review; automate only after observed paper assignments are handled cleanly for a
  quarter.
