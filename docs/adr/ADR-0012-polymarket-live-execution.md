# ADR-0012: Polymarket live execution (gated CLOB broker; sim remains the default)

Status: accepted (2026-08-10) — supersedes ADR-0011 in part (the "never real"
clause); the simulated sleeve remains the non-LIVE default.

## Context

The user has requested a real Polymarket execution path and holds L2 CLOB API
credentials. ADR-0011 deliberately shipped the sleeve as simulation-only
because Polymarket has **no demo exchange** — the only endpoints are the
production CLOB, and real trades settle in USDC on Polygon. Months of paper
validation are still ahead, so the platform needs a live path that exists in
code, is exhaustively tested, and is **unreachable** until an explicit,
signed, human live decision — while every non-LIVE environment keeps running
the ADR-0011 sim sleeve byte-for-byte unchanged.

## Decision

- **One factory, one precondition chain.**
  `orchestration.assets.pm_trading.polymarket_broker_for` is the only route
  to real execution. It returns the real `PolymarketBroker` iff ALL of:
  1. `DYNAMO_ENV == LIVE`;
  2. `polymarket_live_enabled` (new setting, default **False**, deliberately
     independent of `DYNAMO_ENV`);
  3. the broker is fully configured — CLOB L2 creds
     (`polymarket_api_key/_api_secret/_passphrase`), a wallet key file at
     `polymarket_wallet_key_path`, a funder address for proxy signature
     types, and the optional `polymarket` extra importable;
  4. the doc 09 §7 live gate passes (signed HMAC approval artifact for
     today scoped `all`, kill-switch absent, readiness met) — the same
     machinery Kalshi and Alpaca use.
  **Any** missing precondition falls back to `PolymarketSimBroker` with a
  loud structlog warning — fail closed to simulation, never to live. In the
  expected non-LIVE, non-enabled state the fallback is silent and identical
  to ADR-0011 behavior. Each individual submission is *additionally*
  gate-checked through the order manager with scope
  `POLYMARKET:<market_id>`, exactly like Kalshi's flow, and LIVE never
  auto-approves (doc 09 §2), so a human approves each order.

- **The adapter** (`dynamo/execution/polymarket_broker.py`) implements the
  `PMBroker` port over the official `py-clob-client` (optional extra —
  pulls eth-account/web3 signing machinery, kept out of the base install).
  Limit orders map our cents model to CLOB terms (Gamma `clobTokenIds`
  resolved through the read adapter, still keyless; price = cents/100;
  size = contracts; BUY only, like the Kalshi sleeve). Error discipline
  mirrors `kalshi_broker`: rejects terminal, 429 an incident, one jittered
  retry then `HaltSignal`/`BrokerUnavailable`. The CLOB has no
  client-order-id, so idempotency is enforced locally (DB-backed
  `broker_order_lookup`; a known id raises `DuplicateClientOrderId` rather
  than re-ordering). `polymarket_live_max_order_usd` (default $25) is a
  hard per-order notional cap enforced **inside the broker**,
  belt-and-suspenders beyond risk sizing.

- **Custody & secrets.** The Polygon wallet private key lives in a file
  under gitignored `secrets/`, referenced by path only; it is read lazily
  at client construction, never logged, and reprs redact all credentials.
  UI-created Polymarket accounts trade through a proxy wallet:
  `polymarket_signature_type` 1 (email/Magic proxy) or 2 (browser-wallet
  proxy) with `polymarket_funder_address` set to the proxy that holds the
  USDC; 0 = direct EOA.

- **Reconciliation.** When the live broker is active, the 15-minute PM
  reconciliation pass also syncs Polymarket orders and reconciles positions
  against CLOB truth (positions via the keyless Data API keyed on the
  funder address) with the same sticky-halt-on-mismatch semantics as Kalshi
  (`POLYMARKET:<market_id>` scopes, `pm_live:*` run scopes — caught by the
  existing sticky `pm*` halt check, which also blocks new live submissions).
  In sim mode there is no external truth and nothing changes.

## Consequences

- Paper validation continues exactly as before: in every non-LIVE
  environment the factory returns the sim broker unconditionally (regression
  test enforced), and no unit test touches the network.
- The live path is fully wired and tested (mocked CLOB client) before any
  money moves, so the eventual live decision changes configuration, not code.

## Risks to revisit at the live decision (the human approval MUST cover these)

- **No sandbox exists.** The first real order is real money; there is no
  dress rehearsal against a demo CLOB. Start with the $25 per-order cap and
  a small USDC balance.
- **USDC/Polygon custody.** Funds sit in a self-custodied (or Magic-proxy)
  Polygon wallet; key loss or compromise is unrecoverable. The wallet key
  file custody procedure in `secrets/` needs an explicit review.
- **US-person eligibility.** Post-CFTC settlement, Polymarket access for US
  persons is via the regulated intermediary path; confirm account
  eligibility and terms before enabling.
- **Sizing input.** The sleeve is still sized against the sim-bankroll
  equity, not the real USDC balance; acceptable under the per-order cap,
  but must be re-based on broker balance before meaningful size.
- **Venue taxonomy.** `is_simulated_venue("polymarket")` remains True
  (rows are simulated-by-default); live rows are marked via
  `simulated: false` in the DecisionContext payload. Dashboards/reports
  labeling the sleeve "SIMULATED" must be updated before live enablement.
