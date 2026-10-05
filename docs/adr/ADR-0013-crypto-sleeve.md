# ADR-0013 — Crypto sleeve (Alpaca crypto, 24/7 stops, equity-session replay)

Status: accepted. Date: 2026-08-17.

## Context

The platform trades equities/ETFs plus three satellite sleeves (options,
Kalshi, simulated Polymarket) through Alpaca. Alpaca also supports crypto
pairs in PAPER mode today, with free historical daily bars since ~2021 and no
data API key requirement. The backtest campaign plan adds crypto as a fourth
sleeve so it can join both nightly paper trading and the replay campaign with
real data. Crypto differs from equities in ways the platform must absorb:
24/7 sessions, realized vol 3-5x equities, fractional-native quantities, no
PDT rule, no fundamentals/filings/transcripts, and broker payloads that
sometimes drop the slash from pair symbols.

## Decision

- **Broker/data: Alpaca crypto.** Same `TradingClient`, same order-request
  classes, same reconciliation loop. A new `AlpacaCryptoData` adapter
  (`src/dynamo/data/alpaca_crypto.py`) serves daily bars into `bars_1d` with
  `source='alpaca_crypto'`; keys optional.
- **Canonical symbol form is `"BASE/QUOTE"`** (`"BTC/USD"`) everywhere
  internally. Adapters normalize the slashless broker form at the edge
  (`dynamo.domain.symbols`); `"/" in symbol` is the platform-wide crypto
  discriminator.
- **Sleeve caps: 10% collective, 5% per pair** (`crypto_sleeve_cap`,
  `crypto_max_position`; defaults mirrored in `RiskConfig`). Vol-parity
  sizing already shrinks crypto naturally (3-5x sigma); the caps are
  belt-and-suspenders, enforced twice — in the sizing cascade and in the
  constructor's verdict-compliance pass (so retained lots are bounded too).
  `Sleeve.CRYPTO` tags lots/orders/decision contexts; crypto candidates ride
  the long-term selection path (ranked watchlist + entry verdicts).
- **Research: crypto pairs are Tier-1 core universe members**
  (`CRYPTO_STARTERS`, seeded when `crypto_enabled`). Quant features work off
  bars (momentum/vol/volume); fundamentals/insider/earnings features NaN by
  design with `feature_coverage` reflecting it. Committee dossiers render
  "none" for missing sections honestly; instruments rows carry
  `sector="crypto"` (one shared sector bucket — conservative for a highly
  correlated asset class).
- **Execution: GTC-only** (Alpaca rejects DAY/OPG for crypto);
  `MARKET_ON_OPEN` maps to a plain GTC market order (no auction in a 24/7
  market); plain stops map to GTC stop-limits 1% through the stop (Alpaca
  crypto supports market/limit/stop-limit only). Marketable-limit pricing
  applies unchanged. Fractional qty is native. Same order state machine,
  DecisionContext requirement, and gate.
- **PDT exemption:** crypto fills are filtered out of PDT accounting
  (`pdt_relevant_fills`) — FINRA's rule covers margin securities accounts
  only, so crypto exits are never deferred by the guard.
- **24/7 stop coverage:** the existing 15-min weekday stops loop covers
  market hours; a new `crypto_stops_schedule` (every 2h Sat/Sun) runs the
  FULL stops job on weekends — the simpler correct option (equity GTC stop
  replaces queue harmlessly off-hours). Nothing is enabled by default;
  schedules are registered only.
- **Replay convention (v1): crypto trades and marks on equity sessions
  only.** The NYSE calendar stays the replay clock; weekend bars are stored
  but never matched to a session, so weekend moves surface at the Monday
  close. Annualization stays sqrt(252) on trading-day series for all assets
  (documented in `risk/sizing.py`).

## Consequences

- Paper-capable immediately: crypto orders work against Alpaca paper today,
  reconciliation and sticky halts included (symbol normalization makes
  broker truth comparable).
- The 10% sleeve cap bounds the blast radius of the most volatile sleeve; a
  full crypto wipeout costs at most ~10% of equity plus slippage.
- Weekend gap risk is mitigated (not eliminated) by the 2h weekend stop
  cadence — a genuine advantage over equities, which cannot exit at all
  off-hours.
- Replay understates crypto vol/opportunity by ignoring weekend sessions;
  acceptable for v1 and stamped in the campaign audit caveats. Switching to
  a 365-day crypto calendar would require calendar-day return panels and a
  dual-calendar runner — deliberately deferred.
- No schema migration was needed: `bars_1d`/`instruments`/`positions` key on
  Text symbols, and `"BTC/USD"` fits every existing PK.
