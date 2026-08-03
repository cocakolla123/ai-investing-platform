---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 02 — Data Platform & Three-Tier Universe

## 1. Provider matrix

| Provider | Data | Cadence | Cost | Limits / notes |
|---|---|---|---|---|
| **SEC EDGAR** | Filings (10-K/Q, 8-K, S-1…), XBRL company facts, Form 4 insider tx, 13F institutional | Poll every 10 min (RSS/full-text index) | Free | 10 req/s, declared User-Agent required |
| **FRED** | Macro series: rates, CPI, PCE, unemployment, yield curve, credit spreads, financial-conditions indices | Daily | Free | API key, generous limits |
| **Alpaca Market Data (free)** | Daily + minute bars (IEX feed), quotes, corporate actions, tradable-asset list | EOD full universe; intraday for held positions | Free | IEX-only depth — fine for EOD strategies |
| **Tiingo ($10/mo)** | Adjusted EOD history incl. **delisted names**, fundamentals, news | Daily | ~$10 | Second price source + survivorship-clean history |
| **Finnhub (free)** | Company news headlines, earnings calendar, analyst actions | Daily + event poll | Free | 60 req/min |
| **Kalshi API** | Prediction markets: markets, orderbooks/last, settlements | Daily snapshot | Free | Signed requests; read-only |
| **Polymarket Gamma/CLOB** | Prediction markets: markets, prices, volume | Daily snapshot | Free | No key for reads; read-only |
| yfinance | — | — | — | **Break-glass only.** Never in the production path (ADR-0004) |

Rules:
- Every record carries **provenance**: `(source, fetched_at, source_id)` columns.
- Every provider sits behind a domain port; swapping/adding providers is an adapter, not a rewrite.
- Rate limits are enforced client-side (token bucket per provider) and exported as Prometheus
  counters; hitting a provider 429 is a logged incident, not a retry storm.

## 2. Three-tier universe

The point-in-time membership table is the backbone for both live scanning and survivorship-safe
backtests.

```sql
CREATE TABLE universe_membership (
    symbol        text NOT NULL,
    tier          smallint NOT NULL,          -- 1 core | 2 surveillance | 3 promoted
    valid_from    date NOT NULL,
    valid_to      date,                       -- NULL = current
    reason        text NOT NULL,              -- entry/exit rationale, always populated
    approved_by   text,                       -- human approval for permanent tier-1 changes
    PRIMARY KEY (symbol, valid_from)
);
```

### Tier 1 — Core universe (~500–600 names)

Selection criteria (initial seed, then evolves via Tier 3):
- Member of S&P 500 or Nasdaq-100, **or** market cap ≥ $2B with 3-month median dollar volume
  ≥ $10M and price ≥ $5.
- Plus a curated ETF set (~30: broad index, sector SPDRs, factor, treasury/credit, commodity).

Treatment: full pipeline — daily bars + fundamentals + filings + news ingestion, quant features
and scores, Claude committee eligibility, portfolio optimization membership, continuous risk
monitoring.

### Tier 2 — Broad-market surveillance (rest of US listed, ~5–6k names)

Free data only (Alpaca/Tiingo EOD, EDGAR, Finnhub calendar), **code-only detectors — zero LLM
cost**. Detector catalog (each configurable, each emits a typed `SurveillanceFlag` with a
severity score):

| Detector | Signal |
|---|---|
| `price_zscore` | |1-day return| > k·σ (60d), k default 4 |
| `volume_spike` | Dollar volume > m× 60d median, m default 5 |
| `rel_strength` | 20d return vs sector ETF > threshold |
| `range_break` | 52-week high/low breach with volume confirmation |
| `earnings_surprise` | Reported vs consensus gap (Finnhub calendar) |
| `filing_event` | 8-K items (1.01, 2.01, 5.02…), S-1, 13D/G |
| `insider_cluster` | ≥ 2 distinct insiders buying ≥ $100k within 14d (Form 4) |
| `analyst_action` | Multi-notch upgrades/downgrades |
| `news_burst` | Headline count z-score; M&A keyword match |

Nightly: detectors → flags table → **Haiku triage batch** (one cheap call per flagged name,
typically 20–60/night) classifying materiality → promotion candidates.

### Tier 3 — Dynamic promotion

- Trigger: Haiku triage severity ≥ threshold, or configurable auto-triggers (M&A announcement,
  earnings surprise > x%).
- Promoted names receive **full Tier-1 treatment for a configurable window** (default 30
  calendar days, extendable by continued flags).
- Exit paths: (a) window lapses quietly → back to Tier 2; (b) sustained interest — committee
  conviction and quality scores above configurable bars for ≥ 2 weeks → platform emits a
  **universe-change recommendation** (add to core, with full rationale) for human approval;
  (c) Tier-1 names failing liquidity/quality floors for 60d → demotion recommendation with
  suggested replacement.
- All transitions write `universe_membership` rows with reasons; permanent changes require
  `approved_by`. The dashboard shows a universe-changes queue.

## 3. Storage design

One Postgres 16 instance: TimescaleDB for time series, pgvector for journal embeddings, plain
relational for the rest (ADR-0001). Alembic migrations from day one.

### Hypertables

```sql
-- EOD bars, full market (Tier 1 + 2), ~6k symbols × 250d/yr ≈ 1.5M rows/yr
CREATE TABLE bars_1d (
    symbol text, ts timestamptz, open double precision, high double precision,
    low double precision, close double precision, volume bigint,
    adj_close double precision,          -- split+dividend adjusted (Tiingo method)
    source text, fetched_at timestamptz,
    PRIMARY KEY (symbol, ts, source)     -- both sources stored; canonical view resolves
);
SELECT create_hypertable('bars_1d', 'ts');
-- compression after 90d; continuous aggregates: weekly/monthly bars, rolling vol

-- Minute bars only for Tier-1 + held positions, retained 2y then dropped
CREATE TABLE bars_1m (LIKE bars_1d INCLUDING ALL);
```

### Relational core (abridged DDL; full schema in migrations)

- `instruments(symbol, name, exchange, sector, industry, cik, figi, status, first_seen, delisted_at)`
- `filings(accession_no PK, cik, symbol, form_type, filed_at, period, url, sections jsonb, parsed_at)`
- `xbrl_facts(cik, tag, period_end, value, unit, filed_at)` — point-in-time fundamentals:
  **keyed by `filed_at`, never restated backwards** (look-ahead defense)
- `insider_tx(accession_no, symbol, insider_name, role, tx_type, shares, price, filed_at)`
- `news(id PK, symbol[], headline, summary, source, published_at, url, cluster_id)`
- `macro_series(series_id, ts, value, vintage_ts)` — FRED vintages kept for point-in-time macro
- `earnings_events(symbol, event_date, period, eps_est, eps_act, revenue_est, revenue_act, call_transcript_ref)`
- `prediction_markets(venue, market_id, question, category, close_ts, resolved_outcome)` +
  `prediction_market_prices(venue, market_id, ts, implied_prob, volume, liquidity)`
- `surveillance_flags(id, symbol, detector, severity, details jsonb, ts, triage_verdict, triaged_at)`
- Research/portfolio/learning tables are specified in docs `03`, `06`–`10`.

### DuckDB/Parquet research sidecar

Nightly Dagster asset exports `bars_1d` (canonical view), features, and scores to partitioned
Parquet (`data/parquet/…`, gitignored). Backtests and feature research read via DuckDB —
columnar full-history scans without hammering the OLTP instance. Postgres remains the single
system of record; the sidecar is disposable and rebuildable.

## 4. Ingestion cadences (Dagster schedules/sensors)

| Job | When | Scope |
|---|---|---|
| EOD bars (Alpaca + Tiingo) | 21:30 ET | Full market |
| Corporate actions | 21:30 ET | Full market |
| EDGAR poller | Every 10 min, market days | Full market (sensor → event assets) |
| Form 4 / insider | Hourly | Full market |
| News (Finnhub/Tiingo) | Every 30 min for Tier 1; EOD sweep for Tier 2 | Tiered |
| Macro (FRED) | Daily 08:00 ET + release-calendar sensor | ~40 series |
| Earnings calendar | Daily | Full market |
| Prediction markets snapshot | Daily 20:00 ET | Filtered: volume/liquidity above floor |
| Tier-2 detectors → Haiku triage | 22:30 ET after bars land | Tier 2 |
| Feature build → quant scores | 23:00 ET | Tier 1 + promoted |
| Committee batches | 23:30 ET submit; sensor collects | Deep-dive set |
| Parquet export | 02:00 ET | Derived tables |

## 5. Corporate actions & adjustment policy

- Raw unadjusted OHLCV stored as received; `adj_close` maintained via Tiingo's adjustment
  factors; splits/dividends stored in `corporate_actions` table from Alpaca + Tiingo.
- On any new corporate action: recompute adjusted series for that symbol, bump
  `adjustment_version`, and invalidate downstream features for affected windows (Dagster asset
  dependency does this naturally).
- Feature/label computation **always uses adjusted series**; execution uses raw prices.

## 6. Data quality (`data/quality.py`)

Checks run as a Dagster asset between ingestion and features; failures quarantine the symbol,
not the pipeline:

1. **Dual-source close check:** |Alpaca close − Tiingo adj-unwound close| > 50 bps → flag symbol,
   exclude from signal generation until resolved; alert if > 20 symbols flagged (systemic).
2. **Gap detection:** missing trading days vs exchange calendar → backfill attempt → flag.
3. **Corporate-action sanity:** overnight return > 40% without a corp action or news flag → hold.
4. **Staleness:** any Tier-1 symbol without fresh EOD bar by 22:00 ET → alert.
5. **Fundamentals sanity:** XBRL facts unit/scale checks (revenue in USD, shares > 0, etc.).
6. **Synthetic-fault drills:** quarterly, deliberately inject a bad close in DEV; the check
   suite must catch it (this is a tested test).

Quarantined symbols are visible on the dashboard with reasons; signals for them are suppressed
and the suppression is itself logged (never silent).

## 7. Prediction-market data policy

- Read-only, snapshot-based; venues behind `PredictionMarketProvider` port.
- **Thin-market filter:** implied probabilities are stored but marked `untrusted` unless volume
  and liquidity floors are met (configurable; default: $25k volume, 3¢ spread).
- Mapping table links markets to research topics (macro: Fed/CPI/recession; single-name events:
  M&A, FDA, earnings) consumed by the daily macro digest and committee dossiers (`03`).
- ToS-compliant API use; no order endpoints wired anywhere.

## 8. Rate-limit and cost budgets (per night, steady state)

| Provider | Requests | Headroom |
|---|---|---|
| Alpaca | ~10 (bulk bar endpoints) | Large |
| Tiingo | ~6.5k EOD symbol calls → batched endpoint ~65 | Fits free-tier RPM at paid plan |
| EDGAR | ≤ 10 req/s poller, ~2k/day | Compliant |
| Finnhub | ~1.5k/day spread hourly | Within 60/min |
| FRED | ~50/day | Trivial |
| Kalshi/Polymarket | ~200/day | Trivial |

All budgets are config values with Prometheus counters; exceeding 80% of any budget raises a
warning alert.
