# Runbook — provider outage

**Use when:** a data provider is down, rate-limiting, or returning gaps.
Triggering alerts: `provider_outage` (warn), `data_quality_systemic` (page),
or you simply noticed empty metadata on an ingestion asset.

**Severity:** warn, unless it is the price feed on a trading day.
**Expected time:** 15 minutes to confirm and decide.

---

## 0. What already happens without you

Degradation is per-asset and automatic (doc 01 §6):

- Every provider accessor on `DynamoResource` returns `None` when its key is
  missing or its client cannot be built. Assets log a skip and materialize
  with empty metadata rather than failing the run. **A skipped asset is not
  a failed pipeline** — do not go looking for a crash.
- `dynamo.data.errors` distinguishes `RateLimited`, `ProviderDown` and
  `DataGap`. Retries and backoff live in `dynamo/data/http.py` and
  `ratelimit.py`.
- Bad or missing symbol data goes to the `quarantine` table; the risk engine
  reads open quarantines and **vetoes those symbols**. A quarantined name
  cannot be bought — that protection needs nothing from you.
- Over 20 symbols quarantined in one run raises `data_quality_systemic`
  (page). That is the "the provider, not the symbol" signal.

## 1. Identify which provider, and whether it matters today

```console
$ curl -s http://127.0.0.1:8000/ops/summary | jq '.data.quarantine_open | length'
$ curl -s http://127.0.0.1:8000/alerts | jq '.items[:20]'
```

Then rank by consequence:

| Provider | Feeds | If it is down |
|---|---|---|
| Alpaca (data) | daily/intraday bars, crypto bars, **quotes for execution** | **Highest.** No marks, no sizing, no safe execution |
| Tiingo | second price source for the canonical-bars cross-check | Single-source data gets labelled, not hidden. Tolerable for days |
| Finnhub | earnings calendar, estimates | Research degrades; nothing trades on it directly |
| FRED | macro series | Regime inputs fall back to defaults (`_aligned(..., default)`) — the regime becomes less informative, not wrong |
| EDGAR | filings, XBRL, insider | Fundamentals go stale. Tolerable for days |
| Alpha Vantage / defeatbeta | transcripts | Optional enrichment |
| Kalshi / Polymarket | prediction-market sleeve | That sleeve idles |

## 2. Confirm it is them, not you

1. The provider's status page.
2. `docker compose logs --since 1h dagster-daemon | grep -i -E 'ratelimit|providerdown|datagap'`
3. Check your key has not expired or hit a plan cap — `RateLimited` and
   "your free tier ended" look identical from inside the process.
4. Confirm the host has network: `curl -sS https://api.alpaca.markets/v2/clock`.

## 3. Decide

**If it is the price feed and the market is open:** engage the kill switch
([kill-switch.md](kill-switch.md)). Sizing, stops and reconciliation all read
prices; acting on stale marks is worse than not acting.

**If it is anything else:** do nothing. Let the assets skip. The nightly
report renders partial results and lists the gaps. Re-running an ingestion
asset later backfills, because every ingest upsert is idempotent.

**Never** substitute a different data source by hand to "fill in". ADR-0004
makes yfinance break-glass only, and any use of it requires a logged decision
in `docs/ops-log.md`. Data that entered outside the normal path is
indistinguishable from good data six months later.

## 4. After it recovers

```console
# Re-run the affected assets for the missed partitions.
$ uv run dagster job execute -f orchestration/definitions.py -j eod_ingest_job
```

Then clear the quarantine entries whose cause was the outage
(`POST /admin/...` or directly, with a note), and confirm the next
`data_quality` run is clean.

## 5. Log it

`docs/ops-log.md`: provider, window, what degraded, whether you halted
trading, whether any decision was made on partial data.

---

## Known gaps

- There is **no automatic failover** between price providers, by design: two
  sources disagreeing is information (the canonical-bars cross-check), and
  silently swapping one for the other destroys it.
- `provider_budget_utilization` and `provider_errors_total` are in the doc 12
  §6 metrics catalogue but have no producer yet, so the 80%-of-budget warn
  will not fire. Watch the daemon logs during an outage instead.
