# Operations log

The append-only record of everything that happened to the running platform.
Four separate mechanisms reference this file:

- **Runbooks** (`docs/runbooks/`) — every one ends by writing a line here.
- **The quarterly restore drill** (doc 12 §4) — its results are logged here,
  and the drill is a live-gate dependency (doc 00 §5.4).
- **The alert-fatigue rule** (doc 12 §7, doc 13) — any alert that fires ≥3
  times in a month without prompting action is re-tiered or fixed at the
  source, and that decision is tracked here.
- **Break-glass decisions** (ADR-0004 and friends) — anything done outside the
  normal path, e.g. using yfinance, is only permitted with a logged decision.

## How to write an entry

Newest first. One entry per incident, drill, or deliberate exception. Keep it
short and factual; the value is in being able to reconstruct a month later
what was true and what you did.

```markdown
## YYYY-MM-DD — <short title>

- **Kind:** incident | drill | maintenance | exception | alert-review
- **Severity:** page | warn | info | n/a
- **Runbook:** docs/runbooks/<file>.md (or "none — no runbook covers this")
- **Trigger:** what you saw first (alert name, log line, or "noticed manually")
- **Impact:** what was degraded, and for how long. Trading halted? Money at
  risk? Data lost?
- **Actions:** numbered, what you actually did — including the things that did
  not work
- **Root cause:** or "unknown", which is an honest answer
- **Follow-ups:** concrete, with an owner. "Be more careful" is not one.
- **Elapsed:** detection → resolution
```

Rules:

1. **Append, never edit.** Corrections are a new entry that references the old
   one. This file is evidence, not a wiki page.
2. **Log the boring ones too.** The alert-fatigue rule needs the count of
   times an alert fired and you did nothing — that is the whole signal.
3. **Never paste secrets.** No API keys, no `DYNAMO_APPROVAL_KEY`, no
   `POSTGRES_PASSWORD`, no broker credentials. Reference them by name.
4. **Log every break-glass action** — manual database edits, manual broker
   orders, disabling a check, a data source used outside its ADR. If you did
   something the platform would not have done, it goes here.

## Recurring entries expected

| Cadence | Entry |
|---|---|
| Quarterly | Restore drill (`make backup-verify`): dump size, dump seconds, restore seconds, row counts, surprises |
| Quarterly | Secret rotation, piggybacked on the drill (doc 12 §2) |
| Monthly | Alert review: what fired, how often, what you did — re-tier the noise |
| Per incident | Everything above |

---

# Entries

## 2026-08-20 — First backup and restore drill

- **Kind:** drill
- **Severity:** n/a
- **Runbook:** docs/runbooks/db-restore.md
- **Trigger:** first execution of the newly implemented backup path — before
  this, no backup of any kind existed and the `pgdata` Docker volume was the
  only copy of the database.
- **Impact:** none. The drill restores into a throwaway database
  (`dynamo_restore_drill_<stamp>`) and drops it; `dynamo_dev` was never
  touched.
- **Actions:**
  1. `make backup` → `dynamo-dynamo_dev-20260820T033955Z.dump`,
     **265,318,379 bytes in 15.7s**.
  2. `make backup-verify` → restored into scratch in **30.8s**, all five
     critical tables matched exactly:
     `bars_1d` 6,196,596 · `instruments` 11,505 · `xbrl_facts` 944,859 ·
     `corporate_actions` 104,835 · `short_interest` 3,904,106.
  3. Scratch database dropped; `dynamo_dev` confirmed intact at 2,380 MB.
  4. `backup_last_success_timestamp{kind=logical}` and `{kind=restore_drill}`
     written to `metrics_daily`. The drill metric is what now sets
     `Readiness.backup_drill_ok`.
- **Root cause:** n/a — this is the first drill, not a failure.
- **Follow-ups:**
  1. **Off-host copies.** `backups/` sits on the same disk as `pgdata`; a disk
     loss still loses both. Add a second cron step shipping the newest archive
     off the box. *This is the largest remaining data risk.*
  2. **WAL archiving / wal-g** is documented in doc 12 §4 and not implemented.
     Recovery point is currently the last nightly dump — up to 24h of loss.
  3. Install the 03:00 ET cron line from the runbook on the real host.
- **Elapsed:** ~1 minute.

## 2026-09-13 — backup + restore drill (post-campaign)

- `make backup`: 539,224,037 bytes in 43.5s (dynamo-dynamo_dev-20260914T030234Z.dump) — first dump containing the wide campaign.
- First drill attempt CRASHED the server: timescaledb-ha auto-tuned maintenance_work_mem ~1GB inside a 2.5GiB-capped container; the xbrl_facts PK build OOM-killed Postgres mid-restore (crash recovery clean, no data loss). Fixed: restores now run under PGOPTIONS maintenance_work_mem=256MB.
- Re-drill PASSED: full restore in 126.2s, all five critical tables exact (bars 7,427,210 / instruments 11,505 / xbrl 21,787,755 / corp_actions 104,835 / short_interest 3,904,106).
- Still open: 03:00 cron not installed; no off-host copy.


## 2026-09-15 — backup scheduling moved into Dagster

- The 03:00 cron line fired on time and FAILED: `make: getcwd: Operation not permitted` — macOS privacy protection blocks /usr/sbin/cron from reading ~/Documents. A LaunchAgent hit the same wall (exit 78). Both would need Full Disk Access granted to cron/launchd in System Settings.
- Fix: `backup_job` + `backup_schedule` (03:00 ET) now live in Dagster, which already runs as the user with the folder open and is the scheduler that demonstrably works here. Failure raises, so the run goes red and the dispatcher pages.
- launchd job and cron entry removed. Schedules now 34.


## 2026-09-16 — host power loss, boot race, and the news 429

- **Host was off** ~2026-09-14 late → 2026-09-16 10:36 PDT (user: "my computer was
  dead"). No schedule fired in between, so 09-15's backup, both days' committee
  batches, and Mon/Tue/Wed `paper_execution` (09:35 ET) never ran. The orders
  table is still empty: no paper trades yet, which is the power loss, not a defect.
- **Boot race on the catch-up.** The daemon came up and immediately tried to
  launch ~16 missed schedules against a code server that was already shutting
  down: `DagsterLaunchFailedError: Tried to start a run on a server after telling
  it to shut down`. That batch — `backup_job`, `risk_portfolio_job`,
  `paper_execution_job`, `healthcheck_job` and a dozen more — all went FAILURE
  without executing. It self-corrected seconds later (24 consecutive SUCCESSes
  after), so this is a startup-ordering wart, not a persistent fault. Worth a
  guard if it recurs; a missed tick is silently a failed run in the history.
- **Backup caught up manually:** `dynamo-dynamo_dev-20260916T184445Z.dump`,
  541,509,953 bytes, exit 0.
- **`news_job` failed 3×** on `RateLimited: [finnhub] 429 from /company-news` —
  NOT a power-loss artifact. Finnhub free tier is 60 req/min and the asset sweeps
  tier-1 + promoted one request per symbol, so a 429 partway through is expected
  operating condition. `get_json` deliberately does not retry a 429 (doc 01 §6,
  "no retry storm"), so it escaped into the asset and killed the whole run —
  every symbol after the limit AND TickerTick's headlines, a separate keyless API
  that had not been rate-limited at all but whose upsert sat downstream of the
  raising generator. Fixed in `43ad375`: each source fetches under its own guard
  that stops iteration on a 429; partial results persist (`upsert_news`
  materialises the generator) and the truncation is reported in `rate_limited`
  metadata. 5 regression tests. This mattered before 23:30 ET — the committee
  reads news into its dossiers.
- **Verified against the live failure** (run `933374be`): Finnhub 429'd mid-sweep
  at 11:50:43, exactly as before. The guard caught it, TickerTick carried on for
  another 7 minutes, and the run finished RUN_SUCCESS with
  `rows=2429 finnhub=1296 tickertick=1133 symbols=63 rate_limited=['finnhub']`.
  All 2,429 of those headlines were lost on each of the three prior runs: the
  raise landed inside `upsert_news`'s list comprehension, so even Finnhub's
  already-fetched 1,296 never reached the database and TickerTick never ran.
- **Clean restart** to load the fix: killed all, verified 0 procs, relaunched.
  38 instigators RUNNING (34 schedules + 4 sensors), no boot race this time.
- **Still open:** off-host backup copy; stale `crontab` line (removal hangs under
  macOS privacy gating — harmless, fails instantly with the getcwd error).
