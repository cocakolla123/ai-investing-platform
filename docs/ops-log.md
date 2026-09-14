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

