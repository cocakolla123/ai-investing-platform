# Runbook — database restore (and the quarterly drill)

**Use when:** Postgres is lost, corrupt, or you are performing the quarterly
restore drill that doc 00 §5.4 makes a dependency of the live (real-money) gate.

**Severity:** page. **Expected time:** drill ~5 minutes; real recovery
30-90 minutes depending on where the dump lives.

---

## 0. What exists, and what does not

**Exists:**

- `make backup` — `pg_dump -Fc` of the whole database into `backups/`, prunes
  to 30 daily + 12 monthly, writes
  `backup_last_success_timestamp{kind=logical}` into `metrics_daily`.
- `make backup-verify` — restores the newest dump into a **throwaway**
  `dynamo_restore_drill_<stamp>` database, asserts row counts on five critical
  tables, drops it, writes `…{kind=restore_drill}`. That metric is what
  `Readiness.backup_drill_ok` reads.
- `make backup-status` — inventory + staleness; exits non-zero past 26h.

**Does NOT exist yet** (doc 12 §4 describes them; nothing implements them):

- WAL archiving / wal-g point-in-time recovery. **Recovery point is the last
  nightly dump**, so you can lose up to a day.
- Off-host copies. `backups/` is on the same disk as `pgdata`. A disk loss
  loses both. Copying the dump off-box is a manual step until this is fixed
  — see the last section.

## 1. Take a backup right now (before touching anything)

Even a corrupt database is evidence.

```console
$ make backup
pg_dump -Fc dynamo_dev -> .../backups/dynamo-dynamo_dev-20260820T033955Z.dump
  ok: 265,318,379 bytes in 15.7s
  retention: 1 kept, 0 pruned
  metric: backup_last_success_timestamp{kind=logical} = ...
```

If `pg_dump` itself fails, skip to step 4 and restore from the newest existing
dump instead.

## 2. Stop everything that writes

```console
$ docker compose stop dagster-daemon dagster-webserver api dashboard
```

Leave `postgres` running — you restore *through* it, into a different
database.

Engage the kill switch too if any orders could be in flight
([kill-switch.md](kill-switch.md)).

## 3. Find the dump you want

```console
$ make backup-status
backup dir: /.../backups
dumps: 14
  dynamo-dynamo_dev-20260820T033955Z.dump  265,318,379 bytes
  ...
last logical backup: 2026-08-20T03:39:55+00:00
last restore drill:  2026-08-20T03:40:19+00:00
stale (> 26h): False
```

Filenames carry a UTC timestamp: `dynamo-<db>-<YYYYmmddTHHMMSS>Z.dump`.

## 4. Verify the dump before you trust it

**Never restore over the working database to find out whether the dump is
good.** `make backup-verify` restores into a scratch database and drops it:

```console
$ make backup-verify
restore drill: dynamo-dynamo_dev-20260820T033955Z.dump -> scratch database dynamo_restore_drill_20260820034019
  bars_1d: 6,196,596 -> 6,196,596
  instruments: 11,505 -> 11,505
  xbrl_facts: 944,859 -> 944,859
  corporate_actions: 104,835 -> 104,835
  short_interest: 3,904,106 -> 3,904,106
  restore took 30.8s from a 265,318,379-byte archive
  metric: backup_last_success_timestamp{kind=restore_drill} = ...
restore drill PASSED — log it in docs/ops-log.md
```

Any mismatch, or a table at zero rows on both sides, fails the drill loudly
and raises a `restore_drill_failed` page.

Add `--dump <path>` to verify a specific archive, `--keep-scratch` to leave
the scratch database around for a look.

## 5. Restore for real

Only after step 4 passed on the archive you intend to use.

```console
# 5a. Rename the damaged database out of the way — do NOT drop it.
$ docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" <pg-container> \
    psql -h localhost -U dynamo -d postgres \
    -c 'ALTER DATABASE dynamo_dev RENAME TO dynamo_dev_broken_20260820'

# 5b. Fresh target.
$ docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" <pg-container> \
    psql -h localhost -U dynamo -d postgres -c 'CREATE DATABASE dynamo_dev'

# 5c. Restore into it.
$ docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" <pg-container> \
    pg_restore --no-owner --no-acl --exit-on-error \
    -h localhost -U dynamo -d dynamo_dev < backups/<the dump>
```

Container name: `docker ps --format '{{.Names}}' | grep postgres`.
`--exit-on-error` matters — a half-restored database that "succeeded" is how
you end up trading on partial history.

## 6. Bring the schema to head and restart

```console
$ uv run alembic upgrade head          # dumps restore the schema they had
$ docker compose up -d
$ curl -s http://127.0.0.1:8000/health
```

## 7. Reconcile before you unfreeze

The database is now as of the dump. The **broker is not**. Positions, orders
and fills between the dump and now are missing.

1. `POST /admin/reconciliation/run` (or wait for the 15-minute
   reconciliation op) and read the divergences.
2. Resolve them deliberately. A divergent run is a sticky halt and stays one
   until you `POST /admin/reconciliation/{run_id}/ack` with a note.
3. Only then clear the kill switch ([kill-switch.md](kill-switch.md)).

## 8. Log it

`docs/ops-log.md`: which dump, elapsed time, row counts, what was lost
between the dump and the incident, and what you had to do by hand.

---

## The quarterly drill (doc 12 §4, doc 00 §5.4)

Once a quarter, on a calm day:

1. `make backup` — fresh archive.
2. `make backup-verify` — record the archive size and the restore time.
3. `curl -s http://127.0.0.1:8000/health` to confirm nothing was disturbed.
4. Append to `docs/ops-log.md`: date, dump size, dump seconds, restore
   seconds, row counts, surprises.

Passing the drill writes `backup_last_success_timestamp{kind=restore_drill}`,
which is the only thing that sets `Readiness.backup_drill_ok`. It expires
after 100 days — a stale drill fails the live (real-money) gate closed, by design.

## Scheduling the nightly backup

Not a compose service (a resident container costs RAM all day for a 16-second
job, and the dump you most want is the one taken while the app containers are
stopped). Host cron, 03:00 ET:

```cron
0 3 * * *  cd /opt/dynamo && /usr/local/bin/uv run python -m dynamo.cli.backup run --quiet >> /var/log/dynamo-backup.log 2>&1
```

A failure exits non-zero **and** raises a `backup_failed` page through the
notifier. `make backup-status` exits 1 once the last backup is over 26h old —
wire that into your monitoring if you have any.

## Known gap: the dump is on the same disk as the data

Until off-host copies exist, add a second cron line that ships the newest
archive somewhere else (`rclone`, `rsync`, `aws s3 cp`, a USB disk — anything
that is not this disk). Until you do, **a disk failure loses everything**, and
that is the single largest remaining risk in this runbook.
