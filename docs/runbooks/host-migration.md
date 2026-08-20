# Runbook — host migration / host loss

**Use when:** the host is dead or unreachable (dead-man silence), or you are
deliberately moving to new hardware.

**Severity:** page when unplanned. **Expected time:** 1-2 hours, dominated by
restoring the database and re-fetching secrets.

---

## 0. Before anything: is the book unattended?

If the host is down, nothing is watching your positions — no stops are being
maintained, no reconciliation is running. Decide within minutes:

- **Broker-side protection.** Any stops resting *at the broker* still work.
  Stops the platform maintains in software do not.
- If you cannot restore quickly and the market is open, flatten or set
  protective orders **manually in the broker UI**. Doing nothing is a choice.

## 1. Confirm it is really the host

```console
$ ping -c 3 <host>
$ ssh <host> uptime
$ curl -sS --max-time 5 http://<host>:8000/health
```

Provider status page for the VPS. A dead-man silence with a live SSH usually
means the daemon died, not the host → [batch-failure.md](batch-failure.md)
step 4.

## 2. What you need to rebuild, and where it lives

| Thing | Where it lives | Recoverable without the host? |
|---|---|---|
| Code | git (GitHub) | Yes |
| Compose topology, Prometheus/Grafana config | git | Yes |
| **Postgres data** | `pgdata` Docker volume | **Only from a `backups/` dump** — see below |
| `backups/*.dump` | the host's disk, next to `pgdata` | **No, if the disk is gone** |
| Secrets (`.env.<env>`) | the host, `chmod 600`, outside the working tree | Only from your password manager |
| `DYNAMO_APPROVAL_KEY` | password manager ONLY, deliberately not in the repo | Only from your password manager |
| Parquet sidecar | `parquet` volume | Rebuildable from Postgres |
| Dagster run history | `dagster-home` volume | No — and it does not matter |

**The uncomfortable one:** until off-host backup copies exist
([db-restore.md](db-restore.md), last section), a total disk loss loses both
`pgdata` and `backups/`. If you have a copy anywhere else — another machine, a
bucket, a USB disk — this is the moment it pays for itself.

## 3. Provision the new host

Target from doc 12 §1: 4 vCPU / 8 GB / ≥160 GB. Docker + Docker Compose.

```console
$ git clone <repo> /opt/dynamo && cd /opt/dynamo
$ cp .env.example .env      # then fill it in from your password manager
$ chmod 600 .env
```

Set `DYNAMO_HOME=/opt/dynamo` if the checkout is not where you will run
commands from — it is what anchors the kill switch and the risk config.

**Engage the kill switch before you start anything**, so nothing trades while
you are mid-migration:

```console
$ mkdir -p /opt/dynamo && date -u +'engaged_by=migration\nengaged_at=%FT%TZ' > /opt/dynamo/KILL_SWITCH
```

## 4. Bring up Postgres and restore

```console
$ docker compose up -d postgres
$ docker compose ps          # wait for healthy
```

Copy your newest dump to `/opt/dynamo/backups/`, then follow
[db-restore.md](db-restore.md) steps 4-6 (verify into scratch, then restore
for real, then `alembic upgrade head`).

If you have **no** dump, the database is gone. What that costs:
~6.2M bars, ~945k XBRL facts, ~105k corporate actions, ~3.9M short-interest
rows and the entire replay/journal/learning history. Bars and corporate
actions can be re-ingested from providers over days; the learning history and
replay campaigns cannot be recreated at all. Say so explicitly in the ops log.

## 5. Start the rest and verify

```console
$ docker compose up -d
$ curl -s http://127.0.0.1:8000/health
$ curl -s http://127.0.0.1:8000/risk/limits | jq '.data.config_version'
$ make backup-status
```

Checklist before you consider the migration done:

- [ ] `/health` OK, `alembic current` at head.
- [ ] Dagster UI reachable; schedules present. **They start STOPPED** — turn
      on only the ones you had on before, deliberately.
- [ ] `DYNAMO_ENV` is what you intend. Default is `DEV`; PAPER/LIVE is a
      choice you make.
- [ ] `make backup` succeeds on the new host, and cron is installed
      ([db-restore.md](db-restore.md)).
- [ ] `TELEGRAM_BOT_TOKEN`/`CHAT_ID` set, and you have seen a test alert
      arrive. Unconfigured, the notifier degrades silently after one log line
      — a quiet new host proves nothing.
- [ ] `HEALTHCHECK_URL` set, or the dead-man switch is not watching you.
- [ ] `DYNAMO_APPROVAL_KEY` exported if you intend to go back to LIVE
      (`python -m dynamo.cli.approve verify --file <artifact>` to prove it).

## 6. Reconcile against the broker before unfreezing

The restored database is as of the dump; the broker is current. Run
reconciliation, resolve every divergence, ack it with a note, then clear the
kill switch ([kill-switch.md](kill-switch.md)).

## 7. Log it

`docs/ops-log.md`: why you migrated, what was lost, elapsed time, and every
checklist item that was NOT satisfied when you resumed.

---

## Planned migration (not an emergency)

Same steps, better ordering: engage the kill switch → `make backup` on the old
host → copy the dump → restore and verify on the new host → cut DNS/access
over → run one full nightly cycle in PAPER on the new host with the switch
still engaged → then clear it. Keep the old host powered off but intact for a
week.
