# Runbooks

Doc 12 §8 requires these six, written **before the first paper trade**. Each
one is a checklist for an operator at 02:00 who is not going to read source
code first. They describe what this platform *actually does today* — where a
mechanism does not exist, the runbook says so rather than describing the
design.

| File | Fires when |
|---|---|
| [provider-outage.md](provider-outage.md) | a market-data / filings / news provider is down or rate-limiting |
| [batch-failure.md](batch-failure.md) | the Claude committee batch fails, expires, or the nightly pipeline errors |
| [db-restore.md](db-restore.md) | Postgres is lost or corrupt — and the quarterly restore drill |
| [host-migration.md](host-migration.md) | the host is dead, or you are moving to a new one |
| [kill-switch.md](kill-switch.md) | you need trading to stop NOW, and how to start it again |
| [budget-exhaustion.md](budget-exhaustion.md) | the Claude monthly budget hits 70% / 100% |

**Every runbook ends by writing a line in [`../ops-log.md`](../ops-log.md).**
That file is what the alert-fatigue rule (doc 13) and the quarterly review
read. An incident that is not in the log did not happen.

## Facts every runbook assumes

- One host, Docker Compose (`docker-compose.yml`). Postgres is the only
  stateful service; everything else is rebuildable.
- `DYNAMO_ENV` selects the environment. `PAPER` is the normal state. `LIVE`
  additionally requires a signed approval artifact (`approvals/README.md`).
- The kill switch is ONE file, at `$DYNAMO_HOME/KILL_SWITCH` unless
  `DYNAMO_KILL_SWITCH_PATH` is absolute. Every process resolves it the same
  way (`Settings.kill_switch_file`).
- Alerts land in the `alerts` table AND, when `TELEGRAM_BOT_TOKEN` /
  `TELEGRAM_CHAT_ID` are set, in Telegram. Unconfigured, the notifier logs one
  warning per process and never crashes anything — so **a silent night does
  not mean a healthy night** unless Telegram is configured.
