# Runbook — kill switch

**Use when:** you want all order submission to stop, immediately, for any
reason. This is the blunt instrument. It is safe to engage; it is deliberately
awkward to disengage.

**Severity:** page. **Expected time:** 1 minute to stop, 10 minutes to review.

---

## 0. What the kill switch actually does

`dynamo.execution.gate`:

- **Presence of the file is the switch.** `kill_switch_engaged(path)` is
  `Path(path).exists()`. Nothing else. No daemon has to be alive, no API call
  has to succeed.
- It is checked **inside every gate call**, on every single order submission —
  equities/crypto, options and prediction markets all route through the same
  `live_orders_permitted()`. A switch thrown mid-job blocks the very next
  order, not the next job.
- It applies in **every environment**, including PAPER and DEV. There is
  exactly one halt pathway.
- Engaging via `engage_kill_switch()` also **cancels every open broker order**.
- **Positions are NOT liquidated.** That is a human decision, always.

## 1. Where the file is

One file, resolved identically by the Dagster daemon, the API, and your shell:

```console
$ uv run python -c "from dynamo.config.settings import Settings; print(Settings().kill_switch_file)"
/Users/you/Documents/ai-investing-platform/KILL_SWITCH
```

Relative `DYNAMO_KILL_SWITCH_PATH` values anchor to `$DYNAMO_HOME` (default:
the repo root), **not** to the process working directory. If you set an
absolute path, every process must see the same filesystem.

## 2. Engage it — three ways, pick whichever works

**A. Shell (works with everything else on fire):**

```console
$ printf 'engaged_by=operator\nengaged_at=%s\n' "$(date -u +%FT%TZ)" \
    > "$(uv run python -c 'from dynamo.config.settings import Settings; print(Settings().kill_switch_file)')"
```

This stops new orders instantly. It does **not** cancel resting broker orders
— do step 3.

**B. API (also cancels open orders, writes the alert):**

```console
$ curl -X POST http://127.0.0.1:8000/admin/kill-switch/engage \
    -H 'X-Dynamo-Confirm: kill-switch' \
    -H 'Content-Type: application/json' \
    -d '{"note": "why you are doing this"}'
```

**C. It engaged itself.** A drawdown-ladder transition to **DD20 (-20%)**
engages it automatically and pages you (`ladder_transition` +
`kill_switch_engaged` alerts). See `docs/runbooks/README.md` and step 6.

## 3. Confirm nothing is resting at the broker

If you used the shell path, cancel manually in the broker UI, or:

```console
$ curl http://127.0.0.1:8000/portfolio/orders | jq '.items[] | select(.state != "filled")'
```

Cancel anything open. Then check positions against the broker — the platform
will not move them for you.

## 4. Verify it is really engaged

```console
$ curl -s http://127.0.0.1:8000/admin/kill-switch | jq
$ ls -l "$(uv run python -c 'from dynamo.config.settings import Settings; print(Settings().kill_switch_file)')"
```

The Dagster execution ops log `execution: kill switch engaged — refusing all
orders` on their next run. That log line is your confirmation that the
*daemon* sees the same file you do.

## 5. Understand why before you clear it

Do not clear the switch to "see if it happens again".

- `GET /alerts` — what fired, in order.
- `GET /risk/ladder/halts` — is a drawdown rung awaiting acknowledgment?
- `GET /risk/limits` — what caps were in force (`config_version` tells you
  which risk config ran).
- Dagster run logs for the failing asset.

## 6. Disengage

Two separate acts, and they are separate on purpose.

**6a. If a ladder rung is pending, acknowledge it first.** DD8/DD10/DD15/DD20
leave an unacked `ladder_events` row that blocks orders independently of the
kill-switch file. Clearing only the file leaves you halted and confused.

```console
$ curl -s http://127.0.0.1:8000/risk/ladder/halts | jq
$ curl -X POST http://127.0.0.1:8000/admin/ladder/<event_id>/ack \
    -H 'X-Dynamo-Confirm: ladder-ack' -H 'Content-Type: application/json' \
    -d '{"note": "reviewed the drawdown; thesis intact, resuming at DD15 caps"}'
```

The note is mandatory and is the audit record of who resumed trading and why.

**6b. Then clear the switch:**

```console
$ curl -X POST http://127.0.0.1:8000/admin/kill-switch/disengage \
    -H 'X-Dynamo-Confirm: kill-switch' -H 'Content-Type: application/json' \
    -d '{"note": "cause understood and fixed: <what>"}'
```

`disengage_kill_switch()` refuses a blank acknowledgment. If the API is down,
`rm` the file — but then write the ops-log line by hand, because nothing else
will.

## 7. Log it

Append to `docs/ops-log.md`: when, who, why engaged, what you found, what you
changed, when cleared.

---

## Notes and known gaps

- The signing key for live approvals, `DYNAMO_APPROVAL_KEY`, is deliberately
  **not** stored in the repo or in `.env` (see `approvals/README.md`). If you
  are re-enabling LIVE after an incident and it is not exported, the gate
  refuses with `approval_bad_signature` and prints how to fix it.
- Engaging the kill switch does **not** stop research, ingestion, or the
  nightly pipeline. It stops *orders*. To stop spending money on Claude, see
  [budget-exhaustion.md](budget-exhaustion.md).
