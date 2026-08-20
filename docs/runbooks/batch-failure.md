# Runbook — batch / nightly-pipeline failure

**Use when:** the Claude committee batch errored or expired, or the nightly
Dagster chain did not complete. Triggering alerts: `batch_expiry` (warn),
nightly-pipeline failure (page), or the dead-man switch went quiet.

**Severity:** page if the whole nightly chain failed; warn for a single batch.
**Expected time:** 20 minutes.

---

## 0. What already happens without you

- The committee submits through the Anthropic **Message Batches** API with
  **deterministic `custom_id`s**. Entries that error, expire or are cancelled
  come back as `(custom_id, None)` — they are *identified*, not lost.
- `batch_jobs` rows carry the batch id and status; a batch over 24h unresolved
  is marked `expired` (`DynamoResource.mark_batch_expired`).
- The risk engine and execution path **do not depend on the committee at
  all** (doc 06). Stops, reconciliation and the drawdown ladder run whether or
  not any research happened. A failed batch means no *new* theses, not an
  unmanaged book.
- The daily report renders partial results and lists what is missing.

## 1. Find out what actually failed

```console
$ curl -s http://127.0.0.1:8000/ops/summary | jq '.data.batch_jobs'
$ curl -s http://127.0.0.1:8000/alerts | jq '.items[:20]'
```

Dagster UI (`make dagster`, or `http://127.0.0.1:3000`) → Runs → the failed
run → the failed asset's logs. Note **which asset** failed; the chain is
ordered (features → scores → signals → regime → selection → committee →
meta-scoring → risk verdict → target portfolio), and everything downstream of
the failure simply did not run.

## 2. Classify

| Symptom | Cause | Action |
|---|---|---|
| Batch status `expired`, no results | Anthropic-side backlog or incident | Wait. Resubmit next night; the `custom_id`s are stable so nothing duplicates |
| `BudgetExceededError` | Monthly Claude budget hit | [budget-exhaustion.md](budget-exhaustion.md) |
| `PlanLimitExhausted` | `RESEARCH_BACKEND=claude_code` and the subscription window is spent | Wait for the window, or set `claude_code_fallback_to_api` with a key |
| Schema-validation failures on responses | Prompt or schema drift | Inspect the stored raw responses. **Never hand-edit an output.** Fix the prompt/schema, re-run |
| An ingestion asset failed first | Provider problem | [provider-outage.md](provider-outage.md) |
| The whole run never started | Daemon dead | Step 4 |

## 3. Re-run — safely

Every asset in this pipeline is idempotent (upserts keyed on natural keys), so
re-running is safe. Re-run **from the failed asset**, not the whole chain, so
you do not burn budget re-doing committee calls that already succeeded:

```console
$ uv run dagster job execute -f orchestration/definitions.py -j scores_job
$ uv run dagster job execute -f orchestration/definitions.py -j risk_portfolio_job
```

Resubmit only the *failed* `custom_id`s. Resubmitting the whole batch doubles
the spend for no new information.

## 4. If the daemon itself is dead

```console
$ docker compose ps
$ docker compose logs --tail 200 dagster-daemon
$ docker compose up -d dagster-daemon
```

If it is OOM-killed (exit 137), check `mem_limit` in `docker-compose.yml` —
the daemon runs jobs **in-process** (doc 12 §1), so a heavy asset is the
daemon's own memory.

If the host is unreachable at all → [host-migration.md](host-migration.md).

## 5. Decide about today's trading

A missing committee night is not an emergency. `target_portfolio` will simply
not be refreshed, and execution acts on the most recent target it has.

Halt only if the **risk verdict** failed to materialize — without it there are
no current caps, and the constructor has nothing to clamp against.

## 6. Log it

`docs/ops-log.md`: which asset, cause, whether you re-ran, spend impact,
whether the night's decisions were made on partial research.

---

## Known gaps

- `pipeline_run_success{job}` is in the doc 12 §6 catalogue but has no
  producer, so **"nightly pipeline failure" does not page you today**. Until
  it does, the dead-man ping is the real detector: set `HEALTHCHECK_URL` to a
  Healthchecks.io check with a 2h grace, and the `healthcheck_job` (02:30 ET,
  after the chain) pings it. No ping, no chain.
- `schema_validation_failures_total` likewise has no producer; count them in
  the daemon logs.
