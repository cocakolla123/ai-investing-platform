---
status: reviewed
owner: lead
last-updated: 2026-08-03
---

# 12 — Operations

Operations for a one-person platform must be **boring, automated, and rehearsed**. The design
target: the owner spends < 30 minutes/week on ops in steady state, and every failure mode either
self-heals, degrades gracefully, or pages loudly. Reliability is a feature (doc 00 §4:
pipeline completion ≥ 99% unattended).

## 1. Compose topology and resource budget

One host, Docker Compose, per doc 01 §3. Budget sized for a **4 vCPU / 8 GB** machine.

| Container | Image | Mem limit | CPU (soft) | Notes |
|---|---|---|---|---|
| `postgres` | `timescale/timescaledb-ha:pg16` (pinned digest) + pgvector | 2.5 GB | 1.5 | `shared_buffers=1GB`, `work_mem=32MB`; the only stateful service |
| `dagster-webserver` | project image | 512 MB | 0.5 | UI only; can be stopped without affecting runs |
| `dagster-daemon` | project image | 1 GB | 1.0 | Schedules/sensors/run launcher; runs jobs in-process (no per-run containers — RAM) |
| `api` | project image (`infra/docker/Dockerfile.api`) | 512 MB | 0.5 | Stateless Uvicorn |
| `dashboard` | Streamlit (M1–M5) → nginx+React (M6) | 512 MB → 128 MB | 0.5 | Doc 11 §2 |
| `prometheus` | `prom/prometheus` (pinned) | 512 MB | 0.25 | 30d local retention |
| `grafana` | `grafana/grafana-oss` (pinned) | 256 MB | 0.25 | Dashboards provisioned from `infra/monitoring/` |
| **Total limits** | | **~5.8 GB** | | ~2 GB headroom for OS, page cache, and nightly job spikes |

Host options (doc 00 §6, $0–15/mo line):

| Option | Cost | Trade-off |
|---|---|---|
| **Hetzner CPX31** (4 vCPU, 8 GB, 160 GB) | ~$14/mo | Default. Unattended, snapshot-able, boring |
| Home server (any 8 GB box) | ~$0 | Free but adds residential-network/power failure modes; acceptable, dead-man switch covers silent death |

Disk: bars for ~6k symbols ≈ 1.5M rows/yr plus filings/news — < 40 GB in year one with
Timescale compression; the 160 GB CPX31 disk is years of headroom. Parquet sidecar lives on the
same disk and is disposable (doc 02 §3).

## 2. Environments and secrets

Four environments per doc 01 §5 (`DEV`/`TEST`/`PAPER`/`LIVE`), selected by `DYNAMO_ENV`,
config from `.env.<env>` (template: `.env.example` at repo root).

- **On host:** plaintext `.env.<env>` files, `chmod 600`, outside the repo working tree.
- **In repo:** `infra/secrets/<env>.env.sops` — the same files encrypted with
  **sops + age**; committed so a host loss doesn't lose the secret set (backup parity).
  The age private key lives in the owner's password manager and on the host at
  `~/.config/sops/age/keys.txt` — it is the one secret with no automated backup, by design.
  ADR-0008: no Vault — a secrets *server* is more operational surface than the secrets it
  would protect.
- **In CI:** GitHub Actions secrets (`ANTHROPIC_API_KEY` small-cap DEV key, deploy SSH key,
  GHCR token). CI never sees paper/live broker credentials; deploys ship images, not env files.
- Rotation: quarterly, piggybacked on the restore drill (§4); structlog redaction (§5) is the
  last line against leakage.

## 3. CI/CD (GitHub Actions)

### 3.1 PR pipeline (`.github/workflows/pr.yml`)

Mirrors `make check` plus what needs services:

| Step | Gate |
|---|---|
| `ruff check` + `ruff format --check` | Style |
| `lint-imports` | The dependency rule (doc 01 §2) — architectural drift fails CI |
| `pyright` | Strict on `domain` + `risk` |
| Unit tests (incl. hypothesis risk suites) | `tests/unit` |
| Integration tests | `tests/integration` against a `timescale/timescaledb-ha:pg16` service container + vcrpy cassettes; no live provider calls |
| Docker build (api, dashboard, dagster) | Images must build on every PR |
| `pip-audit` | Known-vuln gate (advisory → blocking for critical) |

### 3.2 Main pipeline

Everything above, then: build and push images to **GHCR** tagged `sha-<short>` + `latest`;
deploy = `ssh host 'cd /opt/dynamo && docker compose pull && docker compose up -d'` followed by
a health-endpoint smoke check. Rollback is `docker compose up` with the previous sha tag —
images are the unit of deployment, the database schema migrates forward-only (Alembic,
expand-then-contract for anything touching live tables).

### 3.3 Scheduled

- **Weekly full nightly-pipeline dry-run in CI**: the complete Dagster nightly job against
  recorded provider data (cassettes + a frozen DB snapshot fixture), Claude calls mocked from
  golden outputs. Catches wiring/schema rot that unit tests can't (doc 01 §6 testing summary).
- **Dependabot** weekly for GitHub Actions and Docker base images; uv lockfile handled
  manually per §9.

## 4. Backups and the restore drill

Postgres is the single system of record; everything else is rebuildable.

| What | How | Where | Retention |
|---|---|---|---|
| Nightly logical | `pg_dump -Fc` per database at 03:00 ET (after Parquet export) | Object storage (Hetzner Object Storage or B2, ~$5/mo, doc 00 §6) | 30 daily, 12 monthly |
| Continuous physical | **wal-g** base backup weekly + WAL archiving | Same bucket, separate prefix | 2 base backups + WAL |
| Parquet sidecar | **Not backed up** — rebuilt from Postgres by the export asset | — | — |
| Repo + sops secrets + Grafana/Prometheus config | Git (GitHub) | — | — |

Backup success/failure is a Prometheus metric and a §7 alert; a backup that silently stops is
the classic solo-operator failure.

**Quarterly restore drill** — a live-gate dependency (doc 00 §5.4), executed as a checklist
(`docs/runbooks/db-restore.md`), results logged in `docs/ops-log.md`:

1. Provision scratch container; restore latest `pg_dump` → row-count and spot-check assertions.
2. wal-g point-in-time restore to a chosen timestamp → verify a known transaction boundary.
3. Decrypt `infra/secrets/paper.env.sops` with the password-manager copy of the age key.
4. Run the nightly pipeline in DEV against the restored DB — it must complete.
5. Record elapsed time (target: < 2h to operational) and any surprises as issues.

## 5. Structured logging

structlog, JSON to stdout, collected by `docker compose logs` / journald; shipped nowhere (one
host — Grafana's Loki is deliberately out of scope, ADR non-goal parity with doc 00 §8.4).

Schema (every line):

```json
{"ts": "2026-08-03T03:12:44.120Z", "level": "info", "env": "PAPER",
 "run_id": "dagster-run-uuid", "module": "dynamo.llm.batch_runner",
 "event": "batch_submitted", "symbol": "NVDA", "cost_usd": 0.083}
```

- `run_id` joins application logs to Dagster runs and asset materializations (doc 01 §6).
- `symbol` and `cost_usd` optional; all other keys mandatory (enforced by a processor).
- **Redaction processor**: known secret keys (`*_API_KEY`, `*_SECRET`, `*TOKEN*`,
  `POSTGRES_PASSWORD`) and any value matching loaded secret values are replaced with
  `[redacted]` before serialization. Unit-tested with real-shaped fakes.
- Retention: 14 days via Docker log rotation (`max-size: 50m`, `max-file: 7` per service).
  Anything worth keeping longer is a metric or a database row, not a log line.

## 6. Monitoring

Prometheus scrapes `api`, `dagster` (via exporter asset), and node-exporter on the host.
Grafana dashboards are **provisioned as JSON** in `infra/monitoring/dashboards/` — the UI is
never the source of truth.

### Metrics catalog (prefix `dynamo_`)

| Metric | Type | Purpose |
|---|---|---|
| `pipeline_stage_duration_seconds{stage}` | histogram | Nightly stage timing; regression detection |
| `pipeline_run_success{job}` | gauge | Completion-rate SLO (doc 00 §4) |
| `provider_errors_total{provider,kind}` | counter | `RateLimited` / `ProviderDown` / `DataGap` (doc 01 §6) |
| `provider_budget_utilization{provider}` | gauge | vs doc 02 §8 request budgets; warn at 80% |
| `claude_tokens_total{model,call}` / `claude_cost_usd_total` | counter | From `cost_meter` (doc 03 §5) |
| `claude_budget_utilization` | gauge | Monthly spend / `CLAUDE_MONTHLY_BUDGET_USD` |
| `schema_validation_failures_total{agent}` | counter | The doc 00 §4 ≥ 99% validity metric |
| `risk_limit_utilization{limit}` | gauge | Per doc 06 §2 limit |
| `abstention_rate_30d` | gauge | Near-zero is a red flag, not a win |
| `reconciliation_divergences_total` | counter | Doc 09; any increment alerts |
| `drift_indicator{model,feature}` | gauge | Quant input/output drift (doc 08) |
| `backup_last_success_timestamp{kind}` | gauge | Staleness-based alerting |

Plus node-exporter disk/RAM/CPU. **Dead-man switch:** the nightly job's final step pings
Healthchecks.io (free tier); a missed ping alerts via Healthchecks' own delivery —
independent of the entire host, which is the point.

## 7. Alert catalog

Delivery: Grafana alert rules → Telegram bot (`TELEGRAM_BOT_TOKEN`/`CHAT_ID` in env).
Severities: **page** (act now), **warn** (act within a day), **info** (awareness). The
dashboard lists all alerts; ack semantics in doc 11 §4. Every page-class alert names a runbook.

| Alert | Severity | Trigger | Response |
|---|---|---|---|
| Nightly pipeline failure | page | `pipeline_run_success == 0` for nightly job | Runbook: batch-failure; rerun after fix; report renders partial results meanwhile |
| Dead-man silence | page | Healthchecks.io missed ping (grace 2h) | Host/daemon down — SSH in; runbook: host-migration if unreachable |
| Data-quality systemic flag | page | > 20 symbols quarantined in one run (doc 02 §6) | Provider-level issue; runbook: provider-outage; signals already suppressed |
| Claude budget 100% | page | `claude_budget_utilization ≥ 1.0` | cost_meter refuses new batch submissions automatically (doc 03 §5 — **research-only stop; orders and stops are untouched**, the global kill-switch is a separate mechanism); runbook: budget-exhaustion (degraded mode §8) |
| Reconciliation divergence | page | `reconciliation_divergences_total` increment | Positions vs broker mismatch; halt new orders until resolved (doc 09) |
| Drawdown ladder transition (−10% and below) | page | Ladder rung change | Doc 06 §3 responses are automatic; human reviews and acks the state |
| Backup failure | page | `backup_last_success_timestamp` stale > 26h | Fix tonight — never run without yesterday's backup |
| Drawdown ladder transition (−5%, −8%) | warn | Ladder rung change | Review positions; automatic responses already applied |
| VaR exception cluster | warn | Kupiec traffic light leaves green (doc 06 §6) | Caps auto-tighten 25%; review risk model |
| Claude budget 70% | warn | `claude_budget_utilization ≥ 0.7` | Reduce deep-dive cap or accept degradation date (doc 03 §5 knobs) |
| Batch expiry | warn | Batch > 24h unresolved | Auto-resubmit of failures already ran (doc 03 §3); investigate if repeated |
| Provider outage | warn | `provider_errors_total{kind="ProviderDown"}` sustained 30m | Runbook: provider-outage; per-asset degradation is automatic |
| Provider budget 80% | warn | `provider_budget_utilization ≥ 0.8` | Check for a retry loop or scope creep |
| Disk > 80% / RAM sustained > 90% | warn | node-exporter | Compress/expand; check Parquet growth |
| Pipeline stage slow (> 2× median) | info | Duration histogram | Watch; often a provider slowdown |
| Universe/improvement queue item pending > 7d | info | Queue age | Decide or reject; queues must not rot |

Alert-fatigue rule: any alert that fires ≥ 3× in a month without prompting action is
re-tiered or fixed at the source — tracked in `docs/ops-log.md` (see doc 13, alert fatigue).

## 8. Runbooks

Live in `docs/runbooks/`, one file each, written **before** the first paper trade (M5 gate).
Summaries:

- **provider-outage.md** — confirm via provider status page; pipelines already degrade
  per-asset (doc 01 §6); if Tiingo is down the dual-source check is suspended and flagged in
  the report (single-source data is labeled, not hidden); yfinance remains break-glass only
  (ADR-0004) and its use requires a logged decision.
- **batch-failure.md** — inspect quarantined calls (raw responses stored); resubmit failures
  only via deterministic `custom_id`s; if Anthropic-side incident, wait — the report renders
  partial results and lists gaps (doc 03 §3); never hand-edit outputs.
- **db-restore.md** — the §4 drill checklist, doubling as the real-incident procedure.
- **host-migration.md** — provision new host → `docker compose up` from GHCR images →
  restore §4 backups → decrypt sops secrets → repoint tailscale + Healthchecks → verify with a
  DEV pipeline run. Target: < 1 day, tested implicitly by the restore drill.
- **kill-switch.md** — engage: `POST /killswitch` or touch the kill-switch file on host
  (works with API down); verify orders refused (doc 01 §5 gate). Disengage: requires the
  triggering alert resolved + a one-line rationale appended to `docs/ops-log.md`; the file
  removal is deliberate and manual, never automated.
- **budget-exhaustion.md** — when `cost_meter` hits the cap it refuses new batch submissions
  (doc 03 §5). **Degraded mode is defined, not improvised:** the nightly pipeline still runs
  ingestion, data quality, Tier-2 detectors (code-only), quant scores, and the risk engine —
  reports become **quant-only** (scores, flags, risk metrics; no committee verdicts) and are
  labeled as degraded. Existing verdicts age normally and staleness is displayed. Options:
  wait for month rollover, or raise `CLAUDE_MONTHLY_BUDGET_USD` as a deliberate config change.
  Risk management never degrades — it uses no LLM (doc 06).

## 9. Update and patch policy

| Component | Policy |
|---|---|
| Python deps | uv lockfile upgraded **monthly via PR** (`uv lock --upgrade`); full CI + weekly dry-run must pass; no auto-merge |
| Security patches | pip-audit/dependabot criticals: out-of-band PR within 48h |
| Postgres/Timescale | Image **pinned by digest**; minor upgrades quarterly: restore latest backup into the new image in scratch, run integration suite, then upgrade host (piggybacks the restore drill). Major versions: planned migration with ADR |
| Prometheus/Grafana/base images | Dependabot PRs, monthly batch |
| Anthropic models | Pinned model ids in config; migrations are deliberate, ADR-gated (doc 13) |
| Host OS | unattended-upgrades (security only); reboot window Sunday 04:00 ET with compose restart + post-reboot health check |

## 10. Open questions (carried to 14-design-review)

1. Object-storage vendor: Hetzner Object Storage (same provider, simpler) vs Backblaze B2
   (provider-diverse backups). Leaning B2 — a Hetzner account issue shouldn't take host *and*
   backups.
2. Is journald + docker log rotation enough, or does the first real debugging session justify
   Loki after all? Decide after M3, not before.
