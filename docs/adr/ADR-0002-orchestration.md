# ADR-0002 — Dagster OSS for orchestration (Prefect as designated fallback)

**Status:** accepted · 2026-08-03

## Context
The nightly pipeline is asset-shaped: bars → features → quant scores → dossiers → Claude batches
→ reports → watchlist, plus pollers, sensors (Message Batch completion), backfills, and run
history. Single host, 8 GB RAM budget shared with everything else.

## Decision
Dagster OSS: software-defined assets, daily partitions, backfills, sensors, and a run UI map 1:1
to the problem. Dagster storage reuses the main Postgres. Orchestration code stays a thin layer
in `orchestration/` calling core functions, so the scheduler is swappable.

## Alternatives rejected
- **APScheduler:** would require hand-rolling retries, run history, dependency ordering,
  backfills, and alerting — precisely the plumbing that must be boring and reliable here.
- **Prefect 3:** good middle ground but task-centric; weaker "this asset is stale, rebuild it
  and downstreams" story. **Designated fallback** if Dagster's ~2 containers strain host RAM.

## Consequences
Two extra containers (webserver + daemon) and a steeper API; in exchange: observability,
backfills, and event-driven sensors for free. Fallback path documented and cheap because
orchestration is thin.
