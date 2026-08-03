# ADR-0001 — One Postgres instance (TimescaleDB + pgvector) with a DuckDB/Parquet research sidecar

**Status:** accepted · 2026-08-03

## Context
Single-host deployment, one user, ~6k symbols of EOD data + minute bars for a few hundred names,
relational research/portfolio/journal data, and embedding search for the investment journal.
Backtests and feature research need columnar full-history scans.

## Decision
One Postgres 16 instance running TimescaleDB (hypertables, compression, continuous aggregates)
and pgvector. A nightly job exports derived time-series tables to partitioned Parquet; backtests
and feature research read them via DuckDB. Postgres is the sole system of record; the sidecar is
disposable.

## Alternatives rejected
- **ClickHouse/QuestDB for ticks:** overkill for EOD/minute bars at this scale; second system to
  operate and back up.
- **SQLite:** concurrent writers (ingestion, Dagster, API) make it painful.
- **Dedicated vector DB:** pgvector is sufficient at journal scale (thousands of entries).

## Consequences
One backup story and one container; OLTP protected from analytical scans; an extra export job
to maintain. Revisit only if minute-bar retention or symbol scope grows ~10×.
