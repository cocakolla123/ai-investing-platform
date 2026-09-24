.PHONY: dev test lint typecheck check up down migrate fmt backup backup-verify backup-status

dev:            ## Install deps into .venv
	uv sync

test:           ## Run unit tests
	uv run pytest tests/unit

test-all:       ## Unit + integration tests
	uv run pytest

lint:           ## Ruff lint + import-linter dependency rule
	uv run ruff check src tests
	uv run lint-imports

fmt:            ## Format
	uv run ruff format src tests

typecheck:      ## Pyright (strict on domain + risk)
	uv run pyright

check: lint typecheck test  ## Full local gate (mirrors CI)

up:             ## Start the stack
	docker compose up -d

down:
	docker compose down

migrate:        ## Apply DB migrations (M1+)
	uv run alembic upgrade head

# --- Orchestration (M1) ---------------------------------------------------
.PHONY: dagster dagster-host dagster-host-stop dagster-host-status pipeline-dry api dash

# Always the persistent instance. Without this, `dagster dev` silently creates a
# throwaway .tmp_dagster_home_* with every schedule off (2026-09-16 outage).
export DAGSTER_HOME ?= $(CURDIR)/.dagster_home

dagster:        ## Dagster dev UI for LOCAL DEVELOPMENT only (temp workspace; not for the always-on host)
	uv run dagster dev -w workspace.yaml

dagster-host:   ## Always-on scheduler for paper/live: daemon + webserver, supervised, persistent state
	./scripts/dagster_host.sh start

dagster-host-stop:  ## Stop the always-on scheduler
	./scripts/dagster_host.sh stop

dagster-host-status:  ## Is the scheduler alive, and how many schedules are RUNNING in the instance it uses?
	./scripts/dagster_host.sh status

pipeline-dry:   ## Wiring dry-run job: no DB or API keys needed
	uv run dagster job execute -f orchestration/definitions.py -j dry_run_job

api:            ## Serve the FastAPI app on 127.0.0.1:8000
	uv run python -m dynamo.api

dash:           ## Serve the Streamlit dashboard on 127.0.0.1:8501
	uv run streamlit run dashboard/app.py

# --- Backups (doc 12 section 4; runbook docs/runbooks/db-restore.md) -------
# Deliberately NOT a compose service: a resident container costs RAM on a
# 8 GB host all day for a job that runs 16 seconds a night, and the backup
# you most want is the one taken while the app containers are stopped.
# Schedule with host cron -- the crontab line is in the runbook.

backup:         ## pg_dump -Fc into backups/, prune to 30 daily + 12 monthly, write the metric
	uv run python -m dynamo.cli.backup run

backup-verify:  ## Quarterly restore drill: restore the newest dump into a SCRATCH db and assert row counts
	uv run python -m dynamo.cli.backup verify

backup-status:  ## Inventory + staleness (exit 1 when the last backup is > 26h old)
	uv run python -m dynamo.cli.backup status
