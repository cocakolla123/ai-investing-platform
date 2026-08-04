.PHONY: dev test lint typecheck check up down migrate fmt

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
.PHONY: dagster pipeline-dry api dash

dagster:        ## Dagster dev UI (webserver + daemon, local)
	uv run dagster dev -f orchestration/definitions.py

pipeline-dry:   ## Wiring dry-run job: no DB or API keys needed
	uv run dagster job execute -f orchestration/definitions.py -j dry_run_job

api:            ## Serve the FastAPI app on 127.0.0.1:8000
	uv run python -m dynamo.api

dash:           ## Serve the Streamlit dashboard on 127.0.0.1:8501
	uv run streamlit run dashboard/app.py
