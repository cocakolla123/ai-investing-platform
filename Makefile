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
