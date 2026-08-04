# Project image for dagster-webserver, dagster-daemon, api, and dashboard.
# Design: docs/design/12-operations.md §1 (one image, 12-factor, doc 01 §3).
FROM python:3.12-slim

# uv pinned by minor version (doc 12 §9: deps move via PRs, not floating tags).
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /usr/local/bin/

# libgomp1: lightgbm runtime (OpenMP); curl: container healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependency layer first (cache-friendly): lockfile only, no project.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project source + entrypoints.
COPY src ./src
COPY orchestration ./orchestration
COPY dashboard ./dashboard
COPY infra/migrations ./infra/migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p "$DAGSTER_HOME"
COPY infra/docker/dagster.yaml $DAGSTER_HOME/dagster.yaml

# Default command is the API; compose overrides per service (doc 12 §1).
CMD ["python", "-m", "dynamo.api"]
