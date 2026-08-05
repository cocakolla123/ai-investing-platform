"""Dagster entrypoint: ``dagster dev -f orchestration/definitions.py``.

Everything is built lazily from ``Settings`` (orchestration/resources.py); the
definitions load with no database and no API keys, and each asset degrades
per-provider at runtime (doc 01 §6).
"""

from __future__ import annotations

from dagster import Definitions

from orchestration.assets.ingestion import INGESTION_ASSETS, filings_poll_job
from orchestration.assets.intraday import INTRADAY_JOBS
from orchestration.assets.learning_ops import LEARNING_ASSETS, healthcheck_job
from orchestration.assets.options_trading import OPTIONS_JOBS
from orchestration.assets.pm_trading import PM_ASSETS, PM_JOBS
from orchestration.assets.portfolio_exec import EXECUTION_JOBS, PORTFOLIO_ASSETS
from orchestration.assets.reporting import REPORTING_ASSETS
from orchestration.assets.research import COMMITTEE_JOBS, RESEARCH_ASSETS
from orchestration.assets.social import SOCIAL_ASSETS
from orchestration.dry_run import dry_run_job
from orchestration.resources import DynamoResource
from orchestration.schedules import ALL_JOBS, ALL_SCHEDULES
from orchestration.sensors import ALL_SENSORS

defs = Definitions(
    assets=[
        *INGESTION_ASSETS,
        *RESEARCH_ASSETS,
        *SOCIAL_ASSETS,
        *PORTFOLIO_ASSETS,
        *PM_ASSETS,
        *LEARNING_ASSETS,
        *REPORTING_ASSETS,
    ],
    jobs=[
        filings_poll_job,
        healthcheck_job,
        dry_run_job,
        *COMMITTEE_JOBS,
        *EXECUTION_JOBS,
        *PM_JOBS,
        *OPTIONS_JOBS,
        *INTRADAY_JOBS,
        *ALL_JOBS,
    ],
    schedules=ALL_SCHEDULES,
    sensors=ALL_SENSORS,
    resources={"dynamo": DynamoResource()},
)
