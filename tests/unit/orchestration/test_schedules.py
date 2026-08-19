"""Schedule table assertions — doc 02 §4 cadences, all America/New_York."""

from __future__ import annotations

import pytest
from orchestration.schedules import ALL_SCHEDULES

TZ = "America/New_York"

EXPECTED_CRONS = {
    "eod_ingest_schedule": "30 21 * * 1-5",
    # Crypto sleeve (ADR-0013): bars every day (24/7 market), stops every 2h
    # on weekends (the weekday 15-min loop covers market hours).
    "crypto_bars_schedule": "35 21 * * *",
    "crypto_stops_schedule": "0 */2 * * 6,0",
    "insider_schedule": "0 * * * *",
    "news_schedule": "*/30 * * * 1-5",
    "macro_ingest_schedule": "0 8 * * *",
    "macro_digest_schedule": "30 8 * * *",
    "earnings_calendar_schedule": "0 7 * * *",
    "transcripts_schedule": "30 7 * * *",
    "prediction_market_schedule": "0 20 * * *",
    "pm_trading_schedule": "30 20 * * *",
    "pm_resolution_schedule": "45 20 * * *",
    "pm_reconciliation_schedule": "*/15 9-16 * * 1-5",
    "options_trading_schedule": "45 0 * * *",
    "options_reconciliation_schedule": "*/15 9-16 * * 1-5",
    "intraday_entries_schedule": "*/15 9-16 * * 1-5",
    "filings_poll_schedule": "*/10 * * * 1-5",
    "tier2_schedule": "30 22 * * 1-5",
    "social_schedule": "10 22 * * 1-5",
    "scores_schedule": "0 23 * * 1-5",
    "committee_schedule": "30 23 * * 1-5",
    "meta_scoring_schedule": "45 23 * * 1-5",
    "risk_portfolio_schedule": "30 0 * * *",
    "paper_execution_schedule": "35 9 * * 1-5",
    "reconciliation_schedule": "*/15 9-16 * * 1-5",
    "stops_schedule": "*/15 9-16 * * 1-5",
    "nightly_rollup_schedule": "0 2 * * *",
    "daily_report_schedule": "15 2 * * *",
    "safety_checks_schedule": "20 2 * * *",
    "healthcheck_schedule": "30 2 * * *",
    "weekly_learning_schedule": "0 8 * * 0",
    # Monthly quant retrain-cadence check (proposal-gated; registered like
    # every schedule, STOPPED until a human enables it).
    "quant_retrain_check_schedule": "0 9 1 * *",
}

BY_NAME = {s.name: s for s in ALL_SCHEDULES}


def test_every_schedule_has_expected_cron() -> None:
    assert set(BY_NAME) == set(EXPECTED_CRONS)
    for name, cron in EXPECTED_CRONS.items():
        assert BY_NAME[name].cron_schedule == cron, name


@pytest.mark.parametrize("name", sorted(EXPECTED_CRONS))
def test_every_schedule_runs_in_eastern_time(name: str) -> None:
    assert BY_NAME[name].execution_timezone == TZ


def test_weekly_learning_job_includes_learning_metrics() -> None:
    """Plan Phase 6: the weekly Sunday chain now ends in learning_metrics
    (deps order it after calibration_update)."""
    from dagster import AssetSelection
    from orchestration.schedules import weekly_learning_job

    assert weekly_learning_job.selection == AssetSelection.assets(
        "post_trade_reviews",
        "experience_resolution",
        "calibration_update",
        "drift_monitors",
        "learning_metrics",
    )


def test_failure_clustering_job_registered_but_never_scheduled() -> None:
    """The clustering job is sensor-triggered only (plan Phase 6)."""
    from orchestration.schedules import ALL_JOBS

    names = {j.name for j in ALL_JOBS}
    assert "failure_clustering_job" in names
    scheduled_jobs = {s.job.name for s in ALL_SCHEDULES}
    assert "failure_clustering_job" not in scheduled_jobs


def test_crypto_stops_schedule_runs_the_full_stops_job() -> None:
    """ADR-0013: weekend crypto stop coverage reuses the existing stops job —
    the simpler correct option (equity GTC stop replaces queue harmlessly
    off-hours; crypto stops get real weekend maintenance)."""
    assert BY_NAME["crypto_stops_schedule"].job.name == "stops_job"
    assert BY_NAME["stops_schedule"].job.name == "stops_job"


def test_quant_retrain_check_job_registered() -> None:
    """The monthly cadence check is a registered job + schedule; enabling it
    stays a human action (Dagster schedules default to STOPPED)."""
    from orchestration.schedules import ALL_JOBS

    assert "quant_retrain_check_job" in {j.name for j in ALL_JOBS}
