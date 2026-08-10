"""Schedule table assertions — doc 02 §4 cadences, all America/New_York."""

from __future__ import annotations

import pytest
from orchestration.schedules import ALL_SCHEDULES

TZ = "America/New_York"

EXPECTED_CRONS = {
    "eod_ingest_schedule": "30 21 * * 1-5",
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
    "risk_portfolio_schedule": "30 0 * * *",
    "paper_execution_schedule": "35 9 * * 1-5",
    "reconciliation_schedule": "*/15 9-16 * * 1-5",
    "stops_schedule": "*/15 9-16 * * 1-5",
    "nightly_rollup_schedule": "0 2 * * *",
    "daily_report_schedule": "15 2 * * *",
    "healthcheck_schedule": "30 2 * * *",
    "weekly_learning_schedule": "0 8 * * 0",
}

BY_NAME = {s.name: s for s in ALL_SCHEDULES}


def test_every_schedule_has_expected_cron() -> None:
    assert set(BY_NAME) == set(EXPECTED_CRONS)
    for name, cron in EXPECTED_CRONS.items():
        assert BY_NAME[name].cron_schedule == cron, name


@pytest.mark.parametrize("name", sorted(EXPECTED_CRONS))
def test_every_schedule_runs_in_eastern_time(name: str) -> None:
    assert BY_NAME[name].execution_timezone == TZ
