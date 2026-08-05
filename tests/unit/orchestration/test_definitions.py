"""The Definitions object loads and every job/schedule/sensor resolves."""

from __future__ import annotations

from dagster import Definitions
from orchestration.definitions import defs


def test_definitions_validate_loadable() -> None:
    Definitions.validate_loadable(defs)


def test_expected_assets_present() -> None:
    names = {
        key.to_user_string()
        for assets_def in defs.assets or []
        for key in getattr(assets_def, "keys", [])
    }
    expected = {
        # ingestion (doc 02 §4)
        "eod_bars",
        "corporate_actions",
        "insider_tx",
        "news",
        "macro_series",
        "earnings_calendar",
        "prediction_market_snapshot",
        # research (docs 02 §6, 03, 04, 05)
        "data_quality",
        "tier2_scan",
        "quant_features_scores",
        "quality_scores",
        "strategy_signals",
        "regime_state",
        "macro_digest",
        "deep_dive_selection",
        "watchlist",
        # portfolio / risk (docs 06, 07)
        "risk_verdict",
        "target_portfolio",
        # prediction markets — Kalshi sleeve (ADR-0009)
        "pm_resolution",
        # learning / ops (docs 10, 12)
        "post_trade_reviews",
        "calibration_update",
        "drift_monitors",
        "metrics_daily",
        "equity_curve",
        "parquet_export",
    }
    assert expected <= names


def test_op_jobs_resolve() -> None:
    for name in (
        "filings_poll_job",
        "committee_job",
        "committee_advance_job",
        "committee_sync_job",
        "paper_execution_job",
        "reconciliation_job",
        "stops_job",
        "pm_trading_job",
        "pm_reconciliation_job",
        "healthcheck_job",
    ):
        assert defs.get_job_def(name).name == name


def test_sensors_registered() -> None:
    names = {s.name for s in defs.sensors or []}
    assert {"batch_completion_sensor", "quarantine_alert_sensor", "budget_sensor"} <= names
