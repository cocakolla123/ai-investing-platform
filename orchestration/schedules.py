"""Schedules — the doc 02 §4 cadence table, all times America/New_York.

| Job | When (ET) |
|---|---|
| EOD bars + corporate actions | 21:30 market days |
| Crypto bars (24/7 market) | 21:35 every day |
| Crypto weekend stops (full stops job) | every 2h Sat/Sun |
| EDGAR filings poller | every 10 min, market days |
| Form 4 / insider | hourly |
| News | every 30 min, market days |
| Macro (FRED) | 08:00 daily |
| Macro digest (Sonnet) | 08:30 daily |
| Earnings calendar | 07:00 daily |
| Earnings-call transcripts | 07:30 daily |
| Prediction-market snapshot (+ OddsPipe spreads) | 20:00 daily |
| PM trading: Kalshi (gated, demo-first) + Polymarket shadow (SIMULATED) | 20:30 daily |
| PM resolution: Kalshi + calibration, Polymarket sim settlement | 20:45 daily |
| Kalshi PM reconciliation | every 15 min 09:30-16:00 market days |
| Options trading (defined-risk, gated) | 00:45 daily |
| Options reconciliation | every 15 min 09:30-16:00 market days |
| Intraday same-day entries | every 15 min 09:30-16:00 market days |
| Tier-2 detectors + triage (after data quality) | 22:30 market days |
| Features -> scores -> signals -> regime -> selection | 23:00 market days |
| Committee batch submission | 23:30 market days |
| Meta-model advisory scoring | 23:45 market days |
| Risk verdict + target portfolio | 00:30 daily |
| Paper execution | 09:35 market days |
| Reconciliation + stops | every 15 min 09:30-16:00 market days |
| Parquet export + metrics + equity curve | 02:00 daily |
| Daily activity report | 02:15 daily |
| Chain verify + experience completeness | 02:20 daily |
| Healthcheck dead-man ping | 02:30 daily |
| Post-trade reviews + experiences + calibration + drift + learning metrics | Sunday 08:00 |
| Monte Carlo + 7-scenario stress suite (doc 06 §6) | Sunday 09:00 |
| Quant retrain-cadence check (proposal-gated, never trains) | 09:00, 1st of month |
| Failure clustering | sensor-triggered (~150 new resolved failures), never scheduled |
"""

from dagster import (
    AssetSelection,
    RunRequest,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    define_asset_job,
    schedule,
)

from orchestration.assets.ingestion import filings_poll_job
from orchestration.assets.intraday import intraday_entries_job
from orchestration.assets.learning_ops import failure_clustering_job, healthcheck_job
from orchestration.assets.options_trading import (
    options_reconciliation_job,
    options_trading_job,
)
from orchestration.assets.pm_trading import pm_reconciliation_job, pm_trading_job
from orchestration.assets.portfolio_exec import (
    paper_execution_job,
    reconciliation_job,
    stops_job,
)
from orchestration.assets.research import committee_job
from orchestration.partitions import TZ

MARKET_DAYS = "1-5"  # Mon-Fri; exchange holidays no-op harmlessly (idempotent upserts)

# --- asset jobs -----------------------------------------------------------

eod_ingest_job = define_asset_job(
    "eod_ingest_job",
    selection=AssetSelection.assets("eod_bars", "corporate_actions"),
)
crypto_bars_job = define_asset_job(
    "crypto_bars_job",
    selection=AssetSelection.assets("crypto_bars"),
)
insider_job = define_asset_job(
    "insider_job",
    selection=AssetSelection.assets("insider_tx"),
)
news_job = define_asset_job("news_job", selection=AssetSelection.assets("news"))
macro_ingest_job = define_asset_job(
    "macro_ingest_job",
    selection=AssetSelection.assets("macro_series"),
)
earnings_calendar_job = define_asset_job(
    "earnings_calendar_job",
    selection=AssetSelection.assets("earnings_calendar"),
)
transcripts_job = define_asset_job(
    "transcripts_job",
    selection=AssetSelection.assets("transcripts"),
)
prediction_market_job = define_asset_job(
    "prediction_market_job",
    # OddsPipe spreads run in the same 20:00 chain, after the primary snapshot.
    selection=AssetSelection.assets("prediction_market_snapshot", "oddspipe_spreads"),
)
pm_resolution_job = define_asset_job(
    "pm_resolution_job",
    # Kalshi settlement + the SIMULATED Polymarket sleeve settlement (ADR-0011)
    # run in the same 20:45 slot; the sim asset never touches a real venue.
    selection=AssetSelection.assets("pm_resolution", "pm_sim_resolution"),
)
macro_digest_job = define_asset_job(
    "macro_digest_job", selection=AssetSelection.assets("macro_digest")
)
tier2_job = define_asset_job(
    "tier2_job", selection=AssetSelection.assets("data_quality", "tier2_scan")
)
scores_job = define_asset_job(
    "scores_job",
    selection=AssetSelection.assets(
        "quant_features_scores",
        "quality_scores",
        "strategy_signals",
        "regime_state",
        "deep_dive_selection",
        "watchlist",
    ),
)
risk_portfolio_job = define_asset_job(
    "risk_portfolio_job",
    selection=AssetSelection.assets("risk_verdict", "target_portfolio"),
)
meta_scoring_job = define_asset_job(
    "meta_scoring_job",
    # 23:45 ET — after the committee batch (23:30), before risk/portfolio
    # (00:30). target_portfolio also declares meta_scoring as a lineage dep;
    # the schedule-time ordering here is the operational guarantee.
    selection=AssetSelection.assets("meta_scoring"),
)
nightly_rollup_job = define_asset_job(
    "nightly_rollup_job",
    selection=AssetSelection.assets("metrics_daily", "equity_curve", "parquet_export"),
)
daily_report_job = define_asset_job(
    "daily_report_job",
    selection=AssetSelection.assets("daily_report"),
)
weekly_learning_job = define_asset_job(
    "weekly_learning_job",
    # Deps order the chain: post_trade_reviews -> experience_resolution ->
    # calibration_update -> learning_metrics (plan Phases 1 + 6);
    # drift_monitors is independent.
    selection=AssetSelection.assets(
        "post_trade_reviews",
        "experience_resolution",
        "calibration_update",
        "drift_monitors",
        "learning_metrics",
    ),
)
risk_scenarios_job = define_asset_job(
    "risk_scenarios_job",
    # Weekly Monte Carlo + 7-scenario deterministic stress suite (doc 06 §6).
    # Registered but, like every Dagster schedule here, STOPPED until a human
    # turns it on in the daemon/UI.
    selection=AssetSelection.assets("risk_scenarios"),
)
quant_retrain_check_job = define_asset_job(
    "quant_retrain_check_job",
    # Monthly retrain-cadence check (proposal-gated per the iron law: emits
    # an OPEN RETRAIN proposal, NEVER trains). Registered like every other
    # schedule but NOT enabled — Dagster schedules start STOPPED until a
    # human turns them on in the daemon/UI.
    selection=AssetSelection.assets("quant_retrain_check"),
)
safety_checks_job = define_asset_job(
    "safety_checks_job",
    # Nightly learning-loop safety: hash-chain verification (02:20) and the
    # review-completeness check, between the daily report and the dead-man ping.
    selection=AssetSelection.assets("chain_verify_check", "experience_completeness_check"),
)

# --- partitioned schedules (partition key = the ET calendar day) ----------


def _partitioned_schedule(name: str, cron: str, job):
    @schedule(name=name, cron_schedule=cron, job=job, execution_timezone=TZ)
    def _sched(context: ScheduleEvaluationContext) -> RunRequest:
        at = context.scheduled_execution_time
        return RunRequest(partition_key=at.date().isoformat(), run_key=f"{name}-{at.isoformat()}")

    return _sched


eod_ingest_schedule = _partitioned_schedule(
    "eod_ingest_schedule", f"30 21 * * {MARKET_DAYS}", eod_ingest_job
)
# Crypto trades 24/7 (ADR-0013): bars ingest EVERY day, 5 minutes after the
# equities EOD ingest slot.
crypto_bars_schedule = _partitioned_schedule(
    "crypto_bars_schedule", "35 21 * * *", crypto_bars_job
)
insider_schedule = _partitioned_schedule("insider_schedule", "0 * * * *", insider_job)
news_schedule = _partitioned_schedule("news_schedule", f"*/30 * * * {MARKET_DAYS}", news_job)
macro_ingest_schedule = _partitioned_schedule(
    "macro_ingest_schedule", "0 8 * * *", macro_ingest_job
)
earnings_calendar_schedule = _partitioned_schedule(
    "earnings_calendar_schedule", "0 7 * * *", earnings_calendar_job
)
# Transcripts run 30 minutes after the earnings calendar refresh.
transcripts_schedule = _partitioned_schedule(
    "transcripts_schedule", "30 7 * * *", transcripts_job
)
prediction_market_schedule = _partitioned_schedule(
    "prediction_market_schedule", "0 20 * * *", prediction_market_job
)

# --- plain cron schedules -------------------------------------------------

filings_poll_schedule = ScheduleDefinition(
    name="filings_poll_schedule",
    cron_schedule=f"*/10 * * * {MARKET_DAYS}",
    job=filings_poll_job,
    execution_timezone=TZ,
)
# Kalshi PM sleeve (ADR-0009): trade after the 20:00 snapshot, then settle.
pm_trading_schedule = ScheduleDefinition(
    name="pm_trading_schedule",
    cron_schedule="30 20 * * *",
    job=pm_trading_job,
    execution_timezone=TZ,
)
pm_resolution_schedule = ScheduleDefinition(
    name="pm_resolution_schedule",
    cron_schedule="45 20 * * *",
    job=pm_resolution_job,
    execution_timezone=TZ,
)
pm_reconciliation_schedule = ScheduleDefinition(
    name="pm_reconciliation_schedule",
    cron_schedule=f"*/15 9-16 * * {MARKET_DAYS}",
    job=pm_reconciliation_job,
    execution_timezone=TZ,
)
# Options sleeve (ADR-0010): trade nightly after target_portfolio (00:30),
# reconcile + evaluate same-day entries on the 15-minute intraday loop.
options_trading_schedule = ScheduleDefinition(
    name="options_trading_schedule",
    cron_schedule="45 0 * * *",
    job=options_trading_job,
    execution_timezone=TZ,
)
options_reconciliation_schedule = ScheduleDefinition(
    name="options_reconciliation_schedule",
    cron_schedule=f"*/15 9-16 * * {MARKET_DAYS}",
    job=options_reconciliation_job,
    execution_timezone=TZ,
)
intraday_entries_schedule = ScheduleDefinition(
    name="intraday_entries_schedule",
    cron_schedule=f"*/15 9-16 * * {MARKET_DAYS}",
    job=intraday_entries_job,
    execution_timezone=TZ,
)
macro_digest_schedule = ScheduleDefinition(
    name="macro_digest_schedule",
    cron_schedule="30 8 * * *",
    job=macro_digest_job,
    execution_timezone=TZ,
)
social_job = define_asset_job("social_job", selection=AssetSelection.assets("social_scan"))
social_schedule = ScheduleDefinition(
    name="social_schedule",
    cron_schedule=f"10 22 * * {MARKET_DAYS}",
    job=social_job,
    execution_timezone=TZ,
)
tier2_schedule = ScheduleDefinition(
    name="tier2_schedule",
    cron_schedule=f"30 22 * * {MARKET_DAYS}",
    job=tier2_job,
    execution_timezone=TZ,
)
scores_schedule = ScheduleDefinition(
    name="scores_schedule",
    cron_schedule=f"0 23 * * {MARKET_DAYS}",
    job=scores_job,
    execution_timezone=TZ,
)
committee_schedule = ScheduleDefinition(
    name="committee_schedule",
    cron_schedule=f"30 23 * * {MARKET_DAYS}",
    job=committee_job,
    execution_timezone=TZ,
)
meta_scoring_schedule = ScheduleDefinition(
    name="meta_scoring_schedule",
    cron_schedule=f"45 23 * * {MARKET_DAYS}",
    job=meta_scoring_job,
    execution_timezone=TZ,
)
risk_portfolio_schedule = ScheduleDefinition(
    name="risk_portfolio_schedule",
    cron_schedule="30 0 * * *",
    job=risk_portfolio_job,
    execution_timezone=TZ,
)
paper_execution_schedule = ScheduleDefinition(
    name="paper_execution_schedule",
    cron_schedule=f"35 9 * * {MARKET_DAYS}",
    job=paper_execution_job,
    execution_timezone=TZ,
)
reconciliation_schedule = ScheduleDefinition(
    name="reconciliation_schedule",
    cron_schedule=f"*/15 9-16 * * {MARKET_DAYS}",
    job=reconciliation_job,
    execution_timezone=TZ,
)
stops_schedule = ScheduleDefinition(
    name="stops_schedule",
    cron_schedule=f"*/15 9-16 * * {MARKET_DAYS}",
    job=stops_job,
    execution_timezone=TZ,
)
# Crypto weekend stop coverage (ADR-0013): crypto trades through the weekend,
# so the FULL stops job runs every 2h Sat/Sun. Running it unrestricted is the
# simpler correct option: crypto stop specs get maintained, while equity stop
# replaces are harmless (GTC stops queue at the broker outside market hours)
# and synthetic-stop evaluation is quote-driven and quotes do not move for
# closed markets. The 15-min weekday loop above already covers crypto during
# market hours.
crypto_stops_schedule = ScheduleDefinition(
    name="crypto_stops_schedule",
    cron_schedule="0 */2 * * 6,0",
    job=stops_job,
    execution_timezone=TZ,
)
nightly_rollup_schedule = ScheduleDefinition(
    name="nightly_rollup_schedule",
    cron_schedule="0 2 * * *",
    job=nightly_rollup_job,
    execution_timezone=TZ,
)
# Daily activity report: after the 02:00 rollup, before the 02:30 dead-man ping.
daily_report_schedule = ScheduleDefinition(
    name="daily_report_schedule",
    cron_schedule="15 2 * * *",
    job=daily_report_job,
    execution_timezone=TZ,
)
safety_checks_schedule = ScheduleDefinition(
    name="safety_checks_schedule",
    cron_schedule="20 2 * * *",
    job=safety_checks_job,
    execution_timezone=TZ,
)
healthcheck_schedule = ScheduleDefinition(
    name="healthcheck_schedule",
    cron_schedule="30 2 * * *",
    job=healthcheck_job,
    execution_timezone=TZ,
)
weekly_learning_schedule = ScheduleDefinition(
    name="weekly_learning_schedule",
    cron_schedule="0 8 * * 0",
    job=weekly_learning_job,
    execution_timezone=TZ,
)
# Weekly (Sunday 09:00 ET) risk scenario suite — after the 08:00 learning
# job, so a week's realized outcomes are already resolved.
risk_scenarios_schedule = ScheduleDefinition(
    name="risk_scenarios_schedule",
    cron_schedule="0 9 * * 0",
    job=risk_scenarios_job,
    execution_timezone=TZ,
)
# Monthly (1st, 09:00 ET) quant retrain-cadence check — proposal-gated.
quant_retrain_check_schedule = ScheduleDefinition(
    name="quant_retrain_check_schedule",
    cron_schedule="0 9 1 * *",
    job=quant_retrain_check_job,
    execution_timezone=TZ,
)

ALL_JOBS = [
    eod_ingest_job,
    crypto_bars_job,
    insider_job,
    news_job,
    macro_ingest_job,
    earnings_calendar_job,
    transcripts_job,
    prediction_market_job,
    pm_resolution_job,
    macro_digest_job,
    tier2_job,
    social_job,
    scores_job,
    meta_scoring_job,
    risk_portfolio_job,
    nightly_rollup_job,
    daily_report_job,
    weekly_learning_job,
    risk_scenarios_job,
    safety_checks_job,
    quant_retrain_check_job,
    # Sensor target only (failure_clustering_sensor) — deliberately unscheduled.
    failure_clustering_job,
]

ALL_SCHEDULES = [
    eod_ingest_schedule,
    crypto_bars_schedule,
    insider_schedule,
    news_schedule,
    macro_ingest_schedule,
    earnings_calendar_schedule,
    transcripts_schedule,
    prediction_market_schedule,
    pm_trading_schedule,
    pm_resolution_schedule,
    pm_reconciliation_schedule,
    options_trading_schedule,
    options_reconciliation_schedule,
    intraday_entries_schedule,
    filings_poll_schedule,
    macro_digest_schedule,
    tier2_schedule,
    social_schedule,
    scores_schedule,
    committee_schedule,
    meta_scoring_schedule,
    risk_portfolio_schedule,
    paper_execution_schedule,
    reconciliation_schedule,
    stops_schedule,
    crypto_stops_schedule,
    nightly_rollup_schedule,
    daily_report_schedule,
    safety_checks_schedule,
    healthcheck_schedule,
    weekly_learning_schedule,
    risk_scenarios_schedule,
    quant_retrain_check_schedule,
]
