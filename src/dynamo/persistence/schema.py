"""Database schema — SQLAlchemy Core tables, single source of truth.

DDL design: docs/design/02 (data), 03 (research), 06 (risk), 09 (execution),
10 (learning). Timescale hypertables + pgvector are applied in the Alembic
migration (infra/migrations/), not here.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

metadata = MetaData()


def _ts(name: str = "ts") -> Column:
    return Column(name, DateTime(timezone=True), nullable=False)


# --- Instruments & universe (doc 02 §2) ---

instruments = Table(
    "instruments",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("name", Text, nullable=False, default=""),
    Column("exchange", Text),
    Column("sector", Text),
    Column("industry", Text),
    Column("cik", Text, index=True),
    Column("figi", Text),
    Column("is_etf", Boolean, nullable=False, default=False),
    Column("status", Text, nullable=False, default="active"),
    Column("first_seen", Date),
    Column("delisted_at", Date),
)

universe_membership = Table(
    "universe_membership",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("valid_from", Date, primary_key=True),
    Column("tier", SmallInteger, nullable=False),
    Column("valid_to", Date),
    Column("reason", Text, nullable=False),
    Column("approved_by", Text),
)

# --- Time series (hypertables; doc 02 §3) ---

bars_1d = Table(
    "bars_1d",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("source", Text, primary_key=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("adj_close", Float),
    Column("fetched_at", DateTime(timezone=True)),
)

bars_1m = Table(
    "bars_1m",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("source", Text, primary_key=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", BigInteger, nullable=False),
    Column("adj_close", Float),
    Column("fetched_at", DateTime(timezone=True)),
)

corporate_actions = Table(
    "corporate_actions",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("ex_date", Date, primary_key=True),
    Column("action_type", Text, primary_key=True),
    Column("source", Text, primary_key=True),
    Column("ratio", Float),
    Column("amount", Float),
    Column("fetched_at", DateTime(timezone=True)),
)

# --- Filings & fundamentals (doc 02 §3; point-in-time by filed_at) ---

filings = Table(
    "filings",
    metadata,
    Column("accession_no", Text, primary_key=True),
    Column("cik", Text, nullable=False, index=True),
    Column("symbol", Text, index=True),
    Column("form_type", Text, nullable=False, index=True),
    Column("filed_at", DateTime(timezone=True), nullable=False, index=True),
    Column("period", Date),
    Column("url", Text, nullable=False),
    Column("items", ARRAY(Text), nullable=False, default=[]),
    Column("fetched_at", DateTime(timezone=True)),
)

xbrl_facts = Table(
    "xbrl_facts",
    metadata,
    Column("cik", Text, primary_key=True),
    Column("tag", Text, primary_key=True),
    Column("period_end", Date, primary_key=True),
    Column("filed_at", DateTime(timezone=True), primary_key=True),
    Column("value", Float, nullable=False),
    Column("unit", Text, nullable=False),
)

insider_tx = Table(
    "insider_tx",
    metadata,
    Column("accession_no", Text, primary_key=True),
    Column("symbol", Text, primary_key=True),
    Column("insider_name", Text, primary_key=True),
    Column("tx_type", Text, primary_key=True),
    Column("role", Text),
    Column("shares", Float, nullable=False),
    Column("price", Float),
    Column("filed_at", DateTime(timezone=True), nullable=False, index=True),
)

news = Table(
    "news",
    metadata,
    Column("id", Text, primary_key=True),
    Column("symbols", ARRAY(Text), nullable=False, default=[]),
    Column("headline", Text, nullable=False),
    Column("summary", Text),
    Column("source", Text, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False, index=True),
    Column("url", Text),
    Column("fetched_at", DateTime(timezone=True)),
)

macro_series = Table(
    "macro_series",
    metadata,
    Column("series_id", Text, primary_key=True),
    Column("ts", Date, primary_key=True),
    Column("vintage_ts", DateTime(timezone=True), primary_key=True),
    Column("value", Float, nullable=False),
)

earnings_events = Table(
    "earnings_events",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("event_date", Date, primary_key=True),
    Column("period", Text),
    Column("eps_est", Float),
    Column("eps_act", Float),
    Column("revenue_est", Float),
    Column("revenue_act", Float),
    Column("call_transcript_ref", Text),
    Column("fetched_at", DateTime(timezone=True)),
)

# --- Prediction markets (doc 02 §7; research-only) ---

prediction_markets = Table(
    "prediction_markets",
    metadata,
    Column("venue", Text, primary_key=True),
    Column("market_id", Text, primary_key=True),
    Column("question", Text, nullable=False),
    Column("category", Text),
    Column("close_ts", DateTime(timezone=True)),
    Column("resolved_outcome", Boolean),
)

prediction_market_prices = Table(
    "prediction_market_prices",
    metadata,
    Column("venue", Text, primary_key=True),
    Column("market_id", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("implied_prob", Float, nullable=False),
    Column("volume", Float),
    Column("liquidity", Float),
    Column("trusted", Boolean, nullable=False, default=False),
)

# --- Prediction-market trading — Kalshi sleeve (ADR-0009; migration 0002) ---
# Mirrors orders/order_events/positions/trades; prices in integer-ish cents,
# contracts are integers. Polymarket has NO rows here — research only.

pm_orders = Table(
    "pm_orders",
    metadata,
    Column("client_order_id", Text, primary_key=True),
    Column("broker_order_id", Text),
    Column("decision_context_id", Text, nullable=False),
    Column("venue", Text, nullable=False, default="kalshi"),
    Column("market_id", Text, nullable=False, index=True),
    Column("side", Text, nullable=False),  # yes | no
    Column("contracts", Integer, nullable=False),
    Column("limit_price_cents", Integer, nullable=False),
    Column("state", Text, nullable=False, default="proposed"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

pm_order_events = Table(
    "pm_order_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, nullable=False, index=True),
    Column("from_state", Text),
    Column("to_state", Text, nullable=False),
    Column("reason", Text, nullable=False, default=""),
    Column("payload", JSONB),
    _ts(),
)

pm_positions = Table(
    "pm_positions",
    metadata,
    Column("venue", Text, primary_key=True),
    Column("market_id", Text, primary_key=True),
    Column("side", Text, nullable=False),
    Column("contracts", Integer, nullable=False),
    Column("avg_price_cents", Float, nullable=False),
    Column("question", Text, nullable=False, default=""),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

pm_trades = Table(
    "pm_trades",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, index=True),  # null for settlements
    Column("venue", Text, nullable=False, default="kalshi"),
    Column("market_id", Text, nullable=False, index=True),
    Column("side", Text, nullable=False),
    Column("contracts", Integer, nullable=False),
    Column("price_cents", Float, nullable=False),
    Column("fees_cents", Float, nullable=False, default=0.0),
    Column("kind", Text, nullable=False, default="fill"),  # fill | settlement
    Column("realized_pnl_cents", Float),
    Column("filled_at", DateTime(timezone=True), nullable=False),
)

# --- Options trading — defined-risk sleeve (ADR-0010; migration 0003) ---
# Mirrors orders/order_events/positions/trades; prices in dollars per share,
# contracts are integers. Only the four defined-risk structures ever appear in
# ``structure`` (enforced by the domain validators, not the database).

options_orders = Table(
    "options_orders",
    metadata,
    Column("client_order_id", Text, primary_key=True),
    Column("broker_order_id", Text),
    Column("decision_context_id", Text, nullable=False),
    Column("occ_symbol", Text, nullable=False, index=True),
    Column("underlying", Text, nullable=False, index=True),
    # long_call | long_put | covered_call | cash_secured_put
    Column("structure", Text, nullable=False),
    Column("side", Text, nullable=False),  # buy | sell
    Column("right", Text, nullable=False),  # call | put
    Column("expiry", Date, nullable=False),
    Column("strike", Float, nullable=False),
    Column("contracts", Integer, nullable=False),
    Column("limit_price", Float, nullable=False),
    Column("state", Text, nullable=False, default="proposed"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

options_order_events = Table(
    "options_order_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, nullable=False, index=True),
    Column("from_state", Text),
    Column("to_state", Text, nullable=False),
    Column("reason", Text, nullable=False, default=""),
    Column("payload", JSONB),
    _ts(),
)

options_positions = Table(
    "options_positions",
    metadata,
    Column("occ_symbol", Text, primary_key=True),
    Column("underlying", Text, nullable=False, index=True),
    Column("structure", Text, nullable=False),
    Column("side", Text, nullable=False),
    Column("right", Text, nullable=False),
    Column("expiry", Date, nullable=False),
    Column("strike", Float, nullable=False),
    Column("contracts", Integer, nullable=False),
    Column("avg_price", Float, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

options_trades = Table(
    "options_trades",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, index=True),  # null for expiry close-outs
    Column("occ_symbol", Text, nullable=False, index=True),
    Column("underlying", Text, nullable=False, index=True),
    Column("side", Text, nullable=False),
    Column("right", Text, nullable=False),
    Column("contracts", Integer, nullable=False),
    Column("price", Float, nullable=False),
    Column("fees", Float, nullable=False, default=0.0),
    Column("kind", Text, nullable=False, default="fill"),  # fill | expiry
    Column("filled_at", DateTime(timezone=True), nullable=False),
)

# Minimal chain persistence: only rows near the selection targets (spot ±20%,
# target expiries) to keep volume small.
options_chain_snapshots = Table(
    "options_chain_snapshots",
    metadata,
    Column("underlying", Text, primary_key=True),
    Column("as_of", DateTime(timezone=True), primary_key=True),
    Column("expiry", Date, primary_key=True),
    Column("strike", Float, primary_key=True),
    Column("cp", Text, primary_key=True),  # call | put
    Column("bid", Float),
    Column("ask", Float),
    Column("oi", Integer),
    Column("iv", Float),
    Column("delta", Float),
)

options_iv_history = Table(
    "options_iv_history",
    metadata,
    Column("underlying", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("atm_iv", Float, nullable=False),
)

# --- Surveillance & quality (doc 02 §2, §6) ---

surveillance_flags = Table(
    "surveillance_flags",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("detector", Text, nullable=False),
    Column("severity", Float, nullable=False),
    Column("details", JSONB, nullable=False, default={}),
    _ts(),
    Column("triage_verdict", Text),
    Column("triaged_at", DateTime(timezone=True)),
)

quarantine = Table(
    "quarantine",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("scope", Text, nullable=False),  # symbol | llm_call | record
    Column("key", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column("payload", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True)),
)

# --- Research (doc 03) ---

dossiers = Table(
    "dossiers",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("payload", JSONB, nullable=False),  # DossierAnalysis
    Column("prompt_version", Text, nullable=False, default=""),
    Column("model_id", Text, nullable=False, default=""),
)

agent_opinions = Table(
    "agent_opinions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("as_of", Date, nullable=False, index=True),
    Column("role", Text, nullable=False),  # dossier | bull | bear | macro | triage | ...
    Column("payload", JSONB, nullable=False),
    Column("prompt_version", Text, nullable=False, default=""),
    Column("model_id", Text, nullable=False, default=""),
)

committee_verdicts = Table(
    "committee_verdicts",
    metadata,
    Column("id", Text, primary_key=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("as_of", Date, nullable=False, index=True),
    Column("action", Text, nullable=False),
    Column("conviction", Float, nullable=False),
    Column("disagreement_score", Float, nullable=False, default=0.0),
    Column("payload", JSONB, nullable=False),  # full CommitteeVerdict
)

macro_digests = Table(
    "macro_digests",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("payload", JSONB, nullable=False),
)

prediction_market_views = Table(
    "prediction_market_views",
    metadata,
    Column("venue", Text, primary_key=True),
    Column("market_id", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("payload", JSONB, nullable=False),
)

batch_jobs = Table(
    "batch_jobs",
    metadata,
    Column("batch_id", Text, primary_key=True),
    Column("stage", Text, nullable=False),  # screen | dossier | advocates | arbiter | review
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("status", Text, nullable=False, default="submitted"),
    Column("request_count", Integer, nullable=False, default=0),
    Column("resolved_at", DateTime(timezone=True)),
)

cost_ledger = Table(
    "cost_ledger",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    _ts(),
    Column("model_id", Text, nullable=False),
    Column("call_kind", Text, nullable=False),
    Column("input_tokens", BigInteger, nullable=False, default=0),
    Column("output_tokens", BigInteger, nullable=False, default=0),
    Column("cache_read_tokens", BigInteger, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("batch", Boolean, nullable=False, default=False),
)

# --- Quant & strategies (docs 04, 05) ---

quant_scores = Table(
    "quant_scores",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("model_version", Text, primary_key=True),
    Column("expected_excess_return", Float),
    Column("downside_prob", Float),
    Column("vol_forecast", Float),
    Column("rank", Integer),
    Column("confidence", Float, nullable=False, default=0.0),
    Column("feature_coverage", Float, nullable=False, default=1.0),
)

quality_scores = Table(
    "quality_scores",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("composite", Float),
    Column("subscores", JSONB, nullable=False, default={}),
    Column("config_hash", Text, nullable=False, default=""),
)

signals = Table(
    "signals",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("strategy_id", Text, nullable=False),
    Column("symbol", Text, nullable=False),
    Column("as_of", Date, nullable=False),
    Column("direction", Text, nullable=False, default="long"),
    Column("entry_ref_price", Float),
    Column("stop_hint", Float),
    Column("horizon_days", Integer, nullable=False, default=20),
    Column("strength", Float, nullable=False, default=0.5),
    Column("rule_snapshot", JSONB, nullable=False, default={}),
    Column("status", Text, nullable=False, default="active"),
)

watchlist = Table(
    "watchlist",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("sleeve", Text, primary_key=True),
    Column("symbol", Text, primary_key=True),
    Column("rank", Integer, nullable=False),
    Column("blend_score", Float, nullable=False),
    Column("committee_component", Float),
    Column("quant_component", Float),
    Column("quality_component", Float),
    Column("badges", ARRAY(Text), nullable=False, default=[]),
    Column("blend_config_hash", Text, nullable=False, default=""),
    Column("verdict_ref", Text),
    Column("signal_ref", BigInteger),
)

# --- Registry & promotion (docs 04 §7, 08 §7) ---

model_versions = Table(
    "model_versions",
    metadata,
    Column("id", Text, primary_key=True),
    Column("model_type", Text, nullable=False),
    Column("algo", Text, nullable=False),
    Column("features_hash", Text, nullable=False, default=""),
    Column("train_start", Date),
    Column("train_end", Date),
    Column("hyperparams", JSONB, nullable=False, default={}),
    Column("metrics", JSONB, nullable=False, default={}),
    Column("artifact_path", Text, nullable=False, default=""),
    Column("status", Text, nullable=False, default="candidate"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("promoted_at", DateTime(timezone=True)),
    Column("promoted_by", Text),
)

promotion_records = Table(
    "promotion_records",
    metadata,
    Column("id", Text, primary_key=True),
    Column("target_id", Text, nullable=False, index=True),
    Column("from_status", Text, nullable=False),
    Column("to_status", Text, nullable=False),
    Column("actor", Text, nullable=False),
    Column("evidence_bundle_ref", Text),
    Column("trial_count", Integer, nullable=False, default=0),
    Column("git_commit", Text, nullable=False, default=""),
    Column("config_hash", Text, nullable=False, default=""),
    Column("at", DateTime(timezone=True), nullable=False),
)

trial_counter = Table(
    "trial_counter",
    metadata,
    Column("family", Text, primary_key=True),
    Column("trials", Integer, nullable=False, default=0),
)

# --- Risk (doc 06) ---

regime_states = Table(
    "regime_states",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("regime", Text, nullable=False),
    Column("composite_score", Float, nullable=False),
    Column("inputs", JSONB, nullable=False, default={}),
    Column("sessions_in_state", Integer, nullable=False, default=0),
)

risk_verdicts = Table(
    "risk_verdicts",
    metadata,
    Column("id", Text, primary_key=True),
    Column("as_of", DateTime(timezone=True), nullable=False, index=True),
    Column("payload", JSONB, nullable=False),
)

ladder_states = Table(
    "ladder_states",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("rung", Text, nullable=False),
    Column("drawdown", Float, nullable=False),
    Column("hwm", Float, nullable=False),
)

# --- Portfolio & execution (docs 07, 09) ---

target_portfolios = Table(
    "target_portfolios",
    metadata,
    Column("id", Text, primary_key=True),
    Column("as_of", DateTime(timezone=True), nullable=False, index=True),
    Column("payload", JSONB, nullable=False),
    Column("approved_at", DateTime(timezone=True)),
    Column("approved_by", Text),
)

positions = Table(
    "positions",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("qty", Float, nullable=False),
    Column("avg_entry_price", Float, nullable=False),
    Column("lots", JSONB, nullable=False, default=[]),
    Column("current_stop", Float),
    Column("stop_kind", Text),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

orders = Table(
    "orders",
    metadata,
    Column("client_order_id", Text, primary_key=True),
    Column("broker_order_id", Text),
    Column("decision_context_id", Text, nullable=False),
    Column("symbol", Text, nullable=False, index=True),
    Column("side", Text, nullable=False),
    Column("qty", Float, nullable=False),
    Column("order_type", Text, nullable=False),
    Column("limit_price", Float),
    Column("stop_price", Float),
    Column("tif", Text, nullable=False, default="day"),
    Column("state", Text, nullable=False, default="proposed"),
    Column("sleeve", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

order_events = Table(
    "order_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, nullable=False, index=True),
    Column("from_state", Text),
    Column("to_state", Text, nullable=False),
    Column("reason", Text, nullable=False, default=""),
    Column("payload", JSONB),
    _ts(),
)

trades = Table(
    "trades",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("client_order_id", Text, nullable=False, index=True),
    Column("symbol", Text, nullable=False, index=True),
    Column("side", Text, nullable=False),
    Column("qty", Float, nullable=False),
    Column("fill_price", Float, nullable=False),
    Column("decision_price", Float),
    Column("slippage_bps", Float),
    Column("fees", Float, nullable=False, default=0.0),
    Column("filled_at", DateTime(timezone=True), nullable=False),
)

reconciliation_runs = Table(
    "reconciliation_runs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    _ts(),
    Column("scope", Text, nullable=False),
    Column("result", Text, nullable=False),  # clean | divergent
    Column("divergences", JSONB, nullable=False, default=[]),
)

reconciliation_acks = Table(
    "reconciliation_acks",
    metadata,
    Column("run_id", BigInteger, primary_key=True),
    Column("acked_by", Text, nullable=False),
    Column("acked_at", DateTime(timezone=True), nullable=False),
    Column("note", Text, nullable=False, default=""),
)

equity_curve = Table(
    "equity_curve",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("equity", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("gross_exposure", Float, nullable=False, default=0.0),
)

# --- Learning (doc 10) ---

decision_context = Table(
    "decision_context",
    metadata,
    Column("id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("env", Text, nullable=False),
    Column("symbol", Text, nullable=False, index=True),
    Column("sleeve", Text, nullable=False),
    Column("verdict_id", Text),
    Column("target_portfolio_id", Text, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("expected_upside_pct", Float, nullable=False),
    Column("expected_downside_pct", Float, nullable=False),
    Column("horizon_days", Integer, nullable=False),
    Column("prev_hash", Text, nullable=False),  # hash chain (doc 10 §2)
    Column("row_hash", Text, nullable=False),
)

post_trade_reviews = Table(
    "post_trade_reviews",
    metadata,
    Column("trade_id", Text, primary_key=True),
    Column("decision_context_id", Text, nullable=False),
    Column("symbol", Text, nullable=False, index=True),
    Column("as_of", Date, nullable=False),
    Column("payload", JSONB, nullable=False),
    Column("failure_primary", Text),
)

journal_entries = Table(
    "journal_entries",
    metadata,
    Column("id", Text, primary_key=True),
    Column("kind", Text, nullable=False, index=True),
    Column("symbols", ARRAY(Text), nullable=False, default=[]),
    Column("strategy", Text),
    Column("regime_at_write", Text),
    Column("sectors", ARRAY(Text), nullable=False, default=[]),
    Column("text", Text, nullable=False),
    Column("refs", JSONB, nullable=False, default={}),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("embedding", Vector(384)),  # bge-small; null until embedded
    Column("embedding_version", Text, nullable=False, default=""),
)

improvement_proposals = Table(
    "improvement_proposals",
    metadata,
    Column("id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("kind", Text, nullable=False),
    Column("target", Text, nullable=False),
    Column("evidence", JSONB, nullable=False, default=[]),
    Column("expected_effect", Text, nullable=False, default=""),
    Column("proposed_change", JSONB, nullable=False, default={}),
    Column("status", Text, nullable=False, default="open"),
)

calibration_points = Table(
    "calibration_points",
    metadata,
    Column("scope_type", Text, primary_key=True),
    Column("scope_id", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("brier", Float, nullable=False),
    Column("n", Integer, nullable=False),
    Column("bucket_stats", JSONB, nullable=False, default={}),
)

metrics_daily = Table(
    "metrics_daily",
    metadata,
    Column("metric", Text, primary_key=True),
    Column("scope_type", Text, primary_key=True),
    Column("scope_id", Text, primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("value", Float, nullable=False),
)

# --- Ops (docs 11, 12) ---

admin_actions = Table(
    "admin_actions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    _ts(),
    Column("action", Text, nullable=False),
    Column("payload", JSONB, nullable=False, default={}),
)

alerts = Table(
    "alerts",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("severity", Text, nullable=False),  # page | warn | info
    Column("name", Text, nullable=False),
    Column("message", Text, nullable=False),
    Column("ackable", Boolean, nullable=False, default=True),
    Column("acked_at", DateTime(timezone=True)),
)

# Rendered daily activity report (migration 0004): both the DailyReport JSON
# and the exact markdown shown to the owner, keyed by report day.
daily_reports = Table(
    "daily_reports",
    metadata,
    Column("as_of", Date, primary_key=True),
    Column("payload", JSONB, nullable=False),  # DailyReport (dynamo.reporting.daily)
    Column("markdown", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
