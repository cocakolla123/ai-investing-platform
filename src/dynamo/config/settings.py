"""Platform configuration. Design: doc 01 section 5 (environments), doc 00 section 6 (budget)."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DynamoEnv(StrEnum):
    DEV = "DEV"
    TEST = "TEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    dynamo_env: DynamoEnv = Field(default=DynamoEnv.DEV, alias="DYNAMO_ENV")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://dynamo:change-me@localhost:5432/dynamo_dev",
        alias="DATABASE_URL",
    )

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_monthly_budget_usd: float = Field(default=90.0, alias="CLAUDE_MONTHLY_BUDGET_USD")
    # Tier models: Sonnet 5 (intro $2/$10 through 2026-08-31, then $3/$15) is
    # cheaper AND newer than Sonnet 4.6; Opus 5 matches Opus 4.6 pricing with a
    # far lower prompt-cache minimum (512 vs 4096 tokens).
    model_screen: str = "claude-haiku-4-5"
    model_analyst: str = "claude-sonnet-5"
    model_arbiter: str = "claude-opus-5"
    deep_dive_cap_per_night: int = 8
    tier2_triage_cap_per_night: int = 60

    # --- Broker (paper first; live requires the gate, doc 09 section 7) ---
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )
    execution_auto_approve: bool = True  # paper default; always False in LIVE
    approval_ttl_minutes: int = 30
    reconciliation_cash_tolerance: float = 1.00

    # --- Market data ---
    tiingo_api_key: str = Field(default="", alias="TIINGO_API_KEY")
    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    edgar_user_agent: str = Field(
        default="Dynamo research (set EDGAR_USER_AGENT)", alias="EDGAR_USER_AGENT"
    )

    # --- Earnings-call transcripts (Alpha Vantage free key; defeatbeta backfill) ---
    alphavantage_api_key: str = Field(default="", alias="ALPHAVANTAGE_API_KEY")
    transcript_daily_cap: int = 20  # stay under Alpha Vantage's 25 requests/day free quota

    # --- TickerTick second news source (free, no key, 10 req/min) ---
    tickertick_enabled: bool = Field(default=True, alias="TICKERTICK_ENABLED")

    # --- Prediction markets (Kalshi tradable behind the gate; Polymarket reads keyless) ---
    kalshi_api_key_id: str = Field(default="", alias="KALSHI_API_KEY_ID")
    kalshi_private_key_path: str = Field(default="", alias="KALSHI_PRIVATE_KEY_PATH")
    pm_min_volume_usd: float = 25_000.0
    pm_max_spread: float = 0.03

    # --- OddsPipe cross-venue enrichment (ADR-0009 addendum; free tier) ---
    # Enrichment ONLY: the direct Kalshi/Polymarket adapters stay primary.
    # No key -> the OddsPipe adapter skips gracefully and nothing changes.
    oddspipe_api_key: str = Field(default="", alias="ODDSPIPE_API_KEY")
    oddspipe_min_spread: float = 0.03  # minimum YES spread fetched from /v1/spreads
    pm_cross_venue_veto_gap: float = 0.10  # |polymarket - estimated| veto threshold

    # --- Kalshi trading sleeve (ADR-0009; demo first, prod only behind the gate) ---
    # Base URLs exclude the /trade-api/v2 prefix (the adapters append it).
    kalshi_demo_base_url: str = Field(
        default="https://demo-api.kalshi.co", alias="KALSHI_DEMO_BASE_URL"
    )
    kalshi_prod_base_url: str = Field(
        default="https://api.elections.kalshi.com", alias="KALSHI_PROD_BASE_URL"
    )
    pm_sleeve_cap: float = 0.05  # sleeve total <= 5% of equity, hard
    pm_per_market_cap: float = 0.01  # per-market exposure <= 1% of equity
    pm_min_edge: float = 0.08  # |estimated - implied| floor
    pm_min_confidence: float = 0.6  # committee confidence floor
    pm_max_days_to_resolution: int = 90
    pm_kelly_fraction_cap: float = 0.25  # fractional Kelly, same cap as equities
    pm_max_open_markets: int = 10

    # --- Polymarket shadow-trading sleeve (ADR-0011; SIMULATED by default) ---
    # Polymarket has no demo exchange, so the sleeve simulates the broker
    # internally: decisions run through the same pm_* gates against the sim
    # bankroll and fills happen in our own tables at Gamma prices. This is
    # the ONLY behavior outside a fully-satisfied LIVE precondition chain.
    polymarket_sim_enabled: bool = True
    polymarket_sim_bankroll_usd: float = 1000.0
    polymarket_sim_slippage_cents: int = 1  # applied AGAINST us on every simulated fill

    # --- Polymarket LIVE execution (ADR-0012; gated, sim remains the default) ---
    # Real CLOB orders require ALL of: DYNAMO_ENV=LIVE, polymarket_live_enabled,
    # full credentials below, and a passing doc 09 §7 live gate (signed
    # approval + kill-switch absent + readiness). Anything missing falls back
    # to the sim broker with a loud warning — never the other way around.
    polymarket_api_key: str = Field(default="", alias="POLYMARKET_API_KEY")
    polymarket_api_secret: str = Field(default="", alias="POLYMARKET_API_SECRET")
    polymarket_passphrase: str = Field(default="", alias="POLYMARKET_PASSPHRASE")
    # Path to a file in secrets/ holding the Polygon wallet private key used
    # for EIP-712 order signing. Contents are NEVER logged or printed; the
    # broker reads the file lazily at client construction.
    polymarket_wallet_key_path: str = Field(default="", alias="POLYMARKET_WALLET_KEY_PATH")
    # Polymarket UI accounts trade through a proxy wallet: signature_type 1 =
    # email/Magic proxy, 2 = browser-wallet proxy, 0 = direct EOA. Proxy types
    # (1/2) also require the proxy ("funder") address that holds the USDC.
    polymarket_funder_address: str = Field(default="", alias="POLYMARKET_FUNDER_ADDRESS")
    polymarket_signature_type: int = 1
    polymarket_live_enabled: bool = False  # master switch, independent of DYNAMO_ENV
    polymarket_live_max_order_usd: float = 25.0  # hard per-order cap, enforced in the broker

    # --- Options sleeve (ADR-0010; defined-risk only, paper first) ---
    # Hard caps enforced in code (risk/options_sizing.py + domain validators):
    # the sleeve trades ONLY long calls, long puts, covered calls, and
    # cash-secured puts — unbounded-loss structures are unrepresentable.
    options_premium_at_risk_cap: float = 0.05  # long premium + assignment obligations
    options_per_position_cap: float = 0.01  # per-position premium at risk, of equity
    options_min_dte: int = 21
    options_max_dte: int = 90
    options_min_open_interest: int = 100
    options_max_spread_pct: float = 0.10  # (ask - bid) / mid ceiling
    options_min_conviction: float = 0.7
    options_iv_rank_max: float = 0.8  # never buy premium when IV rank is extreme
    options_max_open_positions: int = 8

    # --- Same-day tactical entries (15-min intraday loop) ---
    intraday_entry_cap_per_day: int = 3

    # --- Learning loop (doc 10; journal retrieval feeds dossier contexts) ---
    journal_retrieval_enabled: bool = True
    experience_retrieval_top_k: int = 3  # journal hits per symbol in the dossier
    # Hierarchical (scope-chain) calibration, plan Phase 2. Gates the
    # hier-scope calibration_points emission AND the advisory calibrated-p
    # logging onto decision_experience. ADVISORY ONLY — nothing consumes
    # calibrated_probability for sizing.
    hierarchical_calibration_enabled: bool = True

    # --- Social awareness (Reddit + StockTwits; detector input ONLY) ---
    # Social text is untrusted (pump-and-dump surface): it feeds the Tier-2
    # social_burst detector as an awareness trigger and never moves a score.
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="dynamo-research/0.1", alias="REDDIT_USER_AGENT")
    reddit_subreddits: str = Field(
        default="stocks,investing,wallstreetbets,StockMarket", alias="REDDIT_SUBREDDITS"
    )
    stocktwits_enabled: bool = Field(default=True, alias="STOCKTWITS_ENABLED")
    social_burst_zscore: float = 4.0
    social_min_mentions: int = 10
    social_lookback_days: int = 14

    # --- Paths ---
    parquet_dir: str = Field(default="data/parquet", alias="DYNAMO_PARQUET_DIR")
    artifact_dir: str = Field(default="artifacts", alias="DYNAMO_ARTIFACT_DIR")
    kill_switch_path: str = Field(default="KILL_SWITCH", alias="DYNAMO_KILL_SWITCH_PATH")
    approvals_dir: str = Field(default="approvals", alias="DYNAMO_APPROVALS_DIR")

    # --- Alerting ---
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    @property
    def is_live(self) -> bool:
        return self.dynamo_env == DynamoEnv.LIVE

    @property
    def auto_approve(self) -> bool:
        # Live orders NEVER auto-approve regardless of config (doc 09 section 2).
        return self.execution_auto_approve and not self.is_live


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
