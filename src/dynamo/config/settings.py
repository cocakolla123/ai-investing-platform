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
    model_screen: str = "claude-haiku-4-5"
    model_analyst: str = "claude-sonnet-4-6"
    model_arbiter: str = "claude-opus-4-6"
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

    # --- Prediction markets (read-only) ---
    kalshi_api_key_id: str = Field(default="", alias="KALSHI_API_KEY_ID")
    kalshi_private_key_path: str = Field(default="", alias="KALSHI_PRIVATE_KEY_PATH")
    pm_min_volume_usd: float = 25_000.0
    pm_max_spread: float = 0.03

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
