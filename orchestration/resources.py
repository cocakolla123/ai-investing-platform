"""Composition root for orchestration (doc 01 §2: orchestration wires everything).

``DynamoResource`` builds the object graph lazily from ``Settings`` so that
``dagster dev`` can load definitions without a database or any API key.
Every provider accessor returns ``None`` when its key is missing — assets log
a skip and return empty metadata (the platform must run partially configured).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from dagster import ConfigurableResource
from pydantic import PrivateAttr

from dynamo.config.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from dynamo.data.alpaca_data import AlpacaMarketData
    from dynamo.data.edgar import EdgarClient
    from dynamo.data.finnhub import FinnhubClient
    from dynamo.data.fred import FredClient
    from dynamo.data.kalshi import KalshiClient
    from dynamo.data.oddspipe import OddsPipeProvider
    from dynamo.data.polymarket import PolymarketClient
    from dynamo.data.tickertick import TickerTickClient
    from dynamo.data.tiingo import TiingoClient
    from dynamo.data.transcripts_av import AlphaVantageTranscripts
    from dynamo.data.transcripts_defeatbeta import DefeatBetaTranscripts
    from dynamo.llm.anthropic_client import AnthropicClient
    from dynamo.llm.cost_meter import CostMeter
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.market import MarketDataRepository
    from dynamo.research.committee import CommitteeEngine

# ~40 FRED series per doc 02 §1; a starter set — config value, not code logic.
DEFAULT_FRED_SERIES: tuple[str, ...] = (
    "DGS3MO",
    "DGS2",
    "DGS10",
    "T10Y3M",
    "T10Y2Y",
    "CPIAUCSL",
    "PCEPILFE",
    "UNRATE",
    "BAMLH0A0HYM2",
    "NFCI",
    "DFF",
    "VIXCLS",
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_of_from_partition(partition_key: str | None) -> date:
    """Partition key -> as_of date; falls back to today (UTC)."""
    if partition_key:
        return date.fromisoformat(partition_key)
    return utcnow().date()


class DynamoResource(ConfigurableResource):
    """Lazily-built adapters + repositories for asset bodies.

    Tests subclass this and override accessors with in-memory fakes; asset
    bodies only ever go through these methods, never module globals.
    """

    database_url: str = ""  # empty -> Settings.database_url

    _settings: Any = PrivateAttr(default=None)
    _engine: Any = PrivateAttr(default=None)
    _cost_meter: Any = PrivateAttr(default=None)
    _research_model: Any = PrivateAttr(default=None)

    # --- config / infrastructure ---

    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    def engine(self) -> Engine:
        if self._engine is None:
            from dynamo.persistence.db import make_engine

            url = self.database_url or self.settings().database_url
            self._engine = make_engine(url)
        return self._engine

    def market_repo(self) -> MarketDataRepository:
        from dynamo.persistence.repositories.market import MarketDataRepository

        return MarketDataRepository(self.engine())

    def experience_repo(self) -> ExperienceRepository:
        from dynamo.persistence.repositories.experience import ExperienceRepository

        return ExperienceRepository(self.engine())

    def cost_meter(self) -> CostMeter:
        if self._cost_meter is None:
            from dynamo.llm.cost_meter import CostMeter, SqlAlchemyLedgerWriter

            self._cost_meter = CostMeter(
                SqlAlchemyLedgerWriter(self.engine()),
                self.settings().claude_monthly_budget_usd,
            )
        return self._cost_meter

    # --- research model + committee ---

    def research_model(self) -> AnthropicClient | None:
        """None when ANTHROPIC_API_KEY is unset — degraded (quant-only) mode."""
        if not self.settings().anthropic_api_key:
            return None
        if self._research_model is None:
            from dynamo.llm.anthropic_client import AnthropicClient

            self._research_model = AnthropicClient(self.settings(), self.cost_meter())
        return self._research_model

    def committee_engine(self) -> CommitteeEngine | None:
        model = self.research_model()
        if model is None:
            return None
        from dynamo.llm.schemas import SCHEMAS
        from dynamo.research.committee import CommitteeEngine
        from dynamo.research.prompt_loader import load_prompt

        prompts = {
            name: load_prompt(name)
            for name in ("dossier_analyst", "bull_advocate", "bear_advocate", "arbiter")
        }
        return CommitteeEngine(model, prompts, SCHEMAS)

    # --- data providers (each None when unconfigured) ---

    def alpaca(self) -> AlpacaMarketData | None:
        s = self.settings()
        if not (s.alpaca_api_key and s.alpaca_api_secret):
            return None
        from dynamo.data.alpaca_data import AlpacaMarketData

        return AlpacaMarketData(s.alpaca_api_key, s.alpaca_api_secret)

    def tiingo(self) -> TiingoClient | None:
        s = self.settings()
        if not s.tiingo_api_key:
            return None
        from dynamo.data.tiingo import TiingoClient

        return TiingoClient(s.tiingo_api_key)

    def edgar(self) -> EdgarClient | None:
        from dynamo.data.edgar import EdgarClient

        return EdgarClient(self.settings().edgar_user_agent)

    def finnhub(self) -> FinnhubClient | None:
        s = self.settings()
        if not s.finnhub_api_key:
            return None
        from dynamo.data.finnhub import FinnhubClient

        return FinnhubClient(s.finnhub_api_key)

    def fred(self) -> FredClient | None:
        s = self.settings()
        if not s.fred_api_key:
            return None
        from dynamo.data.fred import FredClient

        return FredClient(s.fred_api_key)

    def kalshi(self) -> KalshiClient | None:
        s = self.settings()
        if not (s.kalshi_api_key_id and s.kalshi_private_key_path):
            return None
        from dynamo.data.kalshi import KalshiClient

        return KalshiClient(
            s.kalshi_api_key_id,
            s.kalshi_private_key_path,
            min_volume_usd=s.pm_min_volume_usd,
            max_spread=s.pm_max_spread,
        )

    def polymarket(self) -> PolymarketClient | None:
        from dynamo.data.polymarket import PolymarketClient

        s = self.settings()
        return PolymarketClient(min_volume_usd=s.pm_min_volume_usd, max_spread=s.pm_max_spread)

    def tickertick(self) -> TickerTickClient | None:
        """Second news source (free, no key); None when disabled via settings."""
        if not self.settings().tickertick_enabled:
            return None
        from dynamo.data.tickertick import TickerTickClient

        return TickerTickClient()

    def alphavantage_transcripts(self) -> AlphaVantageTranscripts | None:
        """Fresh earnings-call transcripts; None when the AV key is unset."""
        s = self.settings()
        if not s.alphavantage_api_key:
            return None
        from dynamo.data.transcripts_av import AlphaVantageTranscripts

        return AlphaVantageTranscripts(
            s.alphavantage_api_key, daily_cap=s.transcript_daily_cap
        )

    def defeatbeta_transcripts(self) -> DefeatBetaTranscripts | None:
        """Bulk transcript backfill; None when the ``transcripts`` extra is absent."""
        from dynamo.data.transcripts_defeatbeta import DefeatBetaTranscripts

        provider = DefeatBetaTranscripts()
        return provider if provider.available else None

    def oddspipe(self) -> OddsPipeProvider | None:
        """Cross-venue enrichment (ADR-0009 addendum); None when the key is unset."""
        s = self.settings()
        if not s.oddspipe_api_key:
            return None
        from dynamo.data.oddspipe import OddsPipeProvider

        return OddsPipeProvider(s.oddspipe_api_key, min_spread=s.oddspipe_min_spread)

    # --- sensor support (thin queries; fakes override these in tests) ---

    def pending_batches(self) -> list[dict[str, Any]]:
        """batch_jobs rows still awaiting collection (status='submitted')."""
        from sqlalchemy import select

        from dynamo.persistence import schema

        with self.engine().connect() as conn:
            return [
                dict(row._mapping)
                for row in conn.execute(
                    select(schema.batch_jobs).where(schema.batch_jobs.c.status == "submitted")
                )
            ]

    def mark_batch_expired(self, batch_id: str) -> None:
        from sqlalchemy import update

        from dynamo.persistence import schema

        with self.engine().begin() as conn:
            conn.execute(
                update(schema.batch_jobs)
                .where(schema.batch_jobs.c.batch_id == batch_id)
                .values(status="expired", resolved_at=utcnow())
            )

    def alert(self, severity: str, name: str, message: str) -> None:
        self.market_repo().add_alert(severity=severity, name=name, message=message)

    def quarantine_count_24h(self) -> int:
        """Distinct unresolved quarantined symbols created in the last 24h."""
        from sqlalchemy import func, select

        from dynamo.persistence import schema

        q = schema.quarantine
        since = utcnow() - timedelta(hours=24)
        with self.engine().connect() as conn:
            return int(
                conn.execute(
                    select(func.count(func.distinct(q.c.key))).where(
                        q.c.scope == "symbol",
                        q.c.created_at >= since,
                        q.c.resolved_at.is_(None),
                    )
                ).scalar()
                or 0
            )

    def earnings_reporters(self, start: date, end: date) -> list[dict[str, Any]]:
        """earnings_events rows with event_date in [start, end] (transcripts asset)."""
        from sqlalchemy import select

        from dynamo.persistence import schema

        e = schema.earnings_events
        with self.engine().connect() as conn:
            return [
                dict(row._mapping)
                for row in conn.execute(
                    select(e.c.symbol, e.c.event_date, e.c.period, e.c.call_transcript_ref)
                    .where(e.c.event_date >= start, e.c.event_date <= end)
                    .order_by(e.c.event_date, e.c.symbol)
                )
            ]

    # --- universe helpers ---

    def universe_symbols(self, as_of: date, tiers: tuple[int, ...] | None = None) -> list[str]:
        """Active membership symbols at as_of, optionally filtered by tier."""
        members = self.market_repo().universe_at(as_of)
        return sorted({m.symbol for m in members if tiers is None or int(m.tier) in tiers})

    def bars_window(self, symbols: list[str] | None, as_of: date, days: int) -> Any:
        """Canonical bars frame for the trailing ``days`` calendar days."""
        return self.market_repo().bars_frame(symbols, as_of - timedelta(days=days), as_of)
