"""In-memory fakes for orchestration unit tests — no network, no database.

Dagster copies/reconstructs ConfigurableResources during execution, so fake
state cannot live on the instance: it lives in a module registry keyed by a
plain config field (``state_key``) that survives the copy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest
from orchestration.resources import DynamoResource

from dynamo.config.settings import Settings
from dynamo.domain.models.market import Tier, UniverseMembership
from dynamo.llm.anthropic_client import AnthropicClient
from dynamo.llm.cost_meter import CostMeter, InMemoryLedgerWriter
from dynamo.persistence.repositories.market import MarketDataRepository


def make_bars(symbols: list[str], end: date, sessions: int = 70) -> pd.DataFrame:
    """Synthetic canonical bars frame shaped like MarketDataRepository.bars_frame."""
    days = pd.bdate_range(end=pd.Timestamp(end), periods=sessions, tz=UTC)
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        base = 100.0 + hash(sym) % 50
        for i, ts in enumerate(days):
            close = base + i * 0.1
            rows.append(
                {
                    "symbol": sym,
                    "ts": ts,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000 + (100_000 if i == sessions - 1 else 0),
                    "source": "tiingo",
                }
            )
    return pd.DataFrame(rows)


class FakeRepo:
    """The MarketDataRepository slice the tested assets touch."""

    def __init__(self, bars: pd.DataFrame, members: list[UniverseMembership]) -> None:
        self.bars = bars
        self.members = members
        self.quarantines: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.flags: list[Any] = []
        self.inserted_rows: list[tuple[str, list[dict[str, Any]]]] = []

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        return [m for m in self.members if tier is None or m.tier == tier]

    def bars_frame(self, symbols: Any, start: date, end: date) -> pd.DataFrame:
        frame = self.bars
        if symbols is not None and not frame.empty:
            frame = cast(pd.DataFrame, frame[frame["symbol"].isin(list(symbols))])
        return frame.reset_index(drop=True)

    def add_quarantine(self, **kwargs: Any) -> None:
        self.quarantines.append(kwargs)

    def add_alert(self, **kwargs: Any) -> None:
        self.alerts.append(kwargs)

    def insert_surveillance_flags(self, flags: Any) -> int:
        flags = list(flags)
        self.flags.extend(flags)
        return len(flags)

    def insert_rows(self, table: Any, rows: Any) -> int:
        rows = list(rows)
        self.inserted_rows.append((table.name, rows))
        return len(rows)

    def upsert_rows(self, table: Any, rows: Any) -> int:
        return self.insert_rows(table, rows)

    def upsert_cross_venue_spreads(self, spreads: Any) -> int:
        spreads = list(spreads)
        self.inserted_rows.append(("pm_cross_venue_spreads", spreads))
        return len(spreads)

    def latest_quant_scores(self, symbols: Any = None) -> pd.DataFrame:
        return pd.DataFrame()

    def latest_verdicts(self, symbols: Any = None) -> pd.DataFrame:
        return pd.DataFrame()


class _FakeModel:
    def __init__(self, statuses: dict[str, str]) -> None:
        self._statuses = statuses

    def batch_status(self, batch_id: str) -> str:
        return self._statuses.get(batch_id, "in_progress")


@dataclass
class FakeState:
    repo: FakeRepo | None = None
    batches: list[dict[str, Any]] = field(default_factory=list)
    statuses: dict[str, str] | None = None
    alerts: list[dict[str, Any]] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    meter: CostMeter | None = None
    quarantine_count: int = 0
    oddspipe_spreads: list[Any] | None = None  # None -> provider unconfigured


_STATE: dict[str, FakeState] = {}


def offline_settings() -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="",
        ALPACA_API_KEY="",
        ALPACA_API_SECRET="",
        TIINGO_API_KEY="",
        FINNHUB_API_KEY="",
        FRED_API_KEY="",
        KALSHI_API_KEY_ID="",
        ODDSPIPE_API_KEY="",
        _env_file=None,  # type: ignore[call-arg]
    )


class FakeDynamo(DynamoResource):
    """DynamoResource with every external surface replaced by memory fakes."""

    state_key: str = ""

    @property
    def st(self) -> FakeState:
        return _STATE[self.state_key]

    def settings(self) -> Settings:
        return offline_settings()

    def engine(self) -> Any:
        raise AssertionError("unit tests must not touch a database engine")

    def market_repo(self) -> MarketDataRepository:
        return cast(MarketDataRepository, self.st.repo)

    def oddspipe(self) -> Any:
        spreads = self.st.oddspipe_spreads
        if spreads is None:
            return None

        class _StubOddsPipe:
            def spreads(self, min_spread: Any = None) -> list[Any]:
                return list(spreads)

        return _StubOddsPipe()

    def research_model(self) -> AnthropicClient | None:
        if self.st.statuses is None:
            return None
        return cast(AnthropicClient, _FakeModel(self.st.statuses))

    def cost_meter(self) -> CostMeter:
        if self.st.meter is None:
            self.st.meter = CostMeter(InMemoryLedgerWriter(), 100.0)
        return self.st.meter

    # sensor surface
    def pending_batches(self) -> list[dict[str, Any]]:
        return list(self.st.batches)

    def mark_batch_expired(self, batch_id: str) -> None:
        self.st.expired.append(batch_id)

    def alert(self, severity: str, name: str, message: str) -> None:
        self.st.alerts.append({"severity": severity, "name": name, "message": message})

    def quarantine_count_24h(self) -> int:
        return self.st.quarantine_count


@pytest.fixture
def fake_repo() -> FakeRepo:
    today = datetime.now(UTC).date()
    members = [
        UniverseMembership(
            symbol="AAA", tier=Tier.CORE, valid_from=today - timedelta(days=30), reason="seed"
        ),
        UniverseMembership(
            symbol="BBB",
            tier=Tier.SURVEILLANCE,
            valid_from=today - timedelta(days=30),
            reason="seed",
        ),
    ]
    return FakeRepo(make_bars(["AAA", "BBB"], today), members)


@pytest.fixture
def fake_dynamo(fake_repo: FakeRepo) -> Any:
    key = uuid.uuid4().hex
    _STATE[key] = FakeState(repo=fake_repo)
    dynamo = FakeDynamo(state_key=key)
    yield dynamo
    _STATE.pop(key, None)
