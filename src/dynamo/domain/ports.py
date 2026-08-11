"""Domain ports — Protocols that adapters implement. Design: doc 01 section 2.

Core logic depends only on these; adapters (data/, llm/, persistence/,
execution/) provide implementations, injected at the composition root.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from dynamo.domain.models.learning import DecisionExperience
from dynamo.domain.models.market import (
    Bar,
    CorporateAction,
    EarningsEvent,
    Filing,
    InsiderTx,
    Instrument,
    MacroPoint,
    NewsItem,
    PredictionMarket,
    PredictionMarketPrice,
    XbrlFact,
)
from dynamo.domain.models.options import OptionOrder
from dynamo.domain.models.pm import PMOrder
from dynamo.domain.models.portfolio import Order


@runtime_checkable
class MarketDataProvider(Protocol):
    """Bars, quotes, corporate actions (Alpaca / Tiingo adapters)."""

    @property
    def source(self) -> str: ...

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> Iterable[Bar]: ...

    def latest_quotes(self, symbols: Sequence[str]) -> dict[str, float]: ...

    def corporate_actions(self, start: date, end: date) -> Iterable[CorporateAction]: ...

    def tradable_instruments(self) -> Iterable[Instrument]: ...


class FilingsProvider(Protocol):
    """SEC EDGAR: filings index, XBRL facts, Form 4 insider transactions."""

    def recent_filings(self, since: datetime) -> Iterable[Filing]: ...

    def company_facts(self, cik: str) -> Iterable[XbrlFact]: ...

    def insider_transactions(self, since: datetime) -> Iterable[InsiderTx]: ...


class MacroProvider(Protocol):
    def series(self, series_id: str, start: date) -> Iterable[MacroPoint]: ...


class NewsProvider(Protocol):
    def news(self, symbols: Sequence[str], since: datetime) -> Iterable[NewsItem]: ...

    def earnings_calendar(self, start: date, end: date) -> Iterable[EarningsEvent]: ...


class PredictionMarketProvider(Protocol):
    """Read-only market/price snapshots. Kalshi and Polymarket additionally
    have gated trading paths via :class:`PMBroker` (ADR-0009 / ADR-0012);
    this provider itself never places orders."""

    @property
    def venue(self) -> str: ...

    def markets(self) -> Iterable[PredictionMarket]: ...

    def prices(self, market_ids: Sequence[str]) -> Iterable[PredictionMarketPrice]: ...


class BrokerOrderAck(Protocol):
    @property
    def broker_order_id(self) -> str: ...


class Broker(Protocol):
    """Alpaca adapter (paper first) + MockBroker for TEST/backtest."""

    def submit(self, order: Order) -> str: ...  # returns broker_order_id

    def cancel(self, client_order_id: str) -> None: ...

    def replace(self, client_order_id: str, stop_price: float) -> str: ...

    def get_order(self, client_order_id: str) -> dict[str, Any]: ...

    def open_orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> list[dict[str, Any]]: ...

    def account(self) -> dict[str, Any]: ...


class PMBroker(Protocol):
    """Prediction-market broker port (ADR-0009, ADR-0011, ADR-0012).

    KalshiBroker (demo/prod behind the live gate), PolymarketSimBroker (the
    ADR-0011 simulated sleeve, the default), PolymarketBroker (real CLOB
    orders, reachable ONLY through the ADR-0012 precondition chain), and
    MockPMBroker for tests.
    """

    def submit(self, order: PMOrder) -> str: ...  # returns broker_order_id

    def cancel(self, client_order_id: str) -> None: ...

    def get_order(self, client_order_id: str) -> dict[str, Any]: ...

    def open_orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> list[dict[str, Any]]: ...

    def balance(self) -> dict[str, Any]: ...


class OptionsBroker(Protocol):
    """Options broker port — defined-risk sleeve (ADR-0010).

    AlpacaOptionsBroker (paper first, live only behind the doc 09 §7 gate) +
    MockOptionsBroker for tests. Limit orders only — the port has no market
    order surface, by design.
    """

    def submit(self, order: OptionOrder) -> str: ...  # returns broker_order_id

    def cancel(self, client_order_id: str) -> None: ...

    def get_order(self, client_order_id: str) -> dict[str, Any]: ...

    def open_orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> list[dict[str, Any]]: ...  # options positions only


# Segment boundary inside ResearchModel `context` strings. Implementations MAY
# split on it to place prompt-cache breakpoints at stable prefix boundaries
# (doc 03 section 3: stable -> volatile). Callers build contexts as growing
# prefixes (base | dossier | advocates) so later calls re-read earlier segments
# from cache. Plain-text safe: ASCII group separator, never in natural text.
CONTEXT_CACHE_BOUNDARY = "\n\x1d\n"


class ResearchModel(Protocol):
    """Every Claude interaction goes through this port (doc 03).

    Implementations meter cost and enforce the monthly kill-switch.
    """

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        context: str,
        instruction: str,
        schema: dict[str, Any],
        model_tier: str,  # screen | analyst | arbiter
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...

    def submit_batch(
        self, requests: Sequence[dict[str, Any]]
    ) -> str: ...  # returns batch id; custom_ids deterministic

    def batch_status(self, batch_id: str) -> str: ...

    def batch_results(self, batch_id: str) -> Iterable[tuple[str, dict[str, Any] | None]]: ...


class ForecastModel(Protocol):
    """Quant model family behind one interface (ADR-0003 modularity)."""

    def fit(self, x: Any, y: Any, cfg: dict[str, Any]) -> Any: ...

    def predict(self, x: Any) -> Any: ...

    def feature_importances(self) -> dict[str, float]: ...


class ExperienceStore(Protocol):
    """``decision_experience`` persistence (plan Phase 1).

    Firewall discipline: live learning consumers read ONLY through
    ``resolved_live``/``unresolved_live`` (which filter ``source='live'`` by
    design); ``by_run`` is the sole replay read path. ``insert`` is idempotent
    for live rows (one experience per decision context).
    """

    def insert(self, exp: DecisionExperience) -> bool: ...  # True iff a row was written

    def resolved_live(self, since: date | None = None) -> list[DecisionExperience]: ...

    def unresolved_live(self) -> list[DecisionExperience]: ...

    def by_run(self, replay_run_id: str) -> list[DecisionExperience]: ...

    def count(self, source: str | None = None) -> int: ...


class EmbeddingModel(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Clock(Protocol):
    """Injectable time source — risk/portfolio code never reads the wall clock."""

    def now(self) -> datetime: ...

    def today(self) -> date: ...
