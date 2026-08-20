"""Market-data domain entities. Design: docs/design/02-data-platform.md."""

from __future__ import annotations

from datetime import date, datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Base for immutable domain entities."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Tier(IntEnum):
    CORE = 1  # full pipeline
    SURVEILLANCE = 2  # code-only detectors
    PROMOTED = 3  # temporary full treatment


class InstrumentStatus(StrEnum):
    ACTIVE = "active"
    DELISTED = "delisted"
    SUSPENDED = "suspended"


class Instrument(Frozen):
    symbol: str
    name: str
    exchange: str | None = None
    sector: str | None = None  # GICS sector
    industry: str | None = None
    cik: str | None = None
    figi: str | None = None
    is_etf: bool = False
    status: InstrumentStatus = InstrumentStatus.ACTIVE
    first_seen: date | None = None
    delisted_at: date | None = None


class Bar(Frozen):
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None
    source: str = ""


class UniverseMembership(Frozen):
    """Point-in-time tier membership; backbone for survivorship-safe backtests."""

    symbol: str
    tier: Tier
    valid_from: date
    valid_to: date | None = None  # None = current
    reason: str
    approved_by: str | None = None  # required for permanent tier-1 changes


class Filing(Frozen):
    accession_no: str
    cik: str
    symbol: str | None = None
    form_type: str  # 10-K, 10-Q, 8-K, S-1, 13D, 13G, ...
    filed_at: datetime
    period: date | None = None
    url: str
    items: tuple[str, ...] = ()  # 8-K item codes


class XbrlFact(Frozen):
    """Point-in-time fundamental. Keyed by filed_at — never restated backwards."""

    cik: str
    tag: str
    period_end: date
    value: float
    unit: str
    filed_at: datetime


class InsiderTx(Frozen):
    accession_no: str
    symbol: str
    insider_name: str
    role: str | None = None
    tx_type: str  # P (purchase) / S (sale) / A (award) ...
    shares: float
    price: float | None = None
    filed_at: datetime


class NewsItem(Frozen):
    id: str
    symbols: tuple[str, ...]
    headline: str
    summary: str | None = None
    source: str
    published_at: datetime
    url: str | None = None


class MacroPoint(Frozen):
    """FRED observation with vintage for point-in-time discipline."""

    series_id: str
    ts: date
    value: float
    vintage_ts: datetime


class ShortInterest(Frozen):
    """One FINRA consolidated short-interest cycle for one symbol.

    TWO DATES, DELIBERATELY. ``settlement_date`` is what the figures describe;
    ``publication_date`` is when FINRA disseminated them, roughly eight days
    later. Reads MUST bound on ``publication_date`` — bounding on the
    settlement date hands a backtest a week and a half of the future. See
    :mod:`dynamo.data.finra` for how the publication date is derived.
    """

    symbol: str
    settlement_date: date
    publication_date: date
    shares_short: float
    previous_shares_short: float | None = None
    avg_daily_volume: float | None = None
    days_to_cover: float | None = None
    change_pct: float | None = None
    market_class: str | None = None
    issue_name: str | None = None
    split_flag: str | None = None  # "S" when the cycle spans a stock split
    revision_flag: str | None = None  # "R" when the prior cycle was revised


class EarningsEvent(Frozen):
    symbol: str
    event_date: date
    period: str | None = None  # e.g. "2026Q2"
    eps_est: float | None = None
    eps_act: float | None = None
    revenue_est: float | None = None
    revenue_act: float | None = None
    call_transcript_ref: str | None = None


class Transcript(Frozen):
    """One earnings-call transcript (sources: ``alphavantage`` | ``defeatbeta``)."""

    symbol: str
    event_date: date
    source: str
    period: str | None = None  # e.g. "2026Q2"
    content: str
    speakers: tuple[dict[str, str], ...] | None = None

    @property
    def ref(self) -> str:
        """The ``earnings_events.call_transcript_ref`` pointer for this row."""
        return f"transcripts:{self.symbol}:{self.event_date.isoformat()}:{self.source}"


class CorporateAction(Frozen):
    symbol: str
    ex_date: date
    action_type: str  # split | dividend | merger | spinoff
    ratio: float | None = None
    amount: float | None = None
    source: str = ""


class SurveillanceFlag(Frozen):
    """Tier-2 detector output. Design: doc 02 section 2."""

    symbol: str
    detector: str  # price_zscore | volume_spike | rel_strength | ...
    severity: float  # 0-1
    details: dict[str, float | str] = {}
    ts: datetime


class TriageVerdict(StrEnum):
    MATERIAL = "material"
    WATCH = "watch"
    NOISE = "noise"


class PredictionMarket(Frozen):
    venue: str  # kalshi | polymarket
    market_id: str
    question: str
    category: str | None = None
    close_ts: datetime | None = None
    resolved_outcome: bool | None = None


class PredictionMarketPrice(Frozen):
    venue: str
    market_id: str
    ts: datetime
    implied_prob: float
    volume: float | None = None
    liquidity: float | None = None
    trusted: bool = False  # thin-market filter (doc 02 section 7)
