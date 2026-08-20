"""Point-in-time fundamentals for the dossier (campaign findings, item 10).

The committee used to judge a $5T company from ~18 ticker-specific numbers:
``assemble_dossier_context`` had no parameter that could carry a financial
statement, so 944,859 ``xbrl_facts`` rows were compressed into one
``expected_excess_return`` scalar before the model ever saw them. This module
builds the missing evidence — TTM income statement, margins and their trend,
balance sheet, cash flow, and valuation multiples with sector percentiles —
as a typed, citable snapshot.

Three properties matter and are enforced here:

1. **Point-in-time.** Every read goes through the existing catalog helpers
   (:func:`~dynamo.research.quant.catalog.pit_facts`, ``quarterly_series``,
   ``ttm_sum``, ``latest_value``, ``SHARES_TAGS``). There is deliberately no
   second point-in-time implementation: facts filed after ``as_of`` are
   invisible because ``pit_facts`` says so.
2. **Citable.** Every rendered line carries a ``source_ref`` the advocates can
   quote — ``xbrl:WMT:Revenues:2026-04-30:filed-2026-05-29`` for a filed fact,
   ``derived:...`` for an arithmetic combination naming its inputs.
3. **Never fabricated.** A metric that cannot be computed renders ``n/a``.
   Nothing here raises: :func:`build_fundamentals` catches per-metric failures
   and degrades to absence.

Three real obstacles in the data (all verified against the live table):

(a) **No period-type discriminator.** ``Revenues`` rows for a fiscal year and
    for Q1 are indistinguishable except by ``period_end`` — WMT files
    713,163M at 2026-01-31 (FY) next to 177,751M at 2026-04-30 (Q1); GOOGL
    files 402,836M at 2025-12-31 (FY) with no separate Q4 row at all. Summing
    "the last four period_ends" therefore double-counts a year.

    Worse, the scope varies BY TAG and BY FISCAL YEAR inside one filer:
    CAT's 2025 filings report ``Revenues`` as a discrete Q4,
    ``OperatingIncomeLoss`` as the full year and
    ``NetCashProvidedByUsedInOperatingActivities`` as year-to-date (10-Q cash
    flow statements are cumulative by construction); ACN's ``Revenues`` are
    fiscal-year totals through FY2023 and discrete quarters from FY2024. So
    the scope is recovered per tag and per year from the numbers themselves:
    :func:`fiscal_year_end_month` finds the year end, :func:`infer_shape`
    classifies the tag (discrete / cumulative / fiscal-year-in-Q4 /
    annual-only), and :func:`normalized_facts` rewrites every flow row into a
    single quarter — a fiscal-year row becomes ``FY - (Q1+Q2+Q3)``, a
    year-to-date row becomes ``YTD(q) - YTD(q-1)``. Rows whose predecessors
    are missing are dropped, TTM windows must cover four DISTINCT fiscal
    quarters, and a rewrite that turns revenue, cost of revenue or capex
    negative marks the tag unusable. Absence beats a number nobody can
    defend.
(b) **Restatements.** Up to 6 rows share a (cik, tag, period_end) and differ
    only by ``filed_at``. ``quarterly_series``/``latest_value`` already take
    the latest filed row (<= ``as_of`` via ``pit_facts``); this module keeps
    that ordering and reports the winning ``filed_at`` in the source ref.
(c) **Tag normalization.** Revenue splits across ``Revenues`` (36 CIKs) and
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` (25); equity
    across ``StockholdersEquity`` (38) and the
    ``...IncludingPortionAttributableToNoncontrollingInterest`` variant (27) —
    CAT has zero rows of the former. Each concept resolves through an ordered
    candidate list, first tag with usable data wins, and the winning tag is
    named in the source ref so the reader knows which one was used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Any, cast

import pandas as pd

from dynamo.domain.models.market import Frozen
from dynamo.research.quant.catalog import (
    SHARES_TAGS,
    latest_value,
    pit_facts,
    quarterly_series,
    ttm_sum,
)

# --- units -----------------------------------------------------------------

USD_M = "usd_m"  # millions of USD, 1 decimal
USD = "usd"  # per-share dollars, 2 decimals
PCT = "pct"  # percent, 2 decimals
PCT_SERIES = "pct_series"  # oldest -> latest percent series
MULTIPLE = "x"  # valuation multiple, 2 decimals
SHARES_M = "shares_m"  # millions of shares, 1 decimal
PCTILE = "pctile"  # sector percentile, 0 decimals

# A TTM window must cover four consecutive quarters: roughly a year of
# period_ends, never a stale straggler pair.
TTM_MIN_SPAN_DAYS = 250
TTM_MAX_SPAN_DAYS = 400
# A fiscal-year row is at least this multiple of the largest of the three
# quarters preceding it (a real FY is ~4x; a genuine Q4 filed at the FY end is
# ~1x and must stay a quarter).
ANNUAL_MAGNITUDE_RATIO = 1.8
# Sector percentiles are noise below this many peers with a finite multiple.
MIN_PEERS_FOR_PERCENTILE = 3
# A share count below this is an upstream unit artifact, not a company with
# 700 shares outstanding (see :func:`_shares_row`).
MIN_PLAUSIBLE_SHARES = 100_000.0
# Operating cash flow this far outside revenue means the period rewrite did
# not recover the filer's scoping (JPM's filed quarterly cash-flow rows sum to
# -849bn on 182bn of revenue): render nothing rather than that.
MAX_CFO_REVENUE_MULTIPLE = 3.0

# --- tag candidates (obstacle (c)) -----------------------------------------

REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
COGS_TAGS = (
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsSold",
)
GROSS_PROFIT_TAGS = ("GrossProfit",)
OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
)
EPS_DILUTED_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")
CFO_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
ASSETS_TAGS = ("Assets",)
LIABILITIES_TAGS = ("Liabilities",)
EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
LONG_TERM_DEBT_TAGS = ("LongTermDebtNoncurrent", "LongTermDebt")
CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
)

# Flow (duration) concepts — the ones period-type inference applies to.
FLOW_TAGS: tuple[str, ...] = (
    *REVENUE_TAGS,
    *COGS_TAGS,
    *GROSS_PROFIT_TAGS,
    *OPERATING_INCOME_TAGS,
    *NET_INCOME_TAGS,
    *EPS_DILUTED_TAGS,
    *CFO_TAGS,
    *CAPEX_TAGS,
)

# Concepts that cannot legitimately be negative: if the period rewrite makes
# one negative, the inference was wrong and the metric must not be rendered.
NON_NEGATIVE_TAGS: frozenset[str] = frozenset((*REVENUE_TAGS, *COGS_TAGS, *CAPEX_TAGS))

_MODE_QUARTERLY = "quarterly"
_MODE_ANNUAL = "annual"
_MODE_NONE = "none"


# --- typed snapshot --------------------------------------------------------


class FundamentalLine(Frozen):
    """One rendered fundamentals line with its provenance.

    A plain ``Mapping[str, float | str]`` was the other option, and it was
    rejected for one reason: it cannot carry a per-metric ``source_ref``
    without smuggling provenance into stringly-typed values. The advocates are
    required to cite their evidence — unreferenced points are discarded — so
    the citation has to be a first-class field of every number.
    """

    key: str
    unit: str
    value: float | None = None
    values: tuple[float, ...] = ()
    source_ref: str = ""
    note: str = ""


class FundamentalsGroup(Frozen):
    title: str
    lines: tuple[FundamentalLine, ...] = ()


class FundamentalsSnapshot(Frozen):
    """Everything the ``## Fundamentals`` section renders, already resolved."""

    symbol: str
    as_of: date
    groups: tuple[FundamentalsGroup, ...] = ()

    @property
    def has_data(self) -> bool:
        return any(
            line.value is not None or line.values for group in self.groups for line in group.lines
        )


# --- formatting (fixed, deterministic) -------------------------------------


def format_value(line: FundamentalLine) -> str:
    """Fixed formatting per unit — the ``_fmt`` convention, unit-aware."""
    if line.unit == PCT_SERIES:
        if not line.values:
            return "n/a"
        return ", ".join(f"{v:.2f}%" for v in line.values)
    if line.value is None or not math.isfinite(line.value):
        return "n/a"
    value = line.value
    if line.unit == USD_M:
        return f"{value:.1f}M USD"
    if line.unit == SHARES_M:
        return f"{value:.1f}M shares"
    if line.unit == USD:
        return f"{value:.2f} USD"
    if line.unit == PCT:
        return f"{value:.2f}%"
    if line.unit == MULTIPLE:
        return f"{value:.2f}x"
    if line.unit == PCTILE:
        return f"{value:.0f}"
    return f"{value:.4f}"


def render_line(line: FundamentalLine) -> str:
    text = f"{line.key}: {format_value(line)}"
    if line.note:
        text = f"{text} ({line.note})"
    if line.source_ref and (line.value is not None or line.values):
        text = f"{text} [{line.source_ref}]"
    return text


# --- fact access (thin provenance wrappers over the catalog helpers) --------


@dataclass(frozen=True)
class FactRow:
    """One resolved fact: its value plus the provenance for the citation."""

    symbol: str
    tag: str
    period_end: pd.Timestamp
    filed_at: pd.Timestamp
    value: float

    @property
    def source_ref(self) -> str:
        return source_ref(self.symbol, self.tag, self.period_end, self.filed_at)


def source_ref(
    symbol: str,
    tag: str,
    period_end: pd.Timestamp | Sequence[pd.Timestamp],
    filed_at: pd.Timestamp,
) -> str:
    """``xbrl:WMT:Revenues:2026-04-30:filed-2026-05-29`` (a range of
    period_ends collapses to ``first..last`` for TTM aggregates)."""
    if isinstance(period_end, pd.Timestamp):
        period = period_end.date().isoformat()
    else:
        periods = list(period_end)
        period = (
            f"{periods[0].date().isoformat()}..{periods[-1].date().isoformat()}"
            if len(periods) > 1
            else periods[0].date().isoformat()
        )
    return f"xbrl:{symbol}:{tag}:{period}:filed-{filed_at.date().isoformat()}"


def _ts(value: Any) -> pd.Timestamp:
    """``pd.Timestamp`` from a frame/index value (pandas stubs widen these to
    ``Timestamp | NaTType | Hashable``; the callers only ever pass real
    timestamps, and a NaT would simply fail the window checks downstream)."""
    return cast(pd.Timestamp, pd.Timestamp(cast(Any, value)))


def symbol_facts(facts: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One symbol's point-in-time rows (the frame is universe-wide)."""
    if facts.empty:
        return facts
    return facts.loc[facts["symbol"] == symbol]


def _tag_rows(facts: pd.DataFrame, symbol: str, tag: str) -> pd.DataFrame:
    if facts.empty:
        return facts
    mask = (facts["symbol"] == symbol) & (facts["tag"] == tag)
    return facts.loc[mask]


def latest_row(facts: pd.DataFrame, symbol: str, tag: str, as_of: pd.Timestamp) -> FactRow | None:
    """Provenance-carrying twin of ``catalog.latest_value`` (same ordering:
    latest ``filed_at``, then latest ``period_end`` within that filing). The
    value is taken from ``latest_value`` itself so the two can never drift."""
    sub = _tag_rows(facts, symbol, tag)
    if sub.empty:
        return None
    value = latest_value(facts, symbol, tag, as_of)
    if not math.isfinite(value):
        return None
    ordered = sub.sort_values(["filed_at", "period_end"])
    row = ordered.iloc[-1]
    return FactRow(
        symbol=symbol,
        tag=tag,
        period_end=_ts(row["period_end"]),
        filed_at=_ts(row["filed_at"]),
        value=float(value),
    )


def _last_filed_at(facts: pd.DataFrame, symbol: str, tag: str) -> pd.Timestamp | None:
    sub = _tag_rows(facts, symbol, tag)
    if sub.empty:
        return None
    return _ts(sub["filed_at"].max())


# --- obstacle (a): period-type inference -----------------------------------


def annual_period_ends(
    facts: pd.DataFrame, symbol: str, as_of: pd.Timestamp, tag: str
) -> frozenset[pd.Timestamp]:
    """The ``period_end`` values of ``tag`` that carry a FISCAL-YEAR figure.

    A period qualifies when its three predecessors sit inside a year, are all
    positive, and its own value is at least :data:`ANNUAL_MAGNITUDE_RATIO`
    times the largest of them. A genuine Q4 filed at the fiscal-year end fails
    that test (it is ~1x, not ~4x) and correctly stays a quarter.
    """
    series = quarterly_series(facts, symbol, tag, as_of)
    if len(series) < 4:
        return frozenset()
    periods = [_ts(p) for p in series.index]
    found: set[pd.Timestamp] = set()
    for i in range(3, len(periods)):
        period = periods[i]
        prior = periods[i - 3 : i]
        if (period - prior[0]).days > TTM_MAX_SPAN_DAYS:
            continue
        if (period - prior[-1]).days < 40:
            continue
        prior_values = [float(series.iloc[j]) for j in range(i - 3, i)]
        if min(prior_values) <= 0.0:
            continue
        if float(series.iloc[i]) >= ANNUAL_MAGNITUDE_RATIO * max(prior_values):
            found.add(period)
    return frozenset(found)


def fiscal_year_end_month(
    facts: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    *,
    flow_tags: Sequence[str] = FLOW_TAGS,
) -> int | None:
    """The month a fiscal year closes in, inferred from the filed facts.

    Every flow tag is scanned for fiscal-year rows (:func:`annual_period_ends`)
    and for annual-only cadence; the month those land in wins by majority.
    ``None`` when nothing votes — the caller then renders ``n/a`` instead of
    guessing a calendar.
    """
    votes: dict[int, int] = {}
    for tag in flow_tags:
        series = quarterly_series(facts, symbol, tag, as_of)
        if series.empty:
            continue
        months: list[int] = []
        if _annual_cadence(series):
            months = [int(_ts(p).month) for p in series.index]
        else:
            months = [int(p.month) for p in annual_period_ends(facts, symbol, as_of, tag)]
        for month in months:
            votes[month] = votes.get(month, 0) + 1
    if not votes:
        return None
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _annual_cadence(series: pd.Series) -> bool:
    """A series filed once a year (no quarterly rows at all)."""
    if len(series) < 2:
        return False
    periods = [_ts(p) for p in series.index]
    gaps = [(b - a).days for a, b in pairwise(periods)]
    return all(gap > 300 for gap in gaps)


def _fiscal_quarter(period: pd.Timestamp, fye_month: int) -> int:
    """1..4 — which fiscal quarter ``period`` closes (4 = the year end)."""
    offset = (period.month - fye_month) % 12
    index = round(offset / 3.0) % 4
    return 4 if index == 0 else index


def _fiscal_year_key(period: pd.Timestamp, fye_month: int) -> tuple[int, int]:
    """A sortable fiscal-year key: the calendar year the fiscal year ends in."""
    quarter = _fiscal_quarter(period, fye_month)
    ends = _ts(period + pd.DateOffset(months=3 * (4 - quarter)))
    return (int(ends.year), fye_month)


SHAPE_DISCRETE = "discrete"  # each row is one quarter
SHAPE_CUMULATIVE = "cumulative"  # each row is year-to-date (10-Q cash flow)
SHAPE_ANNUAL_AT_Q4 = "annual_at_q4"  # Q1-Q3 quarterly, the Q4 row is the FY
SHAPE_ANNUAL_ONLY = "annual_only"  # one row per fiscal year
SHAPE_UNKNOWN = "unknown"


def infer_shape(series: pd.Series, fye_month: int | None) -> str:
    """Classify how one tag's filed values are scoped.

    The SEC's company-facts feed distinguishes these by the fact's ``start``
    date, which this platform's ``xbrl_facts`` table does not store — so the
    scope is recovered from the numbers themselves, per tag, because one
    filing mixes them: CAT's 2025 10-K reports ``Revenues`` as a discrete Q4,
    ``OperatingIncomeLoss`` as the full year, and
    ``NetCashProvidedByUsedInOperatingActivities`` as year-to-date.
    """
    if series.empty:
        return SHAPE_UNKNOWN
    if _annual_cadence(series):
        return SHAPE_ANNUAL_ONLY
    if fye_month is None:
        return SHAPE_UNKNOWN
    by_year: dict[tuple[int, int], dict[int, float]] = {}
    for period, value in series.items():
        stamp = _ts(period)
        by_year.setdefault(_fiscal_year_key(stamp, fye_month), {})[
            _fiscal_quarter(stamp, fye_month)
        ] = float(value)
    complete = [year for year in sorted(by_year) if len(by_year[year]) == 4]
    if not complete:
        return SHAPE_DISCRETE
    # Classify every complete fiscal year, then take the majority (ties go to
    # the most recent year): one odd year must not flip a whole tag.
    verdicts = [_year_shape(by_year[year]) for year in complete]
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    best = max(counts.values())
    tied = {shape for shape, n in counts.items() if n == best}
    if len(tied) == 1:
        return tied.pop()
    for verdict in reversed(verdicts):
        if verdict in tied:
            return verdict
    return SHAPE_DISCRETE  # pragma: no cover - unreachable, verdicts is non-empty


def _year_shape(quarters: Mapping[int, float]) -> str:
    """Shape verdict for ONE complete fiscal year of a tag.

    Year-to-date values climb with decaying ratios (~2.0, ~1.5, ~1.33); a
    fiscal-year total sitting in the Q4 slot towers over three flat quarters;
    anything else is four discrete quarters.
    """
    values = [quarters[i] for i in (1, 2, 3, 4)]
    if any(v == 0.0 for v in values):
        return SHAPE_DISCRETE
    if len({v > 0 for v in values}) > 1:
        return SHAPE_DISCRETE  # sign flip: not a cumulative or annual pattern
    q1, q2, q3, q4 = (abs(v) for v in values)
    cumulative = q4 >= 2.5 * q1 and q2 >= 1.5 * q1 and q3 >= 1.2 * q2 and q4 >= 1.05 * q3
    if cumulative:
        return SHAPE_CUMULATIVE
    # Annual totals are compared against the MEDIAN of the first three
    # quarters, not the max: JNJ's Q1-2025 net income was twice its other
    # quarters, which is enough to hide a fiscal-year Q4 row behind a
    # max-based threshold (and then double-count a year into TTM).
    median = sorted((q1, q2, q3))[1]
    if q4 >= 2.5 * median and q4 >= 1.5 * max(q1, q2, q3):
        return SHAPE_ANNUAL_AT_Q4
    return SHAPE_DISCRETE


@dataclass(frozen=True)
class NormalizedFacts:
    """``pit_facts``-shaped frame whose flow rows have been rewritten into
    unambiguous single quarters, so the catalog's ``ttm_sum`` /
    ``quarterly_series`` can be used directly. ``shapes`` records how each tag
    was read; ``modes`` reports whether a tag ended up quarterly, annual or
    unusable."""

    frame: pd.DataFrame
    modes: Mapping[str, str]
    shapes: Mapping[str, str]
    fye_month: int | None

    def mode(self, tag: str) -> str:
        return self.modes.get(tag, _MODE_NONE)

    def shape(self, tag: str) -> str:
        return self.shapes.get(tag, SHAPE_UNKNOWN)


def normalized_facts(
    facts: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    *,
    flow_tags: Sequence[str] = FLOW_TAGS,
) -> NormalizedFacts:
    """Rewrite ``symbol``'s flow rows into single quarters.

    ``annual_at_q4`` rows become ``FY - (Q1+Q2+Q3)``; ``cumulative`` rows
    become ``YTD(q) - YTD(q-1)``. A row whose predecessors within the fiscal
    year are missing is DROPPED, never guessed. A concept that must be
    non-negative (revenue, cost of revenue, capex) but comes out negative
    after the rewrite marks the tag unusable — the section renders ``n/a``
    rather than a number nobody can defend. Stock (balance-sheet) rows and
    every other symbol pass through untouched.
    """
    if facts.empty:
        return NormalizedFacts(frame=facts, modes={}, shapes={}, fye_month=None)
    # One symbol's rows only: every consumer of the result reads this symbol,
    # and narrowing here keeps the per-tag scans cheap on a 900k-row table.
    facts = symbol_facts(facts, symbol)
    if facts.empty:
        return NormalizedFacts(frame=facts, modes={}, shapes={}, fye_month=None)
    fye_month = fiscal_year_end_month(facts, symbol, as_of, flow_tags=flow_tags)
    modes: dict[str, str] = {}
    shapes: dict[str, str] = {}
    replacements: list[dict[str, object]] = []
    drop_index: list[object] = []
    for tag in flow_tags:
        series = quarterly_series(facts, symbol, tag, as_of)
        if series.empty:
            modes[tag] = _MODE_NONE
            shapes[tag] = SHAPE_UNKNOWN
            continue
        shape = infer_shape(series, fye_month)
        shapes[tag] = shape
        sub = _tag_rows(facts, symbol, tag)
        if shape == SHAPE_ANNUAL_ONLY:
            modes[tag] = _MODE_ANNUAL
            continue
        if shape == SHAPE_UNKNOWN:
            modes[tag] = _MODE_NONE
            continue
        modes[tag] = _MODE_QUARTERLY
        derived = _derive_quarters(series, shape, fye_month)
        for period in series.index:
            drop_index.extend(sub.index[sub["period_end"] == period].tolist())
        for period, value in derived.items():
            replacements.append(
                {
                    "symbol": symbol,
                    "tag": tag,
                    "period_end": period,
                    "value": value,
                    "filed_at": _ts(sub.loc[sub["period_end"] == period, "filed_at"].max()),
                }
            )
        if not derived or (tag in NON_NEGATIVE_TAGS and any(v < 0 for v in derived.values())):
            modes[tag] = _MODE_NONE  # the rewrite produced nonsense — say nothing
    frame = facts
    if drop_index:
        frame = frame.drop(index=drop_index)
    if replacements:
        frame = pd.concat([frame, pd.DataFrame(replacements)], ignore_index=True)
    return NormalizedFacts(frame=frame, modes=modes, shapes=shapes, fye_month=fye_month)


def _derive_quarters(
    series: pd.Series, tag_shape: str, fye_month: int | None
) -> dict[pd.Timestamp, float]:
    """period_end -> single-quarter value, decided FISCAL YEAR BY FISCAL YEAR.

    A tag's scope changes over time in the real feed: ACN's ``Revenues`` are
    fiscal-year totals through FY2023 and discrete quarters from FY2024;
    CAT's are annual through 2023 and quarterly after. A single verdict for
    the whole series is therefore wrong for part of it — each complete year
    is classified on its own numbers, incomplete years inherit the tag-level
    verdict, and a lone leftover annual row (an incomplete year towering over
    the derived quarters) is dropped rather than passed off as a quarter.
    """
    if fye_month is None:
        return {}
    by_year: dict[tuple[int, int], dict[int, tuple[pd.Timestamp, float]]] = {}
    for period, value in series.items():
        stamp = _ts(period)
        key = _fiscal_year_key(stamp, fye_month)
        by_year.setdefault(key, {})[_fiscal_quarter(stamp, fye_month)] = (stamp, float(value))
    out: dict[pd.Timestamp, float] = {}
    passthrough: dict[pd.Timestamp, float] = {}
    for key in sorted(by_year):
        quarters = by_year[key]
        complete = len(quarters) == 4
        shape = _year_shape({q: v for q, (_, v) in quarters.items()}) if complete else tag_shape
        for q, (stamp, value) in quarters.items():
            if shape == SHAPE_ANNUAL_AT_Q4:
                if q != 4:
                    out[stamp] = value
                    continue
                prior = [quarters.get(i) for i in (1, 2, 3)]
                if any(p is None for p in prior):
                    passthrough[stamp] = value
                    continue
                out[stamp] = value - sum(p[1] for p in prior if p is not None)
            elif shape == SHAPE_CUMULATIVE:
                if q == 1:
                    out[stamp] = value
                    continue
                previous = quarters.get(q - 1)
                if previous is None:
                    continue
                out[stamp] = value - previous[1]
            else:
                if complete:
                    out[stamp] = value
                else:
                    passthrough[stamp] = value
    return _merge_passthrough(out, passthrough)


def _merge_passthrough(
    derived: Mapping[pd.Timestamp, float], passthrough: Mapping[pd.Timestamp, float]
) -> dict[pd.Timestamp, float]:
    """Admit rows from incomplete fiscal years only when they look like the
    quarters around them — a value >= 2.5x the median derived quarter is a
    fiscal-year total in disguise and is dropped."""
    out = dict(derived)
    if not passthrough:
        return out
    scale = float(pd.Series([abs(v) for v in derived.values()]).median()) if derived else 0.0
    for period, value in passthrough.items():
        if scale > 0.0 and abs(value) >= 2.5 * scale:
            continue
        out[period] = value
    return out


# --- obstacle (c): tag resolution ------------------------------------------


def first_available_tag(
    facts: pd.DataFrame, symbol: str, tags: Sequence[str], as_of: pd.Timestamp
) -> str | None:
    """First candidate tag with at least one point-in-time row for ``symbol``.

    This is what saves CAT, which has zero ``StockholdersEquity`` rows and
    reports equity only under the
    ``...IncludingPortionAttributableToNoncontrollingInterest`` variant.
    """
    for tag in tags:
        if not _tag_rows(facts, symbol, tag).empty:
            return tag
    return None


def usable_tag(
    norm: NormalizedFacts, symbol: str, tags: Sequence[str], as_of: pd.Timestamp
) -> str | None:
    """:func:`first_available_tag` restricted to tags whose period scope was
    resolved — a tag the rewrite could not make sense of is skipped so the
    next candidate (or ``n/a``) wins instead of a wrong number."""
    for tag in tags:
        if _tag_rows(norm.frame, symbol, tag).empty:
            continue
        if norm.mode(tag) == _MODE_NONE:
            continue
        return tag
    return None


# --- TTM aggregates --------------------------------------------------------


@dataclass(frozen=True)
class TTMValue:
    symbol: str
    tag: str
    value: float
    periods: tuple[pd.Timestamp, ...]
    filed_at: pd.Timestamp

    @property
    def source_ref(self) -> str:
        return source_ref(self.symbol, self.tag, list(self.periods), self.filed_at)


def ttm(
    norm: NormalizedFacts,
    symbol: str,
    tags: Sequence[str],
    as_of: pd.Timestamp,
    *,
    offset_quarters: int = 0,
) -> TTMValue | None:
    """Trailing-twelve-month sum over the normalized quarters (or the latest
    annual value for annual-only tags), with provenance. ``None`` when the
    window is incomplete or does not span a year — never a partial sum."""
    facts = norm.frame
    tag = usable_tag(norm, symbol, tags, as_of)
    if tag is None:
        return None
    series = quarterly_series(facts, symbol, tag, as_of)
    if series.empty:
        return None
    periods = [_ts(p) for p in series.index]
    if norm.mode(tag) == _MODE_ANNUAL:
        index = len(periods) - 1 - (offset_quarters // 4)
        if index < 0:
            return None
        filed = _last_filed_at(facts, symbol, tag)
        if filed is None:
            return None
        return TTMValue(
            value=float(series.iloc[index]),
            tag=tag,
            periods=(periods[index],),
            filed_at=filed,
            symbol=symbol,
        )
    hi = len(periods) - offset_quarters
    lo = hi - 4
    if lo < 0:
        return None
    window = periods[lo:hi]
    span = (window[-1] - window[0]).days
    if not (TTM_MIN_SPAN_DAYS <= span <= TTM_MAX_SPAN_DAYS):
        return None
    if norm.fye_month is not None:
        quarters = {_fiscal_quarter(p, norm.fye_month) for p in window}
        if quarters != {1, 2, 3, 4}:
            return None  # not four distinct fiscal quarters — refuse to sum
    value = ttm_sum(facts, symbol, tag, as_of, offset_quarters)
    if not math.isfinite(value):
        return None
    filed = _last_filed_at(facts, symbol, tag)
    if filed is None:
        return None
    return TTMValue(
        value=float(value), tag=tag, periods=tuple(window), filed_at=filed, symbol=symbol
    )


def quarter_values(
    norm: NormalizedFacts, symbol: str, tags: Sequence[str], as_of: pd.Timestamp, n: int = 4
) -> tuple[tuple[pd.Timestamp, float], ...]:
    """The last ``n`` normalized quarters (oldest -> latest) for a concept."""
    facts = norm.frame
    tag = usable_tag(norm, symbol, tags, as_of)
    if tag is None or norm.mode(tag) == _MODE_ANNUAL:
        return ()
    series = quarterly_series(facts, symbol, tag, as_of)
    if series.empty:
        return ()
    tail = series.iloc[-n:]
    return tuple((_ts(p), float(v)) for p, v in tail.items())


# --- valuation helpers ------------------------------------------------------


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    ratio = _safe_ratio(numerator, denominator)
    return None if ratio is None else ratio * 100.0


@dataclass(frozen=True)
class _Valuation:
    market_cap: float | None = None
    pe: float | None = None
    ps: float | None = None
    pb: float | None = None
    fcf_yield: float | None = None


def _valuation_for(
    facts: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    close: float | None,
) -> _Valuation:
    """Multiples from ``close`` x shares outstanding — the same inputs the
    quant path uses (``SHARES_TAGS`` preference order), so the dossier and the
    model cannot disagree about market cap."""
    if close is None or not math.isfinite(close):
        return _Valuation()
    facts = symbol_facts(facts, symbol)
    norm = normalized_facts(facts, symbol, as_of)
    shares, _shares_ref = resolve_shares(facts, norm, symbol, as_of)
    if shares is None:
        return _Valuation()
    market_cap = close * shares
    revenue = ttm(norm, symbol, REVENUE_TAGS, as_of)
    net_income = ttm(norm, symbol, NET_INCOME_TAGS, as_of)
    equity_tag = first_available_tag(norm.frame, symbol, EQUITY_TAGS, as_of)
    equity = latest_row(norm.frame, symbol, equity_tag, as_of) if equity_tag is not None else None
    fcf = _fcf(norm, symbol, as_of, None if revenue is None else revenue.value)
    return _Valuation(
        market_cap=market_cap,
        pe=_safe_ratio(market_cap, None if net_income is None else net_income.value),
        ps=_safe_ratio(market_cap, None if revenue is None else revenue.value),
        pb=_safe_ratio(market_cap, None if equity is None else equity.value),
        fcf_yield=_pct(None if fcf is None else fcf[0], market_cap),
    )


def _shares_row(facts: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> FactRow | None:
    """Shares outstanding via the catalog's ``SHARES_TAGS`` preference order,
    rejecting rows below :data:`MIN_PLAUSIBLE_SHARES`.

    That floor is not paranoia: MCD's ``WeightedAverageNumberOfSharesOutstanding
    Basic`` switches from 744,600,000 (2020) to 713.4 (2026) — the same tag,
    silently rescaled to millions upstream. Multiplying the second by a share
    price prices the company at $0.7bn, so the row is refused and the caller
    falls back to net income / diluted EPS or renders ``n/a``.
    """
    for tag in SHARES_TAGS:
        row = latest_row(facts, symbol, tag, as_of)
        if row is not None and row.value >= MIN_PLAUSIBLE_SHARES:
            return row
    return None


def _cfo(
    norm: NormalizedFacts, symbol: str, as_of: pd.Timestamp, revenue: float | None
) -> TTMValue | None:
    """TTM operating cash flow, refused when it is implausible against
    revenue (see :data:`MAX_CFO_REVENUE_MULTIPLE`)."""
    value = ttm(norm, symbol, CFO_TAGS, as_of)
    if value is None:
        return None
    if (
        revenue is not None
        and revenue > 0
        and (abs(value.value) > MAX_CFO_REVENUE_MULTIPLE * revenue)
    ):
        return None
    return value


def resolve_shares(
    facts: pd.DataFrame, norm: NormalizedFacts, symbol: str, as_of: pd.Timestamp
) -> tuple[float | None, str]:
    """(shares outstanding, source ref).

    Filed share counts first (``SHARES_TAGS`` order, plausibility-floored);
    when every filed row is unusable, the diluted count implied by
    ``net income / diluted EPS`` — arithmetic on two cited facts, not a guess.
    """
    row = _shares_row(facts, symbol, as_of)
    if row is not None:
        return row.value, row.source_ref
    net_income = ttm(norm, symbol, NET_INCOME_TAGS, as_of)
    eps = ttm(norm, symbol, EPS_DILUTED_TAGS, as_of)
    if net_income is None or eps is None or eps.value == 0.0:
        return None, ""
    implied = net_income.value / eps.value
    if not math.isfinite(implied) or implied < MIN_PLAUSIBLE_SHARES:
        return None, ""
    return implied, f"derived:{net_income.source_ref} / {eps.source_ref}"


def _fcf(
    norm: NormalizedFacts, symbol: str, as_of: pd.Timestamp, revenue: float | None = None
) -> tuple[float, str, TTMValue] | None:
    """(free cash flow, source ref, the CFO aggregate it came from)."""
    cfo = _cfo(norm, symbol, as_of, revenue)
    if cfo is None:
        return None
    capex = ttm(norm, symbol, CAPEX_TAGS, as_of)
    value = cfo.value - (0.0 if capex is None else capex.value)
    refs = cfo.source_ref if capex is None else f"{cfo.source_ref} + {capex.source_ref}"
    return value, f"derived:{refs}", cfo


def _percentile(value: float | None, peers: Sequence[float]) -> float | None:
    """Ascending percentile of ``value`` within ``peers`` (0 = lowest
    multiple in the sector). ``None`` below :data:`MIN_PEERS_FOR_PERCENTILE`."""
    if value is None or not math.isfinite(value):
        return None
    finite = [p for p in peers if math.isfinite(p)]
    if len(finite) < MIN_PEERS_FOR_PERCENTILE:
        return None
    below = sum(1 for p in finite if p < value)
    return 100.0 * below / len(finite)


# --- the builder -----------------------------------------------------------


def build_fundamentals(
    fundamentals: pd.DataFrame | None,
    *,
    symbol: str,
    as_of: date,
    close: float | None = None,
    sector: str | None = None,
    peer_closes: Mapping[str, float] | None = None,
    peer_sectors: Mapping[str, str] | None = None,
) -> FundamentalsSnapshot:
    """The ``## Fundamentals`` snapshot for one symbol at one date.

    ``fundamentals`` is the long ``xbrl_facts`` frame
    ``[symbol, tag, period_end, value, filed_at]`` (the replay gateway's
    ``feature_inputs(...).fundamentals``, or the equivalent live read). Rows
    filed after ``as_of`` are dropped by ``pit_facts`` before anything is
    computed. Missing inputs render ``n/a``; nothing raises.
    """
    stamp = _ts(as_of)
    empty = FundamentalsSnapshot(symbol=symbol, as_of=as_of, groups=())
    try:
        facts = pit_facts(fundamentals, stamp)
    except Exception:  # pragma: no cover - defensive: never break the dossier
        return empty
    if facts.empty:
        return empty
    try:
        return _build(
            facts,
            symbol=symbol,
            as_of=as_of,
            stamp=stamp,
            close=close,
            sector=sector,
            peer_closes=peer_closes or {},
            peer_sectors=peer_sectors or {},
        )
    except Exception:  # pragma: no cover - defensive: never break the dossier
        return empty


def _line(
    key: str,
    unit: str,
    value: float | None,
    ref: str,
    *,
    note: str = "",
) -> FundamentalLine:
    finite = value if value is not None and math.isfinite(value) else None
    return FundamentalLine(
        key=key,
        unit=unit,
        value=finite,
        source_ref=ref if finite is not None else "",
        note=note,
    )


def _build(
    facts: pd.DataFrame,
    *,
    symbol: str,
    as_of: date,
    stamp: pd.Timestamp,
    close: float | None,
    sector: str | None,
    peer_closes: Mapping[str, float],
    peer_sectors: Mapping[str, str],
) -> FundamentalsSnapshot:
    norm = normalized_facts(facts, symbol, stamp)
    frame = norm.frame

    revenue = ttm(norm, symbol, REVENUE_TAGS, stamp)
    revenue_prior = ttm(norm, symbol, REVENUE_TAGS, stamp, offset_quarters=4)
    revenue_3y = ttm(norm, symbol, REVENUE_TAGS, stamp, offset_quarters=12)
    gross = ttm(norm, symbol, GROSS_PROFIT_TAGS, stamp)
    if gross is None:
        cogs = ttm(norm, symbol, COGS_TAGS, stamp)
        if revenue is not None and cogs is not None:
            gross = TTMValue(
                value=revenue.value - cogs.value,
                tag=f"{revenue.tag}-{cogs.tag}",
                periods=revenue.periods,
                filed_at=max(revenue.filed_at, cogs.filed_at),
                symbol=symbol,
            )
    operating = ttm(norm, symbol, OPERATING_INCOME_TAGS, stamp)
    net_income = ttm(norm, symbol, NET_INCOME_TAGS, stamp)
    eps = ttm(norm, symbol, EPS_DILUTED_TAGS, stamp)

    revenue_value = None if revenue is None else revenue.value
    yoy = (
        None
        if revenue is None or revenue_prior is None or revenue_prior.value <= 0
        else (revenue.value / revenue_prior.value - 1.0) * 100.0
    )
    cagr = (
        None
        if revenue is None or revenue_3y is None or revenue_3y.value <= 0
        else ((revenue.value / revenue_3y.value) ** (1.0 / 3.0) - 1.0) * 100.0
    )

    income = [
        _line(
            "revenue_ttm",
            USD_M,
            None if revenue is None else revenue.value / 1e6,
            "" if revenue is None else revenue.source_ref,
        ),
        _line(
            "revenue_growth_yoy",
            PCT,
            yoy,
            ""
            if revenue is None or revenue_prior is None
            else f"derived:{revenue.source_ref} vs {revenue_prior.source_ref}",
        ),
        _line(
            "revenue_cagr_3y",
            PCT,
            cagr,
            ""
            if revenue is None or revenue_3y is None
            else f"derived:{revenue.source_ref} vs {revenue_3y.source_ref}",
        ),
        _line(
            "gross_profit_ttm",
            USD_M,
            None if gross is None else gross.value / 1e6,
            "" if gross is None else gross.source_ref,
        ),
        _line(
            "gross_margin_ttm",
            PCT,
            _pct(None if gross is None else gross.value, revenue_value),
            "" if gross is None or revenue is None else f"derived:{gross.source_ref}",
        ),
        _line(
            "operating_income_ttm",
            USD_M,
            None if operating is None else operating.value / 1e6,
            "" if operating is None else operating.source_ref,
        ),
        _line(
            "operating_margin_ttm",
            PCT,
            _pct(None if operating is None else operating.value, revenue_value),
            "" if operating is None or revenue is None else f"derived:{operating.source_ref}",
        ),
        _line(
            "net_income_ttm",
            USD_M,
            None if net_income is None else net_income.value / 1e6,
            "" if net_income is None else net_income.source_ref,
        ),
        _line(
            "net_margin_ttm",
            PCT,
            _pct(None if net_income is None else net_income.value, revenue_value),
            "" if net_income is None or revenue is None else f"derived:{net_income.source_ref}",
        ),
        _line(
            "eps_diluted_ttm",
            USD,
            None if eps is None else eps.value,
            "" if eps is None else eps.source_ref,
        ),
    ]
    income.extend(_margin_trend_lines(norm, symbol, stamp, revenue_tag_periods=4))

    # --- balance sheet ---
    balance: list[FundamentalLine] = []
    assets = _resolved_row(frame, symbol, ASSETS_TAGS, stamp)
    equity = _resolved_row(frame, symbol, EQUITY_TAGS, stamp)
    liabilities = _resolved_row(frame, symbol, LIABILITIES_TAGS, stamp)
    liabilities_ref = "" if liabilities is None else liabilities.source_ref
    liabilities_value = None if liabilities is None else liabilities.value
    if liabilities is None and assets is not None and equity is not None:
        liabilities_value = assets.value - equity.value
        liabilities_ref = f"derived:{assets.source_ref} - {equity.source_ref}"
    debt = _resolved_row(frame, symbol, LONG_TERM_DEBT_TAGS, stamp)
    cash = _resolved_row(frame, symbol, CASH_TAGS, stamp)
    balance += [
        _line(
            "total_assets",
            USD_M,
            None if assets is None else assets.value / 1e6,
            "" if assets is None else assets.source_ref,
        ),
        _line(
            "total_liabilities",
            USD_M,
            None if liabilities_value is None else liabilities_value / 1e6,
            liabilities_ref,
        ),
        _line(
            "shareholders_equity",
            USD_M,
            None if equity is None else equity.value / 1e6,
            "" if equity is None else equity.source_ref,
        ),
        _line(
            "long_term_debt",
            USD_M,
            None if debt is None else debt.value / 1e6,
            "" if debt is None else debt.source_ref,
        ),
        _line(
            "cash_and_equivalents",
            USD_M,
            None if cash is None else cash.value / 1e6,
            "" if cash is None else cash.source_ref,
        ),
        _line(
            "debt_to_equity",
            MULTIPLE,
            _safe_ratio(
                None if debt is None else debt.value,
                None if equity is None or equity.value <= 0 else equity.value,
            ),
            ""
            if debt is None or equity is None
            else f"derived:{debt.source_ref} / {equity.source_ref}",
        ),
    ]

    # --- cash flow ---
    cfo = _cfo(norm, symbol, stamp, revenue_value)
    capex = ttm(norm, symbol, CAPEX_TAGS, stamp)
    fcf = _fcf(norm, symbol, stamp, revenue_value)
    cash_flow = [
        _line(
            "cfo_ttm",
            USD_M,
            None if cfo is None else cfo.value / 1e6,
            "" if cfo is None else cfo.source_ref,
        ),
        _line(
            "capex_ttm",
            USD_M,
            None if capex is None else capex.value / 1e6,
            "" if capex is None else capex.source_ref,
        ),
        _line(
            "fcf_ttm",
            USD_M,
            None if fcf is None else fcf[0] / 1e6,
            "" if fcf is None else fcf[1],
        ),
        _line(
            "fcf_margin_ttm",
            PCT,
            _pct(None if fcf is None else fcf[0], revenue_value),
            "" if fcf is None or revenue is None else fcf[1],
        ),
    ]

    # --- valuation + sector percentiles ---
    shares, shares_ref = resolve_shares(facts, norm, symbol, stamp)
    valuation_self = _valuation_for(facts, symbol, stamp, close)
    peers = _peer_valuations(
        facts,
        symbol=symbol,
        sector=sector,
        stamp=stamp,
        peer_closes=peer_closes,
        peer_sectors=peer_sectors,
    )
    peer_note = "" if sector is None else f"vs {len(peers)} {sector} peers, ascending"
    close_ref = f"derived:close@{as_of.isoformat()}"
    valuation = [
        _line("close", USD, close, close_ref),
        _line(
            "shares_outstanding",
            SHARES_M,
            None if shares is None else shares / 1e6,
            shares_ref,
        ),
        _line(
            "market_cap",
            USD_M,
            None if valuation_self.market_cap is None else valuation_self.market_cap / 1e6,
            f"{close_ref} x {shares_ref}" if shares is not None else "",
        ),
    ]
    for key, value, basis in (
        ("pe_ttm", valuation_self.pe, "net_income_ttm"),
        ("ps_ttm", valuation_self.ps, "revenue_ttm"),
        ("pb", valuation_self.pb, "shareholders_equity"),
        ("fcf_yield", valuation_self.fcf_yield, "fcf_ttm"),
    ):
        unit = PCT if key == "fcf_yield" else MULTIPLE
        valuation.append(_line(key, unit, value, f"derived:market_cap / {basis}"))
        peer_values = [
            float(peer_value)
            for peer in peers
            if (peer_value := getattr(peer, _VALUATION_ATTR[key])) is not None
        ]
        pct_rank = _percentile(value, peer_values)
        valuation.append(
            _line(
                f"{key}_sector_pctile",
                PCTILE,
                pct_rank,
                f"derived:{key} vs sector peers",
                note=peer_note,
            )
        )

    groups = (
        FundamentalsGroup(title="Income statement (TTM)", lines=tuple(income)),
        FundamentalsGroup(title="Balance sheet (latest filed)", lines=tuple(balance)),
        FundamentalsGroup(title="Cash flow (TTM)", lines=tuple(cash_flow)),
        FundamentalsGroup(title="Valuation", lines=tuple(valuation)),
    )
    return FundamentalsSnapshot(symbol=symbol, as_of=as_of, groups=groups)


_VALUATION_ATTR = {"pe_ttm": "pe", "ps_ttm": "ps", "pb": "pb", "fcf_yield": "fcf_yield"}


def _resolved_row(
    facts: pd.DataFrame, symbol: str, tags: Sequence[str], stamp: pd.Timestamp
) -> FactRow | None:
    tag = first_available_tag(facts, symbol, tags, stamp)
    return None if tag is None else latest_row(facts, symbol, tag, stamp)


def _margin_trend_lines(
    norm: NormalizedFacts, symbol: str, stamp: pd.Timestamp, *, revenue_tag_periods: int
) -> list[FundamentalLine]:
    """Gross/operating/net margin over the last four normalized quarters."""
    revenue_quarters = dict(quarter_values(norm, symbol, REVENUE_TAGS, stamp, revenue_tag_periods))
    lines: list[FundamentalLine] = []
    for key, tags in (
        ("gross_margin_trend_4q", GROSS_PROFIT_TAGS),
        ("operating_margin_trend_4q", OPERATING_INCOME_TAGS),
        ("net_margin_trend_4q", NET_INCOME_TAGS),
    ):
        quarters = quarter_values(norm, symbol, tags, stamp, revenue_tag_periods)
        values: list[float] = []
        for period, value in quarters:
            revenue = revenue_quarters.get(period)
            margin = _pct(value, revenue)
            if margin is None:
                values = []
                break
            values.append(margin)
        if not values:
            lines.append(FundamentalLine(key=key, unit=PCT_SERIES, values=()))
            continue
        tag = first_available_tag(norm.frame, symbol, tags, stamp)
        periods = [p for p, _ in quarters]
        filed = _last_filed_at(norm.frame, symbol, tag or "")
        ref = (
            ""
            if tag is None or filed is None
            else f"derived:{source_ref(symbol, tag, periods, filed)} / revenue_ttm quarters"
        )
        lines.append(
            FundamentalLine(
                key=key,
                unit=PCT_SERIES,
                values=tuple(values),
                source_ref=ref,
                note="oldest->latest",
            )
        )
    return lines


def _peer_valuations(
    facts: pd.DataFrame,
    *,
    symbol: str,
    sector: str | None,
    stamp: pd.Timestamp,
    peer_closes: Mapping[str, float],
    peer_sectors: Mapping[str, str],
) -> list[_Valuation]:
    if sector is None:
        return []
    peers = sorted(
        sym
        for sym, sec in peer_sectors.items()
        if sec == sector and sym != symbol and sym in peer_closes
    )
    out: list[_Valuation] = []
    for peer in peers:
        try:
            out.append(_valuation_for(facts, peer, stamp, peer_closes.get(peer)))
        except Exception:  # pragma: no cover - a bad peer never breaks a dossier
            continue
    return out
