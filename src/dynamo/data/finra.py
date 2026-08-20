"""FINRA consolidated short-interest adapter — a free, key-less data source.

Design: doc 02 §1 (provider matrix) and §3 (point-in-time keys). Structure,
pacing and error mapping follow ``dynamo.data.edgar`` and ``dynamo.data.fred``:
one :class:`~dynamo.data.ratelimit.TokenBucket` per provider, transport and
5xx failures normalized to :class:`~dynamo.data.errors.ProviderDown`, 429 to
:class:`~dynamo.data.errors.RateLimited`, an empty payload to
:class:`~dynamo.data.errors.DataGap`.

Endpoint (public, no API key, no registration)::

    GET  https://api.finra.org/partitions/group/otcMarket/name/consolidatedShortInterest
    POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest

The partitions endpoint lists every settlement date FINRA currently serves
(a rolling ~2 years, two cycles a month). The data endpoint returns CSV — one
row per (symbol, settlement date) — and is filtered by settlement date with a
JSON body; ``limit``/``offset`` page through it. Columns, verbatim from
FINRA's own metadata endpoint:

``accountingYearMonthNumber, symbolCode, issueName,
issuerServicesGroupExchangeCode, marketClassCode, currentShortPositionQuantity,
previousShortPositionQuantity, stockSplitFlag, averageDailyVolumeQuantity,
daysToCoverQuantity, revisionFlag, changePercent, changePreviousNumber,
settlementDate``

POINT-IN-TIME CORRECTNESS — THE THING THAT MATTERS HERE
=======================================================

A short-interest row describes positions as of its SETTLEMENT DATE, but FINRA
does not publish it until roughly eight days later, once member firms have
reported and the figures are consolidated. Reading the row on its settlement
date is therefore look-ahead: on 2026-07-31 nobody could see the 2026-07-31
short interest, and a backtest that used it would be trading on information
that did not exist for another week and a half.

So both dates are stored, and every read is bounded on ``publication_date``,
never on ``settlement_date``. :func:`publication_date_for` derives the
publication date from the settlement date with a documented
``PUBLICATION_LAG_DAYS`` offset rolled forward to the next weekday. That is an
approximation of FINRA's published reporting-deadline calendar, which shifts
around holidays; callers holding the exact calendar can pass their own
``publication_dates`` mapping to :meth:`FinraClient.short_interest` and the
adapter will use it verbatim. The approximation is only ever used to make data
visible LATER than the settlement date, never earlier, so the failure mode is
a slightly stale feature rather than a leaked one.

This module is an adapter: it does network I/O and returns domain models. It
holds no database handle — the CLI composition layer
(``dynamo.cli.backfill_short_interest``) does the writing.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

from dynamo.data.errors import DataGap, ProviderDown, RateLimited
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import ShortInterest

log = structlog.get_logger(__name__)

FINRA_BASE = "https://api.finra.org"
DATASET_GROUP = "otcMarket"
DATASET_NAME = "consolidatedShortInterest"

# FINRA disseminates a cycle's short interest ~8 calendar days after its
# settlement date (Rule 4560 reporting deadline plus consolidation). See the
# module docstring: this default is rolled FORWARD to the next weekday, so it
# can only ever delay visibility, never advance it.
PUBLICATION_LAG_DAYS = 8

# The API caps a single response; 5,000 rows keeps each request comfortably
# under it while a full cycle (~12k symbols) takes only a handful of pages.
PAGE_SIZE = 5_000
MAX_PAGES = 20


class _Transient(Exception):
    """Internal marker for retryable failures (never escapes this module)."""


def publication_date_for(settlement: date, *, lag_days: int = PUBLICATION_LAG_DAYS) -> date:
    """The date a settlement cycle first became publicly readable.

    ``settlement + lag_days``, rolled forward off Saturday/Sunday. Deliberately
    conservative: an over-estimate makes a feature stale for a day, while an
    under-estimate injects look-ahead into every backtest that reads it.
    """
    published = settlement + timedelta(days=lag_days)
    while published.weekday() >= 5:  # Sat/Sun -> next Monday
        published += timedelta(days=1)
    return published


def _optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_flag(raw: str | None) -> str | None:
    text = (raw or "").strip()
    return text or None


def parse_short_interest_csv(
    text: str,
    *,
    publication_dates: Mapping[date, date] | None = None,
    lag_days: int = PUBLICATION_LAG_DAYS,
) -> Iterator[ShortInterest]:
    """Parse a FINRA consolidated-short-interest CSV body into domain rows.

    Pure: no network, no clock. Rows missing a symbol, a settlement date or a
    short position are skipped rather than raising — a malformed line in a
    12,000-row file must not lose the other 11,999.

    ``publication_dates`` overrides the derived publication date per settlement
    date (pass FINRA's official deadline calendar when you have it).
    """
    reader = csv.DictReader(io.StringIO(text))
    overrides = dict(publication_dates or {})
    for row in reader:
        symbol = (row.get("symbolCode") or "").strip().upper()
        settlement_raw = (row.get("settlementDate") or "").strip()
        shares_short = _optional_float(row.get("currentShortPositionQuantity"))
        if not symbol or not settlement_raw or shares_short is None:
            continue
        try:
            settlement = date.fromisoformat(settlement_raw[:10])
        except ValueError:
            continue
        published = overrides.get(settlement) or publication_date_for(
            settlement, lag_days=lag_days
        )
        # SEC/FINRA spell share classes with a dot or a space; the platform
        # uses the dot form everywhere (mirrors EdgarClient.company_ticker_map).
        yield ShortInterest(
            symbol=symbol.replace("-", "."),
            settlement_date=settlement,
            publication_date=published,
            shares_short=shares_short,
            previous_shares_short=_optional_float(row.get("previousShortPositionQuantity")),
            avg_daily_volume=_optional_float(row.get("averageDailyVolumeQuantity")),
            days_to_cover=_optional_float(row.get("daysToCoverQuantity")),
            change_pct=_optional_float(row.get("changePercent")),
            market_class=_optional_flag(row.get("marketClassCode")),
            issue_name=_optional_flag(row.get("issueName")),
            split_flag=_optional_flag(row.get("stockSplitFlag")),
            revision_flag=_optional_flag(row.get("revisionFlag")),
        )


class FinraClient:
    """FINRA Query API client for consolidated short interest (no key needed)."""

    def __init__(
        self,
        *,
        user_agent: str = "dynamo/0.1 (research)",
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = FINRA_BASE,
        page_size: int = PAGE_SIZE,
        max_pages: int = MAX_PAGES,
        lag_days: int = PUBLICATION_LAG_DAYS,
    ) -> None:
        self._client = client or httpx.Client(timeout=90.0)
        # FINRA publishes no documented per-second limit for the open Query
        # API; the whole backfill is a few dozen requests, so 2 req/s with a
        # small burst is deliberately gentle.
        self._bucket = bucket or TokenBucket(rate=2.0, capacity=4.0)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._base_url = base_url.rstrip("/")
        self._page_size = page_size
        self._max_pages = max_pages
        self._lag_days = lag_days

    # -- internals ---------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One paced request with a single jittered retry, typed on failure.

        Mirrors :func:`dynamo.data.http.get_json`, which is GET/JSON-only —
        this endpoint needs POST and returns CSV.
        """

        @retry(
            retry=retry_if_exception_type(_Transient),
            stop=stop_after_attempt(2),  # 1 retry (doc 01 §6)
            wait=wait_random(0.2, 1.0),
            reraise=True,
        )
        def _do() -> httpx.Response:
            self._bucket.acquire()
            try:
                resp = self._client.request(method, url, headers=self._headers, **kwargs)
            except httpx.HTTPError as exc:
                log.warning("provider_transport_error", provider="finra", url=url, error=str(exc))
                raise _Transient(str(exc)) from exc
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                log.warning("provider_rate_limited", provider="finra", url=url)
                raise RateLimited(
                    "finra",
                    f"429 from {url}",
                    retry_after_s=float(retry_after) if retry_after else None,
                )
            if resp.status_code >= 500:
                log.warning("provider_5xx", provider="finra", url=url, status=resp.status_code)
                raise _Transient(f"{resp.status_code} from {url}")
            if resp.status_code >= 400:
                raise ProviderDown("finra", f"{resp.status_code} from {url}")
            return resp

        try:
            return _do()
        except _Transient as exc:
            raise ProviderDown("finra", str(exc)) from exc

    # -- public API --------------------------------------------------------

    def settlement_dates(self) -> list[date]:
        """Every settlement cycle FINRA currently serves, newest first."""
        url = f"{self._base_url}/partitions/group/{DATASET_GROUP}/name/{DATASET_NAME}"
        payload = self._request("GET", url).json()
        out: list[date] = []
        for entry in payload.get("availablePartitions", []):
            for value in entry.get("partitions", []):
                try:
                    out.append(date.fromisoformat(str(value)[:10]))
                except ValueError:
                    continue
        if not out:
            raise DataGap("finra", "no available partitions for consolidatedShortInterest")
        return sorted(set(out), reverse=True)

    def published_settlement_dates(self, as_of: date) -> list[date]:
        """Settlement cycles whose PUBLICATION date is on or before ``as_of``.

        The point-in-time accessor: a cycle that has settled but not yet been
        disseminated is invisible, which is the whole reason the publication
        date is tracked.
        """
        return [
            d
            for d in self.settlement_dates()
            if publication_date_for(d, lag_days=self._lag_days) <= as_of
        ]

    def short_interest(
        self,
        settlement_dates: Sequence[date],
        *,
        publication_dates: Mapping[date, date] | None = None,
    ) -> Iterable[ShortInterest]:
        """All rows for the given settlement cycles.

        Each cycle is fetched in ``page_size`` pages until a short page ends
        it. An empty first page for a cycle is a :class:`DataGap` — the
        partitions endpoint said the cycle exists, so nothing is a defect
        rather than a normal outcome.
        """
        url = f"{self._base_url}/data/group/{DATASET_GROUP}/name/{DATASET_NAME}"
        for settlement in settlement_dates:
            total = 0
            for page in range(self._max_pages):
                body = {
                    "limit": self._page_size,
                    "offset": page * self._page_size,
                    "compareFilters": [
                        {
                            "fieldName": "settlementDate",
                            "fieldValue": settlement.isoformat(),
                            "compareType": "EQUAL",
                        }
                    ],
                }
                text = self._request("POST", url, json=body).text
                rows = list(
                    parse_short_interest_csv(
                        text,
                        publication_dates=publication_dates,
                        lag_days=self._lag_days,
                    )
                )
                yield from rows
                total += len(rows)
                if len(rows) < self._page_size:
                    break
            if total == 0:
                raise DataGap("finra", f"no short-interest rows for settlement {settlement}")
            log.info("finra_short_interest_cycle", settlement=settlement.isoformat(), rows=total)
