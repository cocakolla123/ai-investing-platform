"""SEC EDGAR adapter — implements the ``FilingsProvider`` port.

Design: doc 02 §1 (provider matrix: 10 req/s, declared User-Agent required)
and doc 02 §3 (point-in-time fundamentals keyed by ``filed_at``).

Endpoints used (all free, JSON):

- Full-text search index ``https://efts.sec.gov/LATEST/search-index`` for the
  recent-filings poll and Form 4 discovery.
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json`` for XBRL
  company facts (``us-gaap`` tags → :class:`XbrlFact`).
- ``https://www.sec.gov/Archives/edgar/data/...`` for Form 4 ownership XML.

Known simplifications (documented per the design's "simplified OK"):

- ``insider_transactions`` parses only *non-derivative* transactions from the
  first reporting owner of each Form 4, aggregated per transaction code, and
  caps the number of filings fetched per poll (``max_insider_filings``).
- The full-text index only covers filings from 2001+; historical backfill is
  out of scope for the poller.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from typing import Any

import httpx
import structlog

from dynamo.data.errors import DataGap, ProviderDown
from dynamo.data.http import get_json
from dynamo.data.ratelimit import TokenBucket
from dynamo.domain.models.market import Filing, InsiderTx, XbrlFact

log = structlog.get_logger(__name__)

SEARCH_BASE = "https://efts.sec.gov/LATEST/search-index"
DATA_BASE = "https://data.sec.gov"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
FILES_BASE = "https://www.sec.gov/files"

_SYMBOL_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]*)\)\s+\(CIK")
_XML_NS_RE = re.compile(r"\{.*\}")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _to_dt(value: str) -> datetime:
    """EDGAR gives dates only; store as UTC midnight (point-in-time floor)."""
    d = _parse_date(value)
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


class EdgarClient:
    """``FilingsProvider`` implementation over EDGAR public JSON endpoints."""

    def __init__(
        self,
        user_agent: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        search_base: str = SEARCH_BASE,
        data_base: str = DATA_BASE,
        archives_base: str = ARCHIVES_BASE,
        files_base: str = FILES_BASE,
        max_pages: int = 10,
        max_insider_filings: int = 50,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        # 10 req/s hard limit with declared User-Agent (doc 02 §1).
        self._bucket = bucket or TokenBucket(rate=10.0, capacity=10.0)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._search_base = search_base
        self._data_base = data_base
        self._archives_base = archives_base
        self._files_base = files_base
        self._max_pages = max_pages
        self._max_insider_filings = max_insider_filings

    # -- internals ---------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return get_json(
            self._client,
            url,
            provider="edgar",
            bucket=self._bucket,
            params=params,
            headers=self._headers,
        )

    def _search_hits(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Iterate full-text-search hits with pagination (capped)."""
        page_size = 100
        for page in range(self._max_pages):
            payload = self._get(self._search_base, params={**params, "from": page * page_size})
            hits = payload.get("hits", {}).get("hits", [])
            yield from hits
            if len(hits) < page_size:
                return

    @staticmethod
    def _hit_symbol(source: dict[str, Any]) -> str | None:
        for name in source.get("display_names", []):
            m = _SYMBOL_RE.search(name)
            if m:
                return m.group(1)
        return None

    # -- FilingsProvider ---------------------------------------------------

    def recent_filings(self, since: datetime) -> Iterable[Filing]:
        """Recent filings via the full-text search index (doc 02 §4: 10-min poll)."""
        params = {
            "q": '""',  # match-all phrase; the index requires a q parameter
            "dateRange": "custom",
            "startdt": since.date().isoformat(),
            "enddt": datetime.now(UTC).date().isoformat(),
        }
        seen: set[str] = set()
        for hit in self._search_hits(params):
            source = hit.get("_source", {})
            adsh = source.get("adsh")
            ciks = source.get("ciks") or []
            file_date = source.get("file_date")
            form = source.get("file_type") or (source.get("root_forms") or [""])[0]
            if not adsh or not ciks or not file_date or adsh in seen:
                continue
            seen.add(adsh)
            filed_at = _to_dt(file_date)
            if filed_at < datetime(since.year, since.month, since.day, tzinfo=UTC):
                continue
            cik = str(int(ciks[0]))
            accession_nodash = adsh.replace("-", "")
            yield Filing(
                accession_no=adsh,
                cik=cik,
                symbol=self._hit_symbol(source),
                form_type=str(form),
                filed_at=filed_at,
                period=None,
                url=f"{self._archives_base}/{cik}/{accession_nodash}/{adsh}-index.htm",
                items=tuple(str(i) for i in source.get("items", [])),
            )

    # -- history sweeps (dynamo.cli.backfill_events) -----------------------
    #
    # The two methods below exist because the poller's shape does not survive
    # a backfill:
    #
    # * ``recent_filings`` takes the FIRST document of each accession, so an
    #   8-K whose exhibit sorted first is stored with ``form_type='EX-99.1'``
    #   and ``filing_event`` never sees it. Over a 30-minute poll that is
    #   noise; over months of history it is most of the table. The sweeps read
    #   ``root_forms`` (the FORM the document belongs to) instead.
    # * ``insider_transactions`` fetches the ownership XML for every Form 4 in
    #   the window, capped at ``max_insider_filings``. Market-wide that is
    #   ~1,000 filings a day; ``ciks`` restricts the fetch to the universe
    #   being backfilled BEFORE any XML is downloaded, which is the difference
    #   between a 20-minute run and a 3-hour one.
    #
    # Both take an explicit ``[start, end]`` day range so the caller can chunk
    # (the full-text index caps pagination at ``max_pages`` * 100 hits, and a
    # single day of Form 4s already runs close to 1,000).

    def _range_params(self, form: str, start: date, end: date) -> dict[str, Any]:
        return {
            "q": '""',
            "forms": form,
            "dateRange": "custom",
            "startdt": start.isoformat(),
            "enddt": end.isoformat(),
        }

    def filings_in_range(self, form: str, start: date, end: date) -> Iterator[Filing]:
        """Filings of ONE root form filed in ``[start, end]``, one per accession.

        ``form_type`` comes from ``root_forms`` so an 8-K stays an 8-K
        whichever of its documents the index returned first, and ``items``
        carries the 8-K item codes ``filing_event`` keys on.
        """
        seen: set[str] = set()
        for hit in self._search_hits(self._range_params(form, start, end)):
            source = hit.get("_source", {})
            adsh = source.get("adsh")
            ciks = source.get("ciks") or []
            file_date = source.get("file_date")
            if not adsh or not ciks or not file_date or adsh in seen:
                continue
            seen.add(adsh)
            root = (source.get("root_forms") or [""])[0] or source.get("file_type") or form
            cik = str(int(ciks[0]))
            accession_nodash = adsh.replace("-", "")
            yield Filing(
                accession_no=adsh,
                cik=cik,
                symbol=self._hit_symbol(source),
                form_type=str(root),
                filed_at=_to_dt(file_date),
                period=None,
                url=f"{self._archives_base}/{cik}/{accession_nodash}/{adsh}-index.htm",
                items=tuple(str(i) for i in source.get("items", [])),
            )

    def insider_transactions_in_range(
        self,
        start: date,
        end: date,
        *,
        ciks: Iterable[str] | None = None,
    ) -> Iterator[InsiderTx]:
        """Form 4 non-derivative transactions filed in ``[start, end]``.

        ``ciks`` (unpadded, as EDGAR's index reports them after ``int()``)
        restricts which filings are downloaded at all: a hit is fetched only
        when one of its CIKs — issuer or reporting owner — is in the set.
        ``None`` means every filing in the window, which is what the nightly
        asset wants and what a backfill almost never does.

        No ``max_insider_filings`` cap here: the caller chunks by day and is
        the one that decides how much history it is willing to pay for.
        """
        wanted = None if ciks is None else {str(c).lstrip("0") for c in ciks}
        seen: set[str] = set()
        for hit in self._search_hits(self._range_params("4", start, end)):
            source = hit.get("_source", {})
            adsh = source.get("adsh")
            hit_ciks = source.get("ciks") or []
            doc_id = hit.get("_id", "")
            filename = doc_id.split(":", 1)[1] if ":" in doc_id else ""
            if not adsh or not hit_ciks or not filename.endswith(".xml"):
                continue
            # One accession can carry several ownership XMLs (a joint filing's
            # primary_doc alongside the form). They parse to the same rows, so
            # fetching the second is a wasted request against the 10 req/s
            # budget and an in-batch duplicate key downstream.
            if adsh in seen:
                continue
            seen.add(adsh)
            if wanted is not None and not any(str(c).lstrip("0") in wanted for c in hit_ciks):
                continue
            cik = str(int(hit_ciks[0]))
            url = f"{self._archives_base}/{cik}/{adsh.replace('-', '')}/{filename}"
            self._bucket.acquire()
            try:
                resp = self._client.get(url, headers=self._headers)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("edgar_form4_fetch_failed", url=url, error=str(exc))
                continue
            yield from self._parse_form4(adsh, resp.text, source.get("file_date"))

    def company_facts(self, cik: str) -> Iterable[XbrlFact]:
        """XBRL us-gaap facts, keyed by ``filed`` — never restated backwards."""
        padded = f"{int(cik):010d}"
        payload = self._get(f"{self._data_base}/api/xbrl/companyfacts/CIK{padded}.json")
        gaap = (payload.get("facts") or {}).get("us-gaap")
        if not gaap:
            raise DataGap("edgar", f"no us-gaap facts for CIK {padded}")
        for tag, body in gaap.items():
            for unit, entries in (body.get("units") or {}).items():
                for entry in entries:
                    end = entry.get("end")
                    val = entry.get("val")
                    filed = entry.get("filed")
                    if end is None or val is None or filed is None:
                        continue
                    yield XbrlFact(
                        cik=padded,
                        tag=tag,
                        period_end=_parse_date(end),
                        value=float(val),
                        unit=unit,
                        filed_at=_to_dt(filed),
                    )

    def company_ticker_map(self) -> dict[str, str]:
        """symbol -> zero-padded CIK from SEC's ``company_tickers.json``.

        The published mapping (updated periodically by the SEC) covers every
        EDGAR filer with a listed ticker. SEC spells share classes with a
        dash (``BRK-B``); the platform uses the dot form (``BRK.B``), so
        dashes are normalized to dots. Used by the backfill CLI's
        ``fundamentals --seed-ciks`` to populate ``instruments.cik``.
        """
        payload = self._get(f"{self._files_base}/company_tickers.json")
        out: dict[str, str] = {}
        for entry in (payload or {}).values():
            ticker = entry.get("ticker")
            cik = entry.get("cik_str")
            if not ticker or cik is None:
                continue
            symbol = str(ticker).upper().replace("-", ".")
            # First occurrence wins: the file is ordered by market cap, so a
            # duplicate ticker (rare) resolves to the larger filer.
            out.setdefault(symbol, f"{int(cik):010d}")
        return out

    def insider_transactions(self, since: datetime) -> Iterable[InsiderTx]:
        """Form 4 non-derivative transactions (simplified parse; see module doc)."""
        params = {
            "q": '""',
            "forms": "4",
            "dateRange": "custom",
            "startdt": since.date().isoformat(),
            "enddt": datetime.now(UTC).date().isoformat(),
        }
        fetched = 0
        for hit in self._search_hits(params):
            if fetched >= self._max_insider_filings:
                log.info("edgar_insider_cap_reached", cap=self._max_insider_filings)
                return
            source = hit.get("_source", {})
            adsh = source.get("adsh")
            ciks = source.get("ciks") or []
            doc_id = hit.get("_id", "")
            filename = doc_id.split(":", 1)[1] if ":" in doc_id else ""
            if not adsh or not ciks or not filename.endswith(".xml"):
                continue
            cik = str(int(ciks[0]))
            url = f"{self._archives_base}/{cik}/{adsh.replace('-', '')}/{filename}"
            self._bucket.acquire()
            try:
                resp = self._client.get(url, headers=self._headers)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("edgar_form4_fetch_failed", url=url, error=str(exc))
                continue
            fetched += 1
            yield from self._parse_form4(adsh, resp.text, source.get("file_date"))

    # -- Form 4 XML parsing ------------------------------------------------

    @staticmethod
    def _text(node: ET.Element | None) -> str | None:
        if node is None:
            return None
        # ownershipDocument wraps scalars in <value> children.
        value = node.find("value")
        target = value if value is not None else node
        return target.text.strip() if target.text else None

    def _parse_form4(
        self, accession_no: str, xml_text: str, file_date: str | None
    ) -> Iterator[InsiderTx]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ProviderDown("edgar", f"unparseable Form 4 {accession_no}: {exc}") from exc
        symbol = self._text(root.find(".//issuerTradingSymbol"))
        owner = root.find(".//reportingOwner")
        if symbol is None or owner is None:
            log.warning("edgar_form4_incomplete", accession_no=accession_no)
            return
        insider = self._text(owner.find(".//rptOwnerName")) or "UNKNOWN"
        role = self._text(owner.find(".//officerTitle"))
        if role is None and self._text(owner.find(".//isDirector")) in ("1", "true"):
            role = "Director"
        filed_at = _to_dt(file_date) if file_date else datetime.now(UTC)
        # Aggregate per transaction code so the (accession, symbol, insider,
        # tx_type) natural key stays idempotent across re-polls.
        agg: dict[str, tuple[float, float]] = {}  # code -> (shares, notional)
        for tx in root.findall(".//nonDerivativeTransaction"):
            code = self._text(tx.find(".//transactionCode"))
            shares_s = self._text(tx.find(".//transactionShares"))
            price_s = self._text(tx.find(".//transactionPricePerShare"))
            if code is None or shares_s is None:
                continue
            shares = float(shares_s)
            price = float(price_s) if price_s else 0.0
            prev_sh, prev_no = agg.get(code, (0.0, 0.0))
            agg[code] = (prev_sh + shares, prev_no + shares * price)
        for code, (shares, notional) in agg.items():
            yield InsiderTx(
                accession_no=accession_no,
                symbol=symbol,
                insider_name=insider,
                role=role,
                tx_type=code,
                shares=shares,
                price=(notional / shares) if shares and notional else None,
                filed_at=filed_at,
            )
