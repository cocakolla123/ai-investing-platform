"""SEC reference data: CIK→SIC for the whole filer population.

The problem
-----------
``instruments.sector`` was populated for 40 of 11,505 rows and
``instruments.cik`` for 43. The CIK gap kills every fundamentals feature
(``xbrl_facts`` is keyed by CIK, so quality/value/growth resolve to NaN);
the sector gap kills ``rel_strength``, ``rs_20d_sector``,
``ps_sector_pctile``, the deep-dive sector buckets, and the risk engine's
sector caps.

The symbol→CIK half already exists: ``EdgarClient.company_ticker_map``
reads SEC's ``company_tickers.json`` in one request. What was missing is a
free, bulk source of *classification*, because SEC's per-filer
``submissions`` API — which does carry ``sic`` — would need ~10,000
requests and tens of gigabytes to sweep the market.

The source used here
--------------------
SEC's **Financial Statement Data Sets** (DERA), one ZIP per calendar
quarter, ~65-85MB each. Their ``sub.txt`` is a tab-separated row per
submission carrying ``cik`` and ``sic`` for every filer that filed an
XBRL financial statement that quarter. Four quarters cover the operating-
company population in four requests instead of ten thousand.

What it does NOT cover, stated rather than discovered later: ETFs, closed-
end funds, trusts, and other non-XBRL filers never appear in ``sub.txt``,
so they get no SIC and therefore no sector from this path. That is the
correct outcome — inventing a sector for SPY would corrupt the very sector
caps the field feeds.

Everything here is I/O; the SIC→sector judgement is pure and lives in
:mod:`dynamo.data.sic`.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date

import httpx
import structlog

from dynamo.data.errors import ProviderDown
from dynamo.data.ratelimit import TokenBucket

log = structlog.get_logger(__name__)

DERA_BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"


def recent_quarters(as_of: date, count: int = 4) -> list[str]:
    """The ``count`` most recent PUBLISHED quarters, newest first.

    DERA publishes a quarter roughly a month after it closes, so the
    quarter containing ``as_of`` is skipped: asking for it 404s for most of
    its own duration.
    """
    year, quarter = as_of.year, (as_of.month - 1) // 3 + 1
    out: list[str] = []
    for _ in range(count + 1):
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
        out.append(f"{year}q{quarter}")
    return out[:count]


class SecReferenceClient:
    """CIK→SIC over SEC's quarterly Financial Statement Data Sets."""

    def __init__(
        self,
        user_agent: str,
        *,
        client: httpx.Client | None = None,
        bucket: TokenBucket | None = None,
        base_url: str = DERA_BASE,
    ) -> None:
        # These are 65-85MB downloads; the default 30s timeout is not enough.
        self._client = client or httpx.Client(timeout=300.0, follow_redirects=True)
        # Same 10 req/s SEC ceiling the EDGAR adapter observes (doc 02 §1).
        self._bucket = bucket or TokenBucket(rate=10.0, capacity=10.0)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._base_url = base_url.rstrip("/")

    def _fetch_quarter(self, quarter: str) -> bytes:
        url = f"{self._base_url}/{quarter}.zip"
        self._bucket.acquire()
        try:
            resp = self._client.get(url, headers=self._headers)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderDown("sec_dera", f"{url}: {exc}") from exc
        return resp.content

    @staticmethod
    def parse_sub_txt(payload: bytes) -> dict[str, str]:
        """CIK (zero-padded) → SIC from one quarterly ZIP's ``sub.txt``."""
        out: dict[str, str] = {}
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.endswith("sub.txt")]
            if not names:
                raise ProviderDown("sec_dera", "quarterly zip has no sub.txt")
            with archive.open(names[0]) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
                for row in csv.DictReader(text, delimiter="\t"):
                    cik, sic = (row.get("cik") or "").strip(), (row.get("sic") or "").strip()
                    if not cik.isdigit() or not sic.isdigit():
                        continue
                    out[f"{int(cik):010d}"] = sic
        return out

    def cik_to_sic(self, quarters: Sequence[str]) -> dict[str, str]:
        """Merged CIK→SIC across ``quarters``; earlier entries win.

        Pass quarters newest-first so the most recent classification of a
        filer that re-classified is the one that survives. A quarter that
        fails to download is logged and skipped — partial reference data
        beats none, and the caller reports the coverage it achieved.
        """
        merged: dict[str, str] = {}
        for quarter in quarters:
            try:
                mapping = self.parse_sub_txt(self._fetch_quarter(quarter))
            except ProviderDown as exc:
                log.warning("sec_dera_quarter_skipped", quarter=quarter, error=str(exc))
                continue
            new = 0
            for cik, sic in mapping.items():
                if cik not in merged:
                    merged[cik] = sic
                    new += 1
            log.info("sec_dera_quarter", quarter=quarter, filers=len(mapping), new=new)
        return merged


def sectors_for_symbols(
    symbol_to_cik: dict[str, str],
    cik_to_sic: dict[str, str],
    symbols: Iterable[str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """(symbol→sector, symbol→industry) for symbols with a mappable SIC.

    Pure composition of the two lookups and :mod:`dynamo.data.sic`. Symbols
    with no CIK, no SIC, or an unmappable SIC are simply absent from both
    maps — never given a placeholder sector.
    """
    from dynamo.data.sic import industry_for_sic, sector_for_sic

    wanted = None if symbols is None else set(symbols)
    sectors: dict[str, str] = {}
    industries: dict[str, str] = {}
    for symbol, cik in symbol_to_cik.items():
        if wanted is not None and symbol not in wanted:
            continue
        sic = cik_to_sic.get(cik)
        if sic is None:
            continue
        sector = sector_for_sic(sic)
        if sector is not None:
            sectors[symbol] = sector
        industry = industry_for_sic(sic)
        if industry is not None:
            industries[symbol] = industry
    return sectors, industries
