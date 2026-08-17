"""EDGAR adapter parsing tests with httpx.MockTransport."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from dynamo.data.edgar import EdgarClient
from dynamo.data.ratelimit import TokenBucket

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>10.5</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>11.5</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

COMPANY_FACTS = {
    "cik": 320193,
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2026-03-31", "val": 1000.0, "filed": "2026-05-01"},
                        {"end": "2026-03-31", "val": None, "filed": "2026-05-01"},
                    ]
                }
            },
            "CommonStockSharesOutstanding": {
                "units": {"shares": [{"end": "2026-03-31", "val": 42.0, "filed": "2026-05-02"}]}
            },
        }
    },
}

SEARCH_HITS = {
    "hits": {
        "hits": [
            {
                "_id": "0000320193-26-000001:doc.htm",
                "_source": {
                    "adsh": "0000320193-26-000001",
                    "ciks": ["0000320193"],
                    "file_date": "2026-07-30",
                    "file_type": "8-K",
                    "items": ["1.01", "9.01"],
                    "display_names": ["Acme Corp  (ACME)  (CIK 0000320193)"],
                },
            }
        ]
    }
}

FORM4_HITS = {
    "hits": {
        "hits": [
            {
                "_id": "0000320193-26-000002:form4.xml",
                "_source": {
                    "adsh": "0000320193-26-000002",
                    "ciks": ["0000320193"],
                    "file_date": "2026-07-31",
                },
            }
        ]
    }
}


def make_client(handler: httpx.MockTransport) -> EdgarClient:
    return EdgarClient(
        "test test@example.com",
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_company_facts_parses_us_gaap_tags() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "companyfacts/CIK0000320193.json" in str(request.url)
        assert request.headers["User-Agent"].startswith("test")
        return httpx.Response(200, json=COMPANY_FACTS)

    facts = list(make_client(httpx.MockTransport(handler)).company_facts("320193"))
    assert len(facts) == 2  # null val skipped
    rev = next(f for f in facts if f.tag == "Revenues")
    assert rev.cik == "0000320193"
    assert rev.value == 1000.0
    assert rev.unit == "USD"
    assert rev.filed_at == datetime(2026, 5, 1, tzinfo=UTC)  # point-in-time by 'filed'
    shares = next(f for f in facts if f.tag == "CommonStockSharesOutstanding")
    assert shares.unit == "shares"


def test_company_ticker_map_normalizes_and_pads() -> None:
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
        "2": {"cik_str": 99999, "ticker": "AAPL", "title": "Dup ticker (ignored)"},
        "3": {"cik_str": None, "ticker": "NOCIK"},
        "4": {"cik_str": 12345, "ticker": ""},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/company_tickers.json")
        return httpx.Response(200, json=payload)

    mapping = make_client(httpx.MockTransport(handler)).company_ticker_map()
    assert mapping == {"AAPL": "0000320193", "BRK.B": "0001067983"}


def test_recent_filings_maps_index_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SEARCH_HITS)

    since = datetime(2026, 7, 29, tzinfo=UTC)
    filings = list(make_client(httpx.MockTransport(handler)).recent_filings(since))
    assert len(filings) == 1
    f = filings[0]
    assert f.accession_no == "0000320193-26-000001"
    assert f.cik == "320193"
    assert f.symbol == "ACME"
    assert f.form_type == "8-K"
    assert f.items == ("1.01", "9.01")
    assert f.filed_at == datetime(2026, 7, 30, tzinfo=UTC)
    assert "000032019326000001" in f.url


def test_insider_transactions_parses_form4() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "form4.xml" in url:
            return httpx.Response(200, text=FORM4_XML)
        return httpx.Response(200, json=FORM4_HITS)

    since = datetime(2026, 7, 30, tzinfo=UTC)
    txs = list(make_client(httpx.MockTransport(handler)).insider_transactions(since))
    assert len(txs) == 1  # two purchases aggregated per tx code
    tx = txs[0]
    assert tx.symbol == "ACME"
    assert tx.insider_name == "DOE JANE"
    assert tx.role == "Director"
    assert tx.tx_type == "P"
    assert tx.shares == 1500.0
    assert tx.price is not None
    assert abs(tx.price - (1000 * 10.5 + 500 * 11.5) / 1500) < 1e-9
