"""SEC DERA reference adapter — no network: zips are built in-memory."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import httpx
import pytest

from dynamo.data.errors import ProviderDown
from dynamo.data.ratelimit import TokenBucket
from dynamo.data.sec_reference import (
    SecReferenceClient,
    recent_quarters,
    sectors_for_symbols,
)

HEADER = "adsh\tcik\tname\tsic\tfye\n"


def make_zip(rows: list[tuple[str, str, str]], *, member: str = "sub.txt") -> bytes:
    body = HEADER + "".join(
        f"000{i}\t{cik}\t{name}\t{sic}\t1231\n" for i, (cik, name, sic) in enumerate(rows)
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(member, body)
    return buf.getvalue()


def make_client(handler: httpx.MockTransport) -> SecReferenceClient:
    # Fake clock/sleep: the bucket must never actually pace a unit test.
    bucket = TokenBucket(rate=1e9, capacity=1e9, clock=lambda: 0.0, sleep=lambda _: None)
    return SecReferenceClient(
        "test-agent",
        client=httpx.Client(transport=handler),
        bucket=bucket,
        base_url="https://sec.test/dera",
    )


def test_parse_sub_txt_pads_ciks_and_drops_junk() -> None:
    payload = make_zip(
        [
            ("320193", "APPLE INC", "3571"),
            ("789019", "MICROSOFT CORP", "7372"),
            ("", "NO CIK", "1311"),  # dropped
            ("111", "NO SIC", ""),  # dropped
            ("222", "NON NUMERIC SIC", "abcd"),  # dropped
        ]
    )
    assert SecReferenceClient.parse_sub_txt(payload) == {
        "0000320193": "3571",
        "0000789019": "7372",
    }


def test_zip_without_sub_txt_is_a_typed_provider_failure() -> None:
    with pytest.raises(ProviderDown):
        SecReferenceClient.parse_sub_txt(make_zip([("1", "X", "1311")], member="num.txt"))


def test_cik_to_sic_merges_quarters_newest_first() -> None:
    quarters = {
        "2026q1": make_zip([("1", "RECLASSIFIED", "7372")]),
        "2025q4": make_zip([("1", "RECLASSIFIED", "2834"), ("2", "OLDER", "6021")]),
    }
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        quarter = request.url.path.rsplit("/", 1)[-1].removesuffix(".zip")
        seen.append(quarter)
        return httpx.Response(200, content=quarters[quarter])

    mapping = make_client(httpx.MockTransport(handler)).cik_to_sic(["2026q1", "2025q4"])
    assert seen == ["2026q1", "2025q4"]
    # Newest wins for CIK 1; CIK 2 only exists in the older quarter.
    assert mapping == {"0000000001": "7372", "0000000002": "6021"}


def test_a_failed_quarter_is_skipped_not_fatal() -> None:
    """Partial reference data beats none — the caller reports coverage."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("2026q1.zip"):
            return httpx.Response(404)
        return httpx.Response(200, content=make_zip([("7", "OK", "1311")]))

    mapping = make_client(httpx.MockTransport(handler)).cik_to_sic(["2026q1", "2025q4"])
    assert mapping == {"0000000007": "1311"}


def test_recent_quarters_skips_the_unpublished_current_quarter() -> None:
    # DERA publishes ~a month after the quarter closes, so 2026q3 (which
    # contains 2026-08-19) must not be requested.
    assert recent_quarters(date(2026, 8, 19), 4) == ["2026q2", "2026q1", "2025q4", "2025q3"]
    assert recent_quarters(date(2026, 1, 15), 2) == ["2025q4", "2025q3"]


def test_sectors_for_symbols_composes_the_two_lookups() -> None:
    sectors, industries = sectors_for_symbols(
        {"AAPL": "0000320193", "MSFT": "0000789019", "SPY": "0000000999", "NOCIK": "0000000123"},
        {"0000320193": "3571", "0000789019": "7372", "0000000123": "9995"},
    )
    assert sectors == {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
    }
    assert industries["AAPL"] == "Industrial Machinery and Computer Equipment"
    # SPY has a CIK but no DERA row (ETFs file no XBRL financial statements),
    # and NOCIK's SIC is the non-classifiable placeholder: both get NOTHING
    # rather than an invented sector that would corrupt the risk caps.
    assert "SPY" not in sectors and "NOCIK" not in sectors


def test_sectors_for_symbols_honors_a_symbol_subset() -> None:
    sectors, _ = sectors_for_symbols(
        {"AAPL": "0000000001", "MSFT": "0000000002"},
        {"0000000001": "3571", "0000000002": "7372"},
        symbols=["MSFT"],
    )
    assert set(sectors) == {"MSFT"}
