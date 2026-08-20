"""FINRA consolidated short-interest adapter.

The fixture is a real, unedited slice of FINRA's own response for the
2026-07-31 settlement cycle (22 rows, every market class the file carries,
including an OTC name whose average daily volume is 0 and whose days-to-cover
is therefore the 999.99 sentinel). Rows carrying ``stockSplitFlag`` /
``revisionFlag`` are rare and absent from that cycle, so those two columns are
exercised by an inline row instead of by doctoring the captured bytes.

The rule this file exists to protect: a cycle is readable on its PUBLICATION
date, never on its settlement date.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from dynamo.data.errors import DataGap, ProviderDown, RateLimited
from dynamo.data.finra import (
    FinraClient,
    parse_short_interest_csv,
    publication_date_for,
)
from dynamo.data.ratelimit import TokenBucket

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "finra"
    / "consolidated_short_interest_2026-07-31.csv"
)
HEADER = (
    '"accountingYearMonthNumber","symbolCode","issueName",'
    '"issuerServicesGroupExchangeCode","marketClassCode",'
    '"currentShortPositionQuantity","previousShortPositionQuantity",'
    '"stockSplitFlag","averageDailyVolumeQuantity","daysToCoverQuantity",'
    '"revisionFlag","changePercent","changePreviousNumber","settlementDate"\n'
)


def _fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _client(handler: httpx.MockTransport) -> FinraClient:
    return FinraClient(
        client=httpx.Client(transport=handler),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


# --------------------------------------------------------------------------
# the parser, against real captured bytes
# --------------------------------------------------------------------------


def test_parses_the_captured_cycle() -> None:
    rows = {r.symbol: r for r in parse_short_interest_csv(_fixture_text())}
    assert len(rows) == 22

    agilent = rows["A"]
    assert agilent.settlement_date == date(2026, 7, 31)
    assert agilent.shares_short == 5_749_623.0
    assert agilent.previous_shares_short == 7_538_437.0
    assert agilent.avg_daily_volume == 2_301_495.0
    assert agilent.days_to_cover == 2.50
    assert agilent.change_pct == -23.73
    assert agilent.market_class == "NYSE"
    assert agilent.issue_name == "Agilent Technologies Inc."
    # Blank optional columns come back as None, not as empty strings.
    assert agilent.split_flag is None
    assert agilent.revision_flag is None

    # The whole file, not just the mega-caps: ETFs and OTC names too.
    assert {r.market_class for r in rows.values()} >= {"NYSE", "NNM", "ARCA", "OTC"}


def test_zero_volume_rows_survive_parsing_with_their_sentinel_intact() -> None:
    """FINRA reports days-to-cover 999.99 when average daily volume is 0. The
    ADAPTER keeps what the file said; deciding that 999.99 is not a number of
    days is the feature layer's job, on data it can see."""
    rows = {r.symbol: r for r in parse_short_interest_csv(_fixture_text())}
    zero_volume = [r for r in rows.values() if r.avg_daily_volume == 0.0]
    assert zero_volume
    assert all(r.days_to_cover == 999.99 for r in zero_volume)


def test_flag_columns_are_carried_through() -> None:
    body = HEADER + (
        '"20260731","ZZZ","Flagged Corp","A","NNM","1000","500",'
        '"S","250","4.00","R","100.00","500","2026-07-31"\n'
    )
    (row,) = list(parse_short_interest_csv(body))
    assert row.split_flag == "S"
    assert row.revision_flag == "R"


def test_malformed_rows_are_skipped_not_fatal() -> None:
    """A bad line in a 12,000-row file must not cost the other 11,999."""
    body = HEADER + (
        '"20260731","","No Symbol","A","NNM","1000","500",,"250","4.00",,"1","1","2026-07-31"\n'
        '"20260731","NODATE","No Date","A","NNM","1000","500",,"250","4.00",,"1","1",\n'
        '"20260731","NOSHORT","No Position","A","NNM",,"500",,"250","4.00",,"1","1","2026-07-31"\n'
        '"20260731","GOOD","Fine","A","NNM","1000","500",,"250","4.00",,"1","1","2026-07-31"\n'
    )
    rows = list(parse_short_interest_csv(body))
    assert [r.symbol for r in rows] == ["GOOD"]


def test_share_class_symbols_are_normalized_to_the_platform_form() -> None:
    body = HEADER + (
        '"20260731","BRK-B","Berkshire B","A","NYSE","1000","500",,'
        '"250","4.00",,"1","1","2026-07-31"\n'
    )
    (row,) = list(parse_short_interest_csv(body))
    assert row.symbol == "BRK.B"


# --------------------------------------------------------------------------
# publication vs settlement — the point-in-time rule
# --------------------------------------------------------------------------


def test_publication_date_lags_settlement_and_never_lands_on_a_weekend() -> None:
    # 2026-07-31 + 8 days is Saturday 2026-08-08 -> rolls to Monday.
    assert publication_date_for(date(2026, 7, 31)) == date(2026, 8, 10)
    # 2026-07-15 + 8 days is Thursday 2026-07-23 — no roll needed.
    assert publication_date_for(date(2026, 7, 15)) == date(2026, 7, 23)
    for settlement in (date(2026, 1, 15), date(2026, 3, 31), date(2025, 11, 28)):
        published = publication_date_for(settlement)
        assert published > settlement
        assert published.weekday() < 5


def test_every_parsed_row_is_stamped_with_a_publication_date_after_settlement() -> None:
    for row in parse_short_interest_csv(_fixture_text()):
        assert row.publication_date > row.settlement_date


def test_an_explicit_publication_calendar_overrides_the_derived_lag() -> None:
    """FINRA's real deadline calendar shifts around holidays. A caller holding
    it must be able to use it verbatim rather than the approximation."""
    official = {date(2026, 7, 31): date(2026, 8, 12)}
    rows = list(parse_short_interest_csv(_fixture_text(), publication_dates=official))
    assert {r.publication_date for r in rows} == {date(2026, 8, 12)}


def test_published_settlement_dates_hides_a_cycle_that_has_not_been_disseminated() -> None:
    """The whole reason both dates exist. On 2026-08-05 the 2026-07-31 cycle
    has SETTLED but is not out yet — it must be invisible. Five days later it
    is not."""
    partitions = {
        "availablePartitions": [
            {"partitions": ["2026-07-31"]},
            {"partitions": ["2026-07-15"]},
            {"partitions": ["2026-06-30"]},
        ]
    }
    client = _client(
        httpx.MockTransport(lambda request: httpx.Response(200, json=partitions))
    )
    assert client.settlement_dates()[0] == date(2026, 7, 31)

    too_early = client.published_settlement_dates(date(2026, 8, 5))
    assert date(2026, 7, 31) not in too_early
    assert date(2026, 7, 15) in too_early

    later = client.published_settlement_dates(date(2026, 8, 10))
    assert date(2026, 7, 31) in later


# --------------------------------------------------------------------------
# HTTP behaviour: pacing, paging, typed errors
# --------------------------------------------------------------------------


def test_short_interest_filters_by_settlement_date_and_pages() -> None:
    text = _fixture_text()
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        # One short page ends the cycle immediately.
        return httpx.Response(200, text=text)

    client = FinraClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
        page_size=1_000,
    )
    rows = list(client.short_interest([date(2026, 7, 31)]))
    assert len(rows) == 22
    assert len(seen) == 1
    assert seen[0]["compareFilters"] == [
        {
            "fieldName": "settlementDate",
            "fieldValue": "2026-07-31",
            "compareType": "EQUAL",
        }
    ]


def test_a_cycle_that_returns_nothing_is_a_data_gap() -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, text=HEADER)))
    with pytest.raises(DataGap):
        list(client.short_interest([date(2026, 7, 31)]))


def test_rate_limit_and_server_errors_map_to_typed_exceptions() -> None:
    limited = _client(httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(RateLimited):
        limited.settlement_dates()

    down = _client(httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(ProviderDown):
        down.settlement_dates()

    missing = _client(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(ProviderDown):
        missing.settlement_dates()


def test_empty_partition_list_is_a_data_gap() -> None:
    client = _client(
        httpx.MockTransport(lambda r: httpx.Response(200, json={"availablePartitions": []}))
    )
    with pytest.raises(DataGap):
        client.settlement_dates()
