"""Point-in-time fundamentals: the section the committee never had.

Every fixture below is shaped like the real ``xbrl_facts`` table (long rows,
no period-type column, restatements as extra ``filed_at`` values) because that
is where the difficulty lives — see the module docstring of
``dynamo.research.fundamentals``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import cast

import pandas as pd
import pytest

from dynamo.research.fundamentals import (
    SHAPE_ANNUAL_AT_Q4,
    SHAPE_CUMULATIVE,
    SHAPE_DISCRETE,
    build_fundamentals,
    fiscal_year_end_month,
    infer_shape,
    normalized_facts,
    render_line,
    ttm,
)
from dynamo.research.quant.catalog import latest_value, pit_facts

AS_OF = date(2026, 6, 16)
STAMP = cast(pd.Timestamp, pd.Timestamp(AS_OF))
# Q2 + Q3 + derived Q4 + Q1 of the fixture below.
TTM_REVENUE = 121.0 + 132.0 + 143.0 + 121.0


def fact(
    symbol: str,
    tag: str,
    period_end: str,
    value: float,
    filed_at: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "tag": tag,
        "period_end": pd.Timestamp(period_end),
        "value": float(value),
        "filed_at": datetime.fromisoformat(filed_at),
    }


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# fixtures: a WMT-shaped filer (Jan fiscal year end, FY row in the Q4 slot)
# --------------------------------------------------------------------------

QUARTERS = (
    ("2024-04-30", 100.0, "2024-06-06"),
    ("2024-07-31", 110.0, "2024-08-30"),
    ("2024-10-31", 120.0, "2024-12-06"),
    ("2025-01-31", 460.0, "2025-03-14"),  # FISCAL YEAR total, not a quarter
    ("2025-04-30", 110.0, "2025-06-06"),
    ("2025-07-31", 121.0, "2025-08-29"),
    ("2025-10-31", 132.0, "2025-12-03"),
    ("2026-01-31", 506.0, "2026-03-13"),  # FY total again
    ("2026-04-30", 121.0, "2026-05-29"),
)


def wmt_rows(tag: str = "Revenues", scale: float = 1.0) -> list[dict[str, object]]:
    return [fact("WMT", tag, period, value * scale, filed) for period, value, filed in QUARTERS]


@pytest.fixture()
def wmt() -> pd.DataFrame:
    rows = wmt_rows()
    rows += [fact("WMT", "NetIncomeLoss", p, v * 0.03, f) for p, v, f in QUARTERS]
    rows += [
        fact("WMT", "Assets", "2026-04-30", 2000.0, "2026-05-29"),
        fact("WMT", "StockholdersEquity", "2026-04-30", 800.0, "2026-05-29"),
        fact("WMT", "CommonStockSharesOutstanding", "2026-04-30", 8_000_000.0, "2026-05-29"),
    ]
    return frame(rows)


def keyed(snapshot) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        line.key: (line.values if line.unit == "pct_series" else line.value)
        for group in snapshot.groups
        for line in group.lines
    }


def refs(snapshot) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {line.key: line.source_ref for group in snapshot.groups for line in group.lines}


# --------------------------------------------------------------------------
# obstacle (a): period-type inference
# --------------------------------------------------------------------------


def test_fiscal_year_row_is_not_summed_as_a_quarter(wmt: pd.DataFrame) -> None:
    """The naive read (sum the last four period_ends) gives 506+132+121+110
    = 869 — a year and a bit double-counted.

    The right answer is the four real quarters ending 2026-04-30: Q2 (121),
    Q3 (132), the DERIVED Q4 (506 - 110 - 121 - 132 = 143) and Q1 (121)."""
    snapshot = build_fundamentals(wmt, symbol="WMT", as_of=AS_OF)
    assert keyed(snapshot)["revenue_ttm"] == pytest.approx(TTM_REVENUE / 1e6, rel=1e-9)


def test_period_type_inference_labels_the_fiscal_year(wmt: pd.DataFrame) -> None:
    facts = pit_facts(wmt, STAMP)
    assert fiscal_year_end_month(facts, "WMT", STAMP) == 1
    series = pd.Series(
        [v for _, v, _ in QUARTERS],
        index=[pd.Timestamp(p) for p, _, _ in QUARTERS],
    )
    assert infer_shape(series, 1) == SHAPE_ANNUAL_AT_Q4


def test_genuine_q4_stays_a_quarter() -> None:
    """A filer that reports a real Q4 (not the FY) must not be 'corrected'."""
    periods = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
    series = pd.Series([100.0, 105.0, 110.0, 115.0], index=[pd.Timestamp(p) for p in periods])
    assert infer_shape(series, 12) == SHAPE_DISCRETE


def test_year_to_date_cash_flow_is_differenced_not_summed() -> None:
    """10-Q cash-flow rows are cumulative: 30/70/110/150 is a 150 year."""
    rows = [
        fact("ACME", "Revenues", p, v, f)
        for p, v, f in (
            ("2025-03-31", 100.0, "2025-04-20"),
            ("2025-06-30", 100.0, "2025-07-20"),
            ("2025-09-30", 100.0, "2025-10-20"),
            ("2025-12-31", 400.0, "2026-02-20"),
            ("2026-03-31", 110.0, "2026-04-20"),
        )
    ]
    rows += [
        fact("ACME", "NetCashProvidedByUsedInOperatingActivities", p, v, f)
        for p, v, f in (
            ("2025-03-31", 30.0, "2025-04-20"),
            ("2025-06-30", 70.0, "2025-07-20"),
            ("2025-09-30", 110.0, "2025-10-20"),
            ("2025-12-31", 150.0, "2026-02-20"),
            ("2026-03-31", 33.0, "2026-04-20"),
        )
    ]
    facts = pit_facts(frame(rows), STAMP)
    norm = normalized_facts(facts, "ACME", STAMP)
    assert norm.shape("NetCashProvidedByUsedInOperatingActivities") == SHAPE_CUMULATIVE
    aggregate = ttm(norm, "ACME", ("NetCashProvidedByUsedInOperatingActivities",), STAMP)
    assert aggregate is not None
    # Q2..Q4 of 2025 (40, 40, 40) + Q1 2026 (33) — never the 150 YTD row.
    assert aggregate.value == pytest.approx(153.0)


# --------------------------------------------------------------------------
# obstacle (b): restatements
# --------------------------------------------------------------------------


def test_restatement_latest_filed_at_or_before_as_of_wins() -> None:
    rows = wmt_rows()
    rows.append(fact("WMT", "Revenues", "2026-04-30", 999.0, "2026-06-01"))  # restated
    snapshot = build_fundamentals(frame(rows), symbol="WMT", as_of=AS_OF)
    assert keyed(snapshot)["revenue_ttm"] == pytest.approx((TTM_REVENUE - 121.0 + 999.0) / 1e6)
    assert "filed-2026-06-01" in refs(snapshot)["revenue_ttm"]


def test_fact_filed_after_as_of_is_invisible() -> None:
    rows = wmt_rows()
    rows.append(fact("WMT", "Revenues", "2026-04-30", 999.0, "2026-06-17"))  # after as_of
    snapshot = build_fundamentals(frame(rows), symbol="WMT", as_of=AS_OF)
    assert keyed(snapshot)["revenue_ttm"] == pytest.approx(TTM_REVENUE / 1e6)
    assert "filed-2026-05-29" in refs(snapshot)["revenue_ttm"]


# --------------------------------------------------------------------------
# obstacle (c): tag normalization, including CAT's missing equity tag
# --------------------------------------------------------------------------


def test_cat_equity_falls_back_to_the_noncontrolling_interest_tag() -> None:
    """CAT has ZERO ``StockholdersEquity`` rows in the real table — equity must
    resolve through the ``...IncludingPortionAttributableToNoncontrolling
    Interest`` variant, and the ref must name the tag that was used."""
    rows = wmt_rows()
    rows = [{**r, "symbol": "CAT"} for r in rows]
    rows += [
        fact("CAT", "Assets", "2026-04-30", 2000.0, "2026-05-29"),
        fact(
            "CAT",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "2026-04-30",
            500.0,
            "2026-05-29",
        ),
    ]
    facts = frame(rows)
    assert latest_value(pit_facts(facts, STAMP), "CAT", "StockholdersEquity", STAMP) != (500.0)
    snapshot = build_fundamentals(facts, symbol="CAT", as_of=AS_OF)
    values, sources = keyed(snapshot), refs(snapshot)
    assert values["shareholders_equity"] == pytest.approx(500.0 / 1e6)
    assert (
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        in (sources["shareholders_equity"])
    )
    # Liabilities are not filed either: assets - equity, with both refs cited.
    assert values["total_liabilities"] == pytest.approx(1500.0 / 1e6)
    assert sources["total_liabilities"].startswith("derived:")


def test_revenue_falls_back_to_the_contract_revenue_tag() -> None:
    rows = [{**r, "tag": "RevenueFromContractWithCustomerExcludingAssessedTax"} for r in wmt_rows()]
    snapshot = build_fundamentals(frame(rows), symbol="WMT", as_of=AS_OF)
    assert keyed(snapshot)["revenue_ttm"] == pytest.approx(TTM_REVENUE / 1e6)
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" in (refs(snapshot)["revenue_ttm"])


# --------------------------------------------------------------------------
# absence, determinism, provenance
# --------------------------------------------------------------------------


def test_missing_metrics_render_absent_and_never_fabricated(wmt: pd.DataFrame) -> None:
    snapshot = build_fundamentals(wmt, symbol="WMT", as_of=AS_OF)
    values = keyed(snapshot)
    assert values["cfo_ttm"] is None  # no cash-flow rows in the fixture
    assert values["fcf_ttm"] is None
    lines = {line.key: render_line(line) for group in snapshot.groups for line in group.lines}
    assert lines["cfo_ttm"] == "cfo_ttm: n/a"
    assert "[" not in lines["fcf_ttm"]  # no citation for a number that is not there


def test_empty_facts_produce_an_empty_snapshot_without_raising() -> None:
    snapshot = build_fundamentals(None, symbol="WMT", as_of=AS_OF)
    assert snapshot.groups == ()
    assert not snapshot.has_data
    empty = build_fundamentals(
        pd.DataFrame(columns=pd.Index(["symbol", "tag", "period_end", "value", "filed_at"])),
        symbol="WMT",
        as_of=AS_OF,
    )
    assert empty.groups == ()


def test_unknown_symbol_renders_nothing(wmt: pd.DataFrame) -> None:
    snapshot = build_fundamentals(wmt, symbol="NOPE", as_of=AS_OF)
    assert not snapshot.has_data


def test_deterministic_byte_identical_sections(wmt: pd.DataFrame) -> None:
    def render() -> str:
        snapshot = build_fundamentals(
            wmt.sample(frac=1.0, random_state=7),
            symbol="WMT",
            as_of=AS_OF,
            close=100.0,
            sector="Consumer Staples",
        )
        return "\n".join(render_line(line) for group in snapshot.groups for line in group.lines)

    first = render()
    assert first == render()
    assert "revenue_ttm:" in first


def test_every_populated_line_carries_a_citable_source_ref(wmt: pd.DataFrame) -> None:
    snapshot = build_fundamentals(
        wmt, symbol="WMT", as_of=AS_OF, close=100.0, sector="Consumer Staples"
    )
    for group in snapshot.groups:
        for line in group.lines:
            if line.value is None and not line.values:
                continue
            assert line.source_ref, f"{line.key} has a value but no citation"
            assert line.source_ref.startswith(("xbrl:", "derived:"))


def test_valuation_uses_close_times_shares_outstanding(wmt: pd.DataFrame) -> None:
    snapshot = build_fundamentals(wmt, symbol="WMT", as_of=AS_OF, close=10.0)
    values = keyed(snapshot)
    assert values["market_cap"] == pytest.approx(10.0 * 8_000_000.0 / 1e6)
    net_income_ttm = TTM_REVENUE * 0.03
    assert values["pe_ttm"] == pytest.approx(10.0 * 8_000_000.0 / net_income_ttm)


def test_sector_percentile_needs_enough_peers(wmt: pd.DataFrame) -> None:
    rows = list(wmt.to_dict(orient="records"))
    peers = {}
    for i, peer in enumerate(("PEER1", "PEER2", "PEER3")):
        rows += [{**r, "symbol": peer} for r in wmt_rows()]
        rows += [fact(peer, "NetIncomeLoss", p, v * 0.03, f) for p, v, f in QUARTERS]
        rows += [
            fact(peer, "CommonStockSharesOutstanding", "2026-04-30", 8_000_000.0, "2026-05-29")
        ]
        peers[peer] = 10.0 * (i + 2)  # richer and richer peers
    sectors = {"WMT": "Consumer Staples", **{p: "Consumer Staples" for p in peers}}
    snapshot = build_fundamentals(
        frame(rows),
        symbol="WMT",
        as_of=AS_OF,
        close=10.0,
        sector="Consumer Staples",
        peer_closes={"WMT": 10.0, **peers},
        peer_sectors=sectors,
    )
    values = keyed(snapshot)
    assert values["pe_ttm_sector_pctile"] == pytest.approx(0.0)  # cheapest of the four

    lonely = build_fundamentals(
        wmt, symbol="WMT", as_of=AS_OF, close=10.0, sector="Consumer Staples"
    )
    assert keyed(lonely)["pe_ttm_sector_pctile"] is None  # no peers -> absent
