"""revenue_tag: which revenue concept a filer is read on, point in time."""

from __future__ import annotations

from typing import cast

import pandas as pd

from dynamo.research.quant.catalog import n_filed_quarters, revenue_tag, ttm_sum

ASC606 = "RevenueFromContractWithCustomerExcludingAssessedTax"
AS_OF = cast(pd.Timestamp, pd.Timestamp("2025-06-30"))


def _rows(tag: str, start: str, n: int, value: float = 100.0) -> list[dict[str, object]]:
    return [
        {
            "symbol": "AAA",
            "tag": tag,
            "period_end": q,
            "value": value,
            "filed_at": q + pd.Timedelta(days=40),
        }
        for q in pd.date_range(start, periods=n, freq="QE")
    ]


def test_contract_only_filer_resolves_and_counts_quarters() -> None:
    facts = pd.DataFrame(_rows(ASC606, "2023-03-31", 8))
    assert revenue_tag(facts, "AAA", AS_OF) == ASC606
    assert n_filed_quarters(facts, "AAA", AS_OF) == 8
    assert ttm_sum(facts, "AAA", revenue_tag(facts, "AAA", AS_OF), AS_OF) == 400.0


def test_switched_filer_is_read_on_its_current_tag() -> None:
    # Stale "Revenues" history that stops in 2022, then the contract tag.
    facts = pd.DataFrame(_rows("Revenues", "2021-03-31", 6) + _rows(ASC606, "2023-03-31", 8))
    assert revenue_tag(facts, "AAA", AS_OF) == ASC606


def test_tie_prefers_revenues_and_future_filings_are_ignored() -> None:
    both = pd.DataFrame(_rows("Revenues", "2023-03-31", 8) + _rows(ASC606, "2023-03-31", 8))
    assert revenue_tag(both, "AAA", AS_OF) == "Revenues"
    # A contract-tag quarter filed after as_of must not decide the tag.
    late = _rows(ASC606, "2025-09-30", 1)
    facts = pd.DataFrame(_rows("Revenues", "2023-03-31", 8) + late)
    assert revenue_tag(facts, "AAA", AS_OF) == "Revenues"


def test_no_revenue_falls_back_to_revenues() -> None:
    facts = pd.DataFrame(_rows("GrossProfit", "2023-03-31", 4))
    assert revenue_tag(facts, "AAA", AS_OF) == "Revenues"
    assert n_filed_quarters(facts, "AAA", AS_OF) == 0
