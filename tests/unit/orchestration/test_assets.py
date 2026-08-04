"""Asset selection smoke: data_quality + tier2_scan on in-memory fakes.

``_read_df`` is monkeypatched to return empty (correctly-shaped) frames, so
no SQL engine is ever built; bars/universe come from the FakeRepo.
"""

from __future__ import annotations

from typing import Any

import orchestration.assets.research as research_mod
import pandas as pd
import pytest
from dagster import materialize
from orchestration.assets.research import data_quality, tier2_scan

from tests.unit.orchestration.conftest import FakeDynamo, FakeRepo


@pytest.fixture(autouse=True)
def no_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_df(dynamo: Any, stmt: Any) -> pd.DataFrame:
        cols = [c.key for c in stmt.selected_columns]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    monkeypatch.setattr(research_mod, "_read_df", fake_read_df)


def test_data_quality_and_tier2_scan_materialize(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    result = materialize([data_quality, tier2_scan], resources={"dynamo": fake_dynamo})
    assert result.success

    dq = result.asset_materializations_for_node("data_quality")
    assert dq and "symbols_quarantined" in dq[0].metadata

    t2 = result.asset_materializations_for_node("tier2_scan")
    assert t2 and "flags" in t2[0].metadata
    # flags that fired were persisted through the repository, not raw SQL
    assert len(fake_repo.flags) == int(t2[0].metadata["flags"].value)  # type: ignore[union-attr]


def test_tier2_scan_skips_triage_without_api_key(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    result = materialize([data_quality, tier2_scan], resources={"dynamo": fake_dynamo})
    assert result.success
    t2 = result.asset_materializations_for_node("tier2_scan")
    assert t2 and int(t2[0].metadata["triaged_symbols"].value) == 0  # type: ignore[union-attr]
    # no agent_opinions written when the research model is unavailable
    assert all(name != "agent_opinions" for name, _ in fake_repo.inserted_rows)
