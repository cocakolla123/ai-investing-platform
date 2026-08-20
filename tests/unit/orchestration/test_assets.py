"""Asset selection smoke: data_quality + tier2_scan on in-memory fakes.

``_read_df`` is monkeypatched to return empty (correctly-shaped) frames, so
no SQL engine is ever built; bars/universe come from the FakeRepo.
"""

from __future__ import annotations

from typing import Any, cast

import orchestration.assets.research as research_mod
import pandas as pd
import pytest
from dagster import MaterializeResult, build_asset_context, materialize
from orchestration.assets.ingestion import oddspipe_spreads
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


def test_oddspipe_spreads_skips_without_key(fake_dynamo: FakeDynamo) -> None:
    """No ODDSPIPE_API_KEY -> graceful skip metadata, nothing persisted."""
    ctx = build_asset_context(partition_key="2026-08-07")
    result = cast(MaterializeResult, oddspipe_spreads(ctx, fake_dynamo))
    assert result.metadata is not None and result.metadata["skipped"] == "oddspipe not configured"


def test_oddspipe_spreads_persists_pairs(fake_dynamo: FakeDynamo, fake_repo: FakeRepo) -> None:
    from datetime import UTC, datetime

    from dynamo.domain.models.pm import CrossVenueSpread

    fake_dynamo.st.oddspipe_spreads = [
        CrossVenueSpread(
            question="Fed funds above 4.00% in December?",
            kalshi_market_id="FED-26DEC-T4.00",
            kalshi_yes=0.47,
            polymarket_id="514729",
            polymarket_yes=0.41,
            yes_diff=0.06,
            as_of=datetime(2026, 8, 7, tzinfo=UTC),
        )
    ]
    ctx = build_asset_context(partition_key="2026-08-07")
    result = cast(MaterializeResult, oddspipe_spreads(ctx, fake_dynamo))
    assert result.metadata is not None and result.metadata["pairs"] == 1
    assert ("pm_cross_venue_spreads", fake_dynamo.st.oddspipe_spreads) in fake_repo.inserted_rows


def test_crypto_bars_skips_when_sleeve_disabled(fake_dynamo: FakeDynamo) -> None:
    """ADR-0013: no provider (sleeve disabled) -> graceful skip metadata."""
    from orchestration.assets.ingestion import crypto_bars

    ctx = build_asset_context(partition_key="2026-08-14")
    result = cast(MaterializeResult, crypto_bars(ctx, fake_dynamo))
    assert result.metadata is not None and result.metadata["skipped"] == "crypto disabled"


def test_crypto_bars_upserts_bars_and_crypto_instruments(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    """ADR-0013: bars land in bars_1d and instrument rows carry sector='crypto'."""
    from datetime import UTC, date, datetime

    from dynamo.domain.models.market import Bar

    class _StubCrypto:
        def daily_bars(self, symbols: Any, start: date, end: date) -> Any:
            for symbol in symbols:
                yield Bar(
                    symbol=symbol,
                    ts=datetime(2026, 8, 14, 5, 0, tzinfo=UTC),
                    open=1.0,
                    high=2.0,
                    low=0.5,
                    close=1.5,
                    volume=100,
                    adj_close=1.5,
                    source="alpaca_crypto",
                )

    from orchestration.assets.ingestion import crypto_bars

    fake_dynamo.st.crypto_provider = _StubCrypto()
    ctx = build_asset_context(partition_key="2026-08-14")
    result = cast(MaterializeResult, crypto_bars(ctx, fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["rows"] == 5  # settings default: 5 pairs
    assert result.metadata["instruments"] == 5
    upserted = dict(fake_repo.inserted_rows)
    assert all(b.source == "alpaca_crypto" for b in upserted["bars_1d"])
    assert all(i.sector == "crypto" for i in upserted["instruments"])
