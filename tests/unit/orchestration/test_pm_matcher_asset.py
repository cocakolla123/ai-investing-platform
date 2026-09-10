"""Matcher fallback in oddspipe_spreads: rows reach pm_cross_venue_spreads
and flow through _latest_spread_pairs -> find_cross_venue_arbs as UNVERIFIED
(never tradable candidates without the pm_arb_verified_pairs settings entry).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from dagster import MaterializeResult, build_asset_context
from orchestration.assets.ingestion import oddspipe_spreads

from dynamo.domain.models.market import PredictionMarket
from dynamo.domain.models.pm import CrossVenueSpread
from tests.unit.orchestration.conftest import FakeDynamo, FakeRepo

KALSHI_ID = "KXFED-26DEC-T4.00"
POLY_ID = "514729"
PAIR_ID = f"{KALSHI_ID}|{POLY_ID}"


def _seed_matcher_feedstock(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.kalshi_markets = [
        PredictionMarket(
            venue="kalshi",
            market_id=KALSHI_ID,
            question="Will the Fed cut rates in December 2026?",
        ),
        PredictionMarket(
            venue="kalshi",
            market_id="KXNOPE-26",
            question="Completely unrelated question about llamas?",
        ),
    ]
    fake_dynamo.st.kalshi_yes = {KALSHI_ID: 0.30}
    fake_dynamo.st.polymarket_markets = [
        (
            PredictionMarket(
                venue="polymarket",
                market_id=POLY_ID,
                question="Fed cut rates in December 2026?",
            ),
            0.85,
        ),
        (
            PredictionMarket(
                venue="polymarket",
                market_id="999999",
                question="Will it rain in Paris tomorrow?",
            ),
            0.10,
        ),
    ]


def _matcher_rows(fake_repo: FakeRepo) -> list[CrossVenueSpread]:
    rows = [r for name, r in fake_repo.inserted_rows if name == "pm_cross_venue_spreads"]
    assert rows, "matcher wrote nothing to pm_cross_venue_spreads"
    return cast(list[CrossVenueSpread], rows[-1])


def test_matcher_fallback_writes_candidate_rows(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    """OddsPipe unconfigured -> matcher pairs land in pm_cross_venue_spreads."""
    _seed_matcher_feedstock(fake_dynamo)
    ctx = build_asset_context(partition_key="2026-09-09")
    result = cast(MaterializeResult, oddspipe_spreads(ctx, fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["source"] == "matcher"
    assert result.metadata["pairs"] == 1
    spreads = _matcher_rows(fake_repo)
    assert len(spreads) == 1
    spread = spreads[0]
    assert spread.kalshi_market_id == KALSHI_ID
    assert spread.polymarket_id == POLY_ID
    assert spread.kalshi_yes == 0.30  # from the stored prediction_market_prices row
    assert spread.polymarket_yes == 0.85  # from the Gamma listing
    assert spread.yes_diff == spread.kalshi_yes - spread.polymarket_yes


def test_oddspipe_stays_primary_when_it_returns_pairs(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo
) -> None:
    """OddsPipe pairs present -> matcher never runs, oddspipe rows written."""
    from datetime import UTC, datetime

    _seed_matcher_feedstock(fake_dynamo)
    oddspipe_pairs = [
        CrossVenueSpread(
            question="Fed funds above 4.00% in December?",
            kalshi_market_id="FED-26DEC-T4.00",
            kalshi_yes=0.47,
            polymarket_id="222222",
            polymarket_yes=0.41,
            yes_diff=0.06,
            as_of=datetime(2026, 9, 9, tzinfo=UTC),
        )
    ]
    fake_dynamo.st.oddspipe_spreads = list(oddspipe_pairs)
    ctx = build_asset_context(partition_key="2026-09-09")
    result = cast(MaterializeResult, oddspipe_spreads(ctx, fake_dynamo))
    assert result.metadata is not None
    assert result.metadata["source"] == "oddspipe"
    assert ("pm_cross_venue_spreads", oddspipe_pairs) in fake_repo.inserted_rows


def test_matcher_rows_flow_to_arb_finder_as_unverified(
    fake_dynamo: FakeDynamo, fake_repo: FakeRepo, tmp_path: Path
) -> None:
    """Written matcher rows -> _latest_spread_pairs -> find_cross_venue_arbs:
    UNVERIFIED without the settings entry, tradable candidates only with it."""
    from orchestration.assets.pm_trading import _latest_spread_pairs
    from sqlalchemy import create_engine

    from dynamo.persistence import schema
    from dynamo.risk.pm_arbitrage import find_cross_venue_arbs

    _seed_matcher_feedstock(fake_dynamo)
    ctx = build_asset_context(partition_key="2026-09-09")
    cast(MaterializeResult, oddspipe_spreads(ctx, fake_dynamo))
    spreads = _matcher_rows(fake_repo)

    # Persist the exact rows the asset produced into a real (sqlite) copy of
    # the pm_cross_venue_spreads table, then read them back the way the
    # pm_arbitrage_sim asset does.
    engine = create_engine(f"sqlite:///{tmp_path / 'spreads.db'}", future=True)
    schema.pm_cross_venue_spreads.create(engine)
    with engine.begin() as conn:
        conn.execute(
            schema.pm_cross_venue_spreads.insert(),
            [
                {
                    "question": s.question,
                    "kalshi_market_id": s.kalshi_market_id,
                    "polymarket_id": s.polymarket_id,
                    "kalshi_yes": s.kalshi_yes,
                    "polymarket_yes": s.polymarket_yes,
                    "yes_diff": s.yes_diff,
                    "ts": s.as_of.replace(tzinfo=None),
                }
                for s in spreads
            ],
        )

    class _EngineOnly:
        def engine(self) -> Any:
            return engine

    pairs = _latest_spread_pairs(cast(Any, _EngineOnly()))
    assert [p.kalshi_market_id for p in pairs] == [KALSHI_ID]

    # Without the settings entry: research row, never a tradable candidate.
    result = find_cross_venue_arbs(pairs, min_net_edge=0.02, verified_pairs=frozenset())
    assert result.candidates == ()
    assert [c.pair_id for c in result.unverified] == [PAIR_ID]

    # Only the explicit human-verified settings entry promotes it.
    verified = find_cross_venue_arbs(pairs, min_net_edge=0.02, verified_pairs={PAIR_ID})
    assert [c.pair_id for c in verified.candidates] == [PAIR_ID]
    assert verified.unverified == ()
