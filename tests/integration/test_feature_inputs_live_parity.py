"""End-to-end: the nightly builder and the replay gateway produce the SAME
``FeatureInputs`` from the same database at the same ``as_of``.

The unit-level companion
(``tests/unit/orchestration/test_feature_inputs_parity.py``) compares the two
constructor calls statically. This one seeds one real Postgres and runs both
builders for real, because the failure that shipped was not a missing keyword
— it was a missing SQL read behind a keyword that was already there.

``as_of`` is set to TODAY, because that is what the nightly pipeline passes:
the live builder has no upper bound (it cannot see the future), so the two
agree only when they read the same tables over the same windows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]

AS_OF = date.today()
SYMBOLS = ["ACME", "PEER"]


def _dt(d: date, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


@pytest.fixture()
def engine() -> Iterator[Any]:
    from sqlalchemy import create_engine, text

    from dynamo.persistence.schema import metadata

    eng = create_engine(INTEGRATION_URL, future=True)
    with eng.begin() as cx:
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    metadata.create_all(eng)
    yield eng
    metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def seeded(engine: Any) -> Any:
    """Rows in every table both builders read, on both sides of ``as_of``."""
    from dynamo.domain.models.market import (
        Bar,
        EarningsEvent,
        InsiderTx,
        Instrument,
        MacroPoint,
        NewsItem,
        XbrlFact,
    )
    from dynamo.persistence import schema
    from dynamo.persistence.repositories.market import MarketDataRepository

    repo = MarketDataRepository(engine)
    repo.upsert_instruments(
        [
            Instrument(symbol="ACME", name="Acme Corp", sector="Industrials", cik="0000000001"),
            Instrument(symbol="PEER", name="Peer Inc", sector="Industrials", cik="0000000002"),
        ]
    )
    repo.upsert_bars(
        [
            Bar(
                symbol=sym,
                ts=_dt(AS_OF - timedelta(days=offset), 20),
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=1_000,
                source="tiingo",
            )
            for sym in SYMBOLS
            for offset in range(1, 30)
        ]
    )
    repo.upsert_xbrl_facts(
        [
            XbrlFact(
                cik="0000000001",
                tag="Revenues",
                period_end=AS_OF - timedelta(days=60),
                value=1_000.0,
                unit="USD",
                filed_at=_dt(AS_OF - timedelta(days=40)),
            )
        ]
    )
    repo.upsert_insider_tx(
        [
            InsiderTx(
                accession_no="0001-24-000001",
                symbol="ACME",
                insider_name="A Director",
                role="Director",
                tx_type="P",
                shares=1_000.0,
                price=10.0,
                filed_at=_dt(AS_OF - timedelta(days=5)),
            )
        ]
    )
    repo.upsert_macro(
        [
            MacroPoint(
                series_id="DGS10",
                ts=AS_OF - timedelta(days=3),
                value=4.2,
                vintage_ts=_dt(AS_OF - timedelta(days=3)),
            )
        ]
    )
    repo.upsert_earnings(
        [
            EarningsEvent(
                symbol="ACME",
                event_date=AS_OF + timedelta(days=10),
                period="Q2",
                eps_est=1.0,
            )
        ]
    )
    repo.upsert_news(
        [
            NewsItem(
                id=f"news-{offset}",
                symbols=("ACME",),
                headline=f"Acme headline {offset}",
                source="test",
                published_at=_dt(AS_OF - timedelta(days=offset)),
            )
            for offset in (1, 30, 200)
        ]
    )
    with engine.begin() as cx:
        cx.execute(
            schema.short_interest.insert(),
            [
                {
                    "symbol": "ACME",
                    "settlement_date": AS_OF - timedelta(days=20),
                    "publication_date": AS_OF - timedelta(days=12),
                    "shares_short": 1_000.0,
                    "previous_shares_short": 900.0,
                    "avg_daily_volume": 500.0,
                    "days_to_cover": 2.0,
                },
                # Settled before as_of, published AFTER it: the ~8-day FINRA
                # dissemination lag. Both builders must be blind to this.
                {
                    "symbol": "ACME",
                    "settlement_date": AS_OF - timedelta(days=1),
                    "publication_date": AS_OF + timedelta(days=7),
                    "shares_short": 5_000.0,
                    "previous_shares_short": 1_000.0,
                    "avg_daily_volume": 500.0,
                    "days_to_cover": 10.0,
                },
            ],
        )
    return repo


class _Dynamo:
    """The two ``DynamoResource`` methods ``_feature_inputs`` actually uses."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        from dynamo.persistence.repositories.market import MarketDataRepository

        self._repo = MarketDataRepository(engine)

    def engine(self) -> Any:
        return self._engine

    def bars_window(self, symbols: Any, as_of: date, days: int) -> pd.DataFrame:
        return self._repo.bars_frame(list(symbols), as_of - timedelta(days=days), as_of)


def _normalize(frame: pd.DataFrame | None) -> Any:
    if frame is None:
        return None
    if frame.empty:
        return "empty"
    cols = sorted(frame.columns)
    return (
        frame.loc[:, cols]
        .sort_values(cols)
        .reset_index(drop=True)
        .astype(str)
        .to_dict(orient="records")
    )


FRAME_FIELDS = (
    "bars",
    "fundamentals",
    "insider",
    "estimates",
    "macro",
    "earnings_calendar",
    "news",
    "short_interest",
)


def test_live_and_gateway_feature_inputs_agree(engine: Any, seeded: Any) -> None:
    from orchestration.assets.research import _feature_inputs

    from dynamo.replay.gateway import HistoricalDataGateway

    live = _feature_inputs(_Dynamo(engine), SYMBOLS, AS_OF)  # type: ignore[arg-type]
    replay = HistoricalDataGateway(engine, AS_OF).feature_inputs(SYMBOLS)

    populated_live = {f for f in FRAME_FIELDS if getattr(live, f) is not None}
    populated_replay = {f for f in FRAME_FIELDS if getattr(replay, f) is not None}
    assert populated_live == populated_replay
    # The four dark features' inputs specifically.
    assert {"news", "short_interest"} <= populated_live
    assert live.sector_map == replay.sector_map

    for field in FRAME_FIELDS:
        assert _normalize(getattr(live, field)) == _normalize(getattr(replay, field)), (
            f"FeatureInputs.{field} differs between the nightly builder and the "
            "replay gateway"
        )


def test_neither_builder_sees_a_cycle_published_after_as_of(
    engine: Any, seeded: Any
) -> None:
    from orchestration.assets.research import _feature_inputs

    from dynamo.replay.gateway import HistoricalDataGateway

    live = _feature_inputs(_Dynamo(engine), SYMBOLS, AS_OF)  # type: ignore[arg-type]
    replay = HistoricalDataGateway(engine, AS_OF).feature_inputs(SYMBOLS)
    for frame in (live.short_interest, replay.short_interest):
        assert frame is not None
        assert 5_000.0 not in set(frame["shares_short"])
        assert set(frame["settlement_date"]) == {AS_OF - timedelta(days=20)}


def test_dark_features_are_no_longer_nan_in_the_live_path(
    engine: Any, seeded: Any
) -> None:
    """The point of the wiring: the five features that were NaN nightly now
    compute from the live builder's frames."""
    from orchestration.assets.research import _feature_inputs

    from dynamo.research.quant.features import build_features

    live = _feature_inputs(_Dynamo(engine), SYMBOLS, AS_OF)  # type: ignore[arg-type]
    frame = build_features(live, AS_OF, SYMBOLS)
    assert not frame.empty
    for name in ("news_count_60d", "short_interest_pct_float", "days_to_cover"):
        if name in frame.columns:
            assert frame[name].notna().any(), f"{name} is still entirely NaN"
