"""Quality score: point-in-time replay, null handling, weight renormalization."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from dynamo.research.quality import SUBSCORE_NAMES, QualityConfig, compute_quality_scores

TAGS = {
    "Revenues": 100.0,
    "GrossProfit": 60.0,
    "OperatingIncomeLoss": 30.0,
    "NetIncomeLoss": 20.0,
    "NetCashProvidedByUsedInOperatingActivities": 28.0,
    "PaymentsToAcquirePropertyPlantAndEquipment": 8.0,
    "InterestExpense": 1.0,
}
BALANCE_TAGS = {
    "StockholdersEquity": 200.0,
    "LongTermDebt": 50.0,
    "CashAndCashEquivalentsAtCarryingValue": 30.0,
    "CommonStockSharesOutstanding": 1000.0,
}


def _facts(
    symbol: str, n_quarters: int, growth: float = 0.02, margin_mult: float = 1.0
) -> list[dict]:
    """Quarterly facts filed ~40 days after each period end."""
    rows = []
    quarters = pd.date_range("2022-03-31", periods=n_quarters, freq="QE")
    for i, q in enumerate(quarters):
        filed = q + pd.Timedelta(days=40)
        scale = (1.0 + growth) ** i
        for tag, base in TAGS.items():
            value = base * scale * (margin_mult if tag != "Revenues" else 1.0)
            rows.append(
                {"symbol": symbol, "tag": tag, "period_end": q, "value": value, "filed_at": filed}
            )
        for tag, base in BALANCE_TAGS.items():
            rows.append(
                {"symbol": symbol, "tag": tag, "period_end": q, "value": base, "filed_at": filed}
            )
    return rows


@pytest.fixture
def fundamentals() -> pd.DataFrame:
    rows = []
    rows += _facts("AAA", 12, growth=0.05, margin_mult=1.2)  # strong grower
    rows += _facts("BBB", 12, growth=0.00, margin_mult=0.8)  # stagnant
    rows += _facts("CCC", 12, growth=0.02, margin_mult=1.0)  # middle
    rows += _facts("DDD", 12, growth=0.03, margin_mult=1.1)  # other sector
    return pd.DataFrame(rows)


SECTOR_MAP = {"AAA": "tech", "BBB": "tech", "CCC": "tech", "DDD": "energy"}
AS_OF = cast(pd.Timestamp, pd.Timestamp("2025-06-30"))


def test_composite_in_bounds_and_subscores_complete(fundamentals: pd.DataFrame) -> None:
    scores = compute_quality_scores(
        fundamentals=fundamentals, sector_map=SECTOR_MAP, as_of=AS_OF
    )
    assert [s.symbol for s in scores] == sorted(SECTOR_MAP)
    for s in scores:
        assert set(s.subscores) == set(SUBSCORE_NAMES)
        for value in s.subscores.values():
            assert value is None or 0.0 <= value <= 100.0
        if s.composite is not None:
            assert 0.0 <= s.composite <= 100.0
        assert s.config_hash == QualityConfig().config_hash
    by = {s.symbol: s for s in scores}
    # Strong grower beats stagnant peer within the same sector.
    assert by["AAA"].composite is not None and by["BBB"].composite is not None
    assert by["AAA"].composite > by["BBB"].composite


def test_point_in_time_replay(fundamentals: pd.DataFrame) -> None:
    """Scores at a historical as_of are identical whether computed then or
    replayed today with later filings present (the filed_at discipline)."""
    historical_as_of = cast(pd.Timestamp, pd.Timestamp("2024-06-30"))
    then_visible = cast(pd.DataFrame, fundamentals[fundamentals["filed_at"] <= historical_as_of])
    computed_then = compute_quality_scores(
        fundamentals=then_visible, sector_map=SECTOR_MAP, as_of=historical_as_of
    )
    replayed_today = compute_quality_scores(
        fundamentals=fundamentals, sector_map=SECTOR_MAP, as_of=historical_as_of
    )
    assert [s.model_dump() for s in computed_then] == [s.model_dump() for s in replayed_today]


def test_insufficient_quarters_nulls_and_composite_ineligible(
    fundamentals: pd.DataFrame,
) -> None:
    rows = _facts("IPO", 4)  # recent IPO: only 4 filed quarters
    fund = pd.concat([fundamentals, pd.DataFrame(rows)], ignore_index=True)
    sector_map = dict(SECTOR_MAP, IPO="tech")
    scores = {s.symbol: s for s in compute_quality_scores(
        fundamentals=fund, sector_map=sector_map, as_of=AS_OF
    )}
    ipo = scores["IPO"]
    assert ipo.subscores["business_quality"] is None
    assert ipo.subscores["revenue_growth"] is None
    assert ipo.subscores["valuation"] is None
    # > 2 null sub-scores -> no composite, ineligible for the long-term sleeve.
    assert ipo.composite is None


def test_weight_renormalization_on_null_subscores(fundamentals: pd.DataFrame) -> None:
    # Without bars, valuation is null for everyone; remaining weights renormalize.
    scores = {s.symbol: s for s in compute_quality_scores(
        fundamentals=fundamentals, sector_map=SECTOR_MAP, as_of=AS_OF
    )}
    s = scores["AAA"]
    assert s.subscores["valuation"] is None
    cfg = QualityConfig()
    live = {k: v for k, v in s.subscores.items() if v is not None}
    total_w = sum(cfg.weights[k] for k in live)
    assert total_w < 1.0  # something was actually null
    expected = sum(cfg.weights[k] / total_w * v for k, v in live.items())
    assert s.composite is not None
    assert abs(s.composite - expected) < 1e-9


def test_qualitative_inputs_blend(fundamentals: pd.DataFrame) -> None:
    base = {s.symbol: s for s in compute_quality_scores(
        fundamentals=fundamentals, sector_map=SECTOR_MAP, as_of=AS_OF
    )}
    with_moat = {s.symbol: s for s in compute_quality_scores(
        fundamentals=fundamentals,
        sector_map=SECTOR_MAP,
        as_of=AS_OF,
        qualitative={"BBB": {"moat": 100.0}},
    )}
    b0 = base["BBB"].subscores["business_quality"]
    b1 = with_moat["BBB"].subscores["business_quality"]
    assert b0 is not None and b1 is not None
    assert b1 == pytest.approx((b0 + 100.0) / 2.0)


def test_config_validation_and_hash() -> None:
    cfg = QualityConfig()
    assert cfg.config_hash == QualityConfig().config_hash
    bad = dict(cfg.weights, business_quality=0.5)
    with pytest.raises(ValueError):
        QualityConfig(weights=bad)
    with pytest.raises(ValueError):
        QualityConfig(weights={"business_quality": 1.0})
    tweaked = QualityConfig(weights=dict(cfg.weights))
    assert tweaked.config_hash == cfg.config_hash  # hash is content-derived
