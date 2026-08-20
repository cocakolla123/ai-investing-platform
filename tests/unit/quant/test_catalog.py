"""Catalog structure, point-in-time discipline, and missing-input behavior."""

from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd

from dynamo.research.quant.catalog import (
    FEATURE_CATALOG,
    GROUP_TARGETS,
    RAW_GROUPS,
    FeatureInputs,
)
from dynamo.research.quant.features import build_raw_features
from tests.unit.quant.conftest import last_session


def test_catalog_matches_doc_04_group_counts() -> None:
    counts = Counter(spec.group for spec in FEATURE_CATALOG.values())
    assert dict(counts) == GROUP_TARGETS
    assert len(FEATURE_CATALOG) == sum(GROUP_TARGETS.values())  # ~51
    for name, spec in FEATURE_CATALOG.items():
        assert spec.name == name
        assert spec.lookback_days > 0


def test_empty_inputs_never_raise_and_yield_nan() -> None:
    raw = build_raw_features(FeatureInputs(), date(2026, 1, 5), ["AAA", "BBB"])
    assert list(raw.index) == ["AAA", "BBB"]
    assert set(raw.columns) == set(FEATURE_CATALOG)
    # Everything except the always-known calendar month is NaN.
    nan_cols = raw.columns[raw.isna().all()]
    assert "ret_20d" in nan_cols
    assert "ev_ebit" in nan_cols
    assert "hy_oas_level" in nan_cols
    assert raw["month_of_year"].tolist() == [1.0, 1.0]


def test_estimate_revision_features_nan_without_data(
    inputs: FeatureInputs, bars: pd.DataFrame
) -> None:
    as_of = last_session(bars)
    raw = build_raw_features(inputs, as_of, ["AAA", "BBB"])
    for col in ("est_rev_breadth_30d", "est_rev_magnitude_90d", "fwd_estimate_momentum"):
        assert bool(raw[col].isna().all())


def test_estimate_revision_features_live_with_vintage_frame(
    inputs: FeatureInputs, bars: pd.DataFrame
) -> None:
    """With an ``earnings_estimate_vintages``-shaped estimates frame (the
    gateway's new wiring, migration 0019) the three estimate-revision
    features stop being structurally NaN."""
    as_of = last_session(bars)
    estimates = pd.DataFrame(
        [
            # 100 days ago: baseline estimate; 10 days ago: upward revision.
            {"symbol": "AAA", "fetched_at": as_of - pd.Timedelta(days=100), "eps_est": 1.00},
            {"symbol": "AAA", "fetched_at": as_of - pd.Timedelta(days=10), "eps_est": 1.20},
            {"symbol": "BBB", "fetched_at": as_of - pd.Timedelta(days=100), "eps_est": 2.00},
            {"symbol": "BBB", "fetched_at": as_of - pd.Timedelta(days=10), "eps_est": 1.50},
        ]
    )
    with_est = FeatureInputs(bars=bars, sector_map=dict(inputs.sector_map), estimates=estimates)
    raw = build_raw_features(with_est, as_of, ["AAA", "BBB"])
    assert raw.loc["AAA", "est_rev_breadth_30d"] == 1.0  # sign of the revision
    assert raw.loc["BBB", "est_rev_breadth_30d"] == -1.0
    assert abs(raw.loc["AAA", "est_rev_magnitude_90d"] - 0.20) < 1e-12
    assert abs(raw.loc["BBB", "fwd_estimate_momentum"] - (-0.25)) < 1e-12


def test_fundamentals_respect_filed_at(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    as_of = last_session(bars)
    quarters = pd.date_range("2023-03-31", periods=8, freq="QE")
    rows = []
    for i, q in enumerate(quarters):
        filed = q + pd.Timedelta(days=40)
        for tag, value in (("Revenues", 100 + i), ("GrossProfit", 50 + i)):
            rows.append(
                {"symbol": "AAA", "tag": tag, "period_end": q, "value": value, "filed_at": filed}
            )
    # A blockbuster quarter filed AFTER as_of must be invisible.
    rows.append(
        {
            "symbol": "AAA",
            "tag": "Revenues",
            "period_end": quarters[-1] + pd.Timedelta(days=91),
            "value": 10_000.0,
            "filed_at": as_of + pd.Timedelta(days=5),
        }
    )
    fundamentals = pd.DataFrame(rows)
    inp = FeatureInputs(bars=bars, fundamentals=fundamentals, sector_map=inputs.sector_map)
    raw = build_raw_features(inp, as_of, ["AAA"])
    # TTM revenue = last four filed quarters: 104+105+106+107
    gm = raw.loc["AAA", "gross_margin_ttm"]
    expected_gm = (54 + 55 + 56 + 57) / (104 + 105 + 106 + 107)
    assert abs(gm - expected_gm) < 1e-12

    # Same build with the future filing removed is identical (nothing leaked).
    raw2 = build_raw_features(
        FeatureInputs(
            bars=bars,
            fundamentals=fundamentals.iloc[:-1],
            sector_map=inputs.sector_map,
        ),
        as_of,
        ["AAA"],
    )
    pd.testing.assert_frame_equal(raw, raw2)


def test_insider_features_form4_filed_at(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    as_of = last_session(bars)
    insider = pd.DataFrame(
        [
            # three distinct buyers within 30 days -> cluster flag
            {
                "symbol": "AAA",
                "insider_name": "a",
                "tx_type": "P",
                "shares": 100.0,
                "price": 10.0,
                "filed_at": as_of - pd.Timedelta(days=5),
            },
            {
                "symbol": "AAA",
                "insider_name": "b",
                "tx_type": "P",
                "shares": 100.0,
                "price": 10.0,
                "filed_at": as_of - pd.Timedelta(days=10),
            },
            {
                "symbol": "AAA",
                "insider_name": "c",
                "tx_type": "P",
                "shares": 100.0,
                "price": 10.0,
                "filed_at": as_of - pd.Timedelta(days=15),
            },
            # filed after as_of -> invisible
            {
                "symbol": "BBB",
                "insider_name": "d",
                "tx_type": "P",
                "shares": 1e6,
                "price": 100.0,
                "filed_at": as_of + pd.Timedelta(days=1),
            },
        ]
    )
    inp = FeatureInputs(bars=bars, insider=insider, sector_map=inputs.sector_map)
    raw = build_raw_features(inp, as_of, ["AAA", "BBB"])
    assert raw.loc["AAA", "insider_buyer_count_90d"] == 3.0
    assert raw.loc["AAA", "insider_cluster_buy"] == 1.0
    assert raw.loc["AAA", "insider_net_buy_90d"] > 0
    assert pd.isna(raw.loc["BBB", "insider_buyer_count_90d"]) or (
        raw.loc["BBB", "insider_buyer_count_90d"] == 0.0
    )


def test_macro_uses_vintage_discipline(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    as_of = last_session(bars)
    macro = pd.DataFrame(
        [
            # original vintage available at as_of
            {
                "series_id": "BAMLH0A0HYM2",
                "ts": as_of - pd.Timedelta(days=3),
                "value": 3.5,
                "vintage_ts": as_of - pd.Timedelta(days=3),
            },
            # revision published after as_of must be ignored
            {
                "series_id": "BAMLH0A0HYM2",
                "ts": as_of - pd.Timedelta(days=3),
                "value": 9.9,
                "vintage_ts": as_of + pd.Timedelta(days=2),
            },
        ]
    )
    inp = FeatureInputs(bars=bars, macro=macro, sector_map=inputs.sector_map)
    raw = build_raw_features(inp, as_of, ["AAA"])
    assert raw.loc["AAA", "hy_oas_level"] == 3.5


def test_raw_groups_flagged_correctly() -> None:
    assert {"macro", "calendar"} == RAW_GROUPS
