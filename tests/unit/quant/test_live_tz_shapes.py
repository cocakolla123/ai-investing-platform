"""The feature path must accept the LIVE timestamp shape, not just the test one.

Repository reads return timestamptz columns (tz-aware UTC). Test fixtures and
the replay gateway hand the catalog tz-naive values. Every data column is
normalized where it is read, but ``as_of`` arrives from the caller — and the
live ``strategy_signals`` asset derived its sessions from aware bar times, so
an aware ``as_of`` met a naive panel and raised ``Cannot compare tz-naive and
tz-aware`` on every live night (2026-09-15 onward) while every test passed.

These tests feed the aware shape and require the SAME numbers as the naive one.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import pytest

from dynamo.research.quant.catalog import FeatureInputs, naive_utc_ts
from dynamo.research.quant.features import build_raw_features
from tests.unit.quant.conftest import last_session, make_bars


def _aware(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.tz_localize("UTC")
    return out


def test_naive_utc_ts_converts_aware_to_naive_utc() -> None:
    aware = pd.Timestamp("2026-09-22 20:00", tz="America/New_York")
    assert naive_utc_ts(aware) == pd.Timestamp("2026-09-23 00:00")
    assert naive_utc_ts(aware).tzinfo is None


@pytest.mark.parametrize("value", [pd.Timestamp("2026-09-22"), date(2026, 9, 22), "2026-09-22"])
def test_naive_utc_ts_leaves_naive_values_alone(value: object) -> None:
    assert naive_utc_ts(value) == pd.Timestamp("2026-09-22")


def test_aware_bars_and_aware_as_of_match_the_naive_computation() -> None:
    naive_bars = make_bars(["AAA", "BBB", "CCC", "XLK"])
    sector_map = {"AAA": "XLK", "BBB": "XLK", "CCC": "XLK"}
    universe = ["AAA", "BBB", "CCC"]
    as_of = last_session(naive_bars)

    expected = build_raw_features(
        FeatureInputs(bars=naive_bars, sector_map=sector_map), as_of, universe
    )
    live = build_raw_features(
        FeatureInputs(bars=_aware(naive_bars), sector_map=sector_map),
        cast(pd.Timestamp, as_of.tz_localize("UTC")),
        universe,
    )
    pd.testing.assert_frame_equal(live, expected)
    assert expected.notna().to_numpy().any(), "the comparison must not be vacuous"
