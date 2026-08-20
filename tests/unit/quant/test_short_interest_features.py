"""Short-interest features (group 15) and their point-in-time bound.

The load-bearing assertion in this file is the one about dates: a cycle whose
publication date is after ``as_of`` must be INVISIBLE, even though its
settlement date is comfortably in the past. Getting that wrong manufactures a
gorgeous short-interest signal out of eight days of look-ahead, twice a month,
and nothing else in the pipeline would notice.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dynamo.data.finra import publication_date_for
from dynamo.research.quant.catalog import (
    FEATURE_CATALOG,
    FeatureInputs,
    applicable_features,
    to_ts,
)
from dynamo.research.quant.features import build_raw_features

AS_OF = to_ts("2026-08-11")
SHORT_INTEREST_FEATURES = (
    "short_interest_pct_float",
    "days_to_cover",
    "short_interest_change",
)


def _bars(symbols: list[str], n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-08-11", periods=n)
    frames = []
    for k, sym in enumerate(symbols):
        px = np.full(n, 50.0 * (k + 1))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "ts": idx,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_close": px,
                    "volume": 1_000_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cycle(
    symbol: str,
    settlement: date,
    shares_short: float,
    previous: float | None = None,
    avg_daily_volume: float | None = 500_000.0,
    days_to_cover: float | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "settlement_date": settlement,
        "publication_date": publication_date_for(settlement),
        "shares_short": shares_short,
        "previous_shares_short": previous,
        "avg_daily_volume": avg_daily_volume,
        "days_to_cover": days_to_cover,
    }


def _shares_outstanding(symbol: str, shares: float) -> dict:
    return {
        "symbol": symbol,
        "tag": "CommonStockSharesOutstanding",
        "period_end": pd.Timestamp("2026-06-30"),
        "value": shares,
        "filed_at": pd.Timestamp("2026-07-20"),
    }


def _raw(
    short_interest: list[dict],
    fundamentals: list[dict] | None = None,
    symbols: list[str] | None = None,
    as_of: pd.Timestamp = AS_OF,
) -> pd.DataFrame:
    universe = symbols or ["AAA"]
    return build_raw_features(
        FeatureInputs(
            bars=_bars(universe),
            fundamentals=pd.DataFrame(fundamentals) if fundamentals else None,
            short_interest=pd.DataFrame(short_interest),
        ),
        as_of,
        universe,
    )


# --------------------------------------------------------------------------
# the point-in-time rule
# --------------------------------------------------------------------------


def test_a_settled_but_unpublished_cycle_is_invisible() -> None:
    """2026-07-31 settles inside the window but publishes on 2026-08-10. Read
    on 2026-08-05 the features must still be reporting the 2026-07-15 cycle;
    read on 2026-08-11 they must have moved on."""
    cycles = [
        _cycle("AAA", date(2026, 7, 15), shares_short=1_000_000.0, previous=900_000.0),
        _cycle("AAA", date(2026, 7, 31), shares_short=4_000_000.0, previous=1_000_000.0),
    ]
    facts = [_shares_outstanding("AAA", 100_000_000.0)]

    early = _raw(cycles, facts, as_of=to_ts("2026-08-05"))
    assert float(early.loc["AAA", "short_interest_pct_float"]) == 0.01  # the OLD cycle
    assert float(early.loc["AAA", "short_interest_change"]) == (1_000_000 / 900_000) - 1

    late = _raw(cycles, facts, as_of=to_ts("2026-08-11"))
    assert float(late.loc["AAA", "short_interest_pct_float"]) == 0.04  # the NEW cycle
    assert float(late.loc["AAA", "short_interest_change"]) == 3.0


def test_features_are_nan_before_any_cycle_has_been_published() -> None:
    cycles = [_cycle("AAA", date(2026, 7, 31), shares_short=1_000_000.0, previous=1.0)]
    raw = _raw(cycles, [_shares_outstanding("AAA", 100_000_000.0)], as_of=to_ts("2026-08-01"))
    for col in SHORT_INTEREST_FEATURES:
        assert np.isnan(float(raw.loc["AAA", col]))


def test_features_are_nan_without_any_short_interest_input() -> None:
    raw = build_raw_features(FeatureInputs(bars=_bars(["AAA"])), AS_OF, ["AAA"])
    for col in SHORT_INTEREST_FEATURES:
        assert np.isnan(float(raw.loc["AAA", col]))


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------


def test_percent_of_float_uses_point_in_time_shares_outstanding() -> None:
    cycles = [_cycle("AAA", date(2026, 7, 31), shares_short=12_500_000.0, previous=1.0)]
    raw = _raw(cycles, [_shares_outstanding("AAA", 50_000_000.0)])
    assert float(raw.loc["AAA", "short_interest_pct_float"]) == 0.25


def test_percent_of_float_is_nan_without_a_filed_share_count() -> None:
    """No XBRL, no denominator — and no guess."""
    cycles = [_cycle("AAA", date(2026, 7, 31), shares_short=12_500_000.0, previous=1.0)]
    raw = _raw(cycles, fundamentals=None)
    assert np.isnan(float(raw.loc["AAA", "short_interest_pct_float"]))
    # ...while the two features that need no share count still work.
    assert float(raw.loc["AAA", "days_to_cover"]) > 0.0


def test_days_to_cover_prefers_the_published_field_and_recomputes_when_absent() -> None:
    published = [
        _cycle(
            "AAA",
            date(2026, 7, 31),
            shares_short=1_000_000.0,
            previous=1.0,
            avg_daily_volume=500_000.0,
            days_to_cover=7.25,  # FINRA's own number wins over 1M/500k = 2.0
        )
    ]
    assert float(_raw(published).loc["AAA", "days_to_cover"]) == 7.25

    absent = [
        _cycle(
            "AAA",
            date(2026, 7, 31),
            shares_short=1_000_000.0,
            previous=1.0,
            avg_daily_volume=500_000.0,
            days_to_cover=None,
        )
    ]
    assert float(_raw(absent).loc["AAA", "days_to_cover"]) == 2.0


def test_the_999_sentinel_on_a_zero_volume_name_is_dropped_not_believed() -> None:
    """FINRA emits days-to-cover 999.99 when average daily volume is 0. That
    is a sentinel, not 999 days: taken literally it would top every
    cross-sectional ranking of short-squeeze pressure with untraded OTC
    shells."""
    cycles = [
        _cycle(
            "AAA",
            date(2026, 7, 31),
            shares_short=50_218.0,
            previous=54_440.0,
            avg_daily_volume=0.0,
            days_to_cover=999.99,
        )
    ]
    raw = _raw(cycles)
    assert np.isnan(float(raw.loc["AAA", "days_to_cover"]))
    # The other two features are unaffected — a zero-volume name still has a
    # real short position and a real change in it.
    assert float(raw.loc["AAA", "short_interest_change"]) < 0.0


def test_change_is_signed_and_reads_the_cycle_over_cycle_move() -> None:
    cycles = [
        _cycle("UP", date(2026, 7, 31), shares_short=1_500_000.0, previous=1_000_000.0),
        _cycle("DOWN", date(2026, 7, 31), shares_short=800_000.0, previous=1_000_000.0),
        _cycle("NEW", date(2026, 7, 31), shares_short=900_000.0, previous=0.0),
    ]
    raw = _raw(cycles, symbols=["UP", "DOWN", "NEW"])
    assert float(raw.loc["UP", "short_interest_change"]) == 0.5
    assert float(raw.loc["DOWN", "short_interest_change"]) == pytest.approx(-0.2)
    # No prior position is not a +inf move; it is unknown.
    assert np.isnan(float(raw.loc["NEW", "short_interest_change"]))


# --------------------------------------------------------------------------
# asset-class applicability
# --------------------------------------------------------------------------


def test_a_crypto_pair_is_not_charged_for_missing_short_interest() -> None:
    """A crypto pair appears in no FINRA file at all, so the whole group is
    structurally inapplicable — coverage must not penalize it. An ETF IS in
    the file, so only the share-count feature drops out for it."""
    names = list(FEATURE_CATALOG)
    crypto = set(applicable_features("crypto", names))
    etf = set(applicable_features("etf", names))
    equity = set(applicable_features("equity", names))

    assert not (crypto & set(SHORT_INTEREST_FEATURES))
    assert "days_to_cover" in etf
    assert "short_interest_change" in etf
    assert "short_interest_pct_float" not in etf
    assert set(SHORT_INTEREST_FEATURES) <= equity


def test_the_latest_cycle_is_one_whole_row_not_a_column_wise_merge() -> None:
    """A pandas trap worth a test. ``groupby("symbol").last()`` skips NaN per
    COLUMN, so a cycle reporting no days-to-cover would inherit the previous
    cycle's figure while its shares_short came from the current one — a row
    stitched from two different fortnights, and nothing would look wrong.

    Here the newer cycle omits both optional fields. days_to_cover must be
    recomputed from the NEW cycle (2.0), never the old cycle's published 9.0,
    and the change must be NaN rather than the old cycle's move.
    """
    cycles = [
        _cycle(
            "AAA",
            date(2026, 7, 15),
            shares_short=5_000_000.0,
            previous=4_000_000.0,
            avg_daily_volume=500_000.0,
            days_to_cover=9.0,
        ),
        _cycle(
            "AAA",
            date(2026, 7, 31),
            shares_short=1_000_000.0,
            previous=None,
            avg_daily_volume=500_000.0,
            days_to_cover=None,
        ),
    ]
    raw = _raw(cycles)
    assert float(raw.loc["AAA", "days_to_cover"]) == 2.0
    assert np.isnan(float(raw.loc["AAA", "short_interest_change"]))
