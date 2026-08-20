"""Multi-timeframe (weekly) features — group 11.

Three things are worth pinning here, and nothing else is:

1. the weekly resample is arithmetically what a human reading a weekly chart
   would compute (a hand-built series with KNOWN weekly structure);
2. it is calendar-invariant — an equity's weekly features do not move when a
   7-day-a-week crypto pair joins the universe (the bug class that killed 16
   daily features; the weekly panel is a second place it could reappear);
3. ``weekly_confirms_daily_breakout`` fires only when BOTH timeframes agree,
   and is NaN — never 0.0 — when either cannot be answered.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import (
    WEEK_END_RULE,
    FeatureInputs,
    build_context,
    to_ts,
)
from dynamo.research.quant.features import build_raw_features


def _all(frame: pd.DataFrame) -> bool:
    """``bool`` of "every cell is True" (pyright cannot narrow DataFrame.all())."""
    return bool(frame.to_numpy().all())


WEEKLY_FEATURES = (
    "weekly_ret_4w",
    "weekly_ret_13w",
    "weekly_ret_26w",
    "weekly_dist_ma10w",
    "weekly_dist_ma30w",
    "weekly_range_contraction_8w",
    "weekly_confirms_daily_breakout",
)


def _bars_from_closes(symbol: str, index: pd.DatetimeIndex, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": index,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "adj_close": closes,
            "volume": 1_000_000,
        }
    )


def _walk(symbols: list[str], freq: str, n: int, seed: int, end: str) -> pd.DataFrame:
    """Random-walk bars on a 5-day (``B``) or 7-day (``D``) calendar."""
    rng = np.random.default_rng(seed)
    idx = (
        pd.bdate_range(end=end, periods=n)
        if freq == "B"
        else pd.date_range(end=end, periods=n, freq="D")
    )
    frames = []
    for k, sym in enumerate(symbols):
        px = 100.0 * (1 + k / 10) * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(idx))))
        frames.append(_bars_from_closes(sym, idx, px))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 1. the resample is what a weekly chart shows
# --------------------------------------------------------------------------


def test_weekly_panel_collapses_each_symbols_own_sessions_to_one_bar_per_week() -> None:
    """One row per calendar week for BOTH a 5-day and a 7-day name, with the
    week's last close, its max high, its min low, and its summed volume."""
    b_idx = pd.bdate_range("2026-01-05", "2026-02-27")  # 8 full Mon-Fri weeks
    d_idx = pd.date_range("2026-01-05", "2026-02-27", freq="D")  # same span, 7d
    bars = pd.concat(
        [
            _bars_from_closes("EQ", b_idx, np.arange(1.0, len(b_idx) + 1.0)),
            _bars_from_closes("CR", d_idx, np.arange(1.0, len(d_idx) + 1.0)),
        ],
        ignore_index=True,
    )
    ctx = build_context(FeatureInputs(bars=bars), to_ts("2026-02-27"), ["EQ", "CR"])
    weekly_close = ctx.weekly("price")
    weekly_volume = ctx.weekly("volume")

    assert len(weekly_close) == 8  # 8 Fridays, shared by both symbols
    assert _all(weekly_close.notna())
    # A ``W-FRI`` week runs Saturday -> Friday, so the second bucket is the
    # first full one: 5 sessions for the equity, 7 for the crypto pair, and
    # both close on the same Friday.
    assert float(weekly_close.iloc[1]["EQ"]) == 10.0  # 10th business day
    assert float(weekly_close.iloc[1]["CR"]) == 12.0  # 12th calendar day
    assert float(weekly_volume.iloc[1]["EQ"]) == 5 * 1_000_000
    assert float(weekly_volume.iloc[1]["CR"]) == 7 * 1_000_000
    assert list(weekly_close.index) == list(
        pd.Series(0.0, index=b_idx).resample(WEEK_END_RULE).last().index
    )


def test_weekly_returns_match_a_hand_computed_weekly_series() -> None:
    """A series built so week ``k`` closes at exactly ``100 * 1.01**k``: the
    4-week return must be ``1.01**4 - 1`` to floating-point precision."""
    idx = pd.bdate_range(end="2026-06-26", periods=5 * 40)  # 40 full weeks
    weeks = np.arange(len(idx)) // 5
    closes = 100.0 * 1.01 ** (weeks + (np.arange(len(idx)) % 5) / 5.0)
    bars = _bars_from_closes("AAA", idx, closes)
    raw = build_raw_features(FeatureInputs(bars=bars), to_ts("2026-06-26"), ["AAA"])

    for weeks_back, col in ((4, "weekly_ret_4w"), (13, "weekly_ret_13w"), (26, "weekly_ret_26w")):
        assert float(raw.loc["AAA", col]) == pytest.approx(1.01**weeks_back - 1.0, rel=1e-9)
    # 10-week MA of a compounding series: the close sits above it.
    assert float(raw.loc["AAA", "weekly_dist_ma10w"]) > 0.0


def test_weekly_lens_is_not_a_smoothed_copy_of_the_daily_one() -> None:
    """The reason this group exists. A name that saws hard inside each week
    but trends up across weeks: the daily distance-from-MA reads deeply
    negative (today is a within-week trough measured against a mean that
    includes every peak) while the weekly one — trough against trough — reads
    positive. Same date, same bars, opposite signs."""
    idx = pd.bdate_range(end="2026-06-26", periods=5 * 40)
    trend = 100.0 * 1.004 ** np.arange(len(idx))
    # A hard sawtooth within each week: Thu/Fri close far below Mon-Wed.
    chop = np.where(np.arange(len(idx)) % 5 >= 3, 0.85, 1.10)
    bars = _bars_from_closes("AAA", idx, trend * chop)
    raw = build_raw_features(FeatureInputs(bars=bars), to_ts("2026-06-26"), ["AAA"])

    assert float(raw.loc["AAA", "dist_ma50"]) < -0.05
    assert float(raw.loc["AAA", "weekly_dist_ma10w"]) > 0.05
    assert float(raw.loc["AAA", "weekly_ret_13w"]) > 0.0


def test_range_contraction_is_higher_for_a_name_that_went_quiet() -> None:
    idx = pd.bdate_range(end="2026-06-26", periods=5 * 40)
    n = len(idx)
    swing = 100.0 + 20.0 * np.sin(np.arange(n) / 4.0)
    # ``quiet`` swings the same way for 30 weeks, then flattens for 10.
    quiet = swing.copy()
    quiet[5 * 30 :] = 100.0 + 0.5 * np.sin(np.arange(n - 5 * 30) / 4.0)
    bars = pd.concat(
        [_bars_from_closes("WIDE", idx, swing), _bars_from_closes("QUIET", idx, quiet)],
        ignore_index=True,
    )
    raw = build_raw_features(FeatureInputs(bars=bars), to_ts("2026-06-26"), ["WIDE", "QUIET"])
    contraction = cast(pd.Series, raw["weekly_range_contraction_8w"])
    assert float(contraction["QUIET"]) > float(contraction["WIDE"])
    assert float(contraction["QUIET"]) > 0.8  # nearly all the range is gone


# --------------------------------------------------------------------------
# 2. calendar invariance
# --------------------------------------------------------------------------


def test_weekly_features_are_invariant_to_crypto_in_the_universe() -> None:
    """The weekly panel is a second union-calendar hazard: a 7-day pair adds
    no weekly rows an equity lacks, but the resample must still run on each
    symbol's own sessions. Same equities, same date, identical values."""
    equities = _walk(["AAA", "BBB", "XLK"], "B", 620, seed=11, end="2026-03-03")
    crypto = _walk(["BTC/USD", "ETH/USD"], "D", 868, seed=12, end="2026-03-03")
    sector_map = {"AAA": "XLK", "BBB": "XLK"}
    as_of = to_ts("2026-03-03")

    alone = build_raw_features(
        FeatureInputs(bars=equities, sector_map=sector_map), as_of, ["AAA", "BBB", "XLK"]
    )
    mixed = build_raw_features(
        FeatureInputs(bars=pd.concat([equities, crypto], ignore_index=True), sector_map=sector_map),
        as_of,
        ["AAA", "BBB", "XLK", "BTC/USD", "ETH/USD"],
    )
    pd.testing.assert_frame_equal(
        mixed.loc[["AAA", "BBB", "XLK"], list(WEEKLY_FEATURES)],
        alone.loc[:, list(WEEKLY_FEATURES)],
    )
    # ...and they are actually live, not invariantly NaN.
    assert _all(mixed.loc[["AAA", "BBB", "XLK"], list(WEEKLY_FEATURES)].notna())
    assert _all(mixed.loc[["BTC/USD", "ETH/USD"], list(WEEKLY_FEATURES)].notna())


# --------------------------------------------------------------------------
# 3. weekly-confirms-daily
# --------------------------------------------------------------------------


def _confirm(closes: np.ndarray) -> float:
    idx = pd.bdate_range(end="2026-06-26", periods=len(closes))
    bars = _bars_from_closes("AAA", idx, closes)
    raw = build_raw_features(FeatureInputs(bars=bars), to_ts("2026-06-26"), ["AAA"])
    return float(raw.loc["AAA", "weekly_confirms_daily_breakout"])


def test_confirmation_fires_only_when_both_timeframes_break_out() -> None:
    n = 5 * 40
    # Both agree: a steady climb makes today the 20-session AND 26-week high.
    assert _confirm(100.0 * 1.002 ** np.arange(n)) == 1.0

    # Daily only: a six-month slide with a sharp three-week bounce. Today is a
    # 20-session high but nowhere near the 26-week high.
    slide = np.concatenate(
        [np.linspace(200.0, 100.0, n - 15), np.linspace(100.0, 130.0, 15)]
    )
    assert _confirm(slide) == 0.0

    # Weekly only: a steady climb (so today IS the highest weekly close) with
    # a one-day intraweek spike three sessions ago that today never took out.
    # The weekly leg passes, the daily leg does not — no confirmation.
    spiked = 100.0 * 1.002 ** np.arange(n)
    spiked[-3] *= 1.05
    assert _confirm(spiked) == 0.0


def test_confirmation_is_nan_not_zero_without_enough_weekly_history() -> None:
    """A name with 12 weeks of bars cannot answer "26-week high?". It must
    come back NaN — reporting 0.0 would score it identically to a name that
    genuinely failed the weekly leg."""
    short = 100.0 * 1.002 ** np.arange(5 * 12)
    assert np.isnan(_confirm(short))


def test_weekly_features_are_nan_without_bars() -> None:
    raw = build_raw_features(FeatureInputs(), to_ts("2026-06-26"), ["AAA", "BBB"])
    assert _all(raw.loc[:, list(WEEKLY_FEATURES)].isna())
