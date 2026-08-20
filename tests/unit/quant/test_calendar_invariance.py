"""THE regression test for the campaign's deepest defect.

``catalog._pivot`` builds ONE price panel indexed by the UNION of every
symbol's bar timestamps. Crypto trades 7 days a week, equities 5, so adding
five crypto pairs to a 50-name equity universe inflated the panel from ~288
real rows to 421 calendar rows — and every rolling window (which required
``min_periods == window`` CALENDAR rows) went NaN for every equity. At
2026-03-03 a 50-day moving average was computable for exactly 5 of 55
symbols: the 5 crypto pairs. 16 features carrying 28.1% of the trained
model's importance mass were dead for all 50 equities.

The invariant that catches it, and that nothing asserted before:

    AN EQUITY'S FEATURES DO NOT DEPEND ON WHO ELSE IS IN THE UNIVERSE.

Cross-sectional NORMALIZATION legitimately depends on the peer set, so this
pins the RAW feature matrix (``build_raw_features``) — the layer where the
calendar bug lived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import FeatureInputs, to_ts
from dynamo.research.quant.features import build_raw_features

EQUITIES = ["AAA", "BBB", "CCC", "XLK"]
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]
SECTOR_MAP = {"AAA": "XLK", "BBB": "XLK", "CCC": "XLK"}

# The features the calendar bug killed outright (doc: campaign findings §1).
ROLLING_FEATURES = (
    "dist_ma50",
    "dist_ma200",
    "ma50_slope",
    "ret_10d_z",
    "dist_20d_mean_sigma",
    "drawdown_20d_high",
    "vol_cc_20d",
    "vol_parkinson_20d",
    "vol_of_vol_60d",
    "vol_pctile_2y",
    "atr14_over_price",
    "dollar_vol_20d",
    "dollar_vol_z_60d",
    "volume_trend_20_60",
    "amihud_20d",
    "days_to_exit",
)
POSITIONAL_FEATURES = ("ret_20d", "ret_60d", "ret_120d", "mom_12_1")
# rs_20d_sector needs a sector ETF, so only the operating companies have one.
OPERATING_COMPANIES = ["AAA", "BBB", "CCC"]


def _bars(symbols: list[str], freq: str, n: int, seed: int) -> pd.DataFrame:
    """Bars on a 5-day (``B``) or 7-day (``D``) calendar, same end date."""
    rng = np.random.default_rng(seed)
    ts = (
        pd.bdate_range(end="2026-03-03", periods=n)
        if freq == "B"
        else pd.date_range(end="2026-03-03", periods=n, freq="D")
    )
    rows = []
    for k, sym in enumerate(symbols):
        px = 100.0 * (1 + k / 10) * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(ts))))
        vol = rng.integers(100_000, 1_000_000, len(ts))
        for t, p, v in zip(ts, px, vol, strict=True):
            rows.append(
                {
                    "symbol": sym,
                    "ts": t,
                    "open": p * 0.999,
                    "high": p * 1.012,
                    "low": p * 0.988,
                    "close": p,
                    "adj_close": p,
                    "volume": int(v),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def equity_bars() -> pd.DataFrame:
    # ~2.5 years of sessions: enough for the 200-day MA and the 2y vol pctile.
    return _bars(EQUITIES, "B", 620, seed=11)


@pytest.fixture(scope="module")
def crypto_bars() -> pd.DataFrame:
    return _bars(CRYPTO, "D", 868, seed=12)  # same span, 7 days a week


AS_OF = to_ts("2026-03-03")


def _raw(bars: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    return build_raw_features(FeatureInputs(bars=bars, sector_map=SECTOR_MAP), AS_OF, universe)


def test_equity_features_are_invariant_to_crypto_in_the_universe(
    equity_bars: pd.DataFrame, crypto_bars: pd.DataFrame
) -> None:
    """Same equities, same date, same bars — with and without crypto in the
    panel. Every raw feature value must be IDENTICAL."""
    without = _raw(equity_bars, EQUITIES)
    with_crypto = _raw(
        pd.concat([equity_bars, crypto_bars], ignore_index=True), [*EQUITIES, *CRYPTO]
    )
    pd.testing.assert_frame_equal(with_crypto.loc[EQUITIES], without)


def test_equity_feature_coverage_is_invariant_to_crypto_in_the_universe(
    equity_bars: pd.DataFrame, crypto_bars: pd.DataFrame
) -> None:
    """The headline symptom, stated as coverage: per-symbol non-NaN counts
    must not move when crypto joins the universe (0.669 -> 0.287 was the
    dated train/serve skew this reproduces)."""
    without = _raw(equity_bars, EQUITIES).notna().mean(axis=1)
    with_crypto = (
        _raw(pd.concat([equity_bars, crypto_bars], ignore_index=True), [*EQUITIES, *CRYPTO])
        .loc[EQUITIES]
        .notna()
        .mean(axis=1)
    )
    pd.testing.assert_series_equal(with_crypto, without)


def test_rolling_features_are_live_for_equities_in_a_mixed_calendar_universe(
    equity_bars: pd.DataFrame, crypto_bars: pd.DataFrame
) -> None:
    """The direct assertion: dist_ma50 (and the other 15 casualties) must be
    non-NaN for EVERY equity when crypto shares the panel. Before the fix
    exactly the crypto pairs had a computable 50-day MA."""
    raw = _raw(pd.concat([equity_bars, crypto_bars], ignore_index=True), [*EQUITIES, *CRYPTO])
    for feature in ROLLING_FEATURES:
        alive = raw.loc[EQUITIES, feature].notna()
        assert bool(alive.all()), f"{feature} NaN for {list(alive[~alive].index)}"
    # ...and for the crypto pairs too — the fix must not break the names that
    # happened to work.
    for feature in ("dist_ma50", "vol_cc_20d", "dollar_vol_20d"):
        assert bool(raw.loc[CRYPTO, feature].notna().all()), feature


def test_positional_lookups_are_counted_in_sessions_not_calendar_days(
    equity_bars: pd.DataFrame, crypto_bars: pd.DataFrame
) -> None:
    """``_ret``/``_mom_12_1``/``rs_20d_sector`` indexed positionally into the
    calendar panel, so an equity's "20-day return" was 21 calendar days and
    NaN'd whenever the offset landed on a weekend."""
    mixed = _raw(pd.concat([equity_bars, crypto_bars], ignore_index=True), [*EQUITIES, *CRYPTO])
    for feature in POSITIONAL_FEATURES:
        assert bool(mixed.loc[EQUITIES, feature].notna().all()), feature
    assert bool(mixed.loc[OPERATING_COMPANIES, "rs_20d_sector"].notna().all())
    # ret_20d must equal the hand-computed 20-SESSION return of the symbol.
    own = equity_bars.loc[equity_bars["symbol"] == "AAA"].sort_values(by="ts")
    closes = np.asarray(own["adj_close"], dtype=float)
    expected = closes[-1] / closes[-21] - 1.0
    assert abs(float(mixed.loc["AAA", "ret_20d"]) - expected) < 1e-12


def test_a_50_day_ma_is_computable_for_every_symbol_in_the_mixed_universe(
    equity_bars: pd.DataFrame, crypto_bars: pd.DataFrame
) -> None:
    """The hand-verified 5-of-55 measurement, in miniature: 7 of 7 now."""
    raw = _raw(pd.concat([equity_bars, crypto_bars], ignore_index=True), [*EQUITIES, *CRYPTO])
    computable = int(raw["dist_ma50"].notna().sum())
    assert computable == len(EQUITIES) + len(CRYPTO)
