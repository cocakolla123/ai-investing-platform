"""Features must be NaN when they cannot be computed — never a constant.

Two silent constant fills hid the calendar outage for a whole 74-day
campaign (campaign findings §1, "collateral damage"):

* ``_rsi_14`` filled exactly ``50.0`` whenever its denominator was NaN, so
  all 50 equities reported a fabricated neutral RSI that then counted as
  "covered" in ``feature_coverage``;
* ``_breadth_above_200dma`` averaged over "whichever names have a valid
  200-day MA", i.e. the 5 crypto pairs, and broadcast that to all 55 rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynamo.research.quant.catalog import (
    ASSET_CLASS_CRYPTO,
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_ETF,
    BREADTH_MIN_NAMES,
    FeatureInputs,
    applicable_features,
    asset_class_of,
    to_ts,
)
from dynamo.research.quant.features import build_raw_features
from tests.unit.quant.conftest import last_session, make_bars


def _crypto_bars(symbols: list[str], n: int, seed: int, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=end, periods=n, freq="D")  # 7 days a week
    rows = []
    for k, sym in enumerate(symbols):
        px = 100.0 * (1 + k / 5) * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
        for t, p in zip(ts, px, strict=True):
            rows.append(
                {
                    "symbol": sym,
                    "ts": t,
                    "open": p,
                    "high": p * 1.02,
                    "low": p * 0.98,
                    "close": p,
                    "adj_close": p,
                    "volume": 1_000,
                }
            )
    return pd.DataFrame(rows)


# -- RSI: 50.0 means "flat", not "unknown" -----------------------------------


def test_rsi_is_nan_without_price_history_not_a_fabricated_50() -> None:
    bars = make_bars(["AAA"], n_sessions=60)
    raw = build_raw_features(FeatureInputs(bars=bars), last_session(bars), ["AAA", "NOBARS"])
    assert np.isnan(raw.loc["NOBARS", "rsi_14"])
    assert not np.isnan(raw.loc["AAA", "rsi_14"])


def test_rsi_is_50_only_when_the_window_is_genuinely_flat() -> None:
    """A perfectly flat price has a zero denominator — that IS 50."""
    ts = pd.bdate_range(end="2026-03-03", periods=60)
    flat = pd.DataFrame(
        [
            {
                "symbol": "FLAT",
                "ts": t,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "adj_close": 10.0,
                "volume": 100,
            }
            for t in ts
        ]
    )
    raw = build_raw_features(FeatureInputs(bars=flat), to_ts("2026-03-03"), ["FLAT"])
    assert raw.loc["FLAT", "rsi_14"] == 50.0


def test_rsi_is_real_for_equities_sharing_a_panel_with_crypto() -> None:
    """The campaign shape: every equity read exactly 50.0 because the
    calendar-panel rolling mean was NaN for all of them."""
    equities = make_bars(["AAA", "BBB"], n_sessions=300, start="2025-01-01")
    end = str(to_ts(str(equities["ts"].max())).date())
    bars = pd.concat([equities, _crypto_bars(["BTC/USD"], 420, 3, end)], ignore_index=True)
    raw = build_raw_features(FeatureInputs(bars=bars), to_ts(end), ["AAA", "BBB", "BTC/USD"])
    rsi = raw["rsi_14"]
    assert bool(rsi.notna().all())
    assert not (rsi == 50.0).any()  # no fabricated neutrals
    assert bool(((rsi > 0.0) & (rsi < 100.0)).all())


# -- breadth: a fixed equity reference set -----------------------------------


def _breadth_universe(n_equities: int) -> tuple[FeatureInputs, list[str], pd.Timestamp]:
    equities = [f"E{i:02d}" for i in range(n_equities)]
    eq_bars = make_bars(equities, n_sessions=300, start="2025-01-01")
    end = str(to_ts(str(eq_bars["ts"].max())).date())
    bars = pd.concat(
        [eq_bars, _crypto_bars(["BTC/USD", "ETH/USD"], 420, 5, end)], ignore_index=True
    )
    inputs = FeatureInputs(bars=bars, sector_map={s: "XLK" for s in equities})
    return inputs, [*equities, "BTC/USD", "ETH/USD"], to_ts(end)


def test_breadth_uses_the_equity_reference_set_not_whoever_has_an_ma() -> None:
    inputs, universe, as_of = _breadth_universe(BREADTH_MIN_NAMES + 2)
    raw = build_raw_features(inputs, as_of, universe)
    breadth = raw["breadth_above_200dma"]
    assert bool(breadth.notna().all())
    assert bool((breadth == breadth.iloc[0]).all())  # market state, broadcast
    # It is a fraction OF THE EQUITIES: computable per name, so the value must
    # match the hand count over the reference set only.
    equities = [s for s in universe if "/" not in s]
    panel = (
        inputs.bars.pivot_table(  # type: ignore[union-attr]
            index="ts", columns="symbol", values="adj_close", aggfunc="last"
        )
        .sort_index()
        .loc[:, equities]
    )
    expected = float(
        np.mean(
            [panel[s].dropna().iloc[-1] > panel[s].dropna().iloc[-200:].mean() for s in equities]
        )
    )
    assert abs(float(breadth.iloc[0]) - expected) < 1e-12


def test_breadth_is_nan_when_the_reference_set_is_too_thin() -> None:
    """Better no reading than a 'market breadth' computed from 3 names."""
    inputs, universe, as_of = _breadth_universe(3)
    raw = build_raw_features(inputs, as_of, universe)
    assert bool(raw["breadth_above_200dma"].isna().all())


# -- asset-class buckets ------------------------------------------------------


def test_asset_class_inference_from_the_sector_map() -> None:
    sector_map = {"AAA": "XLK", "BBB": "XLF"}
    assert asset_class_of("AAA", sector_map) == ASSET_CLASS_EQUITY
    assert asset_class_of("XLK", sector_map) == ASSET_CLASS_ETF  # a sector-ETF value
    assert asset_class_of("SPY", sector_map) == ASSET_CLASS_ETF  # no classification
    assert asset_class_of("BTC/USD", sector_map) == ASSET_CLASS_CRYPTO
    # An explicit override wins (instruments.is_etf, when a caller has it).
    assert asset_class_of("AAA", sector_map, etf_symbols={"AAA"}) == ASSET_CLASS_ETF
    # With no sector map at all nothing can be called a fund.
    assert asset_class_of("SPY", {}) == ASSET_CLASS_EQUITY


def test_applicable_features_drop_only_structurally_impossible_columns() -> None:
    names = [
        "ret_20d",
        "fcf_yield",
        "roic_ttm",
        "insider_net_buy_90d",
        "days_to_next_earnings",
        "rs_20d_sector",
        "month_of_year",
        "f1",
    ]
    equity = applicable_features(ASSET_CLASS_EQUITY, names)
    assert equity == names  # an operating company is expected to have it all
    for cls in (ASSET_CLASS_ETF, ASSET_CLASS_CRYPTO):
        got = applicable_features(cls, names)
        assert "fcf_yield" not in got  # files no XBRL
        assert "roic_ttm" not in got
        assert "insider_net_buy_90d" not in got  # no Form 4s
        assert "days_to_next_earnings" not in got  # no earnings date
        assert "rs_20d_sector" not in got  # no sector peer of its own
        assert "ret_20d" in got and "month_of_year" in got
        assert "f1" in got  # unknown columns are always applicable
