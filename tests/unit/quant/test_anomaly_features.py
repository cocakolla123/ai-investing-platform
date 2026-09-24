"""The published-anomaly features compute what their papers define.

Groups 16-17 come from the classic strategy implementations in
paperswithbacktest/awesome-systematic-trading. Each test plants the effect in
synthetic data with a known answer, and the point-in-time tests plant a fact
that must NOT be visible yet.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import (
    FEATURE_CATALOG,
    MARKET_SYMBOL,
    FeatureInputs,
    build_context,
    latest_value,
    pit_facts,
    to_ts,
)
from dynamo.research.quant.features import build_raw_features

SESSIONS = pd.bdate_range("2021-01-04", "2026-03-03")
AS_OF = cast(pd.Timestamp, SESSIONS[-1])


def _bars(prices: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sym, px in prices.items():
        for ts, p in zip(SESSIONS, px, strict=True):
            rows.append(
                {
                    "symbol": sym,
                    "ts": ts,
                    "open": p,
                    "high": p,
                    "low": p,
                    "close": p,
                    "adj_close": p,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _from_returns(r: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.exp(np.cumsum(r))


def _raw(prices: dict[str, np.ndarray], universe: list[str], **inputs: object) -> pd.DataFrame:
    return build_raw_features(FeatureInputs(bars=_bars(prices), **inputs), AS_OF, universe)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def market() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(0.0003, 0.01, len(SESSIONS))


def test_the_market_symbol_matches_the_adapter_constant() -> None:
    """Core may not import adapters, so the benchmark is declared twice."""
    from dynamo.data.universe import MARKET_BENCHMARK_SYMBOL, feature_input_symbols

    assert MARKET_SYMBOL == MARKET_BENCHMARK_SYMBOL
    assert MARKET_SYMBOL in feature_input_symbols(["AAPL"])


def test_groups_are_registered() -> None:
    price = {n for n, s in FEATURE_CATALOG.items() if s.group == "anomaly_price"}
    fundamental = {n for n, s in FEATURE_CATALOG.items() if s.group == "anomaly_fundamental"}
    assert price == {
        "high_52w_ratio",
        "max_ret_21d",
        "beta_252d",
        "idio_vol_60d",
        "residual_mom_12_1",
        "seasonal_lag12",
    }
    assert fundamental == {
        "asset_growth_yoy",
        "book_to_market",
        "net_share_issuance_1y",
        "rd_to_mcap",
    }


# -- price anomalies ---------------------------------------------------------


def test_52_week_high_ratio() -> None:
    n = len(SESSIONS)
    px = np.full(n, 100.0)
    px[-100] = 120.0  # the high, inside the last year
    px[-1] = 90.0
    raw = _raw({"AAA": px}, ["AAA"])
    assert raw.loc["AAA", "high_52w_ratio"] == pytest.approx(90.0 / 120.0)


def test_max_daily_return_only_counts_the_last_month() -> None:
    n = len(SESSIONS)
    r = np.zeros(n)
    r[-10] = math.log(1.15)  # +15% inside the window
    r[-40] = math.log(1.50)  # +50% outside it
    raw = _raw({"AAA": _from_returns(r)}, ["AAA"])
    assert raw.loc["AAA", "max_ret_21d"] == pytest.approx(0.15)


def test_beta_recovers_a_planted_loading(market: np.ndarray) -> None:
    rng = np.random.default_rng(3)
    stock = 1.5 * market + rng.normal(0, 0.002, len(market))
    raw = _raw(
        {MARKET_SYMBOL: _from_returns(market), "AAA": _from_returns(stock)}, ["AAA", MARKET_SYMBOL]
    )
    assert raw.loc["AAA", "beta_252d"] == pytest.approx(1.5, abs=0.03)
    assert raw.loc[MARKET_SYMBOL, "beta_252d"] == pytest.approx(1.0)


def test_market_features_are_nan_without_the_benchmark(market: np.ndarray) -> None:
    """No SPY in the inputs -> no beta; never a fabricated one."""
    raw = _raw({"AAA": _from_returns(market)}, ["AAA"])
    for col in ("beta_252d", "idio_vol_60d", "residual_mom_12_1"):
        assert math.isnan(raw.loc["AAA", col]), col


def test_idiosyncratic_vol_measures_only_the_unexplained_part(market: np.ndarray) -> None:
    rng = np.random.default_rng(5)
    sigma = 0.02
    stock = 0.8 * market + rng.normal(0, sigma, len(market))
    raw = _raw(
        {MARKET_SYMBOL: _from_returns(market), "AAA": _from_returns(stock)}, ["AAA", MARKET_SYMBOL]
    )
    assert raw.loc["AAA", "idio_vol_60d"] == pytest.approx(sigma * math.sqrt(252), rel=0.25)


def test_residual_momentum_sign_and_the_skipped_month(market: np.ndarray) -> None:
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 0.01, len(market))
    n = len(market)
    formation = np.zeros(n)
    formation[n - 252 : n - 21] = 0.002  # idiosyncratic drift inside 12-1
    recent_only = np.zeros(n)
    recent_only[n - 21 :] = 0.02  # drift ONLY in the skipped month

    up = market + noise + formation
    down = market + noise - formation
    skipped = market + noise + recent_only
    raw = _raw(
        {
            MARKET_SYMBOL: _from_returns(market),
            "UP": _from_returns(up),
            "DOWN": _from_returns(down),
            "SKIP": _from_returns(skipped),
            "FLAT": _from_returns(market + noise),
        },
        ["UP", "DOWN", "SKIP", "FLAT", MARKET_SYMBOL],
    )
    col = raw["residual_mom_12_1"]
    assert col["UP"] > 2.0 and col["DOWN"] < -2.0
    # The most recent month is excluded: a pure last-month move changes nothing.
    assert col["SKIP"] == pytest.approx(col["FLAT"], abs=0.35)


def test_seasonality_reads_last_years_same_month() -> None:
    """A name that rallied last March scores that March. as_of is 2026-03-03,
    so the target month (as_of + 14d) is March; March 2025 is read."""
    r = np.zeros(len(SESSIONS))
    march_2025 = np.asarray([t.year == 2025 and t.month == 3 for t in SESSIONS], dtype=bool)
    march_2024 = np.asarray([t.year == 2024 and t.month == 3 for t in SESSIONS], dtype=bool)
    r[march_2025] = math.log(1.05) / int(march_2025.sum())  # +5% over March 2025
    r[march_2024] = math.log(1.50) / int(march_2024.sum())  # two years back: ignored
    raw = _raw(
        {"AAA": _from_returns(r), "BBB": _from_returns(np.zeros(len(SESSIONS)))}, ["AAA", "BBB"]
    )
    assert raw.loc["AAA", "seasonal_lag12"] == pytest.approx(0.05, rel=1e-6)
    assert raw.loc["BBB", "seasonal_lag12"] == pytest.approx(0.0, abs=1e-12)


def test_seasonality_is_nan_without_last_years_month() -> None:
    short = pd.bdate_range("2025-06-02", AS_OF)
    bars = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "ts": t,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "adj_close": 100.0,
                "volume": 1,
            }
            for t in short
        ]
    )
    raw = build_raw_features(FeatureInputs(bars=bars), AS_OF, ["AAA"])
    assert math.isnan(raw.loc["AAA", "seasonal_lag12"])


def test_seasonality_fits_inside_the_feature_bars_window() -> None:
    """The window every path actually loads — the reason this is lag-12 only.
    If this ever drops below ~13 months the feature silently goes to NaN."""
    from dynamo.replay.gateway import FEATURE_BARS_LOOKBACK_DAYS

    assert FEATURE_BARS_LOOKBACK_DAYS >= 400


# -- fundamental anomalies ---------------------------------------------------


def _fact(tag: str, period_end: str, value: float, filed: str) -> dict[str, object]:
    return {
        "symbol": "AAA",
        "tag": tag,
        "period_end": pd.Timestamp(period_end),
        "value": value,
        "filed_at": pd.Timestamp(filed),
    }


def _flat_price(value: float = 50.0) -> dict[str, np.ndarray]:
    return {"AAA": np.full(len(SESSIONS), value)}


def test_asset_growth_book_to_market_issuance_and_rd() -> None:
    facts = pd.DataFrame(
        [
            _fact("Assets", "2024-12-31", 1_000.0, "2025-02-10"),
            _fact("Assets", "2025-12-31", 1_200.0, "2026-02-10"),
            _fact("StockholdersEquity", "2025-12-31", 400.0, "2026-02-10"),
            _fact("CommonStockSharesOutstanding", "2024-12-31", 100.0, "2025-02-10"),
            _fact("CommonStockSharesOutstanding", "2025-12-31", 90.0, "2026-02-10"),
            *[
                _fact("ResearchAndDevelopmentExpense", pe, 25.0, filed)
                for pe, filed in [
                    ("2025-03-31", "2025-05-05"),
                    ("2025-06-30", "2025-08-05"),
                    ("2025-09-30", "2025-11-05"),
                    ("2025-12-31", "2026-02-10"),
                ]
            ],
        ]
    )
    raw = _raw(_flat_price(50.0), ["AAA"], fundamentals=facts)
    mcap = 90.0 * 50.0
    assert raw.loc["AAA", "asset_growth_yoy"] == pytest.approx(0.2)
    assert raw.loc["AAA", "book_to_market"] == pytest.approx(400.0 / mcap)
    assert raw.loc["AAA", "net_share_issuance_1y"] == pytest.approx(math.log(90.0 / 100.0))
    assert raw.loc["AAA", "rd_to_mcap"] == pytest.approx(100.0 / mcap)


def test_a_fact_filed_after_as_of_is_invisible() -> None:
    """The 2025 annual report lands AFTER as_of: the features must still see
    only the 2024 world, never the future filing."""
    facts = pd.DataFrame(
        [
            _fact("Assets", "2023-12-31", 1_000.0, "2024-02-10"),
            _fact("Assets", "2024-12-31", 1_100.0, "2025-02-10"),
            _fact("Assets", "2025-12-31", 9_999.0, "2026-06-01"),  # future
            _fact("StockholdersEquity", "2025-12-31", -5.0, "2026-06-01"),  # future
            _fact("StockholdersEquity", "2024-12-31", 300.0, "2025-02-10"),
            _fact("CommonStockSharesOutstanding", "2024-12-31", 100.0, "2025-02-10"),
        ]
    )
    raw = _raw(_flat_price(50.0), ["AAA"], fundamentals=facts)
    assert raw.loc["AAA", "asset_growth_yoy"] == pytest.approx(0.1)
    assert raw.loc["AAA", "book_to_market"] == pytest.approx(300.0 / 5_000.0)


def test_negative_book_equity_is_not_extremely_cheap() -> None:
    facts = pd.DataFrame(
        [
            _fact("StockholdersEquity", "2025-12-31", -50.0, "2026-02-10"),
            _fact("CommonStockSharesOutstanding", "2025-12-31", 90.0, "2026-02-10"),
        ]
    )
    raw = _raw(_flat_price(), ["AAA"], fundamentals=facts)
    assert math.isnan(raw.loc["AAA", "book_to_market"])


def test_share_issuance_never_mixes_share_definitions() -> None:
    """Point-in-time count now, weighted-average count a year ago: comparing
    them would measure the definition change, not issuance."""
    facts = pd.DataFrame(
        [
            _fact(
                "WeightedAverageNumberOfSharesOutstandingBasic", "2024-12-31", 50.0, "2025-02-10"
            ),
            _fact("CommonStockSharesOutstanding", "2025-12-31", 100.0, "2026-02-10"),
        ]
    )
    raw = _raw(_flat_price(), ["AAA"], fundamentals=facts)
    assert math.isnan(raw.loc["AAA", "net_share_issuance_1y"])


def test_latest_value_honors_its_as_of() -> None:
    """The bug behind the dead dilution score: as_of was accepted and ignored."""
    facts = pit_facts(
        pd.DataFrame(
            [
                _fact("CommonStockSharesOutstanding", "2024-12-31", 100.0, "2025-02-10"),
                _fact("CommonStockSharesOutstanding", "2025-12-31", 90.0, "2026-02-10"),
            ]
        ),
        AS_OF,
    )
    assert latest_value(facts, "AAA", "CommonStockSharesOutstanding", AS_OF) == 90.0
    year_ago = to_ts(AS_OF - pd.Timedelta(days=365))
    assert latest_value(facts, "AAA", "CommonStockSharesOutstanding", year_ago) == 100.0


def test_the_market_model_is_computed_once_per_context(market: np.ndarray) -> None:
    """Three features share one regression per symbol."""
    from dynamo.research.quant import catalog

    ctx = build_context(
        FeatureInputs(
            bars=_bars({MARKET_SYMBOL: _from_returns(market), "AAA": _from_returns(market)})
        ),
        AS_OF,
        ["AAA"],
    )
    first = catalog._market_model(ctx)
    assert catalog._market_model(ctx) is first
