"""Normalization properties, label construction, determinism (doc 04 section 10)."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.research.quant.catalog import FeatureInputs
from dynamo.research.quant.features import (
    build_features,
    build_labels,
    build_raw_features,
    normalize_features,
)
from tests.unit.quant.conftest import last_session, make_bars, session_at


@settings(max_examples=30, deadline=None)
@given(
    scale=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False),
    shift=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    seed=st.integers(0, 1000),
)
def test_normalization_scale_invariance(scale: float, shift: float, seed: int) -> None:
    """Winsorized z-scores are invariant under positive affine rescaling."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {"ret_20d": rng.normal(size=40), "rsi_14": rng.uniform(0, 100, size=40)},
        index=pd.Index([f"S{i}" for i in range(40)]),
    )
    a = normalize_features(raw)
    b = normalize_features(raw * scale + shift)
    pd.testing.assert_frame_equal(a, b, atol=1e-8, rtol=1e-6)


def test_normalization_leaves_raw_groups_untouched() -> None:
    raw = pd.DataFrame(
        {
            "month_of_year": [3.0, 3.0, 3.0],
            "hy_oas_level": [4.2, 4.2, 4.2],
            "ret_20d": [0.1, 0.0, -0.1],
        },
        index=pd.Index(["A", "B", "C"]),
    )
    norm = normalize_features(raw)
    assert norm["month_of_year"].tolist() == [3.0, 3.0, 3.0]
    assert norm["hy_oas_level"].tolist() == [4.2, 4.2, 4.2]
    assert abs(norm["ret_20d"].mean()) < 1e-12  # z-scored


def test_normalization_preserves_nan_and_handles_constant_columns() -> None:
    raw = pd.DataFrame(
        {"ret_20d": [0.1, np.nan, -0.1, 0.05], "vol_cc_20d": [0.2, 0.2, 0.2, 0.2]},
        index=pd.Index(list("ABCD")),
    )
    norm = normalize_features(raw)
    assert np.isnan(norm.loc["B", "ret_20d"])
    assert (norm["vol_cc_20d"] == 0.0).all()  # zero-variance -> 0, not inf


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.round(12).to_csv().encode("utf-8")).hexdigest()


def test_build_features_deterministic(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    as_of = last_session(bars)
    universe = ["AAA", "BBB", "CCC"]
    h1 = _frame_hash(build_features(inputs, as_of, universe))
    h2 = _frame_hash(build_features(inputs, as_of, universe))
    assert h1 == h2


def test_features_use_only_data_up_to_as_of(inputs: FeatureInputs, bars: pd.DataFrame) -> None:
    """Truncating bars after as_of changes nothing: no future read exists."""
    as_of = session_at(bars, -30)
    truncated = cast(pd.DataFrame, bars[bars["ts"] <= as_of])
    full = build_raw_features(inputs, as_of, ["AAA", "BBB", "CCC"])
    trunc = build_raw_features(
        FeatureInputs(bars=truncated, sector_map=inputs.sector_map), as_of, ["AAA", "BBB", "CCC"]
    )
    pd.testing.assert_frame_equal(full, trunc)


def test_build_labels_excess_downside_and_vol() -> None:
    ts = pd.bdate_range("2025-01-02", periods=30)
    rows = []
    for sym, daily in (("AAA", 0.01), ("BBB", -0.02), ("ETF", 0.0)):
        px = 100.0
        for t in ts:
            rows.append({"symbol": sym, "ts": t, "close": px, "adj_close": px, "volume": 1000})
            px *= 1.0 + daily
    bars = pd.DataFrame(rows)
    sector_map = {"AAA": "ETF", "BBB": "ETF"}
    labels = build_labels(bars, sector_map, cast(pd.Timestamp, ts[0]))
    assert set(labels.index) == {"AAA", "BBB"}
    # AAA: +1%/day for 20 sessions vs flat ETF
    expected_aaa = 1.01**20 - 1.0
    assert abs(labels.loc["AAA", "excess_return_20d"] - expected_aaa) < 1e-9
    assert labels.loc["AAA", "downside_20d"] == 0.0
    # BBB: -2%/day breaches -10% within 20 sessions
    assert labels.loc["BBB", "downside_20d"] == 1.0
    assert labels.loc["BBB", "excess_return_20d"] < -0.3
    # constant daily log-return => (near) zero realized vol
    assert labels.loc["AAA", "realized_vol_20d"] < 1e-8


def test_build_labels_insufficient_forward_history_is_nan() -> None:
    bars = make_bars(["AAA", "ETF"], n_sessions=25)
    as_of = last_session(bars)  # no forward window at all
    labels = build_labels(bars, {"AAA": "ETF"}, as_of)
    assert bool(labels["excess_return_20d"].isna().all())


def test_build_labels_missing_sector_etf_gives_nan_excess() -> None:
    bars = make_bars(["AAA"], n_sessions=60)
    labels = build_labels(bars, {"AAA": "MISSING_ETF"}, session_at(bars, 10))
    assert np.isnan(labels.loc["AAA", "excess_return_20d"])
    # downside/vol only need the symbol's own path
    assert not np.isnan(labels.loc["AAA", "realized_vol_20d"])


# -- factor-residualized labels (residual_return_{h}d) ------------------------


def _factor_bars(
    n_sessions: int = 400,
    beta_mkt: float = 1.3,
    beta_sec: float = 0.7,
    seed: int = 5,
) -> pd.DataFrame:
    """SYM returns are an EXACT linear function of the SPY + sector-excess
    factors (alpha + beta_mkt*m + beta_sec*(s-m)) so OLS recovers the betas
    to float precision and the residual is hand-computable."""
    rng = np.random.default_rng(seed)
    ts = pd.bdate_range("2024-01-02", periods=n_sessions)
    m = rng.normal(0.0005, 0.01, n_sessions)
    s = m + rng.normal(0.0, 0.005, n_sessions)
    sym = 0.0002 + beta_mkt * m + beta_sec * (s - m)
    rows = []
    for name, rets in (("SYM", sym), ("SEC", s), ("SPY", m)):
        px = 100.0 * np.cumprod(1.0 + rets)
        for t, p in zip(ts, px, strict=True):
            rows.append({"symbol": name, "ts": t, "close": p, "adj_close": p, "volume": 1000})
    return pd.DataFrame(rows)


def _price_series(bars: pd.DataFrame, symbol: str) -> pd.Series:
    sub = bars[bars["symbol"] == symbol].set_index("ts")["adj_close"].sort_index()
    return cast(pd.Series, sub)


def test_residual_label_matches_hand_computed_factor_model() -> None:
    bars = _factor_bars()
    as_of = session_at(bars, 300)
    labels = build_labels(bars, {"SYM": "SEC"}, as_of)
    # Hand-compute: betas are exact (1.3, 0.7), so residual = total -
    # (1.3*mkt_total + 0.7*(sec_total - mkt_total)) over [as_of, as_of+20].
    path_idx = [session_at(bars, 300 + i) for i in range(21)]
    sym_p = _price_series(bars, "SYM").loc[path_idx]
    mkt_p = _price_series(bars, "SPY").loc[path_idx]
    sec_p = _price_series(bars, "SEC").loc[path_idx]
    total = float(sym_p.iloc[-1] / sym_p.iloc[0] - 1.0)
    mkt_total = float(mkt_p.iloc[-1] / mkt_p.iloc[0] - 1.0)
    sec_excess = float(sec_p.iloc[-1] / sec_p.iloc[0] - 1.0) - mkt_total
    expected = total - (1.3 * mkt_total + 0.7 * sec_excess)
    assert abs(float(labels.loc["SYM", "residual_return_20d"]) - expected) < 1e-8
    # The sector-relative excess label is untouched by the factor model.
    assert (
        abs(
            float(labels.loc["SYM", "excess_return_20d"])
            - (total - float(sec_p.iloc[-1] / sec_p.iloc[0] - 1.0))
        )
        < 1e-12
    )


def test_trailing_betas_are_pit_honest_against_future_bars() -> None:
    from dynamo.research.quant.features import _trailing_betas

    bars = _factor_bars()
    as_of = session_at(bars, 300)
    sym = _price_series(bars, "SYM")
    mkt = _price_series(bars, "SPY")
    sec = _price_series(bars, "SEC")
    before = _trailing_betas(sym, mkt, sec, as_of)
    assert before is not None
    assert abs(before[0] - 1.3) < 1e-9 and abs(before[1] - 0.7) < 1e-9
    # Wild future bars (after as_of) must not move the betas AT ALL.
    future = pd.bdate_range("2026-06-01", periods=50)
    crash = pd.Series(np.linspace(50.0, 1.0, 50), index=future)
    after = _trailing_betas(
        pd.concat([sym, crash]), pd.concat([mkt, crash]), pd.concat([sec, crash]), as_of
    )
    assert after == before


def test_residual_label_crypto_uses_market_only_factor() -> None:
    """A symbol whose 'sector' benchmark IS the market (crypto -> SPY) gets
    the SPY-only factor model — residual still computes, never raises."""
    rng = np.random.default_rng(9)
    ts = pd.date_range("2024-01-01", periods=400, freq="D")  # 7-day crypto calendar
    bts = pd.bdate_range("2024-01-01", periods=280)
    rows = []
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.03, len(ts)))
    for t, p in zip(ts, px, strict=True):
        rows.append({"symbol": "BTC/USD", "ts": t, "close": p, "adj_close": p, "volume": 10})
    spy = 400.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, len(bts)))
    for t, p in zip(bts, spy, strict=True):
        rows.append({"symbol": "SPY", "ts": t, "close": p, "adj_close": p, "volume": 10})
    bars = pd.DataFrame(rows)
    as_of = cast(pd.Timestamp, pd.Timestamp(cast(object, ts[300])))  # type: ignore[arg-type]
    labels = build_labels(bars, {"BTC/USD": "SPY"}, as_of)
    assert not np.isnan(labels.loc["BTC/USD", "residual_return_20d"])
    assert not np.isnan(labels.loc["BTC/USD", "excess_return_20d"])


def test_residual_label_thin_history_is_nan_never_raises() -> None:
    """< min_beta_sessions of trailing history -> NaN residual while the
    forward-window labels (excess/downside/vol) still compute."""
    bars = _factor_bars(n_sessions=55)
    as_of = session_at(bars, 30)  # 30 trailing sessions < 60 minimum
    labels = build_labels(bars, {"SYM": "SEC"}, as_of)
    assert np.isnan(labels.loc["SYM", "residual_return_20d"])
    assert not np.isnan(labels.loc["SYM", "excess_return_20d"])
    assert not np.isnan(labels.loc["SYM", "realized_vol_20d"])


def test_residual_label_missing_market_is_nan() -> None:
    """No SPY bars -> residual NaN; the sector-relative label is unaffected."""
    bars = make_bars(["AAA", "XLK"], n_sessions=300)
    labels = build_labels(bars, {"AAA": "XLK"}, session_at(bars, 250))
    assert np.isnan(labels.loc["AAA", "residual_return_20d"])
    assert not np.isnan(labels.loc["AAA", "excess_return_20d"])


def test_default_labels_regression_columns_and_values() -> None:
    """Default build_labels output: the original three 20d columns keep their
    exact values and positions; residual_return_20d rides along in parallel."""
    bars = _factor_bars()
    as_of = session_at(bars, 300)
    labels = build_labels(bars, {"SYM": "SEC"}, as_of)
    assert list(labels.columns) == [
        "excess_return_20d",
        "downside_20d",
        "realized_vol_20d",
        "residual_return_20d",
    ]
    # Bit-identical to a by-hand replay of the original label math.
    path_idx = [session_at(bars, 300 + i) for i in range(21)]
    sym_p = _price_series(bars, "SYM").loc[path_idx]
    sec_p = _price_series(bars, "SEC").loc[path_idx]
    cum = sym_p / sym_p.iloc[0] - 1.0
    assert float(labels.loc["SYM", "excess_return_20d"]) == float(cum.iloc[-1]) - float(
        sec_p.iloc[-1] / sec_p.iloc[0] - 1.0
    )
    assert float(labels.loc["SYM", "downside_20d"]) == float(float(cum.iloc[1:].min()) < -0.10)
    lr = np.log(sym_p).diff().dropna()
    assert (
        abs(float(labels.loc["SYM", "realized_vol_20d"]) - float(lr.std() * np.sqrt(252.0))) < 1e-12
    )
