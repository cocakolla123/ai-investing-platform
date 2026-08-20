"""Shared fixtures for quant unit tests. Tiny frames, fixed seeds, fast models."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import FeatureInputs


def last_session(bars: pd.DataFrame) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(str(bars["ts"].max())))


def session_at(bars: pd.DataFrame, pos: int) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(str(sorted(bars["ts"].unique())[pos])))


def make_bars(
    symbols: list[str],
    n_sessions: int = 300,
    seed: int = 7,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.bdate_range(start, periods=n_sessions)
    rows = []
    for k, sym in enumerate(symbols):
        prices = 100.0 * (1 + k / 10) * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_sessions)))
        volumes = rng.integers(100_000, 1_000_000, n_sessions)
        for t, px, v in zip(ts, prices, volumes, strict=True):
            rows.append(
                {
                    "symbol": sym,
                    "ts": t,
                    "open": px * 0.999,
                    "high": px * 1.012,
                    "low": px * 0.988,
                    "close": px,
                    "adj_close": px,
                    "volume": int(v),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars(["AAA", "BBB", "CCC", "XLK"])


@pytest.fixture
def inputs(bars: pd.DataFrame) -> FeatureInputs:
    return FeatureInputs(bars=bars, sector_map={"AAA": "XLK", "BBB": "XLK", "CCC": "XLK"})


def planted_panel(
    n_sessions: int = 320, n_symbols: int = 14, seed: int = 3
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Synthetic (features, label) with one strongly predictive planted feature."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2020-01-01", periods=n_sessions)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([sessions, symbols], names=["ts", "symbol"])
    signal = rng.normal(size=len(idx))
    noise = rng.normal(size=len(idx))
    features = pd.DataFrame({"signal": signal, "noise": noise}, index=idx)
    label = pd.Series(
        2.0 * signal + 0.1 * rng.normal(size=len(idx)),
        index=idx,
        name="excess_return_20d",
    )
    return features, label, sessions
