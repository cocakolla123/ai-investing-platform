"""Feature and label builders — doc 04 sections 2-4.

``build_features`` returns one row per symbol at ``as_of`` with groups 1-8
cross-sectionally winsorized (1st/99th pct) and z-scored, groups 9-10 raw.
``build_labels`` computes the three doc-04 targets from adjusted closes.

Everything takes ``as_of`` explicitly — there is no "latest" accessor here.
Feature functions never raise; missing inputs yield NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from dynamo.research.quant.catalog import (
    FEATURE_CATALOG,
    RAW_GROUPS,
    TRADING_DAYS_PER_YEAR,
    FeatureInputs,
    build_context,
    dt_naive_utc,
)

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99

__all__ = ["build_features", "build_labels", "build_raw_features", "normalize_features"]


def build_raw_features(
    inputs: FeatureInputs, as_of: date | pd.Timestamp, universe: Sequence[str]
) -> pd.DataFrame:
    """Raw (un-normalized) feature matrix, indexed by symbol."""
    ctx = build_context(inputs, as_of, universe)
    cols: dict[str, pd.Series] = {}
    for name, spec in FEATURE_CATALOG.items():
        try:
            series = spec.fn(ctx)
        except Exception:  # missing inputs -> NaN, never raise
            series = pd.Series(np.nan, index=pd.Index(list(universe)), dtype=float)
        cols[name] = series.reindex(list(universe)).astype(float)
    out = pd.DataFrame(cols, index=pd.Index(list(universe)))
    out.index.name = "symbol"
    return out


def normalize_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional winsorize (1/99 pct) + z-score for groups 1-8; 9-10 raw.

    Invariant under per-column affine rescaling with positive slope, so a
    change of units upstream cannot move a normalized feature.
    """
    out = raw.copy()
    for col in raw.columns:
        spec = FEATURE_CATALOG.get(col)
        if spec is not None and spec.group in RAW_GROUPS:
            continue
        s = raw[col].astype(float)
        valid = s.dropna()
        if len(valid) < 2:
            out[col] = pd.Series(np.nan, index=raw.index) if len(valid) == 0 else s * 0.0
            continue
        lo, hi = valid.quantile(WINSOR_LOW), valid.quantile(WINSOR_HIGH)
        w = s.clip(lower=lo, upper=hi)
        mu, sd = w.mean(), w.std()
        out[col] = (w - mu) / sd if sd > 0 else w * 0.0
    return out


def build_features(
    inputs: FeatureInputs, as_of: date | pd.Timestamp, universe: Sequence[str]
) -> pd.DataFrame:
    """Normalized nightly feature matrix indexed by symbol."""
    return normalize_features(build_raw_features(inputs, as_of, universe))


def build_labels(
    bars: pd.DataFrame | None,
    sector_map: Mapping[str, str],
    as_of: date | pd.Timestamp,
    *,
    horizon_sessions: int = 20,
    downside_threshold: float = -0.10,
) -> pd.DataFrame:
    """Forward-looking labels for feature date ``as_of`` (doc 04 section 4).

    Columns: ``excess_return_20d`` (vs sector ETF), ``downside_20d`` (min cum
    return < threshold within the horizon), ``realized_vol_20d`` (annualized).
    Rows: every non-ETF symbol in ``sector_map`` with sufficient forward bars;
    insufficient history yields NaN, never a raised error.
    """
    symbols = [s for s in sector_map if s not in set(sector_map.values())]
    empty = pd.DataFrame(
        np.nan,
        index=pd.Index(symbols, name="symbol"),
        columns=pd.Index(["excess_return_20d", "downside_20d", "realized_vol_20d"]),
    )
    if bars is None or bars.empty or not {"symbol", "ts", "close"} <= set(bars.columns):
        return empty
    df = bars.copy()
    df["ts"] = dt_naive_utc(df["ts"])
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    panel = df.pivot_table(index="ts", columns="symbol", values=price_col, aggfunc="last")
    if price_col == "adj_close":
        close_panel = df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
        panel = panel.combine_first(close_panel)
    panel = panel.sort_index()
    ts_as_of = pd.Timestamp(as_of)

    out = empty.copy()
    for sym in symbols:
        if sym not in panel.columns:
            continue
        # Each symbol's horizon runs on its OWN trading calendar: a mixed
        # equity+crypto panel has weekend rows where equities are NaN, and a
        # union-calendar window would (wrongly) disqualify every equity.
        series = cast(pd.Series, panel[sym]).dropna()
        past_idx = cast(pd.DatetimeIndex, series.index[series.index <= ts_as_of])
        if len(past_idx) == 0:
            continue
        pos_obj = series.index.get_loc(past_idx[-1])
        pos = pos_obj.start if isinstance(pos_obj, slice) else int(pos_obj)  # type: ignore[union-attr]
        path = series.iloc[pos : pos + horizon_sessions + 1]
        if len(path) < horizon_sessions + 1 or path.iloc[0] <= 0:
            continue
        cum = path / path.iloc[0] - 1.0
        total = float(cum.iloc[-1])
        etf = sector_map.get(sym)
        if etf is None or etf not in panel.columns:
            excess = float("nan")
        else:
            # Benchmark aligned to the symbol's dates (as-of joins: last
            # benchmark close at or before each path date — SPY vs 7-day crypto).
            etf_path = cast(pd.Series, panel[etf]).dropna().reindex(path.index, method="ffill")
            if bool(etf_path.isna().any()) or float(etf_path.iloc[0]) <= 0:
                excess = float("nan")
            else:
                excess = total - float(etf_path.iloc[-1] / etf_path.iloc[0] - 1.0)
        lr = np.log(path).diff().dropna()
        vol = float(lr.std() * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(lr) > 2 else float("nan")
        breached = float(cum.iloc[1:].min()) < downside_threshold
        out.loc[sym, "excess_return_20d"] = excess
        out.loc[sym, "downside_20d"] = 1.0 if breached else 0.0
        out.loc[sym, "realized_vol_20d"] = vol
    return out
