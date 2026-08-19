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
    to_ts,
)

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99

# Residual-label factor model (doc 04 addendum): betas from trailing daily
# returns strictly <= as_of. SPY is the market factor everywhere; the sector
# SPDR (when distinct from the market) contributes a sector-excess factor.
MARKET_BENCHMARK = "SPY"
BETA_WINDOW_SESSIONS = 250
MIN_BETA_SESSIONS = 60

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


def _trailing_betas(
    sym_prices: pd.Series,
    market_prices: pd.Series | None,
    sector_prices: pd.Series | None,
    as_of: pd.Timestamp,
    *,
    window: int = BETA_WINDOW_SESSIONS,
    min_sessions: int = MIN_BETA_SESSIONS,
) -> tuple[float, float] | None:
    """``(beta_mkt, beta_sector)`` from trailing daily simple returns.

    PIT-honest by construction: only bars dated ``<= as_of`` enter the
    regression — appending future bars can never move a beta. Returns run on
    the SYMBOL'S own trading calendar (factor prices as-of joined via ffill,
    mirroring the excess-return benchmark alignment — SPY vs 7-day crypto).

    Two-factor OLS with intercept: market return plus the sector-excess
    return (sector minus market). ``sector_prices=None`` fits market-only
    (``beta_sector = 0.0``). Thin history (< ``min_sessions`` overlapping
    return observations) yields ``None``, never a raised error.
    """
    if market_prices is None:
        return None
    sym = cast(pd.Series, sym_prices[sym_prices.index <= as_of])
    if len(sym) < 2:
        return None
    sym = sym.iloc[-(window + 1) :]
    frame: dict[str, pd.Series] = {
        "sym": sym.pct_change(),
        "mkt": market_prices.reindex(sym.index, method="ffill").pct_change(),
    }
    if sector_prices is not None:
        frame["sec"] = sector_prices.reindex(sym.index, method="ffill").pct_change()
    rets = pd.DataFrame(frame).dropna()
    if len(rets) < min_sessions:
        return None
    y = rets["sym"].to_numpy(dtype=float)
    x_mkt = rets["mkt"].to_numpy(dtype=float)
    if sector_prices is not None:
        design = np.column_stack(
            [np.ones(len(rets)), x_mkt, rets["sec"].to_numpy(dtype=float) - x_mkt]
        )
    else:
        design = np.column_stack([np.ones(len(rets)), x_mkt])
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:  # degenerate design — NaN label, never raise
        return None
    beta_mkt = float(coef[1])
    beta_sector = float(coef[2]) if len(coef) > 2 else 0.0
    if not (math.isfinite(beta_mkt) and math.isfinite(beta_sector)):
        return None
    return beta_mkt, beta_sector


def _residual_return(
    total: float,
    path_index: pd.Index,
    market: pd.Series | None,
    sector: pd.Series | None,
    betas: tuple[float, float] | None,
) -> float:
    """``total - (beta_mkt x market return + beta_sector x sector-excess
    return)`` over the forward window; NaN when betas or factor paths are
    unavailable (thin history, missing benchmark bars)."""
    if betas is None or market is None:
        return float("nan")
    beta_mkt, beta_sector = betas
    mkt_path = market.reindex(path_index, method="ffill")
    if bool(mkt_path.isna().any()) or float(mkt_path.iloc[0]) <= 0:
        return float("nan")
    mkt_total = float(mkt_path.iloc[-1] / mkt_path.iloc[0] - 1.0)
    sector_excess_total = 0.0
    if sector is not None:
        sec_path = sector.reindex(path_index, method="ffill")
        if bool(sec_path.isna().any()) or float(sec_path.iloc[0]) <= 0:
            return float("nan")
        sector_excess_total = float(sec_path.iloc[-1] / sec_path.iloc[0] - 1.0) - mkt_total
    return total - (beta_mkt * mkt_total + beta_sector * sector_excess_total)


def build_labels(
    bars: pd.DataFrame | None,
    sector_map: Mapping[str, str],
    as_of: date | pd.Timestamp,
    *,
    horizon_sessions: int = 20,
    downside_threshold: float = -0.10,
    horizons: Sequence[int] | None = None,
    market_symbol: str = MARKET_BENCHMARK,
    beta_window: int = BETA_WINDOW_SESSIONS,
    min_beta_sessions: int = MIN_BETA_SESSIONS,
) -> pd.DataFrame:
    """Forward-looking labels for feature date ``as_of`` (doc 04 section 4).

    Columns per horizon ``h``: ``excess_return_{h}d`` (vs sector ETF),
    ``downside_{h}d`` (min cum return < threshold within the horizon),
    ``realized_vol_{h}d`` (annualized), and ``residual_return_{h}d`` — the
    realized return minus ``beta_mkt x market return + beta_sector x
    sector-ETF excess return``, with betas from ~``beta_window`` trailing
    sessions of data ``<= as_of`` only (:func:`_trailing_betas`). The
    residual label is what real quant shops rank on: idiosyncratic alpha
    with systematic factor exposure stripped out.

    ``horizons`` defaults to ``(horizon_sessions,)`` — the default output
    keeps the original 20d columns bit-identical to the pre-residual builder
    (the residual column rides along in parallel; the trainer picks the rank
    target via ``--label-kind``). Symbols whose sector ETF IS the market
    (crypto, unclassified — mapped to SPY) get a market-only factor model.
    Rows: every non-ETF symbol in ``sector_map`` with sufficient forward
    bars; insufficient history yields NaN, never a raised error.
    """
    hs: tuple[int, ...] = (
        tuple(int(h) for h in horizons) if horizons is not None else (int(horizon_sessions),)
    )
    stems = ("excess_return", "downside", "realized_vol", "residual_return")
    columns = [f"{stem}_{h}d" for h in hs for stem in stems]
    symbols = [s for s in sector_map if s not in set(sector_map.values())]
    empty = pd.DataFrame(
        np.nan,
        index=pd.Index(symbols, name="symbol"),
        columns=pd.Index(columns),
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
    ts_as_of = to_ts(as_of)
    market = (
        cast(pd.Series, panel[market_symbol]).dropna()
        if market_symbol in panel.columns
        else None
    )

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
        etf = sector_map.get(sym)
        etf_series = (
            cast(pd.Series, panel[etf]).dropna()
            if etf is not None and etf in panel.columns
            else None
        )
        # Crypto / unclassified names map to SPY as their "sector" benchmark:
        # a sector factor identical to the market factor is degenerate, so
        # those symbols get the SPY-only factor model.
        sector_for_beta = etf_series if etf is not None and etf != market_symbol else None
        betas = _trailing_betas(
            series,
            market,
            sector_for_beta,
            ts_as_of,
            window=beta_window,
            min_sessions=min_beta_sessions,
        )
        for h in hs:
            path = series.iloc[pos : pos + h + 1]
            if len(path) < h + 1 or path.iloc[0] <= 0:
                continue
            cum = path / path.iloc[0] - 1.0
            total = float(cum.iloc[-1])
            if etf_series is None:
                excess = float("nan")
            else:
                # Benchmark aligned to the symbol's dates (as-of joins: last
                # benchmark close at or before each path date — SPY vs 7-day
                # crypto).
                etf_path = etf_series.reindex(path.index, method="ffill")
                if bool(etf_path.isna().any()) or float(etf_path.iloc[0]) <= 0:
                    excess = float("nan")
                else:
                    excess = total - float(etf_path.iloc[-1] / etf_path.iloc[0] - 1.0)
            lr = np.log(path).diff().dropna()
            vol = (
                float(lr.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
                if len(lr) > 2
                else float("nan")
            )
            breached = float(cum.iloc[1:].min()) < downside_threshold
            out.loc[sym, f"excess_return_{h}d"] = excess
            out.loc[sym, f"downside_{h}d"] = 1.0 if breached else 0.0
            out.loc[sym, f"realized_vol_{h}d"] = vol
            out.loc[sym, f"residual_return_{h}d"] = _residual_return(
                total, path.index, market, sector_for_beta, betas
            )
    return out
