"""Price and peer context for the dossier (campaign findings, items 12-13).

Two thin, pure layers over bars already loaded for the dossier:

* :func:`price_stats` — the four numbers the dossier always had (close, 20d
  return, 20d annualized vol, 20d median dollar volume) plus the ones a human
  analyst reaches for first: 52-week range and distance to each end, 60/120/250
  session returns, the worst drawdown of the last year, and whether liquidity
  is expanding or drying up.
* :func:`peer_context` — relative strength against the name's own sector ETF
  and against SPY over 20/60/120 sessions, plus beta. 287 of 295 verdicts
  complained that no peer or comparable data existed; the sector ETFs, IWM and
  TLT are already in ``bars_1d`` and ``instruments.sector`` is populated, so
  this is arithmetic, not ingestion.

Everything is computed on each symbol's OWN observation sequence (never a
shared calendar — that mistake is what produced the dead feature columns
documented in the campaign findings) and every window that lacks enough
observations is simply absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd

from dynamo.domain.models.market import Frozen

TRADING_DAYS_PER_YEAR = 252
SESSIONS_52W = 252
RETURN_WINDOWS: tuple[int, ...] = (20, 60, 120, 250)
PEER_WINDOWS: tuple[int, ...] = (20, 60, 120)
BETA_WINDOW = 120
BENCHMARK_SYMBOL = "SPY"

# GICS sector -> sector SPDR. Mirrors ``dynamo.data.universe`` (which core
# logic may not import); the peer section needs the mapping, not the adapter.
SECTOR_ETFS: Mapping[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def sector_etf_for(sector: str | None) -> str | None:
    return None if sector is None else SECTOR_ETFS.get(sector)


class PeerContext(Frozen):
    """Relative strength vs the sector ETF and the broad benchmark."""

    symbol: str
    sector: str | None = None
    sector_etf: str | None = None
    benchmark: str = BENCHMARK_SYMBOL
    stats: dict[str, float] = {}


def _closes(bars: pd.DataFrame, symbol: str) -> pd.Series:
    """Close series indexed by ts, ascending, for one symbol."""
    if bars.empty or "symbol" not in bars.columns:
        return pd.Series(dtype=float)
    sub = cast(pd.DataFrame, bars[bars["symbol"] == symbol])
    if sub.empty:
        return pd.Series(dtype=float)
    sub = sub.sort_values("ts")
    series = pd.Series(
        sub["close"].astype(float).to_numpy(), index=pd.to_datetime(sub["ts"]), dtype=float
    )
    deduped = cast(pd.Series, series[~series.index.duplicated(keep="last")])
    return deduped.dropna()


def _window_return_pct(closes: pd.Series, sessions: int) -> float | None:
    """Return over the last ``sessions`` observations of THIS symbol."""
    if len(closes) <= sessions:
        return None
    start = float(closes.iloc[-sessions - 1])
    if start <= 0:
        return None
    return (float(closes.iloc[-1]) / start - 1.0) * 100.0


def price_stats(bars: pd.DataFrame, symbol: str) -> dict[str, float]:
    """Deterministic price/volume statistics for the dossier's stats section.

    The original four keys keep their names and meaning so existing dossiers
    stay comparable; everything else is additive and absent when the history
    is too short.
    """
    closes = _closes(bars, symbol)
    if closes.empty:
        return {}
    stats: dict[str, float] = {"close": float(closes.iloc[-1])}
    for window in RETURN_WINDOWS:
        value = _window_return_pct(closes, window)
        if value is not None:
            stats[f"ret_{window}d_pct"] = value
    if len(closes) > 21:
        stats["vol_20d_ann_pct"] = float(
            closes.pct_change().tail(20).std() * (TRADING_DAYS_PER_YEAR**0.5) * 100.0
        )
    year = closes.tail(SESSIONS_52W)
    if len(year) >= 60:
        high = float(year.max())
        low = float(year.min())
        last = float(year.iloc[-1])
        stats["high_52w"] = high
        stats["low_52w"] = low
        if high > 0:
            stats["dist_from_52w_high_pct"] = (last / high - 1.0) * 100.0
        if low > 0:
            stats["dist_from_52w_low_pct"] = (last / low - 1.0) * 100.0
        running_max = year.cummax()
        drawdown = (year / running_max - 1.0).min()
        if pd.notna(drawdown):
            stats["max_drawdown_52w_pct"] = float(drawdown) * 100.0
    dollar = _dollar_volume(bars, symbol)
    if dollar is not None and len(dollar) >= 20:
        stats["dollar_volume"] = float(dollar.tail(20).median())
        if len(dollar) >= 60:
            adv20 = float(dollar.tail(20).mean())
            adv60 = float(dollar.tail(60).mean())
            if adv60 > 0:
                stats["adv_trend_20_60_pct"] = (adv20 / adv60 - 1.0) * 100.0
    return {k: v for k, v in stats.items() if math.isfinite(v)}


def _dollar_volume(bars: pd.DataFrame, symbol: str) -> pd.Series | None:
    if bars.empty or "volume" not in bars.columns:
        return None
    sub = cast(pd.DataFrame, bars[bars["symbol"] == symbol]).sort_values("ts")
    if sub.empty:
        return None
    return (sub["close"].astype(float) * sub["volume"].astype(float)).reset_index(drop=True)


def _beta(symbol_closes: pd.Series, benchmark_closes: pd.Series, window: int) -> float | None:
    """OLS beta of daily returns over the overlapping trailing window."""
    joined = pd.concat(
        [symbol_closes.rename("sym"), benchmark_closes.rename("bench")], axis=1, join="inner"
    ).dropna()
    if len(joined) < window // 2 + 2:
        return None
    returns = joined.pct_change().dropna().tail(window)
    if len(returns) < 20:
        return None
    bench = cast(pd.Series, returns["bench"])
    own = cast(pd.Series, returns["sym"])
    variance = float(cast(float, bench.var()))
    if variance <= 0:
        return None
    covariance = float(cast(float, own.cov(bench)))
    beta = covariance / variance
    return beta if math.isfinite(beta) else None


def peer_context(
    bars: pd.DataFrame,
    symbol: str,
    *,
    sector: str | None = None,
    sector_etf: str | None = None,
    benchmark: str = BENCHMARK_SYMBOL,
) -> PeerContext | None:
    """Relative strength vs sector ETF and benchmark. ``None`` when the
    symbol's own bars are missing (the section then renders as absent)."""
    closes = _closes(bars, symbol)
    if closes.empty:
        return None
    etf = sector_etf if sector_etf is not None else sector_etf_for(sector)
    stats: dict[str, float] = {}
    references: list[tuple[str, str]] = []
    if etf is not None and etf != symbol:
        references.append(("sector", etf))
    if benchmark and benchmark != symbol:
        references.append(("benchmark", benchmark))
    for window in PEER_WINDOWS:
        own = _window_return_pct(closes, window)
        if own is None:
            continue
        stats[f"ret_{window}d_pct"] = own
        for label, reference in references:
            ref_closes = _closes(bars, reference)
            ref_return = _window_return_pct(ref_closes, window)
            if ref_return is None:
                continue
            stats[f"{label}_{reference}_ret_{window}d_pct"] = ref_return
            stats[f"excess_vs_{reference}_{window}d_pct"] = own - ref_return
    for _label, reference in references:
        beta = _beta(closes, _closes(bars, reference), BETA_WINDOW)
        if beta is not None:
            stats[f"beta_{BETA_WINDOW}d_vs_{reference}"] = beta
    if not stats:
        return None
    return PeerContext(
        symbol=symbol,
        sector=sector,
        sector_etf=etf,
        benchmark=benchmark,
        stats={k: v for k, v in stats.items() if math.isfinite(v)},
    )


def peer_symbols(sectors: Mapping[str, str], symbols: Sequence[str]) -> list[str]:
    """Every reference series the peer/percentile sections need: the sector
    ETFs of the given symbols, their same-sector peers, and the benchmark."""
    wanted = {BENCHMARK_SYMBOL}
    sectors_wanted = {sectors[s] for s in symbols if s in sectors}
    for symbol, sector in sectors.items():
        if sector in sectors_wanted:
            wanted.add(symbol)
    for sector in sectors_wanted:
        etf = SECTOR_ETFS.get(sector)
        if etf is not None:
            wanted.add(etf)
    return sorted(wanted)
