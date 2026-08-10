"""Tier-2 surveillance detectors — pure functions over pandas frames.

Design: doc 02 §2 (detector catalog). Each detector is configurable, costs
zero LLM tokens, and emits typed :class:`SurveillanceFlag` objects with a
0-1 severity score. ``run_detectors`` assembles the nightly flag set which
downstream feeds the Haiku triage batch.

Input conventions:

- ``bars``: DataFrame with columns ``symbol, ts, close, volume`` (canonical
  adjusted closes from ``MarketDataRepository.bars_frame``; ``high``/``low``
  are used by ``range_break`` when present, else ``close`` substitutes).
- Event inputs are sequences of the corresponding domain entities.
- ``as_of``: the flag timestamp (aware UTC); detectors only look backwards
  from it (as-of discipline, doc 01 §6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from dynamo.domain.models.market import (
    EarningsEvent,
    Filing,
    InsiderTx,
    NewsItem,
    SurveillanceFlag,
)

# 8-K items considered material (doc 02 §2: 1.01, 2.01, 5.02, ...).
MATERIAL_8K_ITEMS: frozenset[str] = frozenset(
    {"1.01", "1.03", "2.01", "2.03", "3.01", "4.01", "4.02", "5.01", "5.02"}
)
MATERIAL_FORMS: frozenset[str] = frozenset({"S-1", "13D", "13G", "SC 13D", "SC 13G"})
MNA_KEYWORDS: tuple[str, ...] = (
    "merger", "acquisition", "acquire", "buyout", "takeover", "tender offer",
)


def _sev(value: float) -> float:
    """Clamp a severity into [0, 1]."""
    return max(0.0, min(1.0, value))


def _per_symbol(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(symbol): grp.sort_values("ts").reset_index(drop=True)
        for symbol, grp in bars.groupby("symbol")
    }


def price_zscore(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    k: float = 4.0,
    window: int = 60,
) -> list[SurveillanceFlag]:
    """|1-day return| > k * sigma(60d) (doc 02 §2, default k=4)."""
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _per_symbol(bars).items():
        returns = grp["close"].pct_change().dropna()
        if len(returns) < window // 2 + 1:
            continue
        last = float(returns.iloc[-1])
        sigma = float(returns.iloc[:-1].tail(window).std())
        if sigma <= 0:
            continue
        z = abs(last) / sigma
        if z > k:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="price_zscore",
                    severity=_sev(z / (2.0 * k)),
                    details={"z": round(z, 3), "return_1d": round(last, 5), "k": k},
                    ts=as_of,
                )
            )
    return flags


def volume_spike(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    multiple: float = 5.0,
    window: int = 60,
) -> list[SurveillanceFlag]:
    """Dollar volume > m x 60d median (doc 02 §2, default m=5)."""
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _per_symbol(bars).items():
        dollar = grp["close"] * grp["volume"]
        if len(dollar) < window // 2 + 1:
            continue
        last = float(dollar.iloc[-1])
        median = float(dollar.iloc[:-1].tail(window).median())
        if median <= 0:
            continue
        ratio = last / median
        if ratio > multiple:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="volume_spike",
                    severity=_sev(ratio / (2.0 * multiple)),
                    details={"ratio": round(ratio, 2), "dollar_volume": round(last, 0)},
                    ts=as_of,
                )
            )
    return flags


def rel_strength(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    sector_etf_map: Mapping[str, str],
    threshold: float = 0.10,
    window: int = 20,
) -> list[SurveillanceFlag]:
    """|20d return - sector-ETF 20d return| > threshold (doc 02 §2).

    ``sector_etf_map`` maps each symbol to its sector ETF; ETF bars must be
    present in ``bars`` too.
    """
    frames = _per_symbol(bars)

    def _window_return(symbol: str) -> float | None:
        grp = frames.get(symbol)
        if grp is None or len(grp) < window + 1:
            return None
        closes = grp["close"]
        base = float(closes.iloc[-(window + 1)])
        return (float(closes.iloc[-1]) / base) - 1.0 if base > 0 else None

    flags: list[SurveillanceFlag] = []
    for symbol, etf in sector_etf_map.items():
        if symbol == etf:
            continue
        sym_ret = _window_return(symbol)
        etf_ret = _window_return(etf)
        if sym_ret is None or etf_ret is None:
            continue
        diff = sym_ret - etf_ret
        if abs(diff) > threshold:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="rel_strength",
                    severity=_sev(abs(diff) / (2.0 * threshold)),
                    details={
                        "excess_return_20d": round(diff, 4),
                        "sector_etf": etf,
                        "etf_return_20d": round(etf_ret, 4),
                    },
                    ts=as_of,
                )
            )
    return flags


def range_break(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    lookback: int = 252,
    volume_multiple: float = 1.5,
    volume_window: int = 60,
) -> list[SurveillanceFlag]:
    """52-week high/low breach with volume confirmation (doc 02 §2)."""
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _per_symbol(bars).items():
        if len(grp) < volume_window // 2 + 2:
            continue
        close = float(grp["close"].iloc[-1])
        highs = grp["high"] if "high" in grp.columns else grp["close"]
        lows = grp["low"] if "low" in grp.columns else grp["close"]
        prior_high = float(highs.iloc[:-1].tail(lookback).max())
        prior_low = float(lows.iloc[:-1].tail(lookback).min())
        vol_median = float(grp["volume"].iloc[:-1].tail(volume_window).median())
        vol_ok = vol_median > 0 and float(grp["volume"].iloc[-1]) > volume_multiple * vol_median
        direction: str | None = None
        if close > prior_high:
            direction = "high"
        elif close < prior_low:
            direction = "low"
        if direction and vol_ok:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="range_break",
                    severity=0.6,
                    details={
                        "direction": direction,
                        "close": round(close, 4),
                        "prior_52w_high": round(prior_high, 4),
                        "prior_52w_low": round(prior_low, 4),
                    },
                    ts=as_of,
                )
            )
    return flags


def earnings_surprise(
    events: Sequence[EarningsEvent],
    *,
    as_of: datetime,
    threshold: float = 0.10,
) -> list[SurveillanceFlag]:
    """Reported vs consensus EPS gap (doc 02 §2; Finnhub calendar)."""
    flags: list[SurveillanceFlag] = []
    for e in events:
        if e.eps_est is None or e.eps_act is None:
            continue
        denom = max(abs(e.eps_est), 0.01)
        surprise = (e.eps_act - e.eps_est) / denom
        if abs(surprise) > threshold:
            flags.append(
                SurveillanceFlag(
                    symbol=e.symbol,
                    detector="earnings_surprise",
                    severity=_sev(abs(surprise) / (2.0 * threshold)),
                    details={
                        "surprise": round(surprise, 4),
                        "eps_est": e.eps_est,
                        "eps_act": e.eps_act,
                        "event_date": e.event_date.isoformat(),
                    },
                    ts=as_of,
                )
            )
    return flags


def filing_event(filings: Sequence[Filing], *, as_of: datetime) -> list[SurveillanceFlag]:
    """Material filings: 8-K items (1.01, 2.01, 5.02, ...), S-1, 13D/G (doc 02 §2)."""
    flags: list[SurveillanceFlag] = []
    for f in filings:
        if f.symbol is None:
            continue
        form = f.form_type.upper()
        material_items = sorted(set(f.items) & MATERIAL_8K_ITEMS)
        is_8k_event = form.startswith("8-K") and material_items
        is_material_form = any(form.startswith(m) for m in MATERIAL_FORMS)
        if not is_8k_event and not is_material_form:
            continue
        severity = 0.7 if is_material_form else _sev(0.5 + 0.1 * len(material_items))
        flags.append(
            SurveillanceFlag(
                symbol=f.symbol,
                detector="filing_event",
                severity=severity,
                details={
                    "form_type": f.form_type,
                    "items": ",".join(material_items),
                    "accession_no": f.accession_no,
                },
                ts=as_of,
            )
        )
    return flags


def insider_cluster(
    txs: Sequence[InsiderTx],
    *,
    as_of: datetime,
    min_insiders: int = 2,
    min_notional: float = 100_000.0,
    window_days: int = 14,
) -> list[SurveillanceFlag]:
    """≥2 distinct insiders buying ≥$100k each within 14d (doc 02 §2)."""
    cutoff = as_of - timedelta(days=window_days)
    buyers: dict[str, dict[str, float]] = {}  # symbol -> insider -> notional
    for t in txs:
        if t.tx_type != "P" or t.price is None or t.filed_at < cutoff:
            continue
        notional = t.shares * t.price
        symbol_buyers = buyers.setdefault(t.symbol, {})
        symbol_buyers[t.insider_name] = symbol_buyers.get(t.insider_name, 0.0) + notional
    flags: list[SurveillanceFlag] = []
    for symbol, by_insider in buyers.items():
        qualified = {k: v for k, v in by_insider.items() if v >= min_notional}
        if len(qualified) >= min_insiders:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="insider_cluster",
                    severity=_sev(0.5 + 0.1 * len(qualified)),
                    details={
                        "insiders": len(qualified),
                        "total_notional": round(sum(qualified.values()), 0),
                        "window_days": window_days,
                    },
                    ts=as_of,
                )
            )
    return flags


def news_burst(
    items: Sequence[NewsItem],
    *,
    as_of: datetime,
    z_threshold: float = 3.0,
    baseline_days: int = 30,
) -> list[SurveillanceFlag]:
    """Headline count z-score vs 30d baseline; M&A keyword match (doc 02 §2)."""
    day_start = as_of - timedelta(days=1)
    baseline_start = as_of - timedelta(days=baseline_days + 1)
    per_symbol_days: dict[str, dict[int, int]] = {}
    recent_headlines: dict[str, list[str]] = {}
    for n in items:
        if n.published_at < baseline_start or n.published_at > as_of:
            continue
        age_days = int((as_of - n.published_at).total_seconds() // 86_400)
        for symbol in n.symbols:
            per_symbol_days.setdefault(symbol, {})
            per_symbol_days[symbol][age_days] = per_symbol_days[symbol].get(age_days, 0) + 1
            if n.published_at >= day_start:
                recent_headlines.setdefault(symbol, []).append(n.headline.lower())

    flags: list[SurveillanceFlag] = []
    for symbol, day_counts in per_symbol_days.items():
        today = day_counts.get(0, 0)
        baseline = pd.Series(
            [day_counts.get(d, 0) for d in range(1, baseline_days + 1)], dtype=float
        )
        mean, std = float(baseline.mean()), float(baseline.std())
        z = (today - mean) / std if std > 0 else float(today > max(mean * 3, 2))
        mna = any(
            kw in headline for headline in recent_headlines.get(symbol, []) for kw in MNA_KEYWORDS
        )
        if z > z_threshold or mna:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="news_burst",
                    severity=_sev(max(z / (2.0 * z_threshold), 0.8 if mna else 0.0)),
                    details={
                        "count_1d": today,
                        "z": round(z, 2),
                        "mna_keyword": str(mna).lower(),
                    },
                    ts=as_of,
                )
            )
    return flags


@dataclass(frozen=True)
class DetectorInputs:
    """Everything the nightly Tier-2 scan needs (doc 02 §4: 22:30 ET)."""

    bars: pd.DataFrame
    as_of: datetime
    sector_etf_map: Mapping[str, str] = field(default_factory=dict)
    earnings: Sequence[EarningsEvent] = ()
    filings: Sequence[Filing] = ()
    insider_txs: Sequence[InsiderTx] = ()
    news_items: Sequence[NewsItem] = ()


def run_detectors(inputs: DetectorInputs) -> list[SurveillanceFlag]:
    """Run the full doc 02 §2 catalog and assemble the nightly flag set."""
    flags: list[SurveillanceFlag] = []
    if not inputs.bars.empty:
        flags += price_zscore(inputs.bars, as_of=inputs.as_of)
        flags += volume_spike(inputs.bars, as_of=inputs.as_of)
        flags += range_break(inputs.bars, as_of=inputs.as_of)
        if inputs.sector_etf_map:
            flags += rel_strength(
                inputs.bars, as_of=inputs.as_of, sector_etf_map=inputs.sector_etf_map
            )
    flags += earnings_surprise(inputs.earnings, as_of=inputs.as_of)
    flags += filing_event(inputs.filings, as_of=inputs.as_of)
    flags += insider_cluster(inputs.insider_txs, as_of=inputs.as_of)
    flags += news_burst(inputs.news_items, as_of=inputs.as_of)
    return flags
