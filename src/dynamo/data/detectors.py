"""Tier-2 surveillance detectors — pure functions over pandas frames.

Design: doc 02 §2 (detector catalog). Each detector is configurable, costs
zero LLM tokens, and emits typed :class:`SurveillanceFlag` objects with a
0-1 severity score. ``run_detectors`` assembles the nightly flag set which
downstream feeds the Haiku triage batch.

Input conventions:

- ``bars``: DataFrame with columns ``symbol, ts, close, volume`` (canonical
  adjusted closes from ``MarketDataRepository.bars_frame``; ``high``/``low``
  are used by ``range_break`` when present, else ``close`` substitutes).
  ``range_bound`` needs ~140 sessions of history per symbol and uses
  ``close``/``volume`` only.
- Event inputs are sequences of the corresponding domain entities.
- ``as_of``: the flag timestamp (aware UTC); detectors only look backwards
  from it (as-of discipline, doc 01 §6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dynamo.domain.models.market import (
    EarningsEvent,
    Filing,
    InsiderTx,
    NewsItem,
    SurveillanceFlag,
)
from dynamo.domain.volume_source import (
    consolidated_equivalent_dollar_volume,
    dominant_source,
)

# 8-K items considered material (doc 02 §2: 1.01, 2.01, 5.02, ...).
MATERIAL_8K_ITEMS: frozenset[str] = frozenset(
    {"1.01", "1.03", "2.01", "2.03", "3.01", "4.01", "4.02", "5.01", "5.02"}
)
MATERIAL_FORMS: frozenset[str] = frozenset({"S-1", "13D", "13G", "SC 13D", "SC 13G"})
MNA_KEYWORDS: tuple[str, ...] = (
    "merger", "acquisition", "acquire", "buyout", "takeover", "tender offer",
)

# --- range_bound liquidity floor ------------------------------------------
#
# $2M median daily dollar volume. The number is a judgement call, so here is
# the reasoning rather than just the constant. Range trading earns the width
# of the range and pays the spread twice; a 5-cent spread against a $1 range
# is ~10% of the gross edge per round trip, and below roughly $2M/day US
# equities routinely quote 3-10 cents on sub-$10 names, at which point a
# range that looks profitable on closing prices is not. It is deliberately a
# default argument, not a hard-coded constant: raising it buys tradability
# and costs breadth (small caps oscillate most and are thinnest), and that
# is the operator's trade to make.
RANGE_BOUND_MIN_DOLLAR_VOLUME = 2_000_000.0
# Direct spread guard, applied only to symbols present in the supplied
# ``spreads_bps`` map. 50bps on a $5 name is 2.5 cents.
RANGE_BOUND_MAX_SPREAD_BPS = 50.0


def _sev(value: float) -> float:
    """Clamp a severity into [0, 1]."""
    return max(0.0, min(1.0, value))


def _per_symbol(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``symbol -> its bars, ascending by ts`` — the shape every bar detector
    reads.

    HOIST THE RESULT, do not let each detector recompute it. All five bar
    detectors used to call this themselves, so ``run_detectors`` paid a full
    groupby + per-group sort **five times** over the same frame, every
    session.

    MEASURED, on a synthetic 4,000-symbol x 300-session panel (identical
    flags both ways): 3.93s for the five self-grouping detectors, 2.27s once
    the grouping is hoisted and paid once — **1.7x**. The design note
    predicted 3-4x; the real number is lower because the groupby itself is
    only ~0.37s of the 3.93s and the detectors' own arithmetic dominates.
    Replay does better than 1.7x because
    ``dynamo.replay.discovery.SurveillancePanel`` holds the per-symbol frames
    for the whole run, so the grouping is amortized away entirely and the
    detector stage alone goes 3.93s -> 1.90s (2.1x).

    Every bar detector now takes an optional ``per_symbol`` mapping and
    ``run_detectors`` computes it ONCE; passing it must not change a single
    flag, which is what ``tests/unit/data/test_detectors_per_symbol.py``
    asserts against a golden run.
    """
    return {
        str(symbol): grp.sort_values("ts").reset_index(drop=True)
        for symbol, grp in bars.groupby("symbol")
    }


def _frames(
    bars: pd.DataFrame, per_symbol: Mapping[str, pd.DataFrame] | None
) -> Mapping[str, pd.DataFrame]:
    """The hoisted grouping when the caller supplies one, else group here."""
    return _per_symbol(bars) if per_symbol is None else per_symbol


# --- price_zscore eligibility guards --------------------------------------
#
# The first whole-market sweep (11,258 symbols, 2026-08-18) produced 61
# price_zscore flags. Reading them one by one, only 7 described a real,
# tradable price event. The rest were two failure modes, and each needs its
# own guard because each defeats the other's.

# 1. UNADJUSTED CORPORATE ACTIONS. bars_1d stores Alpaca closes as the feed
#    returns them, and a reverse split lands in the series as a price move:
#    RCON printed 0.0226 -> 3.19 (+14,015%) and CXAI 0.0898 -> 4.23 (+4,610%)
#    on 2026-08-17/18. 7 of the 61 flags were shaped like this. A z-score
#    cannot tell a split from a rally — the ratio is simply enormous — so the
#    only reliable discriminator at this layer is the size of the move.
#
#    The ceiling is stated as a PRICE RATIO, not a return, and it is symmetric
#    in that ratio: a session is rejected when the close is >= 2x or <= 1/2x
#    the previous close. Splits come in both directions and a return-based
#    ceiling is lopsided against them — a 1-for-141 reverse split is +14,015%
#    but an ordinary 2-for-1 forward split is only -50%, which no plausible
#    ceiling on |return| would ever catch. In ratio terms both are exactly one
#    factor of the split away from the previous close.
#
#    The asymmetry of COSTS justifies setting it this tight: routing a genuine
#    -50% biotech blowup to data-quality review costs a day's delay, while
#    treating a split as a price move fabricates a trade on a price that never
#    happened. On the measured date this reroutes 4 of the 61, two of them
#    confirmed reverse splits.
PRICE_ZSCORE_MAX_PRICE_RATIO = 2.0
DATA_QUALITY_DETECTOR = "data_quality"

# 2. COLLAPSED VOLATILITY. Dividing by a 60-day sigma that tends to zero makes
#    any move enormous in sigma units. AACP moved +0.4% and scored 10.4 sigma
#    because its sigma was 0.038% — it barely prints on IEX. 13% of the
#    scanned universe has a 60-day sigma below 0.3%.
#
#    Two independent guards, because either alone is escapable:
#      - a floor under the denominator. 0.5%/session sits at roughly the 18th
#        percentile of the universe's 60-day sigma (measured quantiles:
#        p5 0.12%, p25 0.74%, p50 1.88%), so it binds only on the dead tail
#        and leaves the median name untouched.
#      - a floor under the numerator. Below 2% a single session simply is not
#        a surveillance event on an equity, whatever the sigma says. The two
#        numbers are deliberately consistent: at k=4 the sigma floor already
#        implies a 2% minimum move, so this restates the same bound where a
#        caller lowers k.
PRICE_ZSCORE_SIGMA_FLOOR = 0.005
PRICE_ZSCORE_MIN_ABS_RETURN = 0.02

# 3. UNTRADABLE NAMES. 39 of the 61 flagged symbols had a MEDIAN daily dollar
#    volume of exactly zero over the trailing window — no position could be
#    opened in them at any size. Expressed in CONSOLIDATED-EQUIVALENT dollars
#    (see dynamo.domain.volume_source): $1M/day consolidated is ~$66k/day as
#    observed on the IEX feed. At the risk engine's own exit assumption (20%
#    participation) a $1M/day name absorbs ~$200k/session, which is the point
#    below which a flag cannot become a position worth holding.
PRICE_ZSCORE_MIN_DOLLAR_VOLUME = 1_000_000.0


def _consolidated_dollar_volume(grp: pd.DataFrame, window: int) -> float:
    """Median daily dollar volume over the window, in consolidated-equivalent
    dollars — scaled up from the bar's own feed (doc 02 §5 / D5).

    A frame without a ``source`` column is assumed consolidated, which
    UNDER-states liquidity and so fails toward rejecting a flag.
    """
    dollar = (grp["close"] * grp["volume"]).tail(window)
    observed = float(dollar.median()) if len(dollar) else 0.0
    source = dominant_source(grp["source"]) if "source" in grp.columns else None
    return consolidated_equivalent_dollar_volume(observed, source)


def price_zscore(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    k: float = 4.0,
    window: int = 60,
    sigma_floor: float = PRICE_ZSCORE_SIGMA_FLOOR,
    min_abs_return: float = PRICE_ZSCORE_MIN_ABS_RETURN,
    min_dollar_volume: float = PRICE_ZSCORE_MIN_DOLLAR_VOLUME,
    max_price_ratio: float = PRICE_ZSCORE_MAX_PRICE_RATIO,
    per_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> list[SurveillanceFlag]:
    """|1-day return| > k * sigma(60d), gated for tradability (doc 02 §2).

    Emits two kinds of flag. ``price_zscore`` is a TRADING signal: a move
    large enough to matter, on a name liquid enough to act on, measured
    against a volatility that is not an artifact. ``data_quality`` is not a
    trading signal at all — it marks a session whose printed move is too
    large to be a price move, which in this database means an unadjusted
    corporate action, and asks for a human rather than a position.

    The ceiling is tested FIRST and unconditionally, so no combination of the
    other arguments can route a split into the trading branch. See the
    constants above for what each threshold is worth on measured data.
    """
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _frames(bars, per_symbol).items():
        returns = grp["close"].pct_change().dropna()
        if len(returns) < window // 2 + 1:
            continue
        last = float(returns.iloc[-1])

        # (1) Sanity ceiling, before anything else can make this a signal.
        #     Symmetric in price ratio so both split directions are caught.
        ratio = 1.0 + last
        floor_ratio = 1.0 / max_price_ratio if max_price_ratio > 0 else 0.0
        if ratio >= max_price_ratio or ratio <= floor_ratio:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector=DATA_QUALITY_DETECTOR,
                    severity=0.9,
                    details={
                        "check": "unadjusted_price_move",
                        "return_1d": round(last, 5),
                        "price_ratio": round(ratio, 4),
                        "max_price_ratio": max_price_ratio,
                        "direction": "reverse_split_shaped"
                        if ratio >= max_price_ratio
                        else "forward_split_shaped",
                        "reason": (
                            "single-session price ratio outside the sanity band; "
                            "most likely an unadjusted corporate action, "
                            "NOT a trading signal"
                        ),
                    },
                    ts=as_of,
                )
            )
            continue

        # (2) Dead-sigma guard: floor the denominator AND the numerator.
        raw_sigma = float(returns.iloc[:-1].tail(window).std())
        if raw_sigma != raw_sigma:  # NaN
            continue
        sigma = max(raw_sigma, sigma_floor)
        if sigma <= 0 or abs(last) < min_abs_return:
            continue

        # (3) Tradability: consolidated-equivalent dollars, per D5.
        dollar_volume = _consolidated_dollar_volume(grp, window)
        if dollar_volume < min_dollar_volume:
            continue

        z = abs(last) / sigma
        if z > k:
            flags.append(
                SurveillanceFlag(
                    symbol=symbol,
                    detector="price_zscore",
                    severity=_sev(z / (2.0 * k)),
                    details={
                        "z": round(z, 3),
                        "return_1d": round(last, 5),
                        "k": k,
                        "sigma": round(sigma, 6),
                        "sigma_floored": str(raw_sigma < sigma_floor).lower(),
                        "dollar_volume_consolidated": round(dollar_volume, 0),
                    },
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
    per_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> list[SurveillanceFlag]:
    """Dollar volume > m x 60d median (doc 02 §2, default m=5)."""
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _frames(bars, per_symbol).items():
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
    per_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> list[SurveillanceFlag]:
    """|20d return - sector-ETF 20d return| > threshold (doc 02 §2).

    ``sector_etf_map`` maps each symbol to its sector ETF; ETF bars must be
    present in ``bars`` too.
    """
    frames = _frames(bars, per_symbol)

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
    per_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> list[SurveillanceFlag]:
    """52-week high/low breach with volume confirmation (doc 02 §2)."""
    flags: list[SurveillanceFlag] = []
    for symbol, grp in _frames(bars, per_symbol).items():
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


def range_bound(
    bars: pd.DataFrame,
    *,
    as_of: datetime,
    band_window: int = 20,
    band_sigma: float = 2.0,
    touch_window: int = 60,
    min_band_touches: int = 2,
    vr_lag: int = 10,
    vr_window: int = 120,
    max_variance_ratio: float = 0.90,
    range_window: int = 60,
    max_range_position: float = 0.25,
    min_dollar_volume: float = RANGE_BOUND_MIN_DOLLAR_VOLUME,
    max_spread_bps: float = RANGE_BOUND_MAX_SPREAD_BPS,
    spreads_bps: Mapping[str, float] | None = None,
    per_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> list[SurveillanceFlag]:
    """Established trading range, price currently near the LOW of it.

    This is the whole-market counterpart of the ``range_reversion`` research
    candidate (``research/candidates/range_reversion.py``): the piece that
    lets a $4.30-$5.30 oscillator somewhere in a 6,000-name universe get
    noticed and enter the promotion funnel instead of being invisible because
    nobody was looking at it. Four conditions, all required:

    1. **Mean-reverting** — Lo-MacKinlay variance ratio
       ``Var(10-session log return) / (10 * Var(1-session))`` over the
       trailing ``vr_window`` sessions is at or below ``max_variance_ratio``.
       Under a random walk this is 1.0; below 1 means increments reverse.
    2. **Two-sided oscillation** — at least ``min_band_touches`` distinct
       excursions beyond BOTH the upper and the lower ``band_sigma``
       Bollinger band in the trailing ``touch_window`` sessions. Consecutive
       sessions outside a band count as one excursion; requiring both ends
       is what separates an oscillator from a trend riding one band.
    3. **Near the low** — the close sits in the bottom ``max_range_position``
       of the trailing ``range_window`` high-low range.
    4. **Liquid enough to actually trade** — see below.

    Liquidity is a first-class, configurable parameter here rather than a
    hidden assumption, because for this strategy it IS the strategy. The
    edge being harvested is the width of the range, and the cost of
    harvesting it is the spread, paid twice. A 5-cent spread on a $1 range
    costs about 5% of the range on entry and 5% on exit — roughly **10% of
    the gross edge per round trip** — before commissions, slippage, or the
    fact that a thin book means the quoted spread is optimistic for any size.
    Widen the spread to 10 cents, or narrow the range to 50 cents, and a
    quarter to a half of the edge is gone. ``min_dollar_volume`` (default
    $2M/day median) is the crude proxy that keeps names whose spreads make
    the pattern untradeable out of the funnel; ``max_spread_bps`` is the
    direct test, applied only for symbols present in ``spreads_bps``. Bars
    carry no quote data today, so that guard is inert until a quote feed
    lands — stated rather than silently assumed.

    Raising ``min_dollar_volume`` buys tradability and costs breadth: the
    small and micro caps most likely to oscillate are exactly the ones a
    high floor removes. That trade-off is the operator's to make, which is
    why it is an argument and not a constant.
    """
    flags: list[SurveillanceFlag] = []
    min_sessions = max(vr_window + vr_lag + 1, touch_window + band_window, range_window)
    for symbol, grp in _frames(bars, per_symbol).items():
        if len(grp) < min_sessions:
            continue
        closes = np.asarray(grp["close"], dtype=float)
        volumes = np.asarray(grp["volume"], dtype=float)
        if closes.min() <= 0:
            continue

        dollar_volume = float(np.median((closes * volumes)[-touch_window:]))
        if dollar_volume < min_dollar_volume:
            continue
        spread = None if spreads_bps is None else spreads_bps.get(symbol)
        if spread is not None and float(spread) > max_spread_bps:
            continue

        # Grouped per symbol above, so every window here is already this
        # symbol's OWN session sequence — the calendar-mixing hazard that the
        # wide feature panel has cannot arise in this shape.
        close = pd.Series(closes)
        log_ret = pd.Series(np.diff(np.log(closes)))
        var_1 = float(log_ret.tail(vr_window).var())
        ret_k = pd.Series(log_ret.rolling(vr_lag).sum()).dropna()
        var_k = float(ret_k.tail(vr_window).var())
        if var_1 <= 0:
            continue
        variance_ratio = var_k / (vr_lag * var_1)
        if variance_ratio > max_variance_ratio:
            continue

        mid = pd.Series(close.rolling(band_window).mean())
        sd = pd.Series(close.rolling(band_window).std())
        usable = pd.Series(mid.notna() & (sd > 0))

        def _excursions(touching: pd.Series, usable: pd.Series = usable) -> int:
            """Distinct excursions: a touch whose previous session was inside
            the band. Ten sessions pinned outside a band is one, not ten."""
            touched = touching & usable
            started = touched & ~touched.shift(1, fill_value=False)
            return int(started.tail(touch_window).sum())

        upper_touches = _excursions(pd.Series(close >= mid + band_sigma * sd))
        lower_touches = _excursions(pd.Series(close <= mid - band_sigma * sd))
        if min(upper_touches, lower_touches) < min_band_touches:
            continue

        window = closes[-range_window:]
        low, high = float(window.min()), float(window.max())
        span = high - low
        if span <= 0:
            continue
        position = (float(closes[-1]) - low) / span
        if position > max_range_position:
            continue

        depth = 1.0 - position / max_range_position
        oscillation = min(min(upper_touches, lower_touches) / (2.0 * min_band_touches), 1.0)
        flags.append(
            SurveillanceFlag(
                symbol=symbol,
                detector="range_bound",
                severity=_sev(0.4 + 0.3 * oscillation + 0.3 * depth),
                details={
                    "variance_ratio": round(variance_ratio, 3),
                    "upper_touches": upper_touches,
                    "lower_touches": lower_touches,
                    "range_position": round(position, 3),
                    "range_low": round(low, 4),
                    "range_high": round(high, 4),
                    "dollar_volume": round(dollar_volume, 0),
                    "min_dollar_volume": min_dollar_volume,
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
    # Pre-grouped bars (``symbol -> its frame, ascending by ts``). Supplying
    # this skips the groupby entirely — ``bars`` is then never read by the
    # bar detectors, so a caller that already holds the per-symbol arrays
    # (``dynamo.replay.discovery.SurveillancePanel``) may pass an empty
    # frame. ``None`` = group from ``bars`` here, the live nightly path.
    per_symbol: Mapping[str, pd.DataFrame] | None = None
    sector_etf_map: Mapping[str, str] = field(default_factory=dict)
    # symbol -> quoted spread in bps, for range_bound's tradability guard.
    # Empty until a quote feed exists; the guard is then simply not applied.
    spreads_bps: Mapping[str, float] = field(default_factory=dict)
    earnings: Sequence[EarningsEvent] = ()
    filings: Sequence[Filing] = ()
    insider_txs: Sequence[InsiderTx] = ()
    news_items: Sequence[NewsItem] = ()


def run_detectors(inputs: DetectorInputs) -> list[SurveillanceFlag]:
    """Run the full doc 02 §2 catalog and assemble the nightly flag set.

    The per-symbol grouping is computed ONCE here and threaded into all five
    bar detectors (see :func:`_per_symbol`); a caller holding it already
    passes it on ``inputs.per_symbol``. Flag output is identical either way.
    """
    flags: list[SurveillanceFlag] = []
    frames = _frames(inputs.bars, inputs.per_symbol)
    if frames:
        flags += price_zscore(inputs.bars, as_of=inputs.as_of, per_symbol=frames)
        flags += volume_spike(inputs.bars, as_of=inputs.as_of, per_symbol=frames)
        flags += range_break(inputs.bars, as_of=inputs.as_of, per_symbol=frames)
        flags += range_bound(
            inputs.bars,
            as_of=inputs.as_of,
            spreads_bps=inputs.spreads_bps or None,
            per_symbol=frames,
        )
        if inputs.sector_etf_map:
            flags += rel_strength(
                inputs.bars,
                as_of=inputs.as_of,
                sector_etf_map=inputs.sector_etf_map,
                per_symbol=frames,
            )
    flags += earnings_surprise(inputs.earnings, as_of=inputs.as_of)
    flags += filing_event(inputs.filings, as_of=inputs.as_of)
    flags += insider_cluster(inputs.insider_txs, as_of=inputs.as_of)
    flags += news_burst(inputs.news_items, as_of=inputs.as_of)
    return flags
