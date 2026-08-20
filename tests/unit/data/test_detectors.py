"""Tier-2 detector tests — small pandas frames, assert flags (doc 02 §2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dynamo.data.detectors import (
    RANGE_BOUND_MIN_DOLLAR_VOLUME,
    DetectorInputs,
    earnings_surprise,
    filing_event,
    insider_cluster,
    news_burst,
    price_zscore,
    range_bound,
    range_break,
    rel_strength,
    run_detectors,
    volume_spike,
)
from dynamo.domain.models.market import EarningsEvent, Filing, InsiderTx, NewsItem

AS_OF = datetime(2026, 7, 31, 22, 0, tzinfo=UTC)


def make_bars(
    symbol: str,
    closes: list[float],
    volumes: list[int] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    ts = pd.date_range(end="2026-07-31", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": ts,
            "close": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "volume": volumes if volumes is not None else [100_000] * n,
        }
    )


def noisy_flat(n: int, base: float = 100.0) -> list[float]:
    """Flat series with +/-0.1% alternating noise so sigma > 0."""
    return [base * (1 + (0.001 if i % 2 else -0.001)) for i in range(n)]


def test_price_zscore_flags_extreme_move() -> None:
    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)  # +10% day vs ~0.2% daily noise
    flags = price_zscore(make_bars("ACME", closes), as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].detector == "price_zscore"
    assert flags[0].symbol == "ACME"
    assert 0 < flags[0].severity <= 1

    quiet = price_zscore(make_bars("QUIET", noisy_flat(71)), as_of=AS_OF)
    assert quiet == []


def test_volume_spike_flags_dollar_volume() -> None:
    closes = noisy_flat(70)
    volumes = [100_000] * 69 + [1_000_000]
    flags = volume_spike(make_bars("ACME", closes, volumes), as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].detector == "volume_spike"
    assert float(flags[0].details["ratio"]) > 5


def test_rel_strength_vs_sector_etf() -> None:
    strong = [100.0 * (1.015**i) for i in range(30)]  # ~+35% over 20d window
    flat = noisy_flat(30)
    bars = pd.concat([make_bars("ACME", strong), make_bars("XLK", flat)])
    flags = rel_strength(bars, as_of=AS_OF, sector_etf_map={"ACME": "XLK"})
    assert len(flags) == 1
    assert flags[0].details["sector_etf"] == "XLK"
    assert float(flags[0].details["excess_return_20d"]) > 0.10


def test_range_break_needs_volume_confirmation() -> None:
    closes = [*noisy_flat(70), 110.0]  # clean 52w-high breach
    confirmed = make_bars("ACME", closes, [100_000] * 70 + [300_000])
    flags = range_break(confirmed, as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["direction"] == "high"

    unconfirmed = make_bars("ACME", closes, [100_000] * 71)  # no volume spike
    assert range_break(unconfirmed, as_of=AS_OF) == []


# --- range_bound ----------------------------------------------------------
#
# The detector that would let a $4.30-$5.30 oscillator somewhere in a
# 6,000-name universe get noticed. Needs ~140 sessions of history per symbol.

RANGE_SESSIONS = 420


def oscillator(
    n: int = RANGE_SESSIONS,
    low: float = 4.30,
    high: float = 5.30,
    theta: float = 0.10,
    sigma: float = 0.10,
    seed: int = 4,
) -> list[float]:
    """Ornstein-Uhlenbeck price reverting to the middle of a $4.30-$5.30 band."""
    rng = np.random.default_rng(seed)
    mid = (low + high) / 2.0
    x = mid
    out: list[float] = []
    for _ in range(n):
        x += theta * (mid - x) + rng.normal(0.0, sigma)
        out.append(float(min(max(x, low), high)))
    return out


def trending(n: int = RANGE_SESSIONS, drift: float = 0.0015, seed: int = 5) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))]


def _truncate_at_extreme(prices: list[float], *, kind: str, window: int = 60) -> list[float]:
    """Cut so the last session IS the low (or high) of its own trailing window."""
    series = pd.Series(prices)
    rolling = series.rolling(window, min_periods=window)
    extreme = (rolling.min() if kind == "low" else rolling.max()).to_numpy()
    hits = np.flatnonzero(np.isclose(series.to_numpy(), extreme) & (np.arange(len(series)) >= 200))
    assert hits.size, f"no running {kind} late enough in the series"
    return prices[: int(hits[-1]) + 1]


def liquid(prices: list[float], shares: int = 1_000_000) -> pd.DataFrame:
    return make_bars("OSC", prices, [shares] * len(prices))


def test_range_bound_fires_on_an_oscillator_sitting_near_its_low() -> None:
    flags = range_bound(liquid(_truncate_at_extreme(oscillator(), kind="low")), as_of=AS_OF)
    assert len(flags) == 1
    flag = flags[0]
    assert flag.detector == "range_bound"
    assert flag.symbol == "OSC"
    assert flag.ts == AS_OF
    assert 0 < flag.severity <= 1
    assert float(flag.details["variance_ratio"]) < 1.0
    assert float(flag.details["range_position"]) <= 0.25
    assert float(flag.details["upper_touches"]) >= 2
    assert float(flag.details["lower_touches"]) >= 2
    assert float(flag.details["range_low"]) < float(flag.details["range_high"])


def test_range_bound_is_silent_on_a_trending_name() -> None:
    """Both directions. The downtrend case is the important one: it IS near
    its 60-session low, and the variance-ratio test is what rejects it.
    Buying the dip in a downtrend is what this detector exists not to do."""
    assert range_bound(liquid(trending()), as_of=AS_OF) == []
    assert range_bound(liquid(trending(drift=-0.0025, seed=17)), as_of=AS_OF) == []


def test_range_bound_is_silent_when_the_name_sits_high_in_its_range() -> None:
    at_high = _truncate_at_extreme(oscillator(), kind="high")
    assert range_bound(liquid(at_high), as_of=AS_OF) == []


def test_range_bound_is_silent_below_the_liquidity_floor() -> None:
    """The same name and the same pattern — only the volume differs. A $4.80
    stock trading 100k shares turns over ~$480k/day, under the $2M default."""
    prices = _truncate_at_extreme(oscillator(), kind="low")
    assert range_bound(liquid(prices, shares=1_000_000), as_of=AS_OF) != []
    assert range_bound(liquid(prices, shares=100_000), as_of=AS_OF) == []
    # The floor is a knob, not a hidden constant: lower it and the name returns.
    revived = range_bound(liquid(prices, shares=100_000), as_of=AS_OF, min_dollar_volume=250_000.0)
    assert len(revived) == 1
    assert revived[0].details["min_dollar_volume"] == 250_000.0


def test_range_bound_default_liquidity_floor_is_the_documented_two_million() -> None:
    assert RANGE_BOUND_MIN_DOLLAR_VOLUME == 2_000_000.0


def test_range_bound_spread_guard_applies_only_to_symbols_with_quotes() -> None:
    """A 5-cent spread on a $1 range is ~10% of the gross edge per round
    trip, so a wide quote disqualifies the pattern. Symbols absent from the
    map are NOT penalised — no quote data is not the same as a bad quote."""
    bars = liquid(_truncate_at_extreme(oscillator(), kind="low"))
    assert range_bound(bars, as_of=AS_OF, spreads_bps={"OSC": 20.0}) != []
    assert range_bound(bars, as_of=AS_OF, spreads_bps={"OSC": 120.0}) == []
    assert range_bound(bars, as_of=AS_OF, spreads_bps={"OTHER": 120.0}) != []
    assert range_bound(bars, as_of=AS_OF, spreads_bps=None) != []


def test_range_bound_is_silent_on_short_history() -> None:
    prices = _truncate_at_extreme(oscillator(), kind="low")
    assert range_bound(liquid(prices[-100:]), as_of=AS_OF) == []


def test_earnings_surprise() -> None:
    events = [
        EarningsEvent(symbol="BEAT", event_date=AS_OF.date(), eps_est=1.0, eps_act=1.5),
        EarningsEvent(symbol="MEET", event_date=AS_OF.date(), eps_est=1.0, eps_act=1.02),
        EarningsEvent(symbol="NOEST", event_date=AS_OF.date(), eps_act=1.0),
    ]
    flags = earnings_surprise(events, as_of=AS_OF)
    assert [f.symbol for f in flags] == ["BEAT"]
    assert flags[0].details["surprise"] == 0.5


def test_filing_event_material_forms_and_items() -> None:
    filings = [
        Filing(
            accession_no="a1",
            cik="1",
            symbol="ACME",
            form_type="8-K",
            filed_at=AS_OF,
            url="u",
            items=("5.02", "9.01"),
        ),
        Filing(
            accession_no="a2",
            cik="2",
            symbol="NEWCO",
            form_type="S-1",
            filed_at=AS_OF,
            url="u",
        ),
        Filing(  # non-material 8-K items → no flag
            accession_no="a3",
            cik="3",
            symbol="BORING",
            form_type="8-K",
            filed_at=AS_OF,
            url="u",
            items=("9.01",),
        ),
    ]
    flags = filing_event(filings, as_of=AS_OF)
    assert {f.symbol for f in flags} == {"ACME", "NEWCO"}


def test_insider_cluster_requires_two_big_buyers() -> None:
    def buy(name: str, shares: float, price: float, days_ago: int = 1) -> InsiderTx:
        return InsiderTx(
            accession_no=f"acc-{name}",
            symbol="ACME",
            insider_name=name,
            tx_type="P",
            shares=shares,
            price=price,
            filed_at=AS_OF - timedelta(days=days_ago),
        )

    two_buyers = [buy("CEO", 20_000, 10.0), buy("CFO", 15_000, 10.0)]
    flags = insider_cluster(two_buyers, as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["insiders"] == 2

    # one buyer / small buys / stale buys → no flag
    assert insider_cluster([buy("CEO", 20_000, 10.0)], as_of=AS_OF) == []
    assert insider_cluster([buy("CEO", 100, 10.0), buy("CFO", 100, 10.0)], as_of=AS_OF) == []
    assert (
        insider_cluster(
            [buy("CEO", 20_000, 10.0, days_ago=30), buy("CFO", 15_000, 10.0, days_ago=30)],
            as_of=AS_OF,
        )
        == []
    )


def test_news_burst_zscore_and_mna_keyword() -> None:
    def item(i: int, days_ago: float, headline: str = "Quarterly update") -> NewsItem:
        return NewsItem(
            id=f"n{i}",
            symbols=("ACME",),
            headline=headline,
            source="t",
            published_at=AS_OF - timedelta(days=days_ago),
        )

    baseline = [item(i, days_ago=2 + i) for i in range(5)]  # ~0-1/day baseline
    burst = [item(100 + i, days_ago=0.5) for i in range(8)]
    flags = news_burst(baseline + burst, as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["count_1d"] == 8

    mna = [item(200, days_ago=0.5, headline="XYZ agrees to acquisition of ACME")]
    flags = news_burst(baseline + mna, as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["mna_keyword"] == "true"
    assert flags[0].severity >= 0.8


def test_run_detectors_assembles_catalog() -> None:
    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)
    bars = make_bars("ACME", closes, [100_000] * 70 + [1_000_000])
    inputs = DetectorInputs(
        bars=bars,
        as_of=AS_OF,
        earnings=[EarningsEvent(symbol="BEAT", event_date=AS_OF.date(), eps_est=1.0, eps_act=2.0)],
    )
    flags = run_detectors(inputs)
    detectors = {f.detector for f in flags}
    assert "price_zscore" in detectors
    assert "volume_spike" in detectors
    assert "earnings_surprise" in detectors


def test_run_detectors_registers_range_bound() -> None:
    """Registration parity with the sibling price detectors: the range-bound
    scan must run as part of the nightly assembly, not only when called
    directly, or nothing in the 6,000-name universe ever reaches the funnel."""
    bars = liquid(_truncate_at_extreme(oscillator(), kind="low"))
    flags = run_detectors(DetectorInputs(bars=bars, as_of=AS_OF))
    matched = [f for f in flags if f.detector == "range_bound"]
    assert len(matched) == 1
    assert matched[0].symbol == "OSC"
    assert matched[0].ts == AS_OF


def test_run_detectors_passes_the_spread_map_through_to_range_bound() -> None:
    bars = liquid(_truncate_at_extreme(oscillator(), kind="low"))
    wide = run_detectors(DetectorInputs(bars=bars, as_of=AS_OF, spreads_bps={"OSC": 500.0}))
    assert [f for f in wide if f.detector == "range_bound"] == []


# --------------------------------------------------------------------------
# price_zscore eligibility guards (D1 unadjusted actions, D2 dead sigma,
# D5 source-aware liquidity). Measured against the first whole-market sweep:
# 61 flags on 2026-08-18, of which 7 described a real tradable event.
# --------------------------------------------------------------------------


def zbars(
    symbol: str,
    closes: list[float],
    *,
    volume: int = 100_000,
    source: str | None = None,
) -> pd.DataFrame:
    """Bars for the price_zscore guards, with an optional feed label."""
    frame = make_bars(symbol, closes, [volume] * len(closes))
    if source is not None:
        frame = frame.assign(source=source)
    return frame


def test_price_zscore_ignores_an_unadjusted_reverse_split() -> None:
    """D1: RCON printed 0.0226 -> 3.19 on 2026-08-17 (a 1-for-141 reverse
    split) and reached the detectors as a +14,015% trading signal."""
    closes = [c * 0.0226 / 100.0 for c in noisy_flat(70)]
    closes.append(3.19)
    flags = price_zscore(zbars("RCON", closes, volume=5_000_000), as_of=AS_OF)

    assert [f.detector for f in flags] == ["data_quality"]
    assert not [f for f in flags if f.detector == "price_zscore"]
    detail = flags[0].details
    assert detail["check"] == "unadjusted_price_move"
    assert detail["direction"] == "reverse_split_shaped"
    assert float(detail["price_ratio"]) > 100


def test_price_zscore_ignores_an_unadjusted_forward_split() -> None:
    """A 2-for-1 forward split halves the close — only -50%, which no ceiling
    on |return| would catch. The band is symmetric in PRICE RATIO for exactly
    this reason."""
    closes = noisy_flat(70)
    closes.append(closes[-1] / 2.0)
    flags = price_zscore(zbars("SPLT", closes, volume=5_000_000), as_of=AS_OF)

    assert [f.detector for f in flags] == ["data_quality"]
    assert flags[0].details["direction"] == "forward_split_shaped"
    assert float(flags[0].details["price_ratio"]) == pytest.approx(0.5, abs=0.01)


def test_split_shaped_move_never_reaches_the_trading_branch() -> None:
    """The ceiling is tested FIRST and unconditionally: no combination of the
    other knobs can route a split back into a trading signal."""
    closes = noisy_flat(70)
    closes.append(closes[-1] * 141.0)
    flags = price_zscore(
        zbars("RCON", closes, volume=50_000_000),
        as_of=AS_OF,
        k=0.0001,  # would flag anything
        sigma_floor=0.0,  # no floor
        min_abs_return=0.0,  # no numerator floor
        min_dollar_volume=0.0,  # no liquidity gate
    )
    assert {f.detector for f in flags} == {"data_quality"}


def test_price_zscore_ignores_a_tiny_move_on_dead_volatility() -> None:
    """D2: AACP moved +0.4% and scored 10.4 sigma because its 60-day sigma was
    0.038%. 18% of the sweep's price_zscore flags were this shape."""
    dead = [100.0 * (1 + (0.00002 if i % 2 else -0.00002)) for i in range(70)]
    dead.append(dead[-1] * 1.004)  # +0.4%
    bars = zbars("AACP", dead, volume=100_000)

    # Without the guards this is a > 4 sigma event.
    unguarded = price_zscore(bars, as_of=AS_OF, sigma_floor=0.0, min_abs_return=0.0)
    assert [f.detector for f in unguarded] == ["price_zscore"]
    assert float(unguarded[0].details["z"]) > 4.0

    assert price_zscore(bars, as_of=AS_OF) == []


def test_sigma_floor_alone_kills_the_dead_sigma_case() -> None:
    """The two D2 guards are independent — each must work without the other,
    since a caller can lower either."""
    dead = [100.0 * (1 + (0.00002 if i % 2 else -0.00002)) for i in range(70)]
    dead.append(dead[-1] * 1.004)
    bars = zbars("AACP", dead)
    assert price_zscore(bars, as_of=AS_OF, min_abs_return=0.0) == []
    assert price_zscore(bars, as_of=AS_OF, sigma_floor=0.0) == []


def test_price_zscore_still_flags_a_genuine_move_on_a_liquid_name() -> None:
    """The guards must not cost the real signal. A 4-sigma move on a liquid
    name with ordinary volatility still flags."""
    closes = [100.0]
    for i in range(70):  # ~2%/session noise -> sigma ~0.02
        closes.append(closes[-1] * (1 + (0.02 if i % 2 else -0.02)))
    closes.append(closes[-1] * 1.09)  # +9%, well past 4 * 0.02
    flags = price_zscore(zbars("REAL", closes, volume=200_000), as_of=AS_OF)

    assert [f.detector for f in flags] == ["price_zscore"]
    assert float(flags[0].details["z"]) > 4.0
    assert flags[0].details["sigma_floored"] == "false"
    assert 0 < flags[0].severity <= 1


def test_price_zscore_requires_tradable_liquidity() -> None:
    """39 of the 61 flagged symbols had a median daily dollar volume of
    exactly zero — no position could be opened in them at any size."""
    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)

    assert price_zscore(zbars("THIN", closes, volume=0), as_of=AS_OF) == []
    liquid_flags = price_zscore(zbars("FAT", closes, volume=100_000), as_of=AS_OF)
    assert [f.detector for f in liquid_flags] == ["price_zscore"]


def test_liquidity_gate_reads_the_bar_source() -> None:
    """D5: the floor is consolidated-equivalent dollars. The SAME printed
    volume clears it on a consolidated feed and fails on an IEX-only one,
    because those two numbers do not mean the same thing."""
    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)
    # $100 x 5,000 = $500k printed: clears $1M consolidated-equivalent only
    # after IEX's ~6.6% share is scaled up (~$7.6M).
    assert price_zscore(zbars("IEXNAME", closes, volume=5_000, source="alpaca"), as_of=AS_OF)
    # $100 x 5,000 = $500k on the consolidated tape is genuinely below $1M.
    assert price_zscore(zbars("TAPENAME", closes, volume=5_000, source="tiingo"), as_of=AS_OF) == []


def test_missing_source_column_assumes_consolidated() -> None:
    """Frames without a source column (every legacy caller) are read the
    conservative way: assume the feed already sees the whole tape, which
    under-states liquidity and fails toward rejecting."""
    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)
    bars = zbars("PLAIN", closes, volume=5_000)
    assert "source" not in bars.columns
    assert price_zscore(bars, as_of=AS_OF) == []


def test_data_quality_flags_are_not_price_zscore_flags() -> None:
    """The routed flag carries a different detector name so downstream can
    keep it out of the trading funnel entirely."""
    from dynamo.data.detectors import DATA_QUALITY_DETECTOR

    assert DATA_QUALITY_DETECTOR == "data_quality"
    assert DATA_QUALITY_DETECTOR != "price_zscore"
