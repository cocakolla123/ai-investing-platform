"""Extended price statistics and peer relative strength (arithmetic, exact)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from dynamo.research.market_stats import (
    peer_context,
    peer_symbols,
    price_stats,
    sector_etf_for,
)

START = datetime(2025, 1, 2, tzinfo=UTC)


def bars(
    symbol: str, closes: list[float], volumes: list[float] | None = None
) -> pd.DataFrame:
    volumes = volumes if volumes is not None else [1_000.0] * len(closes)
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "ts": [START + timedelta(days=i) for i in range(len(closes))],
            "close": closes,
            "volume": volumes,
        }
    )


def ramp(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + step * i for i in range(n)]


def test_extended_price_stats_arithmetic() -> None:
    closes = ramp(300)  # 100.0 .. 399.0, one point per session
    stats = price_stats(bars("ACME", closes), "ACME")
    last = closes[-1]
    assert stats["close"] == pytest.approx(last)
    for window in (20, 60, 120, 250):
        assert stats[f"ret_{window}d_pct"] == pytest.approx(
            (last / closes[-window - 1] - 1.0) * 100.0
        )
    year = closes[-252:]
    assert stats["high_52w"] == pytest.approx(max(year))
    assert stats["low_52w"] == pytest.approx(min(year))
    assert stats["dist_from_52w_high_pct"] == pytest.approx(0.0)  # at the high
    assert stats["dist_from_52w_low_pct"] == pytest.approx(
        (last / min(year) - 1.0) * 100.0
    )
    assert stats["max_drawdown_52w_pct"] == pytest.approx(0.0)  # monotone up


def test_max_drawdown_and_distance_from_the_high() -> None:
    closes = ramp(100) + [199.0 - 2.0 * i for i in range(60)]  # rally, then slide
    stats = price_stats(bars("ACME", closes), "ACME")
    peak = max(closes)
    trough = min(closes[closes.index(peak) :])
    assert stats["max_drawdown_52w_pct"] == pytest.approx((trough / peak - 1.0) * 100.0)
    assert stats["dist_from_52w_high_pct"] == pytest.approx(
        (closes[-1] / peak - 1.0) * 100.0
    )


def test_original_four_stats_keep_their_names_and_meaning() -> None:
    closes = ramp(40)
    volumes = [10.0] * 40
    stats = price_stats(bars("ACME", closes, volumes), "ACME")
    assert set(stats) >= {"close", "ret_20d_pct", "vol_20d_ann_pct", "dollar_volume"}
    frame = bars("ACME", closes, volumes)
    dollar = (frame["close"] * frame["volume"]).tail(20).median()
    assert stats["dollar_volume"] == pytest.approx(float(dollar))


def test_adv_trend_compares_20_and_60_session_averages() -> None:
    closes = [100.0] * 120
    volumes = [10.0] * 100 + [20.0] * 20  # volume doubles in the last month
    stats = price_stats(bars("ACME", closes, volumes), "ACME")
    adv20 = 100.0 * 20.0
    adv60 = 100.0 * (40 * 10.0 + 20 * 20.0) / 60.0
    assert stats["adv_trend_20_60_pct"] == pytest.approx((adv20 / adv60 - 1.0) * 100.0)


def test_short_history_omits_windows_it_cannot_compute() -> None:
    stats = price_stats(bars("ACME", ramp(10)), "ACME")
    assert "ret_20d_pct" not in stats
    assert "high_52w" not in stats
    assert stats["close"] == pytest.approx(109.0)
    assert price_stats(pd.DataFrame(), "ACME") == {}


def test_peer_section_returns_excess_and_beta() -> None:
    own = ramp(200, 100.0, 1.0)
    etf = ramp(200, 50.0, 0.25)
    spy = ramp(200, 400.0, 1.0)
    frame = pd.concat(
        [bars("ACME", own), bars("XLI", etf), bars("SPY", spy)], ignore_index=True
    )
    peers = peer_context(frame, "ACME", sector="Industrials")
    assert peers is not None
    assert peers.sector_etf == "XLI"
    assert peers.benchmark == "SPY"
    for window in (20, 60, 120):
        own_ret = (own[-1] / own[-window - 1] - 1.0) * 100.0
        etf_ret = (etf[-1] / etf[-window - 1] - 1.0) * 100.0
        spy_ret = (spy[-1] / spy[-window - 1] - 1.0) * 100.0
        assert peers.stats[f"ret_{window}d_pct"] == pytest.approx(own_ret)
        assert peers.stats[f"sector_XLI_ret_{window}d_pct"] == pytest.approx(etf_ret)
        assert peers.stats[f"excess_vs_XLI_{window}d_pct"] == pytest.approx(
            own_ret - etf_ret
        )
        assert peers.stats[f"excess_vs_SPY_{window}d_pct"] == pytest.approx(
            own_ret - spy_ret
        )
    assert "beta_120d_vs_SPY" in peers.stats


def test_beta_of_a_two_times_levered_series_is_two() -> None:
    spy = [100.0 * (1.01**i) for i in range(200)]
    own = [100.0 * (1.02**i) for i in range(200)]  # ~2x the daily return
    frame = pd.concat([bars("ACME", own), bars("SPY", spy)], ignore_index=True)
    peers = peer_context(frame, "ACME")
    assert peers is not None
    # Constant returns give zero benchmark variance; add a wobble to identify beta.
    spy2 = [100.0]
    own2 = [100.0]
    for i in range(1, 200):
        step = 0.01 if i % 2 else -0.005
        spy2.append(spy2[-1] * (1 + step))
        own2.append(own2[-1] * (1 + 2 * step))
    frame2 = pd.concat([bars("ACME", own2), bars("SPY", spy2)], ignore_index=True)
    peers2 = peer_context(frame2, "ACME")
    assert peers2 is not None
    assert peers2.stats["beta_120d_vs_SPY"] == pytest.approx(2.0, rel=0.05)


def test_peer_section_absent_without_bars() -> None:
    assert peer_context(pd.DataFrame(), "ACME") is None


def test_sector_etf_mapping_and_reference_symbols() -> None:
    assert sector_etf_for("Consumer Staples") == "XLP"
    assert sector_etf_for("Nonexistent") is None
    sectors = {"WMT": "Consumer Staples", "KO": "Consumer Staples", "CAT": "Industrials"}
    assert peer_symbols(sectors, ["WMT"]) == ["KO", "SPY", "WMT", "XLP"]
