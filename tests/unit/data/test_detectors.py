"""Tier-2 detector tests — small pandas frames, assert flags (doc 02 §2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from dynamo.data.detectors import (
    DetectorInputs,
    earnings_surprise,
    filing_event,
    insider_cluster,
    news_burst,
    price_zscore,
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
            accession_no="a1", cik="1", symbol="ACME", form_type="8-K",
            filed_at=AS_OF, url="u", items=("5.02", "9.01"),
        ),
        Filing(
            accession_no="a2", cik="2", symbol="NEWCO", form_type="S-1",
            filed_at=AS_OF, url="u",
        ),
        Filing(  # non-material 8-K items → no flag
            accession_no="a3", cik="3", symbol="BORING", form_type="8-K",
            filed_at=AS_OF, url="u", items=("9.01",),
        ),
    ]
    flags = filing_event(filings, as_of=AS_OF)
    assert {f.symbol for f in flags} == {"ACME", "NEWCO"}


def test_insider_cluster_requires_two_big_buyers() -> None:
    def buy(name: str, shares: float, price: float, days_ago: int = 1) -> InsiderTx:
        return InsiderTx(
            accession_no=f"acc-{name}", symbol="ACME", insider_name=name,
            tx_type="P", shares=shares, price=price,
            filed_at=AS_OF - timedelta(days=days_ago),
        )

    two_buyers = [buy("CEO", 20_000, 10.0), buy("CFO", 15_000, 10.0)]
    flags = insider_cluster(two_buyers, as_of=AS_OF)
    assert len(flags) == 1
    assert flags[0].details["insiders"] == 2

    # one buyer / small buys / stale buys → no flag
    assert insider_cluster([buy("CEO", 20_000, 10.0)], as_of=AS_OF) == []
    assert insider_cluster(
        [buy("CEO", 100, 10.0), buy("CFO", 100, 10.0)], as_of=AS_OF
    ) == []
    assert insider_cluster(
        [buy("CEO", 20_000, 10.0, days_ago=30), buy("CFO", 15_000, 10.0, days_ago=30)],
        as_of=AS_OF,
    ) == []


def test_news_burst_zscore_and_mna_keyword() -> None:
    def item(i: int, days_ago: float, headline: str = "Quarterly update") -> NewsItem:
        return NewsItem(
            id=f"n{i}", symbols=("ACME",), headline=headline, source="t",
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
        earnings=[
            EarningsEvent(symbol="BEAT", event_date=AS_OF.date(), eps_est=1.0, eps_act=2.0)
        ],
    )
    flags = run_detectors(inputs)
    detectors = {f.detector for f in flags}
    assert "price_zscore" in detectors
    assert "volume_spike" in detectors
    assert "earnings_surprise" in detectors
