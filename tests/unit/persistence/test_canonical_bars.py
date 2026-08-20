"""The canonical-view rule (D2): which feed supplies which column, and why.

Pure-frame tests — no database. The rule lives in
``MarketDataRepository.bars_frame``'s helper so it can be asserted here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from dynamo.domain.volume_source import consolidated_equivalent_dollar_volume
from dynamo.persistence.repositories.market import (
    CANONICAL_BAR_COLUMNS,
    canonical_bars,
)

DAY = datetime(2026, 3, 3, tzinfo=UTC)


def raw_row(
    symbol: str,
    source: str,
    *,
    close: float,
    volume: int,
    adj_close: float | None = None,
    ts: datetime = DAY,
) -> dict[str, object]:
    """One stored ``bars_1d`` row, as ``pd.read_sql`` would hand it over."""
    return {
        "symbol": symbol,
        "ts": ts,
        "source": source,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "adj_close": adj_close,
        "fetched_at": DAY,
    }


# Measured mega-cap scale break: Tiingo consolidated vs Alpaca IEX.
TIINGO_VOLUME = 17_400_000
ALPACA_VOLUME = 1_460_000


def test_price_comes_from_tiingo_and_volume_from_alpaca() -> None:
    frame = pd.DataFrame(
        [
            raw_row("AAPL", "tiingo", close=200.0, adj_close=198.0, volume=TIINGO_VOLUME),
            raw_row("AAPL", "alpaca", close=200.5, volume=ALPACA_VOLUME),
        ]
    )
    out = canonical_bars(frame)
    assert list(out.columns) == list(CANONICAL_BAR_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["close"] == 198.0  # Tiingo's ADJUSTED close, not its raw close
    assert row["source"] == "tiingo"
    assert row["volume"] == ALPACA_VOLUME  # the wide panel's tape
    assert row["volume_source"] == "alpaca"


def test_alpaca_only_symbol_keeps_its_own_price_and_volume() -> None:
    frame = pd.DataFrame([raw_row("WIDE", "alpaca", close=12.0, volume=50_000)])
    out = canonical_bars(frame)
    row = out.iloc[0]
    assert row["close"] == 12.0
    assert row["source"] == "alpaca"
    assert row["volume_source"] == "alpaca"


def test_tiingo_only_history_falls_back_and_says_so() -> None:
    """Before 2024 the free Alpaca feed has nothing, so volume falls back to
    Tiingo — the residual tape break. It must be LABELLED, not hidden."""
    frame = pd.DataFrame(
        [raw_row("AAPL", "tiingo", close=100.0, adj_close=99.0, volume=TIINGO_VOLUME)]
    )
    row = canonical_bars(frame).iloc[0]
    assert row["volume"] == TIINGO_VOLUME
    assert row["volume_source"] == "tiingo"


def test_a_mixed_panel_lands_on_one_volume_scale() -> None:
    """The defect this rule exists to kill: a dual-sourced mega-cap and an
    Alpaca-only name must be comparable on volume.

    With the old 'prefer the tiingo row wholesale' rule the mega-cap carried
    consolidated volume and the rest of the panel carried IEX, so the mega-cap
    outranked a genuinely similar name ~12x on nothing but vendor identity."""
    frame = pd.DataFrame(
        [
            raw_row("DUAL", "tiingo", close=200.0, adj_close=200.0, volume=TIINGO_VOLUME),
            raw_row("DUAL", "alpaca", close=200.0, volume=ALPACA_VOLUME),
            raw_row("WIDE", "alpaca", close=200.0, volume=ALPACA_VOLUME),
        ]
    )
    out = canonical_bars(frame).set_index("symbol")
    assert set(out["volume_source"]) == {"alpaca"}
    assert out.loc["DUAL", "volume"] == out.loc["WIDE", "volume"]


def test_volume_source_is_the_column_the_scaler_must_be_handed() -> None:
    """b294501 scales observed dollar volume to consolidated-equivalent using
    the bar's source. Handing it ``source`` (tiingo) instead of
    ``volume_source`` (alpaca) would leave an IEX number unscaled — a 15x
    error, the exact thing that commit set out to stop."""
    frame = pd.DataFrame(
        [
            raw_row("DUAL", "tiingo", close=200.0, adj_close=200.0, volume=TIINGO_VOLUME),
            raw_row("DUAL", "alpaca", close=200.0, volume=ALPACA_VOLUME),
        ]
    )
    row = canonical_bars(frame).iloc[0]
    observed = float(row["close"]) * float(row["volume"])
    by_volume_source = consolidated_equivalent_dollar_volume(
        observed, str(row["volume_source"])
    )
    by_price_source = consolidated_equivalent_dollar_volume(observed, str(row["source"]))
    assert by_volume_source > 10.0 * by_price_source
    assert abs(by_volume_source / observed - 1.0 / 0.066) < 0.01


def test_unknown_source_never_outranks_a_known_one() -> None:
    frame = pd.DataFrame(
        [
            raw_row("SYM", "somevendor", close=9.0, volume=1),
            raw_row("SYM", "alpaca", close=10.0, volume=2),
        ]
    )
    row = canonical_bars(frame).iloc[0]
    assert row["source"] == "alpaca"
    assert row["volume_source"] == "alpaca"


def test_one_row_per_symbol_and_session() -> None:
    other = datetime(2026, 3, 4, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            raw_row("A", "tiingo", close=1.0, adj_close=1.0, volume=1),
            raw_row("A", "alpaca", close=1.0, volume=2),
            raw_row("A", "alpaca", close=1.5, volume=3, ts=other),
            raw_row("B", "alpaca", close=2.0, volume=4),
        ]
    )
    out = canonical_bars(frame)
    assert len(out) == 3
    assert not out.duplicated(subset=["symbol", "ts"]).any()


def test_empty_frame_keeps_the_canonical_columns() -> None:
    out = canonical_bars(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == list(CANONICAL_BAR_COLUMNS)


# --- Session matching ------------------------------------------------------
#
# The two feeds stamp the same trading day differently: Tiingo at 00:00Z,
# Alpaca at 04:00Z (00:00 New York). A (symbol, ts) key never collided, so the
# "canonical" view emitted BOTH rows for every dual-sourced session.

TIINGO_STAMP = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
ALPACA_STAMP = datetime(2026, 8, 12, 4, 0, tzinfo=UTC)


def test_the_two_feeds_stamp_of_one_session_collapses_to_one_row() -> None:
    """Measured on XLE 2026-08-12: close 61.03 @00:00Z beside 61.02 @04:00Z,
    volume 22,584,153 beside 1,752,526. Every Tier-1 name carried two bars a
    day whose volume alternated ~13x — the D2 scale break, inside one
    symbol's own series."""
    frame = pd.DataFrame(
        [
            raw_row("XLE", "tiingo", close=61.03, adj_close=61.03,
                    volume=22_584_153, ts=TIINGO_STAMP),
            raw_row("XLE", "alpaca", close=61.02, volume=1_752_526, ts=ALPACA_STAMP),
        ]
    )
    out = canonical_bars(frame)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ts"] == pd.Timestamp("2026-08-12", tz="UTC")
    assert row["close"] == 61.03
    assert row["volume"] == 1_752_526
    assert row["volume_source"] == "alpaca"


def test_every_symbol_shares_one_session_index() -> None:
    """A cross-symbol pivot on `ts` must not split a session into two
    half-empty rows just because the names came from different feeds."""
    frame = pd.DataFrame(
        [
            raw_row("DUAL", "tiingo", close=61.0, adj_close=61.0,
                    volume=22_000_000, ts=TIINGO_STAMP),
            raw_row("DUAL", "alpaca", close=61.0, volume=1_700_000, ts=ALPACA_STAMP),
            raw_row("WIDE", "alpaca", close=5.0, volume=1_200, ts=ALPACA_STAMP),
        ]
    )
    out = canonical_bars(frame)
    pivot = out.pivot(index="ts", columns="symbol", values="close")
    assert len(pivot) == 1
    assert not pivot.isna().to_numpy().any()


def test_distinct_sessions_are_still_distinct() -> None:
    """Flooring must merge feeds, not merge days."""
    frame = pd.DataFrame(
        [
            raw_row("XLE", "alpaca", close=61.0, volume=1, ts=ALPACA_STAMP),
            raw_row("XLE", "alpaca", close=62.0, volume=2,
                    ts=ALPACA_STAMP + timedelta(days=1)),
        ]
    )
    out = canonical_bars(frame)
    assert len(out) == 2
    assert list(out["close"]) == [61.0, 62.0]
