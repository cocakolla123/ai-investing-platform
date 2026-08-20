"""Alpaca adapter mapping tests with fake SDK clients (no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any, cast

from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from dynamo.data.alpaca_data import DAILY_BAR_ADJUSTMENT, AlpacaMarketData
from dynamo.data.ratelimit import TokenBucket


class FakeDataClient:
    def __init__(self) -> None:
        self.bars_request: Any = None

    def get_stock_bars(self, request: Any) -> Any:
        self.bars_request = request
        bar = SimpleNamespace(
            timestamp=datetime(2026, 7, 31, 4, 0, tzinfo=UTC),
            open=10.0, high=11.0, low=9.5, close=10.5, volume=1234,
        )
        return SimpleNamespace(data={"ACME": [bar]})

    def get_stock_latest_quote(self, request: Any) -> Any:
        return {
            "ACME": SimpleNamespace(bid_price=10.0, ask_price=10.2),
            "ONESIDED": SimpleNamespace(bid_price=0.0, ask_price=5.0),
            "EMPTY": SimpleNamespace(bid_price=0.0, ask_price=0.0),
        }


class FakeTradingClient:
    def get_corporate_announcements(self, request: Any) -> Any:
        return [
            SimpleNamespace(
                initiating_symbol="ACME", target_symbol=None,
                ex_date=date(2026, 8, 1), ca_type="split",
                old_rate=1.0, new_rate=2.0, cash=None,
            ),
            SimpleNamespace(
                initiating_symbol=None, target_symbol="XYZ",
                ex_date=date(2026, 8, 2), ca_type="dividend",
                old_rate=None, new_rate=None, cash=0.5,
            ),
            SimpleNamespace(  # no symbol → dropped
                initiating_symbol=None, target_symbol=None,
                ex_date=date(2026, 8, 3), ca_type="merger",
                old_rate=None, new_rate=None, cash=None,
            ),
        ]

    def get_all_assets(self, request: Any) -> Any:
        return [
            SimpleNamespace(symbol="ACME", name="Acme Corp", exchange="NYSE", tradable=True),
            SimpleNamespace(symbol="HALT", name="Halted", exchange="NASDAQ", tradable=False),
        ]


def make_adapter() -> AlpacaMarketData:
    return AlpacaMarketData(
        "key",
        "secret",
        data_client=cast(StockHistoricalDataClient, FakeDataClient()),
        trading_client=cast(TradingClient, FakeTradingClient()),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )


def test_daily_bars_mapping() -> None:
    adapter = make_adapter()
    assert adapter.source == "alpaca"
    bars = list(adapter.daily_bars(["ACME"], date(2026, 7, 30), date(2026, 7, 31)))
    assert len(bars) == 1
    b = bars[0]
    assert b.symbol == "ACME"
    assert b.close == 10.5
    assert b.volume == 1234
    # Tiingo owns the FULLY adjusted close (doc 02 §5); these are split-only.
    assert b.adj_close is None
    assert b.source == "alpaca"


def test_daily_bars_are_requested_split_adjusted() -> None:
    """D1: raw closes made corporate actions look like price moves — RCON's
    1-for-141 reverse split reached the detectors as +14,015%. The wide feed
    now asks Alpaca to adjust for splits."""
    fake = FakeDataClient()
    adapter = AlpacaMarketData(
        "key",
        "secret",
        data_client=cast(StockHistoricalDataClient, fake),
        trading_client=cast(TradingClient, FakeTradingClient()),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )
    list(adapter.daily_bars(["ACME"], date(2026, 7, 30), date(2026, 7, 31)))
    assert fake.bars_request.adjustment == Adjustment.SPLIT
    assert DAILY_BAR_ADJUSTMENT == Adjustment.SPLIT


def test_daily_bars_adjustment_is_overridable_for_the_raw_series() -> None:
    """The raw series stays reachable: the dual-source close check compares
    unadjusted closes, so it must be able to ask for them."""
    fake = FakeDataClient()
    adapter = AlpacaMarketData(
        "key",
        "secret",
        data_client=cast(StockHistoricalDataClient, fake),
        trading_client=cast(TradingClient, FakeTradingClient()),
        bucket=TokenBucket(rate=10_000.0, capacity=10_000.0),
    )
    list(
        adapter.daily_bars(
            ["ACME"], date(2026, 7, 30), date(2026, 7, 31), adjustment=Adjustment.RAW
        )
    )
    assert fake.bars_request.adjustment == Adjustment.RAW


def test_dividends_are_not_adjusted_away() -> None:
    """Split-only, deliberately: Adjustment.ALL would back out dividends and
    make quality.dual_source_close_check fire on every historical ex-date for
    a difference that is not an error."""
    assert DAILY_BAR_ADJUSTMENT != Adjustment.ALL
    assert DAILY_BAR_ADJUSTMENT != Adjustment.DIVIDEND


def test_latest_quotes_midpoint_and_fallback() -> None:
    quotes = make_adapter().latest_quotes(["ACME", "ONESIDED", "EMPTY"])
    assert abs(quotes["ACME"] - 10.1) < 1e-9
    assert quotes["ONESIDED"] == 5.0
    assert "EMPTY" not in quotes


def test_corporate_actions_mapping() -> None:
    actions = list(make_adapter().corporate_actions(date(2026, 8, 1), date(2026, 8, 5)))
    assert len(actions) == 2
    split = next(a for a in actions if a.action_type == "split")
    assert split.symbol == "ACME"
    assert split.ratio == 2.0
    div = next(a for a in actions if a.action_type == "dividend")
    assert div.symbol == "XYZ"
    assert div.amount == 0.5
    assert all(a.source == "alpaca" for a in actions)


def test_tradable_instruments_filters_untradable() -> None:
    instruments = list(make_adapter().tradable_instruments())
    assert [i.symbol for i in instruments] == ["ACME"]
    assert instruments[0].exchange == "NYSE"
