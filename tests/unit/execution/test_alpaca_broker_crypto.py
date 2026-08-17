"""Crypto order mapping + symbol normalization in AlpacaBroker (ADR-0013)."""

from __future__ import annotations

from typing import Any, cast

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)

from dynamo.domain.models.portfolio import Order, OrderSide, OrderType
from dynamo.execution.alpaca_broker import AlpacaBroker


class FakeTradingClient:
    def get_all_positions(self) -> Any:
        return [
            {"symbol": "BTCUSD", "qty": "0.5", "avg_entry_price": "60000",
             "asset_class": "crypto"},
            {"symbol": "AAPL", "qty": "10", "avg_entry_price": "230",
             "asset_class": "us_equity"},
        ]


def make_broker() -> AlpacaBroker:
    return AlpacaBroker(client=cast(TradingClient, FakeTradingClient()))


def _order(symbol: str, order_type: OrderType, **kwargs: Any) -> Order:
    return Order(
        client_order_id="coid-1",
        symbol=symbol,
        side=kwargs.pop("side", OrderSide.BUY),
        qty=kwargs.pop("qty", 0.25),
        order_type=order_type,
        **kwargs,
    )


def test_crypto_marketable_limit_uses_gtc() -> None:
    broker = make_broker()
    req = broker._build_request(
        _order("BTC/USD", OrderType.MARKETABLE_LIMIT, limit_price=61_000.0)
    )
    assert isinstance(req, LimitOrderRequest)
    assert req.time_in_force == TimeInForce.GTC  # crypto rejects DAY
    assert req.symbol == "BTC/USD"
    assert req.qty == 0.25  # fractional qty native


def test_equity_marketable_limit_still_uses_day() -> None:
    broker = make_broker()
    req = broker._build_request(
        _order("AAPL", OrderType.MARKETABLE_LIMIT, limit_price=230.0, qty=10)
    )
    assert isinstance(req, LimitOrderRequest)
    assert req.time_in_force == TimeInForce.DAY  # equities unchanged


def test_crypto_market_on_open_maps_to_plain_gtc_market() -> None:
    # 24/7 market: no opening auction exists — OPG would be rejected.
    broker = make_broker()
    req = broker._build_request(_order("ETH/USD", OrderType.MARKET_ON_OPEN))
    assert isinstance(req, MarketOrderRequest)
    assert req.time_in_force == TimeInForce.GTC


def test_equity_market_on_open_still_uses_opg() -> None:
    broker = make_broker()
    req = broker._build_request(_order("AAPL", OrderType.MARKET_ON_OPEN, qty=10))
    assert isinstance(req, MarketOrderRequest)
    assert req.time_in_force == TimeInForce.OPG


def test_crypto_stop_maps_to_stop_limit() -> None:
    # Alpaca crypto supports market/limit/stop-limit only — plain stops are
    # rejected; the sell limit sits 1% below the stop (marketable once hit).
    broker = make_broker()
    req = broker._build_request(
        _order("BTC/USD", OrderType.STOP, side=OrderSide.SELL, stop_price=58_000.0)
    )
    assert isinstance(req, StopLimitOrderRequest)
    assert req.time_in_force == TimeInForce.GTC
    assert req.stop_price == 58_000.0
    assert req.limit_price == 57_420.0  # 1% through the stop


def test_equity_stop_stays_a_plain_stop() -> None:
    broker = make_broker()
    req = broker._build_request(
        _order("AAPL", OrderType.STOP, side=OrderSide.SELL, stop_price=200.0, qty=10)
    )
    assert isinstance(req, StopOrderRequest)
    assert req.stop_price == 200.0


def test_positions_normalize_crypto_symbols_for_reconciliation() -> None:
    """Broker returns "BTCUSD"; reconciliation compares against internal
    "BTC/USD" — normalization at the adapter edge keeps them matched."""
    broker = make_broker()
    positions = broker.positions()
    by_symbol = {p["symbol"]: p for p in positions}
    assert "BTC/USD" in by_symbol
    assert "BTCUSD" not in by_symbol
    assert by_symbol["BTC/USD"]["qty"] == 0.5
    assert by_symbol["AAPL"]["qty"] == 10.0  # equities untouched


def test_order_payload_normalizes_crypto_symbol() -> None:
    normalized = AlpacaBroker._normalize_order(
        {"client_order_id": "c1", "id": "b1", "symbol": "ETHUSD",
         "asset_class": "crypto", "qty": "0.1", "filled_qty": "0",
         "status": "new"}
    )
    assert normalized["symbol"] == "ETH/USD"
