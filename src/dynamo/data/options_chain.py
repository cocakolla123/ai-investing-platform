"""Options chain snapshots via alpaca-py (ADR-0010).

Fetches a narrow window per underlying — strikes within ±20% of spot, target
expiries only — from two alpaca-py surfaces:

- ``TradingClient.get_option_contracts`` for the contract universe + open
  interest (the snapshot endpoint does not carry OI), and
- ``OptionHistoricalDataClient.get_option_snapshot`` for quotes, implied
  volatility, and greeks.

Both imports are guarded: if the installed alpaca-py version lacks the
options surface, ``available`` is False and callers skip with a logged reason
(documented degrade — the sleeve simply does not trade without chain data,
fail-closed). The ATM-IV helper feeds the ``options_iv_history`` table from
which :func:`dynamo.risk.options_sizing.iv_rank` computes the IV-rank proxy.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import structlog

from dynamo.domain.models.options import OptionChainRow, parse_occ_symbol

log = structlog.get_logger(__name__)

STRIKE_WINDOW_PCT = 0.20  # strikes fetched within spot ± 20%
SNAPSHOT_BATCH = 100

# Guarded SDK surface: if the installed alpaca-py predates options support,
# these stay None and the adapter reports ``available == False`` (documented
# degrade — the sleeve does not trade without chain data).
OptionHistoricalDataClient: Any = None
OptionSnapshotRequest: Any = None
TradingClient: Any = None
GetOptionContractsRequest: Any = None
try:  # pragma: no cover - import surface probed once, exercised via fakes
    from alpaca.data.historical.option import (
        OptionHistoricalDataClient,  # pyright: ignore[reportAssignmentType]
    )
    from alpaca.data.requests import (
        OptionSnapshotRequest,  # pyright: ignore[reportAssignmentType]
    )
    from alpaca.trading.client import TradingClient  # pyright: ignore[reportAssignmentType]
    from alpaca.trading.requests import (
        GetOptionContractsRequest,  # pyright: ignore[reportAssignmentType]
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - older alpaca-py without options
    _SDK_AVAILABLE = False


def atm_iv(rows: list[OptionChainRow], spot: float) -> float | None:
    """ATM implied-vol proxy: median IV of the strikes nearest to spot.

    Fails closed (None) when spot is unusable or no row carries a finite IV.
    """
    if not math.isfinite(spot) or spot <= 0.0:
        return None
    with_iv = [r for r in rows if r.iv is not None and math.isfinite(r.iv)]
    if not with_iv:
        return None
    nearest = sorted(with_iv, key=lambda r: (abs(r.strike - spot), r.strike))[:4]
    ivs = sorted(r.iv for r in nearest if r.iv is not None)
    mid = len(ivs) // 2
    return ivs[mid] if len(ivs) % 2 == 1 else (ivs[mid - 1] + ivs[mid]) / 2.0


class AlpacaOptionsChain:
    """Chain snapshot fetcher; degrades to unavailable without the SDK/keys."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        trading_client: Any | None = None,
        data_client: Any | None = None,
    ) -> None:
        self._trading: Any = trading_client
        self._data: Any = data_client
        if not _SDK_AVAILABLE:
            log.warning("options_chain_sdk_unavailable", detail="alpaca-py lacks options")
            return
        if self._trading is None and api_key and api_secret:
            self._trading = TradingClient(
                api_key=api_key, secret_key=api_secret, paper=True, raw_data=True
            )
        if self._data is None and api_key and api_secret:
            self._data = OptionHistoricalDataClient(
                api_key=api_key, secret_key=api_secret, raw_data=True
            )

    @property
    def available(self) -> bool:
        return self._trading is not None and self._data is not None

    def chain_snapshot(
        self,
        underlying: str,
        spot: float,
        expiry_start: date,
        expiry_end: date,
        as_of: datetime,
    ) -> list[OptionChainRow]:
        """Chain rows near spot for the target expiry window (fail-soft: [])."""
        if not self.available or not math.isfinite(spot) or spot <= 0.0:
            return []
        contracts = self._contracts(underlying, spot, expiry_start, expiry_end)
        if not contracts:
            return []
        snapshots = self._snapshots(list(contracts))
        rows: list[OptionChainRow] = []
        for occ, oi in contracts.items():
            try:
                sym, expiry, right, strike = parse_occ_symbol(occ)
            except ValueError:
                continue
            snap = snapshots.get(occ) or {}
            quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
            greeks = snap.get("greeks") or {}
            iv = snap.get("impliedVolatility", snap.get("implied_volatility"))
            rows.append(
                OptionChainRow(
                    underlying=sym,
                    as_of=as_of,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    bid=_maybe_float(quote.get("bp", quote.get("bid_price"))),
                    ask=_maybe_float(quote.get("ap", quote.get("ask_price"))),
                    open_interest=oi,
                    iv=_maybe_float(iv),
                    delta=_maybe_float(greeks.get("delta")),
                )
            )
        rows.sort(key=lambda r: (r.expiry, r.right.value, r.strike))
        return rows

    # --- internals ---

    def _contracts(
        self, underlying: str, spot: float, start: date, end: date
    ) -> dict[str, int | None]:
        """OCC symbol -> open interest for the strike/expiry window."""
        request = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            expiration_date_gte=start,
            expiration_date_lte=end,
            strike_price_gte=str(round(spot * (1.0 - STRIKE_WINDOW_PCT), 2)),
            strike_price_lte=str(round(spot * (1.0 + STRIKE_WINDOW_PCT), 2)),
            limit=1000,
        )
        raw = self._trading.get_option_contracts(request)
        payload = raw if isinstance(raw, dict) else {}
        items = payload.get("option_contracts") or []
        out: dict[str, int | None] = {}
        for c in items:
            d = c if isinstance(c, dict) else {}
            symbol = str(d.get("symbol", ""))
            if not symbol:
                continue
            oi = d.get("open_interest")
            out[symbol] = int(oi) if oi is not None else None
        return out

    def _snapshots(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(symbols), SNAPSHOT_BATCH):
            batch = symbols[i : i + SNAPSHOT_BATCH]
            raw = self._data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=batch))
            payload = raw if isinstance(raw, dict) else {}
            # raw_data nests under "snapshots" in some versions; accept both.
            snaps = payload.get("snapshots", payload)
            if isinstance(snaps, dict):
                for occ, snap in snaps.items():
                    if isinstance(snap, dict):
                        out[str(occ)] = snap
        return out


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
