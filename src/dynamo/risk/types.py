"""Shared numeric containers for the risk engine.

Pure data: no I/O, no clock reads. All history is passed in as aligned arrays.
Design: docs/design/06-risk-engine.md section 11.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION = float(np.sqrt(float(TRADING_DAYS_PER_YEAR)))


@dataclass(frozen=True)
class ReturnsPanel:
    """Aligned daily (adjusted) simple-return history, shape (T sessions, N symbols)."""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("ReturnsPanel.values must be 2-dimensional (T, N)")
        if arr.shape[0] != len(self.dates) or arr.shape[1] != len(self.symbols):
            raise ValueError("ReturnsPanel shape does not match dates/symbols")
        object.__setattr__(self, "values", arr)

    @property
    def n_sessions(self) -> int:
        return len(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def index_of(self, symbol: str) -> int | None:
        try:
            return self.symbols.index(symbol)
        except ValueError:
            return None

    def column(self, symbol: str) -> FloatArray | None:
        idx = self.index_of(symbol)
        if idx is None:
            return None
        return np.asarray(self.values[:, idx], dtype=np.float64)

    def subset(self, symbols: Sequence[str]) -> ReturnsPanel:
        """Panel restricted to ``symbols`` (all must be present), preserving order given."""
        indices = [self.symbols.index(s) for s in symbols]
        return ReturnsPanel(self.dates, tuple(symbols), self.values[:, indices])

    def window(self, start: date, end: date) -> ReturnsPanel:
        """Rows with start <= session date <= end (empty panel if no overlap)."""
        mask = [start <= d <= end for d in self.dates]
        rows = np.asarray(mask, dtype=np.bool_)
        kept = tuple(d for d, m in zip(self.dates, mask, strict=True) if m)
        return ReturnsPanel(kept, self.symbols, self.values[rows, :])

    def tail(self, sessions: int) -> ReturnsPanel:
        if sessions >= self.n_sessions:
            return self
        return ReturnsPanel(self.dates[-sessions:], self.symbols, self.values[-sessions:, :])


@dataclass(frozen=True)
class SymbolBars:
    """Aligned OHLC history for one symbol (for ATR and stop computation)."""

    dates: tuple[date, ...]
    high: FloatArray
    low: FloatArray
    close: FloatArray

    def __post_init__(self) -> None:
        n = len(self.dates)
        for field in ("high", "low", "close"):
            arr = np.asarray(getattr(self, field), dtype=np.float64)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"SymbolBars.{field} must be 1-d with len(dates) entries")
            object.__setattr__(self, field, arr)

    @property
    def n_sessions(self) -> int:
        return len(self.dates)


def clean_finite(values: FloatArray) -> FloatArray:
    """Flatten and drop non-finite entries (NaN/inf robustness helper)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def is_finite_positive(x: float) -> bool:
    return bool(np.isfinite(x)) and x > 0.0
