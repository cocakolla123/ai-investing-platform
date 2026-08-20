"""Canonical purged, embargoed walk-forward splitter.

Design: docs/design/08-backtesting-and-validation.md section 4 and
04-quant-models.md section 5. One shared implementation for model training and
strategy validation — two consumers, one splitter.

Defaults (in trading sessions): train 3y (756), purge 20, test 6m (126),
embargo 10, roll 6m (126).

Also owns the config-freeze guard: an :class:`EvaluationRegistry` refuses to
evaluate the same ``(config_hash, test-window)`` pair twice, so no config sees
the same OOS window after being tuned on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

TRADING_DAYS_PER_YEAR = 252

__all__ = [
    "ConfigFrozenError",
    "EvaluationRegistry",
    "Fold",
    "WalkForwardConfig",
    "compute_config_hash",
    "generate_folds",
]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Session-count walk-forward parameters (doc 08 section 4 defaults)."""

    train_sessions: int = 3 * TRADING_DAYS_PER_YEAR  # 756
    test_sessions: int = 126
    purge_sessions: int = 20
    embargo_sessions: int = 10
    roll_sessions: int = 126

    def __post_init__(self) -> None:
        if self.train_sessions <= 0 or self.test_sessions <= 0 or self.roll_sessions <= 0:
            raise ValueError("train/test/roll session counts must be positive")
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise ValueError("purge/embargo session counts must be non-negative")


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. Both indexes are subsets of the sessions index."""

    fold_id: int
    train_sessions: pd.DatetimeIndex
    test_sessions: pd.DatetimeIndex

    @property
    def test_start(self) -> pd.Timestamp:
        return cast(pd.Timestamp, self.test_sessions[0])

    @property
    def test_end(self) -> pd.Timestamp:
        return cast(pd.Timestamp, self.test_sessions[-1])


def generate_folds(
    sessions: pd.DatetimeIndex, config: WalkForwardConfig | None = None
) -> list[Fold]:
    """Rolling folds: [train][purge][test], embargo blocks re-entry of post-test
    sessions into any later fold's training window.

    ``sessions`` is the trading calendar (deduplicated, sorted ascending).
    """
    cfg = config or WalkForwardConfig()
    idx = pd.DatetimeIndex(sessions).unique().sort_values()
    folds: list[Fold] = []
    embargoed: set[pd.Timestamp] = set()
    start = 0
    fold_id = 0
    while True:
        train_lo = start
        train_hi = start + cfg.train_sessions  # exclusive
        test_lo = train_hi + cfg.purge_sessions
        test_hi = test_lo + cfg.test_sessions
        if test_hi > len(idx):
            break
        train_ts = idx[train_lo:train_hi]
        if embargoed:
            train_ts = train_ts[[ts not in embargoed for ts in train_ts]]
        test_ts = idx[test_lo:test_hi]
        if len(train_ts) > 0:  # embargo can empty a degenerate window — skip it
            folds.append(Fold(fold_id=fold_id, train_sessions=train_ts, test_sessions=test_ts))
            fold_id += 1
        embargoed.update(idx[test_hi : test_hi + cfg.embargo_sessions])
        start += cfg.roll_sessions
    return folds


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 of a full model/strategy configuration."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConfigFrozenError(RuntimeError):
    """A config hash already saw this OOS window — protocol restart required."""


@dataclass
class EvaluationRegistry:
    """In-memory freeze guard with an exportable record of every claim.

    ``claim_folds`` is all-or-nothing: it validates every requested window
    before recording any, so a rejected run leaves no partial claims.
    """

    _seen: dict[tuple[str, str, str], dict[str, str | int]] = field(default_factory=dict)

    @staticmethod
    def _key(
        config_hash: str, test_start: pd.Timestamp, test_end: pd.Timestamp
    ) -> tuple[str, str, str]:
        start = cast(pd.Timestamp, pd.Timestamp(test_start))
        end = cast(pd.Timestamp, pd.Timestamp(test_end))
        return (config_hash, start.isoformat(), end.isoformat())

    def is_claimed(
        self, config_hash: str, test_start: pd.Timestamp, test_end: pd.Timestamp
    ) -> bool:
        return self._key(config_hash, test_start, test_end) in self._seen

    def claim(self, config_hash: str, test_start: pd.Timestamp, test_end: pd.Timestamp) -> None:
        key = self._key(config_hash, test_start, test_end)
        if key in self._seen:
            raise ConfigFrozenError(
                f"config {config_hash[:12]} already evaluated on OOS window "
                f"{key[1]} .. {key[2]}; changing config after seeing OOS results "
                "restarts the protocol with a new config hash"
            )
        self._seen[key] = {
            "config_hash": key[0],
            "test_start": key[1],
            "test_end": key[2],
            "claim_order": len(self._seen),
        }

    def claim_folds(self, config_hash: str, folds: Iterable[Fold]) -> None:
        fold_list = list(folds)
        for f in fold_list:
            if self.is_claimed(config_hash, f.test_start, f.test_end):
                raise ConfigFrozenError(
                    f"config {config_hash[:12]} already evaluated on OOS window "
                    f"{f.test_start.date()} .. {f.test_end.date()}"
                )
        for f in fold_list:
            self.claim(config_hash, f.test_start, f.test_end)

    def records(self) -> list[dict[str, str | int]]:
        """Exportable, JSON-serializable claim log (persist alongside metrics)."""
        return sorted(self._seen.values(), key=lambda r: int(r["claim_order"]))
