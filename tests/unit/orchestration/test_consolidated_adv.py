"""ADV units at the orchestration boundary (docs 06 §10, ADR-0013).

``dynamo.risk.liquidity`` declares every ``adv_dollar_20d`` it receives to be
consolidated-equivalent dollars — but the live assets computed raw
``close x volume`` medians off whatever feed supplied the bars. For the ~11k
Alpaca-only equities that under-stated ADV ~15x; for the crypto sleeve it
under-stated it >= 1000x, and the first live risk campaign vetoed every
crypto buy off a ~$4k "dollar volume" for AVAX. ``consolidated_adv_20d`` is
the conversion at the point of computation; these tests pin its contract.
"""

from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest
from orchestration.assets.portfolio_exec import consolidated_adv_20d

from dynamo.domain.volume_source import (
    ALPACA_CRYPTO_CONSOLIDATED_VOLUME_SHARE,
    IEX_CONSOLIDATED_VOLUME_SHARE,
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bars(
    symbol: str, source: str | None, closes: list[float], volume: float
) -> list[dict[str, object]]:
    days = pd.bdate_range(end=pd.Timestamp("2026-09-11", tz=UTC), periods=len(closes))
    rows: list[dict[str, object]] = []
    for ts, close in zip(days, closes, strict=True):
        row: dict[str, object] = {"symbol": symbol, "ts": ts, "close": close, "volume": volume}
        if source is not None:
            row["volume_source"] = source
        rows.append(row)
    return rows


class TestConsolidatedAdv20d:
    def test_empty_frame_is_empty(self) -> None:
        assert consolidated_adv_20d(pd.DataFrame()) == {}

    def test_tiingo_volume_is_left_alone(self) -> None:
        adv = consolidated_adv_20d(_frame(_bars("XLB", "tiingo", [100.0] * 25, 10_000.0)))
        assert adv["XLB"] == pytest.approx(1_000_000.0)

    def test_alpaca_iex_volume_scales_up_to_the_tape(self) -> None:
        adv = consolidated_adv_20d(_frame(_bars("ACME", "alpaca", [100.0] * 25, 660.0)))
        assert adv["ACME"] == pytest.approx(66_000.0 / IEX_CONSOLIDATED_VOLUME_SHARE)
        assert adv["ACME"] == pytest.approx(1_000_000.0)

    def test_crypto_venue_volume_scales_to_global_equivalent(self) -> None:
        """The campaign defect: AVAX at ~$4.9k/day of own-venue volume must
        clear the $1M consolidated ``min_adv_dollar`` floor after scaling."""
        adv = consolidated_adv_20d(_frame(_bars("AVAX/USD", "alpaca_crypto", [24.5] * 25, 200.0)))
        assert adv["AVAX/USD"] == pytest.approx(4_900.0 / ALPACA_CRYPTO_CONSOLIDATED_VOLUME_SHARE)
        assert adv["AVAX/USD"] >= 1_000_000.0

    def test_missing_volume_source_column_assumes_consolidated(self) -> None:
        """Frames without the label (tests, ad-hoc frames) must not invent a
        scale factor — assume-consolidated under-states liquidity and fails
        toward rejecting a trade."""
        adv = consolidated_adv_20d(_frame(_bars("ACME", None, [100.0] * 25, 660.0)))
        assert adv["ACME"] == pytest.approx(66_000.0)

    def test_mixed_sources_take_the_narrowest_view(self) -> None:
        """A partly-IEX tail is scaled as IEX (dominant_source picks the
        smallest share) rather than averaged across incompatible scales."""
        rows = _bars("ACME", "tiingo", [100.0] * 10, 660.0) + _bars(
            "ACME", "alpaca", [100.0] * 10, 660.0
        )
        # Interleave timestamps so both sources land inside the 20-day tail.
        frame = _frame(rows)
        frame["ts"] = pd.date_range(end=pd.Timestamp("2026-09-11", tz=UTC), periods=len(frame))
        adv = consolidated_adv_20d(frame)
        assert adv["ACME"] == pytest.approx(66_000.0 / IEX_CONSOLIDATED_VOLUME_SHARE)

    def test_median_is_over_the_last_20_sessions_only(self) -> None:
        """Old high-volume history outside the 20-session tail must not leak
        into tonight's liquidity estimate."""
        old = _bars("ACME", "tiingo", [100.0] * 30, 1_000_000.0)
        recent = _bars("ACME", "tiingo", [100.0] * 20, 10_000.0)
        frame = _frame(old + recent)
        frame["ts"] = pd.date_range(end=pd.Timestamp("2026-09-11", tz=UTC), periods=len(frame))
        adv = consolidated_adv_20d(frame)
        assert adv["ACME"] == pytest.approx(1_000_000.0)

    def test_unusable_volume_is_zero_not_nan(self) -> None:
        frame = _frame(_bars("ACME", "alpaca", [100.0] * 5, float("nan")))
        assert consolidated_adv_20d(frame)["ACME"] == 0.0
