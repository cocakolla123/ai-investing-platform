"""Source-aware ADV: the IEX->consolidated unit conversion (D5).

The defect this guards: thresholds were written as though every feed's
``volume`` column meant the same thing. Alpaca's free feed is IEX-only and
prints a small fraction of the consolidated tape, so a floor calibrated on
one feed silently means something ~15x different on the other.
"""

from __future__ import annotations

import math

import pytest

from dynamo.domain.volume_source import (
    DEFAULT_VOLUME_SOURCE_SHARE,
    IEX_CONSOLIDATED_VOLUME_SHARE,
    VOLUME_SOURCE_SHARE,
    consolidated_equivalent_dollar_volume,
    dominant_source,
    observed_dollar_volume_threshold,
    volume_source_share,
)


class TestProvenance:
    """The constant has to carry its own evidence — D5 asks for a documented,
    named number, not a magic one."""

    def test_module_source_documents_provenance_caveats_and_remeasure(self) -> None:
        """The number is an estimate off 4 symbols; the file must say so, say
        that it drifts, and say how to redo it. A future reader who only sees
        `0.066` must not mistake it for a calibration."""
        from pathlib import Path

        import dynamo.domain.volume_source as mod

        text = Path(str(mod.__file__)).read_text()
        # sample identity and size
        assert "XLB" in text and "XLC" in text and "XLP" in text and "XLY" in text
        assert "4 symbols" in text
        assert "ESTIMATE, NOT A CALIBRATION" in text
        # the measured per-symbol spread and the headline figure
        assert "6.6%" in text
        # instability over time is recorded, not hidden
        assert "3.2%" in text and "8.1%" in text
        # a pointer to redoing the measurement
        assert "TO RE-MEASURE" in text
        assert "bars_1d" in text

    def test_constant_matches_the_measured_median(self) -> None:
        assert pytest.approx(0.066) == IEX_CONSOLIDATED_VOLUME_SHARE
        # ~15x, the factor the defect report measured.
        assert pytest.approx(15.15, abs=0.05) == 1.0 / IEX_CONSOLIDATED_VOLUME_SHARE


class TestSourceShare:
    def test_alpaca_is_iex_only(self) -> None:
        assert volume_source_share("alpaca") == IEX_CONSOLIDATED_VOLUME_SHARE
        assert VOLUME_SOURCE_SHARE["alpaca"] == IEX_CONSOLIDATED_VOLUME_SHARE

    def test_tiingo_is_the_consolidated_reference(self) -> None:
        assert volume_source_share("tiingo") == 1.0

    def test_crypto_is_not_scaled(self) -> None:
        """Crypto has no consolidated tape and no IEX — scaling it would be a
        fabricated adjustment."""
        assert volume_source_share("alpaca_crypto") == 1.0

    @pytest.mark.parametrize("source", [None, "", "some_new_vendor"])
    def test_unknown_source_assumes_consolidated(self, source: str | None) -> None:
        """The conservative default: assume the feed already sees the whole
        tape, which UNDER-states liquidity and so fails toward rejecting a
        trade rather than admitting one."""
        assert volume_source_share(source) == DEFAULT_VOLUME_SOURCE_SHARE == 1.0


class TestThresholdArithmetic:
    def test_iex_observed_scales_up_to_consolidated_equivalent(self) -> None:
        # $66k/day observed on IEX is ~$1M/day consolidated-equivalent.
        assert consolidated_equivalent_dollar_volume(66_000.0, "alpaca") == pytest.approx(
            1_000_000.0, rel=1e-9
        )

    def test_consolidated_source_is_left_alone(self) -> None:
        assert consolidated_equivalent_dollar_volume(1_000_000.0, "tiingo") == 1_000_000.0

    def test_threshold_converts_back_to_observed_units(self) -> None:
        assert observed_dollar_volume_threshold(1_000_000.0, "alpaca") == pytest.approx(66_000.0)
        assert observed_dollar_volume_threshold(1_000_000.0, "tiingo") == 1_000_000.0

    def test_round_trip_is_lossless(self) -> None:
        for source in ("alpaca", "tiingo", "alpaca_crypto", "unknown"):
            observed = 123_456.0
            back = observed_dollar_volume_threshold(
                consolidated_equivalent_dollar_volume(observed, source), source
            )
            assert back == pytest.approx(observed, rel=1e-9)

    def test_the_same_threshold_admits_the_same_liquidity_on_either_feed(self) -> None:
        """The whole point: one name, two feeds, one verdict.

        A name printing $1M/day consolidated shows ~$66k on IEX. Under the old
        raw comparison a $1M floor passed the Tiingo row and rejected the
        Alpaca row for the SAME name; converted, both agree.
        """
        floor = 1_000_000.0
        tiingo_observed = 1_000_000.0
        alpaca_observed = tiingo_observed * IEX_CONSOLIDATED_VOLUME_SHARE

        assert consolidated_equivalent_dollar_volume(tiingo_observed, "tiingo") >= floor
        assert consolidated_equivalent_dollar_volume(alpaca_observed, "alpaca") >= floor
        # the un-converted comparison is what was broken
        assert alpaca_observed < floor

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_unusable_volume_is_not_evidence_of_liquidity(self, bad: float) -> None:
        result = consolidated_equivalent_dollar_volume(bad, "alpaca")
        assert result == 0.0
        assert math.isfinite(result)


class TestDominantSource:
    def test_single_source_group(self) -> None:
        assert dominant_source(["alpaca", "alpaca"]) == "alpaca"

    def test_mixed_group_takes_the_narrowest_view(self) -> None:
        """Mixed feeds for one symbol: pick the source seeing the SMALLEST
        share, so a partly-IEX history is not read as fully consolidated."""
        assert dominant_source(["tiingo", "alpaca"]) == "alpaca"

    def test_empty_is_unknown(self) -> None:
        assert dominant_source([]) is None
        assert dominant_source([None]) is None


class TestRiskLayerContract:
    def test_risk_liquidity_reexports_the_converter(self) -> None:
        """`dynamo.risk` may not import `dynamo.data` (import-linter), so the
        shared notion lives in domain and risk re-exports it."""
        from dynamo.risk.liquidity import consolidated_equivalent_adv

        assert consolidated_equivalent_adv is consolidated_equivalent_dollar_volume

    def test_risk_liquidity_documents_its_units(self) -> None:
        import dynamo.risk.liquidity as mod

        doc = mod.__doc__ or ""
        assert "consolidated-equivalent" in doc.lower()
        assert "adv_dollar_20d" in doc

    def test_entry_cap_is_15x_wider_once_iex_volume_is_converted(self) -> None:
        """The measured consequence: sizing off raw IEX ADV shrinks every
        entry cap by the tape ratio."""
        from dynamo.domain.models.risk import RiskConfig
        from dynamo.risk.liquidity import consolidated_equivalent_adv, max_entry_weight

        cfg = RiskConfig()
        equity = 1_000_000.0
        iex_observed = 66_000.0

        naive = max_entry_weight(iex_observed, equity, cfg)
        corrected = max_entry_weight(
            consolidated_equivalent_adv(iex_observed, "alpaca"), equity, cfg
        )

        assert corrected > naive
        assert corrected / naive == pytest.approx(1.0 / IEX_CONSOLIDATED_VOLUME_SHARE, rel=1e-9)
