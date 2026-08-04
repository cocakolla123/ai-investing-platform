"""Vol targeting: band behavior and scale bounds."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.vol_target import MAX_SCALE, exposure_scale, scaled_gross_cap

CFG = RiskConfig()  # target 12% ann, band +/-25%


@settings(max_examples=100, deadline=None)
@given(
    realized=st.one_of(
        st.floats(min_value=-1.0, max_value=5.0),
        st.just(float("nan")),
        st.just(float("inf")),
        st.just(float("-inf")),
    )
)
def test_scale_always_bounded(realized: float) -> None:
    scale = exposure_scale(realized, CFG)
    assert 0.0 <= scale <= MAX_SCALE
    assert math.isfinite(scale)
    assert 0.0 <= scaled_gross_cap(realized, CFG) <= CFG.gross_cap


def test_inside_band_no_change() -> None:
    low = CFG.target_vol_annual * (1.0 - CFG.vol_band)
    high = CFG.target_vol_annual * (1.0 + CFG.vol_band)
    for realized in (low, CFG.target_vol_annual, high):
        assert exposure_scale(realized, CFG) == 1.0


def test_vol_spike_scales_down_toward_target() -> None:
    realized = 0.24  # double the 12% target
    scale = exposure_scale(realized, CFG)
    assert math.isclose(scale, 0.5)
    assert math.isclose(scaled_gross_cap(realized, CFG), 0.5 * CFG.gross_cap)


def test_low_vol_scales_up_but_capped() -> None:
    assert math.isclose(exposure_scale(0.08, CFG), CFG.target_vol_annual / 0.08)
    assert exposure_scale(0.01, CFG) == MAX_SCALE  # 12x raw, clamped
    assert scaled_gross_cap(0.01, CFG) == CFG.gross_cap  # never levers past gross


@settings(max_examples=60, deadline=None)
@given(
    r1=st.floats(min_value=0.16, max_value=2.0),
    bump=st.floats(min_value=1e-6, max_value=1.0),
)
def test_monotone_above_band(r1: float, bump: float) -> None:
    assert exposure_scale(r1 + bump, CFG) <= exposure_scale(r1, CFG) + 1e-12


def test_unusable_realized_vol_is_neutral() -> None:
    assert exposure_scale(float("nan"), CFG) == 1.0
    assert exposure_scale(0.0, CFG) == 1.0
    assert exposure_scale(-0.1, CFG) == 1.0
