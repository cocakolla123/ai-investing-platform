"""Funnel throughput arithmetic, and a guard on the shipped defaults."""

from __future__ import annotations

import pytest

from dynamo.config.settings import Settings
from dynamo.research.funnel import FunnelBudget


def budget(**overrides: float | int) -> FunnelBudget:
    base: dict[str, float | int] = {
        "triage_cap_per_night": 16,
        "triage_material_rate": 0.30,
        "deep_dive_cap_per_night": 8,
        "deep_dive_promotion_slots": 5,
        "promotion_window_days": 30,
    }
    base.update(overrides)
    return FunnelBudget(**base)  # type: ignore[arg-type]


def test_the_measured_pre_fix_configuration_is_oversubscribed() -> None:
    """The state the audit found: 60 triaged x ~30% material = ~18
    promotions a night against 8 dives, all claimable by promotions."""
    before = budget(triage_cap_per_night=60, deep_dive_promotion_slots=8)
    assert before.promotions_per_night == pytest.approx(18.0)
    assert not before.is_balanced
    # Two thirds of promotions could never be analyzed at that arrival rate.
    assert before.unanalyzed_fraction == pytest.approx(1 - 8 / 18, abs=1e-9)
    # And the queue saturated in the hundreds against 8 slots.
    assert before.steady_state_backlog == pytest.approx(540.0)
    assert before.max_triage_cap == 26
    assert "OVERSUBSCRIBED" in before.diagnosis()


def test_the_shipped_defaults_are_balanced() -> None:
    """THE guard. Changing any one of the five settings in isolation fails
    here rather than silently rebuilding the backlog."""
    settings = Settings()
    shipped = FunnelBudget(
        triage_cap_per_night=settings.tier2_triage_cap_per_night,
        triage_material_rate=settings.triage_material_rate,
        deep_dive_cap_per_night=settings.deep_dive_cap_per_night,
        deep_dive_promotion_slots=settings.deep_dive_promotion_slots,
        promotion_window_days=settings.tier3_promotion_window_days,
    )
    assert shipped.is_balanced, shipped.diagnosis()
    assert shipped.unanalyzed_fraction == 0.0
    # Promotions must not be able to claim the entire nightly budget, or
    # Tier 1 — the names actually held — is never re-examined.
    assert shipped.deep_dive_promotion_slots < shipped.deep_dive_cap_per_night
    # The reserve left for events/staleness has to cover the core universe's
    # own staleness demand: ~55 Tier-1 names / 14-day trigger ~= 4 a night.
    residual = shipped.deep_dive_cap_per_night - shipped.deep_dive_promotion_slots
    assert residual >= 3


def test_max_triage_cap_is_the_largest_cap_that_stays_balanced() -> None:
    b = budget(triage_cap_per_night=1)
    assert budget(triage_cap_per_night=b.max_triage_cap).is_balanced
    assert not budget(triage_cap_per_night=b.max_triage_cap + 1).is_balanced


def test_lengthening_the_window_does_not_fix_an_arrival_rate_problem() -> None:
    """The window is not a throughput lever — it only changes how long
    names wait before being dropped."""
    short = budget(triage_cap_per_night=60, promotion_window_days=30)
    long = budget(triage_cap_per_night=60, promotion_window_days=365)
    assert short.unanalyzed_fraction == long.unanalyzed_fraction
    assert long.steady_state_backlog > short.steady_state_backlog


def test_balanced_backlog_is_bounded_by_one_nights_promotions() -> None:
    b = budget()
    assert b.is_balanced
    assert b.steady_state_backlog == pytest.approx(b.promotions_per_night)


def test_zero_triage_is_degenerate_not_a_division_by_zero() -> None:
    b = budget(triage_cap_per_night=0)
    assert b.promotions_per_night == 0.0
    assert b.unanalyzed_fraction == 0.0
    assert b.is_balanced


@pytest.mark.parametrize(
    "bad",
    [
        {"triage_material_rate": 0.0},
        {"triage_material_rate": 1.5},
        {"deep_dive_promotion_slots": 9},  # exceeds the nightly cap
        {"promotion_window_days": -1},
    ],
)
def test_incoherent_budgets_are_refused(bad: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        budget(**bad)
