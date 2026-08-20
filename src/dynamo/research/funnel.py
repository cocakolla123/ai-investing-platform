"""Discovery-funnel throughput arithmetic — CORE, pure.

The funnel is a pipeline with one expensive stage, and its stages were
sized independently. Measured on the live platform:

- triage classified 60 symbols a night;
- roughly 30% of them came back ``material``, so ~18 Tier-3 promotions
  landed each night;
- each promotion opened a 30-day window;
- the committee ran 8 deep dives a night, and ``select_deep_dives`` gives
  brand-new promotions absolute priority.

Two consequences, both bad, neither visible from any single number:

1. **Promotions expired unanalyzed.** 18 in and at most 8 out per night is
   a queue growing by 10 a night. ~540 names sat concurrently in Tier 3
   against 8 slots — an average wait around 67 days against a 30-day
   window, so a large fraction of promotions lapsed without the committee
   ever looking at them. Silently: ``expire_promotions`` had no production
   caller, so not even the expiry was recorded.

2. **Tier 1 starved.** Promotions outrank events and staleness, so on any
   night with 8+ new promotions — i.e. every night — the core universe got
   *zero* deep dives. The names actually held in the portfolio were the
   ones never re-examined.

The constraint, stated once
---------------------------
The funnel balances when the rate of promotions does not exceed the rate
at which deep dives can consume them::

    triage_cap * material_rate  <=  deep_dive_promotion_slots

Note what is NOT in that inequality: the promotion window. Lengthening the
window does not fix an arrival rate above the service rate — it only
changes how long names wait before being dropped. The window's real job is
how long a name keeps full quant treatment after its dive, not queueing.

Reserving slots rather than raising the cap
-------------------------------------------
``deep_dive_promotion_slots`` is a ceiling on how many of the nightly deep
dives go to brand-new promotions, so events and staleness on the existing
universe always keep some. The residual is sized off the core universe's
own staleness requirement: ~55 Tier-1 names against a 14-day staleness
trigger need ~4 dives a night just to keep their verdicts fresh, so the
default reserves 3 of 8 for them and gives promotions 5. That is a
deliberate under-provision of the routine refresh in favor of discovery,
not an accident.

Raising ``deep_dive_cap_per_night`` instead was rejected: it is the one
knob that costs real money (Opus/Sonnet committee calls against a $90/month
budget), and the honest reading of the measurement is that the funnel was
generating work it could never afford to do, not that it was under-funded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FunnelBudget:
    """The four numbers that have to agree, and what they imply."""

    triage_cap_per_night: int
    triage_material_rate: float
    deep_dive_cap_per_night: int
    deep_dive_promotion_slots: int
    promotion_window_days: int

    def __post_init__(self) -> None:
        if not 0.0 < self.triage_material_rate <= 1.0:
            raise ValueError("triage_material_rate must be in (0, 1]")
        if self.deep_dive_promotion_slots > self.deep_dive_cap_per_night:
            raise ValueError("deep_dive_promotion_slots cannot exceed deep_dive_cap_per_night")
        if min(self.deep_dive_cap_per_night, self.promotion_window_days) < 0:
            raise ValueError("caps and windows must be non-negative")

    @property
    def promotions_per_night(self) -> float:
        """Expected Tier-3 promotions produced by one night of triage."""
        return self.triage_cap_per_night * self.triage_material_rate

    @property
    def promotion_slots_per_night(self) -> int:
        """Deep dives available to brand-new promotions each night."""
        return self.deep_dive_promotion_slots

    @property
    def is_balanced(self) -> bool:
        """True when promotions cannot outrun the dives that consume them."""
        return self.promotions_per_night <= self.promotion_slots_per_night

    @property
    def max_triage_cap(self) -> int:
        """Largest triage cap this deep-dive budget can actually absorb."""
        return math.floor(self.promotion_slots_per_night / self.triage_material_rate)

    @property
    def steady_state_backlog(self) -> float:
        """Names waiting in Tier 3 once the queue fills.

        Balanced: bounded by one night's promotions. Unbalanced: the queue
        grows until the window expires names as fast as they arrive, i.e.
        it saturates at ``arrivals * window``.
        """
        if self.is_balanced:
            return self.promotions_per_night
        return self.promotions_per_night * self.promotion_window_days

    @property
    def unanalyzed_fraction(self) -> float:
        """Fraction of promotions that expire without a deep dive."""
        arrivals = self.promotions_per_night
        if arrivals <= 0:
            return 0.0
        served = min(arrivals, float(self.promotion_slots_per_night))
        return 1.0 - served / arrivals

    def diagnosis(self) -> str:
        """One line, for asset metadata and CLI output."""
        return (
            f"triage {self.triage_cap_per_night}/night x material "
            f"{self.triage_material_rate:.0%} = {self.promotions_per_night:.1f} "
            f"promotions/night vs {self.promotion_slots_per_night} promotion "
            f"deep-dive slots/night -> "
            f"{'balanced' if self.is_balanced else 'OVERSUBSCRIBED'}, "
            f"{self.unanalyzed_fraction:.0%} of promotions never analyzed "
            f"(max sustainable triage cap: {self.max_triage_cap})"
        )
