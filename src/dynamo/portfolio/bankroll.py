"""Bankroll risk profiles: sizing that depends on how much money the account has.

The committee decides WHAT is worth buying. That doesn't depend on account
size. A profile changes HOW MUCH to hold and how to spread it:

- **small** accounts concentrate. Fees and minimums bite harder, so there
  are fewer, larger positions and no speculative sleeve.
- **medium** accounts hold more names and get a modest risk sleeve.
- **large** accounts keep today's spread and get a larger risk sleeve for
  higher-risk names (crypto, and names the discovery layer added beyond the
  core universe).

OFF unless a profile is named, so nothing changes by default. These defaults
are "common practice", proposed 2026-10-04 and tested in
docs/preregistration_bankroll_profile.md. They are not fitted to any result.
"""

from __future__ import annotations

from dataclasses import dataclass

from dynamo.domain.models.risk import RiskConfig
from dynamo.portfolio.config import ConstructionConfig


@dataclass(frozen=True)
class BankrollTier:
    name: str
    min_equity: float  # tier applies from this account equity upward
    max_positions: int
    per_position_cap: float  # largest single position, fraction of equity
    risk_sleeve_cap: float  # crypto + non-core names together, fraction of equity
    cash_floor: float


PROFILES: dict[str, tuple[BankrollTier, ...]] = {
    "v1": (
        BankrollTier(
            "small",
            0.0,
            max_positions=8,
            per_position_cap=0.15,
            risk_sleeve_cap=0.0,
            cash_floor=0.02,
        ),
        BankrollTier(
            "medium",
            5_000.0,
            max_positions=15,
            per_position_cap=0.10,
            risk_sleeve_cap=0.10,
            cash_floor=0.05,
        ),
        BankrollTier(
            "large",
            50_000.0,
            max_positions=25,
            per_position_cap=0.08,
            risk_sleeve_cap=0.15,
            cash_floor=0.05,
        ),
    ),
}


def tier_for(profile: str, equity: float) -> BankrollTier:
    """The tier of ``profile`` for an account of ``equity`` dollars."""
    if profile not in PROFILES:
        raise ValueError(
            f"unknown bankroll profile {profile!r}; expected one of {sorted(PROFILES)}"
        )
    tiers = sorted(PROFILES[profile], key=lambda t: t.min_equity)
    chosen = tiers[0]
    for tier in tiers:
        if equity >= tier.min_equity:
            chosen = tier
    return chosen


def apply_tier(
    tier: BankrollTier, risk: RiskConfig, construction: ConstructionConfig
) -> tuple[RiskConfig, ConstructionConfig]:
    """Configs with the tier's limits applied.

    The per-position cap must change in the shared RiskConfig: the risk
    engine, the sizing cascade and the final compliance pass all read it, so
    overriding only one of them would be clamped back by the others.
    """
    per_position = min(tier.per_position_cap, risk.per_position_hard)
    new_risk = risk.model_copy(
        update={
            "per_position_cap": per_position,
            "cash_floor": tier.cash_floor,
            "crypto_sleeve_cap": min(risk.crypto_sleeve_cap, tier.risk_sleeve_cap),
            "crypto_max_position": min(risk.crypto_max_position, tier.risk_sleeve_cap),
        }
    )
    new_construction = construction.model_copy(
        update={
            "max_positions": tier.max_positions,
            "risk_sleeve_cap": tier.risk_sleeve_cap,
            "ballast_cash_floor": tier.cash_floor,
        }
    )
    return new_risk, new_construction
