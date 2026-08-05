"""Per-model pricing table and cost calculator. Design: doc 03 section 5.

Prices are USD per million tokens (MTok). Message Batches take 50% off both
input and output; prompt-cache reads bill at 0.1x the input rate; cache writes
at 1.25x the input rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MTOK = 1_000_000

BATCH_DISCOUNT = 0.5
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL
CACHE_WRITE_1H_MULTIPLIER = 2.0  # 1-hour TTL (used for batch requests)


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(1.0, 5.0),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0),
    "claude-opus-4-6": ModelPricing(5.0, 25.0),
    # Sonnet 5 intro pricing ($2/$10) runs through 2026-08-31, then $3/$15;
    # we meter at the POST-intro rate so the budget never under-counts.
    "claude-sonnet-5": ModelPricing(3.0, 15.0),
    "claude-opus-5": ModelPricing(5.0, 25.0),
}


class UnknownModelError(KeyError):
    """Raised when no pricing entry matches the model id."""


def pricing_for(model_id: str) -> ModelPricing:
    """Exact match first, then longest known prefix (dated snapshots)."""
    if model_id in PRICING:
        return PRICING[model_id]
    for known in sorted(PRICING, key=len, reverse=True):
        if model_id.startswith(known):
            return PRICING[known]
    raise UnknownModelError(f"no pricing entry for model {model_id!r}")


def calculate_cost(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> float:
    p = pricing_for(model_id)
    cost = (
        input_tokens * p.input_per_mtok
        + cache_read_tokens * p.input_per_mtok * CACHE_READ_MULTIPLIER
        + cache_write_tokens * p.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + output_tokens * p.output_per_mtok
    ) / MTOK
    if batch:
        cost *= BATCH_DISCOUNT
    return cost


@dataclass(frozen=True)
class UsageCost:
    """Token counts pulled from an API usage object, plus the computed dollars."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float


def cost_from_usage(model_id: str, usage: Any, *, batch: bool = False) -> UsageCost:
    """Compute dollars from a `response.usage`-shaped object (attributes may be absent).

    When the API provides the per-TTL cache-write breakdown
    (`usage.cache_creation.ephemeral_1h_input_tokens`), 1h writes are billed at
    their 2.0x multiplier; otherwise all writes use the 5m 1.25x rate.
    """
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    write_1h = 0
    breakdown = getattr(usage, "cache_creation", None)
    if breakdown is not None:
        write_1h = int(getattr(breakdown, "ephemeral_1h_input_tokens", 0) or 0)
        write_1h = min(write_1h, cache_write)
    cost = calculate_cost(
        model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write - write_1h,
        batch=batch,
    )
    if write_1h:
        p = pricing_for(model_id)
        extra = write_1h * p.input_per_mtok * CACHE_WRITE_1H_MULTIPLIER / MTOK
        cost += extra * (BATCH_DISCOUNT if batch else 1.0)
    return UsageCost(input_tokens, output_tokens, cache_read, cache_write, cost)
