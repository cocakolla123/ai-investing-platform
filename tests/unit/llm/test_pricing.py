"""Pricing table and cost calculator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.llm.pricing import (
    UnknownModelError,
    calculate_cost,
    cost_from_usage,
    pricing_for,
)


class TestTable:
    def test_rates(self) -> None:
        assert pricing_for("claude-haiku-4-5").input_per_mtok == 1.0
        assert pricing_for("claude-haiku-4-5").output_per_mtok == 5.0
        assert pricing_for("claude-sonnet-4-6").input_per_mtok == 3.0
        assert pricing_for("claude-sonnet-4-6").output_per_mtok == 15.0
        assert pricing_for("claude-opus-4-6").input_per_mtok == 5.0
        assert pricing_for("claude-opus-4-6").output_per_mtok == 25.0

    def test_prefix_match_for_dated_snapshots(self) -> None:
        assert pricing_for("claude-sonnet-4-6-20260401") == pricing_for("claude-sonnet-4-6")

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            pricing_for("gpt-9")


class TestCalculator:
    def test_basic_input_output(self) -> None:
        cost = calculate_cost(
            "claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert cost == pytest.approx(3.0 + 15.0)

    def test_batch_halves_both(self) -> None:
        full = calculate_cost("claude-opus-4-6", input_tokens=200_000, output_tokens=50_000)
        half = calculate_cost(
            "claude-opus-4-6", input_tokens=200_000, output_tokens=50_000, batch=True
        )
        assert half == pytest.approx(full / 2)

    def test_cache_read_is_tenth_of_input(self) -> None:
        cached = calculate_cost("claude-sonnet-4-6", cache_read_tokens=1_000_000)
        plain = calculate_cost("claude-sonnet-4-6", input_tokens=1_000_000)
        assert cached == pytest.approx(plain * 0.1)

    def test_cost_from_usage_reads_usage_object(self) -> None:
        usage = SimpleNamespace(
            input_tokens=10_000,
            output_tokens=2_000,
            cache_read_input_tokens=5_000,
            cache_creation_input_tokens=0,
        )
        uc = cost_from_usage("claude-haiku-4-5", usage)
        assert uc.input_tokens == 10_000
        assert uc.output_tokens == 2_000
        assert uc.cache_read_tokens == 5_000
        expected = (10_000 * 1.0 + 5_000 * 0.1 * 1.0 + 2_000 * 5.0) / 1_000_000
        assert uc.cost_usd == pytest.approx(expected)

    def test_cost_from_usage_tolerates_missing_fields(self) -> None:
        uc = cost_from_usage("claude-haiku-4-5", SimpleNamespace(input_tokens=100))
        assert uc.output_tokens == 0
        assert uc.cost_usd == pytest.approx(100 / 1_000_000)
