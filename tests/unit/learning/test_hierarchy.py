"""Hierarchical calibration (plan Phase 2): scope chains, shrinkage math,
fallback order, and the DecisionExperience adapters — all pure, no DB."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.learning.hierarchy import (
    GLOBAL_SCOPE,
    SCOPE_DIMENSIONS,
    HierarchicalCalibrator,
    ScopeKey,
    experience_outcome,
    experience_scope_key,
    hier_observations,
    scope_chain,
    shrink,
)

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)


# --- scope_chain: ordering, omission, determinism ---------------------------


def test_scope_chain_full_key_drops_most_specific_first() -> None:
    key = ScopeKey(
        sleeve="core",
        strategy="momentum-rs",
        regime="risk_off",
        sector="tech",
        quant_alignment="agree",
        event_type="earnings",
    )
    assert scope_chain(key) == [
        "sleeve=core|strategy=momentum-rs|regime=risk_off|sector=tech"
        "|quant_alignment=agree|event_type=earnings",
        "sleeve=core|strategy=momentum-rs|regime=risk_off|sector=tech|quant_alignment=agree",
        "sleeve=core|strategy=momentum-rs|regime=risk_off|sector=tech",
        "sleeve=core|strategy=momentum-rs|regime=risk_off",
        "sleeve=core|strategy=momentum-rs",
        "sleeve=core",
        GLOBAL_SCOPE,
    ]


def test_scope_chain_omits_empty_dims() -> None:
    # strategy/event_type unset (not recorded on live experiences yet) and a
    # blank sector: none of them appear in any node.
    key = ScopeKey(sleeve="core", regime="risk_off", sector="  ", quant_alignment="agree")
    assert scope_chain(key) == [
        "sleeve=core|regime=risk_off|quant_alignment=agree",
        "sleeve=core|regime=risk_off",
        "sleeve=core",
        GLOBAL_SCOPE,
    ]


def test_scope_chain_empty_key_is_just_global() -> None:
    assert scope_chain(ScopeKey()) == [GLOBAL_SCOPE]


def test_scope_chain_is_deterministic() -> None:
    key = ScopeKey(regime="neutral", sector="energy")
    chains = {tuple(scope_chain(key)) for _ in range(50)}
    assert chains == {("regime=neutral|sector=energy", "regime=neutral", GLOBAL_SCOPE)}


def test_scope_dimensions_order_general_to_specific() -> None:
    assert SCOPE_DIMENSIONS == (
        "sleeve",
        "strategy",
        "regime",
        "sector",
        "quant_alignment",
        "event_type",
    )


# --- shrinkage math: child-dominates / parent-exactly / monotonic -----------


def test_shrink_n_zero_returns_parent_exactly() -> None:
    assert shrink(0.9, 0, 0.4) == 0.4


def test_shrink_child_dominates_as_n_grows() -> None:
    assert shrink(0.9, 1_000_000, 0.4) == pytest.approx(0.9, abs=1e-4)


def test_shrink_monotonic_blend_between() -> None:
    p_child, p_parent = 0.9, 0.4
    values = [shrink(p_child, n, p_parent) for n in (0, 1, 5, 30, 100, 1000, 100_000)]
    assert values == sorted(values)  # child > parent -> monotonically rising
    assert all(p_parent <= v <= p_child for v in values)
    # Exact formula spot-check at n = k = 30: midpoint.
    assert shrink(p_child, 30, p_parent, k=30) == pytest.approx((p_child + p_parent) / 2)


# --- calibrator: fallback order + identity root ------------------------------

# Injected node fit: calibrated value = observed hit rate (deterministic,
# independent of mapie) so the shrinkage/fallback arithmetic is exact.


def _hit_rate(p_raw: float, history: Sequence[tuple[float, bool]]) -> float:
    return sum(1.0 for _, o in history if o) / len(history)


def _obs(n: int, hit_rate: float) -> list[tuple[float, bool]]:
    wins = round(n * hit_rate)
    return [(0.5, i < wins) for i in range(n)]


KEY = ScopeKey(sleeve="core", regime="risk_off")


def test_empty_global_root_is_identity() -> None:
    calibrator = HierarchicalCalibrator({}, node_fit=_hit_rate)
    result = calibrator.calibrate(0.7, KEY)
    assert result.value == 0.7
    assert result.scope_used == GLOBAL_SCOPE
    assert result.n_effective == 0


def test_node_below_min_n_uses_parent_conformal_value() -> None:
    # Global has a real fit (60 obs, 50% hit rate); the leaf exists but is
    # below min_n=30 -> NO leaf fit, parent (global) value flows through.
    calibrator = HierarchicalCalibrator(
        {GLOBAL_SCOPE: _obs(60, 0.5), "sleeve=core|regime=risk_off": _obs(5, 1.0)},
        node_fit=_hit_rate,
    )
    result = calibrator.calibrate(0.9, KEY)
    assert result.value == pytest.approx(0.5)
    assert result.scope_used == GLOBAL_SCOPE
    assert result.n_effective == 60


def test_leaf_at_min_n_shrinks_toward_parent() -> None:
    # Global 50% (n=100), leaf 90% (n=30, k=30) -> blend midway: 0.7.
    calibrator = HierarchicalCalibrator(
        {GLOBAL_SCOPE: _obs(100, 0.5), "sleeve=core|regime=risk_off": _obs(30, 0.9)},
        node_fit=_hit_rate,
    )
    result = calibrator.calibrate(0.8, KEY)
    assert result.value == pytest.approx((30 * 0.9 + 30 * 0.5) / 60)
    assert result.scope_used == "sleeve=core|regime=risk_off"
    assert result.n_effective == 30


def test_intermediate_node_fit_feeds_leaf_shrinkage() -> None:
    # global 50% -> sleeve=core 80% (n=60) -> leaf below min_n: the leaf
    # inherits the SLEEVE-blended value, not global's.
    calibrator = HierarchicalCalibrator(
        {
            GLOBAL_SCOPE: _obs(100, 0.5),
            "sleeve=core": _obs(60, 0.8),
            "sleeve=core|regime=risk_off": _obs(3, 0.0),
        },
        node_fit=_hit_rate,
    )
    result = calibrator.calibrate(0.6, KEY)
    expected_sleeve = (60 * 0.8 + 30 * 0.5) / 90
    assert result.value == pytest.approx(expected_sleeve)
    assert result.scope_used == "sleeve=core"
    assert result.n_effective == 60


def test_child_dominates_at_large_n() -> None:
    calibrator = HierarchicalCalibrator(
        {GLOBAL_SCOPE: _obs(100, 0.5), "sleeve=core|regime=risk_off": _obs(30_000, 0.9)},
        node_fit=_hit_rate,
    )
    assert calibrator.calibrate(0.8, KEY).value == pytest.approx(0.9, abs=1e-3)


def test_total_on_empty_data_and_real_conformal_path() -> None:
    # Default node fit (conformal_adjust) with degenerate history: identity
    # per calibration.py contract; never raises; output stays in [0, 1].
    calibrator = HierarchicalCalibrator({GLOBAL_SCOPE: [(0.6, True)] * 40})
    result = calibrator.calibrate(0.7, ScopeKey())
    assert 0.0 <= result.value <= 1.0
    # Mixed-outcome real fit is deterministic run-to-run.
    history = [(0.3 + 0.01 * (i % 40), i % 3 == 0) for i in range(80)]
    calibrator = HierarchicalCalibrator({GLOBAL_SCOPE: history})
    first = calibrator.calibrate(0.7, ScopeKey())
    second = calibrator.calibrate(0.7, ScopeKey())
    assert first == second
    assert 0.0 <= first.value <= 1.0


# --- DecisionExperience adapters ---------------------------------------------


def _exp(**overrides: object) -> DecisionExperience:
    base: dict[str, object] = {
        "id": "exp:live:dc-1",
        "created_at": NOW,
        "symbol": "AAPL",
        "as_of": date(2026, 8, 1),
        "sleeve": "long_term",
        "regime": "risk_off",
        "sector": "tech",
        "quant_alignment": "agree",
        "conviction": 0.7,
        "resolved": True,
        "target_achieved": True,
        "label": "win",
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


def test_experience_scope_key_maps_columns() -> None:
    key = experience_scope_key(_exp())
    assert key == ScopeKey(
        sleeve="long_term", regime="risk_off", sector="tech", quant_alignment="agree"
    )
    assert key.strategy is None and key.event_type is None  # not recorded yet


def test_experience_outcome_prefers_target_achieved_then_label() -> None:
    assert experience_outcome(_exp(target_achieved=False, label="win")) is False
    assert experience_outcome(_exp(target_achieved=None, label="win")) is True
    assert experience_outcome(_exp(target_achieved=None, label="loss")) is False
    assert experience_outcome(_exp(target_achieved=None, label=None)) is None


def test_hier_observations_emit_every_chain_prefix() -> None:
    obs = hier_observations([_exp()])
    assert [o.scope_id for o in obs] == [
        "sleeve=long_term|regime=risk_off|sector=tech|quant_alignment=agree",
        "sleeve=long_term|regime=risk_off|sector=tech",
        "sleeve=long_term|regime=risk_off",
        "sleeve=long_term",
        GLOBAL_SCOPE,
    ]
    assert all(o.scope_type == "hier" for o in obs)
    assert all(o.conviction == 0.7 and o.outcome is True for o in obs)


def test_hier_observations_skip_unusable_rows() -> None:
    unusable = [
        _exp(conviction=None),
        _exp(target_achieved=None, label=None),
    ]
    assert hier_observations(unusable) == []
