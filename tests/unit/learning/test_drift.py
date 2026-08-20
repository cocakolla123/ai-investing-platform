"""Drift monitors (doc 10 §7, §9): a deliberately degraded series MUST trip
the alarm; a healthy series must not. Monitors only propose, never apply."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.learning import ProposalKind, ProposalStatus
from dynamo.learning.drift import (
    feature_psi_monitor,
    ic_decay_monitor,
    incident_trend_monitor,
    mann_kendall_z,
    slippage_monitor,
    strategy_expectation_monitor,
)

AS_OF = date(2026, 8, 3)
NOW = datetime(2026, 8, 3, 22, 0, tzinfo=UTC)


def test_ic_decay_degraded_model_trips_within_window() -> None:
    healthy = [0.06] * 10
    degraded = [0.01] * 30  # < 50% of promotion IC (0.05) for 30 sessions
    proposal = ic_decay_monitor(healthy + degraded, 0.05, "ranker-v1", AS_OF, NOW)
    assert proposal is not None
    assert proposal.kind == ProposalKind.RETRAIN
    assert proposal.target == "ranker-v1"
    assert proposal.status == ProposalStatus.OPEN
    assert proposal.evidence and "50%" in proposal.evidence[0].claim


def test_ic_decay_healthy_model_silent() -> None:
    healthy = [0.05 + 0.01 * ((i % 5) - 2) / 10 for i in range(60)]
    assert ic_decay_monitor(healthy, 0.05, "ranker-v1", AS_OF, NOW) is None


def test_ic_decay_single_recovery_session_resets() -> None:
    series = [0.01] * 29 + [0.06] + [0.01]  # never 30 consecutive bad sessions
    assert ic_decay_monitor(series, 0.05, "ranker-v1", AS_OF, NOW) is None


def test_feature_psi_trips_only_on_top_features() -> None:
    psi = {"momentum_63d": 0.40, "obscure_feature": 0.90, "value_score": 0.05}
    proposal = feature_psi_monitor(psi, ["momentum_63d", "value_score"], "ranker-v1", AS_OF, NOW)
    assert proposal is not None
    assert proposal.kind == ProposalKind.RETRAIN
    assert "momentum_63d" in str(proposal.proposed_change["drifted_features"])
    # obscure_feature is not top-importance → alone it must not trip.
    assert feature_psi_monitor(psi, ["value_score"], "ranker-v1", AS_OF, NOW) is None


def test_strategy_outside_band_20_sessions_proposes_retirement() -> None:
    outside = [-0.5] * 20  # below the 90% band for 20 sessions
    proposal = strategy_expectation_monitor(outside, (0.0, 1.5), "trend-1", AS_OF, NOW)
    assert proposal is not None
    assert proposal.kind == ProposalKind.STRATEGY_RETIREMENT
    assert proposal.target == "trend-1"


def test_strategy_inside_band_silent() -> None:
    inside = [0.7, 0.9, 1.1] * 10
    assert strategy_expectation_monitor(inside, (0.0, 1.5), "trend-1", AS_OF, NOW) is None
    # One in-band session inside the window resets the streak.
    mixed = [-0.5] * 19 + [0.5]
    assert strategy_expectation_monitor(mixed, (0.0, 1.5), "trend-1", AS_OF, NOW) is None


def test_slippage_realized_over_1_5x_modeled_trips() -> None:
    realized = [16.0] * 30
    modeled = [10.0] * 30
    proposal = slippage_monitor(realized, modeled, AS_OF, NOW)
    assert proposal is not None
    assert proposal.kind == ProposalKind.THRESHOLD_CHANGE
    assert proposal.proposed_change["observed_bps"] == 16.0


def test_slippage_within_model_silent() -> None:
    assert slippage_monitor([12.0] * 30, [10.0] * 30, AS_OF, NOW) is None
    assert slippage_monitor([16.0] * 10, [10.0] * 10, AS_OF, NOW) is None  # short window


def test_incident_uptrend_proposes_new_data_source() -> None:
    rising = [float(i) + (0.5 if i % 3 == 0 else 0.0) for i in range(30)]
    proposal = incident_trend_monitor(rising, "tiingo", AS_OF, NOW)
    assert proposal is not None
    assert proposal.kind == ProposalKind.NEW_DATA_SOURCE
    assert proposal.target == "tiingo"


def test_incident_flat_series_silent() -> None:
    flat = [2.0, 1.0, 2.0, 1.0] * 8
    assert incident_trend_monitor(flat, "tiingo", AS_OF, NOW) is None


def test_mann_kendall_direction() -> None:
    assert mann_kendall_z(list(range(20))) > 1.645
    assert mann_kendall_z(list(range(20, 0, -1))) < -1.645
    assert mann_kendall_z([1.0, 1.0, 1.0]) == 0.0
    assert mann_kendall_z([1.0]) == 0.0  # too short


def test_proposals_are_deterministic_and_never_applied() -> None:
    degraded = [0.0] * 30
    a = ic_decay_monitor(degraded, 0.05, "ranker-v1", AS_OF, NOW)
    b = ic_decay_monitor(degraded, 0.05, "ranker-v1", AS_OF, NOW)
    assert a is not None and b is not None
    assert a.id == b.id  # same evidence → same proposal id (dedupe-friendly)
    assert a.status == ProposalStatus.OPEN  # only ever proposed, never applied
