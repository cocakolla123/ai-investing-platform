"""ConstructionConfig presets (doc 07 section 3) and rebalance triggers (s5)."""

from __future__ import annotations

import pytest

from dynamo.domain.models.portfolio import RebalanceTrigger
from dynamo.domain.models.research import Action
from dynamo.domain.models.risk import Regime, RiskBucket
from dynamo.portfolio.config import ConstructionConfig
from dynamo.portfolio.rebalance import TriggerInputs, evaluate_triggers
from tests.unit.portfolio.builders import make_verdict


class TestConfigPresets:
    @pytest.mark.parametrize(
        ("bucket", "split", "tac_max", "max_pos", "max_tac", "turnover"),
        [
            (RiskBucket.LOW, (0.85, 0.15), 0.20, 15, 4, 0.08),
            (RiskBucket.MEDIUM, (0.70, 0.30), 0.35, 25, 8, 0.15),
            (RiskBucket.HIGH, (0.60, 0.40), 0.50, 30, 12, 0.25),
        ],
    )
    def test_bucket_preset_table(self, bucket, split, tac_max, max_pos, max_tac, turnover):
        cfg = ConstructionConfig.for_bucket(bucket)
        assert (cfg.long_term_fraction, cfg.tactical_fraction) == split
        assert cfg.tactical_max == tac_max
        assert cfg.max_positions == max_pos
        assert cfg.max_tactical_positions == max_tac
        assert cfg.turnover_budget_monthly == turnover
        assert cfg.min_trade_weight == 0.0025  # 0.25% all buckets
        assert cfg.drift_band_rel == 0.25

    def test_config_hash_stable_and_sensitive(self):
        a = ConstructionConfig.for_bucket(RiskBucket.MEDIUM)
        b = ConstructionConfig.for_bucket(RiskBucket.MEDIUM)
        c = ConstructionConfig.for_bucket(RiskBucket.HIGH)
        assert a.config_hash == b.config_hash
        assert a.config_hash != c.config_hash

    def test_frozen(self):
        cfg = ConstructionConfig()
        with pytest.raises(Exception):  # noqa: B017 - pydantic frozen error
            cfg.max_positions = 99  # type: ignore[misc]


class TestRebalanceTriggers:
    def test_drift_uses_relative_band(self):
        # 8% target: 10.1% is > 25% relative drift; 9.5% is not.
        target = {"AAA": 0.08}
        verdict = make_verdict(per_position_cap={"AAA": 0.2})
        hot = evaluate_triggers(
            TriggerInputs(
                current_weights={"AAA": 0.101},
                last_target_weights=target,
                verdict=verdict,
            )
        )
        cold = evaluate_triggers(
            TriggerInputs(
                current_weights={"AAA": 0.095},
                last_target_weights=target,
                verdict=verdict,
            )
        )
        assert RebalanceTrigger.DRIFT in hot.triggers and hot.reconstruct
        assert RebalanceTrigger.DRIFT not in cold.triggers and not cold.reconstruct

    def test_verdict_change_for_held_and_new_buy(self):
        decision = evaluate_triggers(
            TriggerInputs(
                current_weights={"AAA": 0.05},
                last_target_weights={"AAA": 0.05},
                verdict=make_verdict(per_position_cap={"AAA": 0.2}),
                previous_actions={"AAA": Action.HOLD},
                current_actions={"AAA": Action.TRIM, "NEW": Action.BUY},
            )
        )
        assert RebalanceTrigger.VERDICT_CHANGE in decision.triggers

    def test_risk_breach_and_regime_and_scheduled_and_vol(self):
        decision = evaluate_triggers(
            TriggerInputs(
                current_weights={"AAA": 0.30},  # above its cap
                last_target_weights={"AAA": 0.30},
                verdict=make_verdict(per_position_cap={"AAA": 0.08}),
                previous_regime=Regime.RISK_ON,  # verdict regime is neutral
                scheduled_review=True,
                vol_scale=0.8,
            )
        )
        assert {
            RebalanceTrigger.RISK_BREACH,
            RebalanceTrigger.REGIME_CHANGE,
            RebalanceTrigger.SCHEDULED,
            RebalanceTrigger.VOL_TARGET,
        } <= decision.triggers
        assert decision.reconstruct
        assert decision.details

    def test_quiet_night_no_triggers(self):
        decision = evaluate_triggers(
            TriggerInputs(
                current_weights={"AAA": 0.08},
                last_target_weights={"AAA": 0.08},
                verdict=make_verdict(per_position_cap={"AAA": 0.10}),
                previous_regime=Regime.NEUTRAL,
                previous_actions={"AAA": Action.HOLD},
                current_actions={"AAA": Action.HOLD},
            )
        )
        assert not decision.triggers
        assert not decision.reconstruct
