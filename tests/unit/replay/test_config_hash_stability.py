"""Adding a ReplayConfig field must not change any persisted run's id.

``config_hash`` is sha256 over the model's JSON, so every field ever added
shifted every run id — ten stored runs no longer rebuild to their own ids
for exactly that reason. The retrieval-ablation fields (``journal_retrieval``,
``replicate``) are hashed only when set away from their defaults; these tests
pin a real campaign config to its real run id so the next field added has to
follow the same rule or fail here.
"""

from __future__ import annotations

from datetime import date

import pytest

from dynamo.replay.runner import ReplayConfig

# replay_runs.config_json of the committee campaign run, ReplayConfig fields only.
CAMPAIGN_COMMITTEE = {
    "ballast_symbol": "",
    "code_version": "wide-2026-08b",
    "commission_per_order": 0.0,
    "committee_deep_dives_per_day": 6,
    "discovery_min_dollar_volume": 2000000.0,
    "discovery_min_price": 5.0,
    "discovery_mode": "haiku",
    "discovery_promotion_days": 30,
    "discovery_promotions_per_day": 15,
    "end": date(2026, 8, 14),
    "generate_macro_digest": True,
    "horizon_sessions": 20,
    "initial_equity": 100000.0,
    "label": "wide-committee-mar-aug",
    "min_feature_coverage": 0.35,
    "min_order_notional": 0.0,
    "mode": "committee",
    "quant_model_version": "campaign-v8",
    "quant_score_max_age_days": 5,
    "shortlist_candidate_percentile": 0.6,
    "shortlist_cooldown_sessions": 5,
    "shortlist_crypto_slots": 1,
    "shortlist_rotation_lookback_sessions": 20,
    "shortlist_rotation_slots": 1,
    "source_run_id": "",
    "start": date(2026, 3, 3),
    "universe_tiers": (1, 3),
    "whole_shares": False,
    "within_run_learning": True,
}


def test_the_campaign_committee_config_still_has_its_run_id() -> None:
    assert ReplayConfig(**CAMPAIGN_COMMITTEE).run_id == "replay-63220a3a5e9a"


def test_explicit_defaults_hash_like_omitted_ones() -> None:
    a = ReplayConfig(**CAMPAIGN_COMMITTEE)
    b = ReplayConfig(**CAMPAIGN_COMMITTEE, journal_retrieval="vector", replicate=0)
    assert a.run_id == b.run_id


def test_every_ablation_arm_and_replicate_gets_its_own_run_id() -> None:
    ids = {
        ReplayConfig(**CAMPAIGN_COMMITTEE, journal_retrieval=mode, replicate=rep).run_id
        for mode in ("vector", "off", "hybrid")
        for rep in (0, 1)
    }
    assert len(ids) == 6


def test_unknown_retrieval_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="journal_retrieval"):
        ReplayConfig(**CAMPAIGN_COMMITTEE, journal_retrieval="bm25")
