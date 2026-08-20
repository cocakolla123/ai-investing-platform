"""Risk limits come from config, are versioned, and are reported (doc 06 §11).

Before this, every production construction was a bare ``RiskConfig()``,
``/risk/limits`` reported ``RiskConfig().model_dump()`` (the defaults, not
what ran), and ``RiskBucket.{LOW,HIGH}`` changed nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dynamo.config.risk_config import (
    BUCKET_MULTIPLIERS,
    BUCKET_SCALED_FIELDS,
    DEFAULT_VERSION,
    RiskConfigError,
    apply_bucket,
    build_risk_config,
    load_risk_config,
    version_for,
)
from dynamo.config.settings import Settings
from dynamo.domain.models.risk import RiskBucket, RiskConfig


def _settings(tmp_path: Path, **env: str) -> Settings:
    return Settings(DYNAMO_HOME=str(tmp_path), _env_file=None, **env)  # type: ignore[call-arg]


# --- defaults stay defaults ---------------------------------------------


def test_nothing_configured_is_the_shipped_default(tmp_path: Path) -> None:
    config = load_risk_config(_settings(tmp_path))
    assert config.version == DEFAULT_VERSION
    assert config.per_position_cap == RiskConfig().per_position_cap
    assert config.bucket is RiskBucket.MEDIUM


def test_the_default_version_is_stable() -> None:
    assert version_for({}) == DEFAULT_VERSION


# --- overrides change behaviour -----------------------------------------


def test_a_yaml_override_changes_the_limit(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text(
        yaml.safe_dump({"per_position_cap": 0.04, "gross_cap": 0.8})
    )
    config = load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml"))
    assert config.per_position_cap == 0.04
    assert config.gross_cap == 0.8
    assert config.sector_cap == RiskConfig().sector_cap  # untouched fields keep defaults


def test_a_json_override_works_too(tmp_path: Path) -> None:
    (tmp_path / "risk.json").write_text(json.dumps({"cash_floor": 0.20}))
    assert load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.json")).cash_floor == 0.20


def test_an_override_is_versioned_by_its_content(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"per_position_cap": 0.04}))
    config = load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml"))
    assert config.version != DEFAULT_VERSION
    assert config.version.startswith("cfg-")


def test_the_same_limits_always_report_the_same_version() -> None:
    a = build_risk_config({"per_position_cap": 0.04, "sector_cap": 0.2})
    b = build_risk_config({"sector_cap": 0.2, "per_position_cap": 0.04})
    assert a.version == b.version


def test_different_limits_report_different_versions() -> None:
    a = build_risk_config({"per_position_cap": 0.04})
    b = build_risk_config({"per_position_cap": 0.05})
    assert a.version != b.version


def test_a_relative_path_anchors_to_home_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"per_position_cap": 0.03}))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert (
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml")).per_position_cap
        == 0.03
    )


# --- refusals ------------------------------------------------------------


def test_a_dangling_path_refuses_rather_than_falling_back(tmp_path: Path) -> None:
    """Running on defaults while the operator believes an override is live
    is the failure mode this exists to prevent."""
    with pytest.raises(RiskConfigError, match="does not exist"):
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="missing.yaml"))


def test_an_unknown_field_is_refused(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"per_positon_cap": 0.04}))
    with pytest.raises(RiskConfigError, match="per_positon_cap"):
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml"))


def test_a_non_mapping_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(RiskConfigError, match="mapping"):
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml"))


def test_an_empty_file_is_simply_no_overrides(tmp_path: Path) -> None:
    (tmp_path / "risk.yaml").write_text("")
    assert (
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml")).version
        == DEFAULT_VERSION
    )


def test_an_unknown_bucket_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RiskConfigError):
        load_risk_config(_settings(tmp_path, DYNAMO_RISK_BUCKET="reckless"))


# --- the bucket multipliers (doc 06 §2) ---------------------------------


def test_medium_is_the_identity() -> None:
    base = RiskConfig()
    assert apply_bucket(base) == base
    assert BUCKET_MULTIPLIERS[RiskBucket.MEDIUM] == 1.0


def test_low_shrinks_the_risk_budget() -> None:
    low = apply_bucket(RiskConfig(bucket=RiskBucket.LOW))
    base = RiskConfig()
    assert low.per_position_cap == pytest.approx(base.per_position_cap * 0.6)
    assert low.kelly_fraction_cap == pytest.approx(base.kelly_fraction_cap * 0.6)
    assert low.target_vol_annual == pytest.approx(base.target_vol_annual * 0.6)


def test_high_widens_the_risk_budget() -> None:
    high = apply_bucket(RiskConfig(bucket=RiskBucket.HIGH))
    base = RiskConfig()
    assert high.per_position_cap == pytest.approx(base.per_position_cap * 1.4)
    assert high.sector_cap == pytest.approx(base.sector_cap * 1.4)


def test_the_bucket_never_introduces_leverage_or_lowers_the_cash_floor() -> None:
    """gross_cap and cash_floor are structure, not appetite."""
    base = RiskConfig()
    for bucket in (RiskBucket.LOW, RiskBucket.HIGH):
        scaled = apply_bucket(RiskConfig(bucket=bucket))
        assert scaled.gross_cap == base.gross_cap
        assert scaled.cash_floor == base.cash_floor


def test_the_bucket_never_widens_a_hard_bound() -> None:
    base = RiskConfig()
    high = apply_bucket(RiskConfig(bucket=RiskBucket.HIGH))
    assert high.per_position_hard == base.per_position_hard
    assert high.sector_hard == base.sector_hard
    assert high.var95_hard == base.var95_hard


def test_a_scaled_soft_cap_is_clamped_to_its_hard_bound() -> None:
    """HIGH on an already-near-hard soft cap must stop at the hard bound."""
    config = apply_bucket(
        RiskConfig(bucket=RiskBucket.HIGH, per_position_cap=0.14, per_position_hard=0.15)
    )
    assert config.per_position_cap == 0.15  # 0.14 * 1.4 = 0.196, clamped


def test_every_scaled_field_actually_exists_on_the_model() -> None:
    """A typo here would silently disable part of the bucket."""
    for name in BUCKET_SCALED_FIELDS:
        assert name in RiskConfig.model_fields, name


def test_the_bucket_participates_in_the_version() -> None:
    assert build_risk_config(bucket=RiskBucket.HIGH).version != DEFAULT_VERSION
    assert (
        build_risk_config(bucket=RiskBucket.HIGH).version
        != build_risk_config(bucket=RiskBucket.LOW).version
    )


# --- an override really reaches the engine ------------------------------


def test_an_override_changes_what_the_engine_produces(tmp_path: Path) -> None:
    """The end-to-end claim of doc 06 §11: change the file, change the caps.

    Reuses the engine's own fixture builders, so this is the real engine
    computing a real verdict — not a re-derivation of the loader.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_engine import ENGINE, _inputs

    (tmp_path / "risk.yaml").write_text(yaml.safe_dump({"per_position_cap": 0.02}))
    tightened = load_risk_config(_settings(tmp_path, DYNAMO_RISK_CONFIG="risk.yaml"))

    baseline = ENGINE.compute_verdict(_inputs(config=RiskConfig()))
    overridden = ENGINE.compute_verdict(_inputs(config=tightened))

    assert max(overridden.per_position_cap.values()) < max(baseline.per_position_cap.values())
    # ...and the verdict records WHICH config produced it.
    assert overridden.config is not None
    assert overridden.config.version == tightened.version
    assert baseline.config is not None and baseline.config.version == DEFAULT_VERSION


def test_a_high_bucket_loosens_what_the_engine_allows(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_engine import ENGINE, _inputs

    low = ENGINE.compute_verdict(
        _inputs(config=load_risk_config(_settings(tmp_path, DYNAMO_RISK_BUCKET="low")))
    )
    high = ENGINE.compute_verdict(
        _inputs(config=load_risk_config(_settings(tmp_path, DYNAMO_RISK_BUCKET="high")))
    )
    assert max(low.per_position_cap.values()) < max(high.per_position_cap.values())
