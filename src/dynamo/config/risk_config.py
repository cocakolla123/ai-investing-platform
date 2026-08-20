"""Loading and versioning :class:`RiskConfig` (doc 06 §11).

Doc 06 §11 says "All thresholds live in ``RiskConfig``, **versioned**; changes
follow the same promotion pipeline as models." What was actually true: every
production construction was a bare ``RiskConfig()``, and ``/risk/limits``
reported ``RiskConfig().model_dump()`` — the *defaults*, not what ran. Changing
a limit meant editing ``domain/models/risk.py`` and redeploying, which is
precisely the shortcut doc 13 §4 claims does not exist.

Three things this module provides:

1. **Overrides from configuration**, not from source: a YAML/JSON file named
   by ``$DYNAMO_RISK_CONFIG``.
2. **A version stamp** derived from the content, so the config that ran is
   identifiable after the fact and two hosts running different limits are
   distinguishable. ``default`` when nothing is overridden.
3. **The risk-bucket multipliers** of doc 06 §2 (LOW 0.6x / MEDIUM 1.0x /
   HIGH 1.4x), which existed as an enum that changed nothing.

The domain model stays pure: this module reads files and settings, and hands
the core an ordinary frozen ``RiskConfig``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dynamo.domain.models.risk import RiskBucket, RiskConfig

DEFAULT_VERSION = "default"

#: doc 06 §2. The bucket scales the *risk budget*, not the structural bounds.
BUCKET_MULTIPLIERS: dict[RiskBucket, float] = {
    RiskBucket.LOW: 0.6,
    RiskBucket.MEDIUM: 1.0,
    RiskBucket.HIGH: 1.4,
}

#: Fields the bucket multiplier scales: the per-name/per-sleeve operating
#: budget. Deliberately NOT here:
#:
#: - ``gross_cap`` / ``cash_floor`` — portfolio structure. 1.4 x a 1.00 gross
#:   cap is leverage, which no bucket setting should be able to introduce.
#: - every ``*_hard`` bound — those are the absolute limits the soft caps are
#:   clamped against; a bucket that widened them would make them not hard.
#: - stop distances, liquidity floors, hysteresis — microstructure and
#:   mechanics, not appetite.
BUCKET_SCALED_FIELDS: tuple[str, ...] = (
    "per_position_cap",
    "sector_cap",
    "cluster_cap",
    "tactical_position_cap",
    "var95_cap",
    "target_vol_annual",
    "kelly_fraction_cap",
    "crypto_sleeve_cap",
    "crypto_max_position",
)

#: soft cap -> the hard bound it may never exceed after scaling.
_SOFT_TO_HARD: dict[str, str] = {
    "per_position_cap": "per_position_hard",
    "sector_cap": "sector_hard",
    "cluster_cap": "cluster_hard",
    "var95_cap": "var95_hard",
}


class RiskConfigError(ValueError):
    """A risk-config override could not be loaded. Never silently ignored:
    running on defaults while believing an override is in force is worse than
    refusing to start."""


def apply_bucket(config: RiskConfig) -> RiskConfig:
    """Apply the doc 06 §2 bucket multiplier to the risk-budget fields.

    ``MEDIUM`` is the identity, so the shipped default is unchanged. Soft caps
    are clamped to their hard siblings afterwards: ``HIGH`` widens the
    operating budget, never the absolute bound.
    """
    multiplier = BUCKET_MULTIPLIERS[config.bucket]
    if multiplier == 1.0:
        return config
    scaled: dict[str, Any] = {
        name: getattr(config, name) * multiplier for name in BUCKET_SCALED_FIELDS
    }
    for soft, hard in _SOFT_TO_HARD.items():
        scaled[soft] = min(scaled[soft], getattr(config, hard))
    return config.model_copy(update=scaled)


def _load_overrides(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RiskConfigError(f"{path} must contain a mapping of RiskConfig fields")
    unknown = set(data) - set(RiskConfig.model_fields)
    if unknown:
        raise RiskConfigError(
            f"{path} sets unknown RiskConfig fields: {', '.join(sorted(unknown))}"
        )
    return dict(data)


def version_for(overrides: Mapping[str, Any]) -> str:
    """Content-addressed version stamp for a set of overrides.

    Deterministic and order-independent, so the same limits always report the
    same version and two hosts running different limits never collide.
    """
    if not overrides:
        return DEFAULT_VERSION
    canonical = json.dumps(dict(overrides), sort_keys=True, separators=(",", ":"), default=str)
    return "cfg-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_risk_config(
    overrides: Mapping[str, Any] | None = None, *, bucket: str | RiskBucket | None = None
) -> RiskConfig:
    """``RiskConfig`` from overrides + bucket, stamped with its version."""
    data = dict(overrides or {})
    if bucket:
        try:
            data["bucket"] = RiskBucket(bucket) if isinstance(bucket, str) else bucket
        except ValueError as exc:
            valid = ", ".join(b.value for b in RiskBucket)
            raise RiskConfigError(
                f"unknown risk bucket {bucket!r}; expected one of: {valid}"
            ) from exc
    version = version_for(data)
    try:
        config = RiskConfig(**data)
    except Exception as exc:  # pydantic validation, bad enum value, ...
        raise RiskConfigError(f"invalid risk config override: {exc}") from exc
    return apply_bucket(config).model_copy(update={"version": version})


def load_risk_config(settings: Any) -> RiskConfig:
    """The effective ``RiskConfig`` for this process.

    ``$DYNAMO_RISK_CONFIG`` names a YAML/JSON file of ``RiskConfig`` field
    overrides (relative paths anchor to ``Settings.home_dir``, like the kill
    switch); ``$DYNAMO_RISK_BUCKET`` selects the doc 06 §2 bucket. Nothing set
    returns the shipped defaults with ``version="default"`` — byte-identical
    to the previous behaviour.

    A named file that is missing or malformed RAISES. Falling back to defaults
    while the operator believes an override is in force is the failure mode
    this refuses to have.
    """
    path_str = str(getattr(settings, "risk_config_path", "") or "")
    overrides: dict[str, Any] = {}
    if path_str:
        anchored = getattr(settings, "anchored", None)
        path = Path(str(anchored(path_str))) if callable(anchored) else Path(path_str)
        if not path.exists():
            raise RiskConfigError(
                f"DYNAMO_RISK_CONFIG points at {path}, which does not exist. "
                "Unset it to run on documented defaults; do not leave it dangling."
            )
        overrides = _load_overrides(path)
    return build_risk_config(overrides, bucket=getattr(settings, "risk_bucket", "") or None)
