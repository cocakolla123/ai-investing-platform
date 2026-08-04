"""Risk read endpoints: limits vs caps, metrics, ladder, stress."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dynamo.api.deps import DepsDep
from dynamo.api.envelopes import AsOf, Page
from dynamo.domain.models.risk import RiskConfig

router = APIRouter(prefix="/risk", tags=["risk"])

_CAP_KEYS = (
    "gross_exposure_cap",
    "net_exposure_cap",
    "cash_floor",
    "per_position_cap",
    "sector_caps",
    "cluster_caps",
    "new_risk_allowed",
    "degraded",
    "ladder",
    "vetoes",
)


@router.get("/limits")
def get_limits(deps: DepsDep) -> AsOf[dict[str, Any]]:
    """Latest risk-verdict caps rendered against the configured RiskConfig."""
    latest = deps.risk.latest_risk_verdict()
    payload = latest["payload"] if latest else {}
    return AsOf(
        data={
            "verdict": {k: payload.get(k) for k in _CAP_KEYS},
            "config": RiskConfig().model_dump(),
        },
        as_of=latest["as_of"] if latest else None,
    )


@router.get("/metrics")
def get_metrics(deps: DepsDep) -> AsOf[dict[str, Any]]:
    latest = deps.risk.latest_risk_verdict()
    payload = latest["payload"] if latest else {}
    return AsOf(
        data={
            "metrics": payload.get("metrics"),
            "clusters": payload.get("clusters"),
            "regime": deps.risk.latest_regime(),
        },
        as_of=latest["as_of"] if latest else None,
    )


@router.get("/ladder")
def get_ladder(deps: DepsDep, limit: int = 90) -> Page[dict[str, Any]]:
    return Page.of(deps.risk.ladder_history(min(limit, 500)))


@router.get("/stress")
def get_stress(deps: DepsDep) -> AsOf[dict[str, Any]]:
    """Latest stress results from the risk-verdict payload, when present."""
    latest = deps.risk.latest_risk_verdict()
    payload = latest["payload"] if latest else {}
    stress = payload.get("stress") or payload.get("stress_results") or []
    return AsOf(
        data={"results": stress},
        as_of=latest["as_of"] if latest else None,
    )
