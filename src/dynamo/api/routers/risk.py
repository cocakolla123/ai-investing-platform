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
    """Latest weekly stress suite + Monte Carlo (doc 06 §6).

    Reads the ``risk_scenarios`` table written by the weekly
    ``risk_scenarios`` asset. It used to read a ``stress`` key of the risk
    verdict that no producer ever wrote, so this endpoint always returned an
    empty list.
    """
    rows = deps.risk.latest_scenarios()
    stress = [r for r in rows if r.get("kind") == "stress"]
    monte_carlo = [r for r in rows if r.get("kind") == "monte_carlo"]
    return AsOf(
        data={
            "results": stress,
            "monte_carlo": monte_carlo[0] if monte_carlo else None,
        },
        as_of=rows[0]["as_of"] if rows else None,
    )


@router.get("/ladder/events")
def get_ladder_events(deps: DepsDep, limit: int = 50) -> Page[dict[str, Any]]:
    """Drawdown-ladder transitions, newest first (doc 06 §3)."""
    return Page.of(deps.risk.ladder_events(min(limit, 500)))


@router.get("/ladder/halts")
def get_ladder_halts(deps: DepsDep) -> AsOf[dict[str, Any]]:
    """Ack-requiring rungs nobody has cleared. Non-empty means orders are halted."""
    pending = deps.risk.unacked_ladder_events()
    return AsOf(data={"halted": bool(pending), "pending": pending}, as_of=None)
