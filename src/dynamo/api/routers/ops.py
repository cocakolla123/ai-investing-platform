"""Ops-lite read endpoints: pipeline runs, spend, quarantine, queues, alerts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from dynamo.api.deps import DepsDep
from dynamo.api.envelopes import AsOf, Page

router = APIRouter(tags=["ops"])


@router.get("/ops/summary")
def ops_summary(deps: DepsDep) -> AsOf[dict[str, Any]]:
    mtd = deps.ops.cost_month_to_date()
    budget = deps.ops.monthly_budget()
    data = {
        "batch_jobs": deps.ops.recent_batch_jobs(20),
        "claude_spend": {
            "month_to_date_usd": mtd,
            "monthly_budget_usd": budget,
            "budget_utilization": (mtd / budget) if budget > 0 else 1.0,
        },
        "quarantine_open": deps.ops.open_quarantine(),
        "alerts_unacked": deps.ops.unacked_alerts(),
    }
    return AsOf(data=data, as_of=None)


@router.get("/alerts")
def list_alerts(deps: DepsDep, limit: int = 200) -> Page[dict[str, Any]]:
    return Page.of(deps.ops.alerts(min(limit, 500)))


@router.get("/universe/queue")
def universe_queue(deps: DepsDep) -> Page[dict[str, Any]]:
    """Pending (unacked) universe-change recommendations (doc 02 §2)."""
    return Page.of(deps.ops.unacked_universe_recommendations())


@router.get("/improvements/queue")
def improvements_queue(deps: DepsDep) -> Page[dict[str, Any]]:
    """Open improvement proposals awaiting approve/reject (doc 10)."""
    return Page.of(deps.ops.open_improvements())
