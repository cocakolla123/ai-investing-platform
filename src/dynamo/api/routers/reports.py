"""Research read endpoints: watchlist, verdicts, dossiers, nightly reports."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException

from dynamo.api.deps import DepsDep
from dynamo.api.envelopes import AsOf, Page
from dynamo.domain.models.research import CommitteeVerdict, WatchlistRow

router = APIRouter(tags=["reports"])


@router.get("/watchlist")
def get_watchlist(deps: DepsDep) -> AsOf[list[WatchlistRow]]:
    as_of, rows = deps.reports.latest_watchlist()
    return AsOf(data=rows, as_of=as_of)


@router.get("/verdicts/{symbol}")
def get_verdicts(symbol: str, deps: DepsDep) -> Page[CommitteeVerdict]:
    history = deps.reports.verdict_history(symbol.upper())
    if not history:
        raise HTTPException(404, f"no committee verdicts for {symbol.upper()}")
    return Page.of(history)


@router.get("/dossiers/{symbol}")
def get_dossiers(symbol: str, deps: DepsDep) -> Page[dict[str, Any]]:
    rows = deps.reports.dossier_history(symbol.upper())
    if not rows:
        raise HTTPException(404, f"no dossiers for {symbol.upper()}")
    return Page.of(rows)


@router.get("/reports")
def list_reports(deps: DepsDep) -> Page[dict[str, Any]]:
    return Page.of(deps.reports.report_dates())


# NOTE: /reports/daily must be registered BEFORE /reports/{report_date} so the
# literal segment wins over the date path parameter.
@router.get("/reports/daily")
def list_daily_reports(deps: DepsDep) -> Page[dict[str, Any]]:
    """Dates that have a stored daily activity report."""
    return Page.of(deps.reports.daily_report_dates())


@router.get("/reports/daily/{report_date}")
def get_daily_report(report_date: date, deps: DepsDep) -> AsOf[dict[str, Any]]:
    """One day's activity report: rendered markdown + the DailyReport JSON."""
    report = deps.reports.daily_report(report_date)
    if report is None:
        raise HTTPException(404, f"no daily report for {report_date.isoformat()}")
    return AsOf(
        data={"markdown": report["markdown"], "report": report["payload"]},
        as_of=report_date,
    )


@router.get("/reports/{report_date}")
def get_report(report_date: date, deps: DepsDep) -> AsOf[dict[str, Any]]:
    report = deps.reports.report_for(report_date)
    if not report["verdicts"] and not report["watchlist"]:
        raise HTTPException(404, f"no nightly report for {report_date.isoformat()}")
    return AsOf(data=report, as_of=report_date)
