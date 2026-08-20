"""Router dependencies: small store Protocols + auth (doc 11 §1.4).

Routers depend on these Protocol interfaces only. Real implementations live in
``dynamo.api.queries`` (wired in ``dynamo.api.wiring``); tests supply fakes.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Protocol

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.learning import JournalEntry
from dynamo.domain.models.research import CommitteeVerdict, WatchlistRow

BASIC_AUTH_USERNAME = "dynamo"


# --- Store protocols (implemented by queries.py; faked in tests) ---


class ReportsStore(Protocol):
    def latest_watchlist(self) -> tuple[date | None, list[WatchlistRow]]: ...

    def verdict_history(self, symbol: str) -> list[CommitteeVerdict]: ...

    def dossier_history(self, symbol: str) -> list[dict[str, Any]]: ...

    def report_dates(self) -> list[dict[str, Any]]: ...

    def report_for(self, as_of: date) -> dict[str, Any]: ...

    def daily_report_dates(self) -> list[dict[str, Any]]: ...

    def daily_report(self, as_of: date) -> dict[str, Any] | None: ...


class PortfolioStore(Protocol):
    def positions(self) -> list[dict[str, Any]]: ...

    def equity_series(self) -> list[dict[str, Any]]: ...

    def attribution(self) -> list[dict[str, Any]]: ...


class RiskStore(Protocol):
    def latest_risk_verdict(self) -> dict[str, Any] | None: ...

    def ladder_history(self, limit: int) -> list[dict[str, Any]]: ...

    def latest_regime(self) -> dict[str, Any] | None: ...

    def latest_scenarios(self) -> list[dict[str, Any]]: ...

    def ladder_events(self, limit: int) -> list[dict[str, Any]]: ...

    def unacked_ladder_events(self) -> list[dict[str, Any]]: ...


class JournalStore(Protocol):
    def search(
        self, q: str | None, kind: str | None, symbol: str | None, limit: int
    ) -> list[JournalEntry]: ...


class OpsStore(Protocol):
    def recent_batch_jobs(self, limit: int) -> list[dict[str, Any]]: ...

    def cost_month_to_date(self) -> float: ...

    def monthly_budget(self) -> float: ...

    def open_quarantine(self) -> list[dict[str, Any]]: ...

    def alerts(self, limit: int) -> list[dict[str, Any]]: ...

    def unacked_alerts(self) -> list[dict[str, Any]]: ...

    def unacked_universe_recommendations(self) -> list[dict[str, Any]]: ...

    def open_improvements(self) -> list[dict[str, Any]]: ...


class AdminStore(Protocol):
    """Write paths for the §1.2 admin actions. Every mutation is audited."""

    def audit(self, action: str, payload: dict[str, Any]) -> None: ...

    def get_alert(self, alert_id: int) -> dict[str, Any] | None: ...

    def ack_alert(self, alert_id: int, now: datetime) -> None: ...

    def approve_universe_change(
        self, symbol: str, change: str, reason: str, approved_by: str, as_of: date
    ) -> None: ...

    def get_target_portfolio(self, tp_id: str) -> dict[str, Any] | None: ...

    def approve_target_portfolio(self, tp_id: str, approved_by: str, now: datetime) -> None: ...

    def reject_target_portfolio(self, tp_id: str, note: str, now: datetime) -> None: ...

    def get_improvement(self, proposal_id: str) -> dict[str, Any] | None: ...

    def set_improvement_status(self, proposal_id: str, status: str) -> None: ...

    def get_reconciliation_run(self, run_id: int) -> dict[str, Any] | None: ...

    def reconciliation_acked(self, run_id: int) -> bool: ...

    def get_ladder_event(self, event_id: int) -> dict[str, Any] | None: ...

    def ack_ladder_event(self, event_id: int, acked_by: str, note: str, now: datetime) -> bool: ...

    def ack_reconciliation(self, run_id: int, acked_by: str, note: str, now: datetime) -> None: ...


class KillSwitch(Protocol):
    def engaged(self) -> bool: ...

    def engage(self, note: str, engaged_by: str) -> None: ...

    def disengage(self, note: str) -> None: ...


# --- Composition ---


@dataclass(frozen=True)
class AuthConfig:
    password: str | None  # DYNAMO_DASHBOARD_PASSWORD; None = unset
    env: DynamoEnv


@dataclass(frozen=True)
class Deps:
    reports: ReportsStore
    portfolio: PortfolioStore
    risk: RiskStore
    journal: JournalStore
    ops: OpsStore
    admin: AdminStore
    kill: KillSwitch
    ping: Callable[[], bool]  # DB health probe
    auth: AuthConfig


def get_deps(request: Request) -> Deps:
    return request.app.state.deps  # type: ignore[no-any-return]


DepsDep = Annotated[Deps, Depends(get_deps)]

# --- Auth (doc 11 §1.4: single credential, constant-time compare) ---

_basic = HTTPBasic(auto_error=False)


def require_auth(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic)] = None,
) -> None:
    cfg: AuthConfig = request.app.state.deps.auth
    if cfg.password is None:
        if cfg.env in (DynamoEnv.DEV, DynamoEnv.TEST):
            return  # auth disabled outside PAPER/LIVE when no password is set
        raise HTTPException(
            status_code=401,
            detail="DYNAMO_DASHBOARD_PASSWORD is not set; refusing all requests",
            headers={"WWW-Authenticate": "Basic"},
        )
    supplied_user = credentials.username if credentials else ""
    supplied_pass = credentials.password if credentials else ""
    user_ok = secrets.compare_digest(supplied_user.encode(), BASIC_AUTH_USERNAME.encode())
    pass_ok = secrets.compare_digest(supplied_pass.encode(), cfg.password.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
