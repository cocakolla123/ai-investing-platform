"""Composition root for the API (doc 01 §2: api wires everything).

``build_deps`` assembles the real stores from an engine + settings;
``create_wired_app`` is what ``python -m dynamo.api`` serves.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import Engine

from dynamo.api.deps import AuthConfig, Deps
from dynamo.api.queries import (
    SqlAdminStore,
    SqlJournalStore,
    SqlOpsStore,
    SqlPortfolioStore,
    SqlReportsStore,
    SqlRiskStore,
    make_ping,
)
from dynamo.config.settings import Settings, get_settings
from dynamo.domain.ports import Broker
from dynamo.execution.gate import (
    disengage_kill_switch,
    engage_kill_switch,
    kill_switch_engaged,
)
from dynamo.observability.notify import AlertDispatcher, DisabledNotifier, build_notifier
from dynamo.persistence.db import make_engine
from dynamo.persistence.repositories.market import MarketDataRepository

DASHBOARD_PASSWORD_ENV = "DYNAMO_DASHBOARD_PASSWORD"


class _NullBroker:
    """Stands in when no broker is configured: nothing to cancel."""

    def open_orders(self) -> list[dict[str, object]]:
        return []

    def cancel(self, client_order_id: str) -> None:  # pragma: no cover - never called
        return None


class FileKillSwitch:
    """Kill-switch controller over the gate helpers (doc 09 §7).

    When no broker is configured, engaging just writes the file and raises the
    alert — there are no open orders to cancel.
    """

    def __init__(
        self,
        path: str | Path,
        market_repo: MarketDataRepository,
        broker: Broker | None = None,
        notifier: object | None = None,
    ) -> None:
        self._path = Path(path)
        self._repo = market_repo
        self._broker = broker
        self._notifier = notifier if notifier is not None else DisabledNotifier()

    def engaged(self) -> bool:
        return kill_switch_engaged(self._path)

    def engage(self, note: str, engaged_by: str) -> None:
        dispatch = AlertDispatcher(self._repo.add_alert, self._notifier)

        def alert(name: str, message: str) -> None:
            dispatch("page", name, message, ackable=False)

        broker = self._broker if self._broker is not None else _NullBroker()
        engage_kill_switch(
            self._path,
            broker,  # type: ignore[arg-type]  # _NullBroker covers the used subset
            alert,
            engaged_by=f"{engaged_by}: {note}" if note else engaged_by,
            now=datetime.now(UTC),
        )

    def disengage(self, note: str) -> None:
        disengage_kill_switch(self._path, ack=note)
        AlertDispatcher(self._repo.add_alert, self._notifier)(
            "info", "kill_switch_disengaged", note, ackable=True
        )


def build_deps(
    engine: Engine | None = None,
    *,
    settings: Settings | None = None,
    broker: Broker | None = None,
) -> Deps:
    settings = settings or get_settings()
    engine = engine if engine is not None else make_engine(settings.database_url)
    market_repo = MarketDataRepository(engine)
    return Deps(
        reports=SqlReportsStore(engine),
        portfolio=SqlPortfolioStore(engine),
        risk=SqlRiskStore(engine),
        journal=SqlJournalStore(engine),
        ops=SqlOpsStore(engine, settings),
        admin=SqlAdminStore(engine),
        # settings.kill_switch_file, NOT settings.kill_switch_path: the API,
        # the Dagster daemon and the operator's shell run from different
        # working directories and must agree on one file (doc 09 §7).
        kill=FileKillSwitch(
            settings.kill_switch_file, market_repo, broker, build_notifier(settings)
        ),
        ping=make_ping(engine),
        auth=AuthConfig(
            password=os.environ.get(DASHBOARD_PASSWORD_ENV) or None,
            env=settings.dynamo_env,
        ),
    )


def create_wired_app(engine: Engine | None = None) -> FastAPI:
    from dynamo.api.app import create_app

    return create_app(engine)
