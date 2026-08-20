"""Fake stores implementing the ``dynamo.api.deps`` Protocols (doc 11 §5).

No database: the routers depend on small Protocol interfaces, so tests wire a
``Deps`` of in-memory fakes into ``create_app(deps=...)``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dynamo.api.app import create_app
from dynamo.api.deps import AuthConfig, Deps
from dynamo.config.settings import DynamoEnv
from dynamo.domain.models.learning import JournalEntry, JournalKind
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    Dissent,
    Sleeve,
    Stance,
    WatchlistRow,
)

AS_OF = date(2026, 8, 3)


def sample_verdict(symbol: str = "AAPL") -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"cv-{symbol.lower()}-1",
        symbol=symbol,
        as_of=AS_OF,
        action=Action.BUY,
        conviction=0.7,
        thesis="Durable franchise, expanding services margin.",
        bull_case="Services growth re-rates the multiple.",
        bear_case="Hardware cycle rolls over.",
        dissents=(Dissent(agent="bear", stance=Stance.SELL, reason="valuation stretched"),),
        self_critique="May be anchored on prior verdicts.",
    )


def sample_watchlist_row(symbol: str = "AAPL", rank: int = 1) -> WatchlistRow:
    return WatchlistRow(
        as_of=AS_OF,
        sleeve=Sleeve.LONG_TERM,
        symbol=symbol,
        rank=rank,
        blend_score=0.82,
        badges=("conflict",),
        verdict_ref=f"cv-{symbol.lower()}-1",
    )


def sample_journal_entry() -> JournalEntry:
    return JournalEntry(
        id="j-1",
        kind=JournalKind.THESIS,
        symbols=("AAPL",),
        text="Bought AAPL on committee buy verdict.",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


class FakeReportsStore:
    def __init__(self) -> None:
        self.watchlist_as_of: date | None = AS_OF
        self.watchlist_rows = [sample_watchlist_row()]
        self.verdicts: dict[str, list[CommitteeVerdict]] = {"AAPL": [sample_verdict()]}
        self.dossiers: dict[str, list[dict[str, Any]]] = {
            "AAPL": [{"symbol": "AAPL", "as_of": AS_OF, "payload": {"business_summary": "x"}}]
        }
        self.daily_reports: dict[date, dict[str, Any]] = {
            AS_OF: {
                "as_of": AS_OF,
                "markdown": f"# Dynamo Daily Report — {AS_OF.isoformat()} (Monday)\n",
                "payload": {"day": AS_OF.isoformat(), "trades": []},
                "created_at": datetime(2026, 8, 4, 6, 15, tzinfo=UTC),
            }
        }

    def latest_watchlist(self) -> tuple[date | None, list[WatchlistRow]]:
        return self.watchlist_as_of, self.watchlist_rows

    def verdict_history(self, symbol: str) -> list[CommitteeVerdict]:
        return self.verdicts.get(symbol, [])

    def dossier_history(self, symbol: str) -> list[dict[str, Any]]:
        return self.dossiers.get(symbol, [])

    def report_dates(self) -> list[dict[str, Any]]:
        return [{"as_of": AS_OF, "verdict_count": 1}]

    def report_for(self, as_of: date) -> dict[str, Any]:
        if as_of != AS_OF:
            return {"verdicts": [], "watchlist": [], "gaps": []}
        return {
            "verdicts": [sample_verdict()],
            "watchlist": [dict(symbol="AAPL", rank=1)],
            "gaps": [{"scope": "llm_call", "key": "batch-1/MSFT", "reason": "schema_invalid"}],
        }

    def daily_report_dates(self) -> list[dict[str, Any]]:
        return [
            {"as_of": as_of, "created_at": r["created_at"]}
            for as_of, r in sorted(self.daily_reports.items(), reverse=True)
        ]

    def daily_report(self, as_of: date) -> dict[str, Any] | None:
        return self.daily_reports.get(as_of)


class FakePortfolioStore:
    def __init__(self) -> None:
        self.rows = [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry_price": 200.0,
                "current_stop": 180.0,
                "stop_kind": "atr",
                "decision_context_id": "dc-1",
                "verdict_id": "cv-aapl-1",
            }
        ]
        self.series = [
            {"as_of": date(2026, 7, d), "equity": e, "cash": 5000.0, "gross_exposure": 0.5}
            for d, e in ((1, 100_000.0), (2, 101_000.0), (3, 99_500.0), (4, 102_000.0))
        ]

    def positions(self) -> list[dict[str, Any]]:
        return self.rows

    def equity_series(self) -> list[dict[str, Any]]:
        return self.series

    def attribution(self) -> list[dict[str, Any]]:
        return [
            {
                "sleeve": "long_term",
                "symbol": "AAPL",
                "strategy_id": None,
                "bought": 2000.0,
                "sold": 0.0,
                "fees": 1.0,
                "trades": 1,
                "net_cash_flow": -2001.0,
            }
        ]


class FakeRiskStore:
    def __init__(self) -> None:
        self.verdict: dict[str, Any] | None = {
            "id": "rv-1",
            "as_of": datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
            "payload": {
                "ladder": "normal",
                "new_risk_allowed": True,
                "gross_exposure_cap": 1.0,
                "net_exposure_cap": 1.0,
                "cash_floor": 0.05,
                "metrics": {"var95_1d": 0.012},
                "clusters": {"AAPL": "c1"},
                "stress": [{"scenario": "2008", "equity_impact_pct": -0.31}],
            },
        }

    def latest_risk_verdict(self) -> dict[str, Any] | None:
        return self.verdict

    def ladder_history(self, limit: int) -> list[dict[str, Any]]:
        return [{"as_of": AS_OF, "rung": "normal", "drawdown": -0.01, "hwm": 102_000.0}]

    def latest_regime(self) -> dict[str, Any] | None:
        return {"as_of": AS_OF, "regime": "neutral", "composite_score": 0.2}


class FakeJournalStore:
    def __init__(self) -> None:
        self.entries = [sample_journal_entry()]
        self.calls: list[dict[str, Any]] = []

    def search(
        self, q: str | None, kind: str | None, symbol: str | None, limit: int
    ) -> list[JournalEntry]:
        self.calls.append({"q": q, "kind": kind, "symbol": symbol, "limit": limit})
        return self.entries


class FakeOpsStore:
    def recent_batch_jobs(self, limit: int) -> list[dict[str, Any]]:
        return [{"batch_id": "b-1", "stage": "arbiter", "status": "resolved"}]

    def cost_month_to_date(self) -> float:
        return 45.0

    def monthly_budget(self) -> float:
        return 90.0

    def open_quarantine(self) -> list[dict[str, Any]]:
        return []

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        return [{"id": 1, "severity": "info", "name": "x", "ackable": True, "acked_at": None}]

    def unacked_alerts(self) -> list[dict[str, Any]]:
        return self.alerts(10)

    def unacked_universe_recommendations(self) -> list[dict[str, Any]]:
        return []

    def open_improvements(self) -> list[dict[str, Any]]:
        return []


class FakeAdminStore:
    def __init__(self) -> None:
        self.audits: list[tuple[str, dict[str, Any]]] = []
        self.alerts: dict[int, dict[str, Any]] = {}
        self.target_portfolios: dict[str, dict[str, Any]] = {}
        self.improvements: dict[str, dict[str, Any]] = {}
        self.universe_calls: list[dict[str, Any]] = []
        self.reconciliation_runs: dict[int, dict[str, Any]] = {}
        self.reconciliation_ack_rows: list[dict[str, Any]] = []

    def audit(self, action: str, payload: dict[str, Any]) -> None:
        self.audits.append((action, payload))

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        return self.alerts.get(alert_id)

    def ack_alert(self, alert_id: int, now: datetime) -> None:
        self.alerts[alert_id]["acked_at"] = now

    def approve_universe_change(
        self, symbol: str, change: str, reason: str, approved_by: str, as_of: date
    ) -> None:
        self.universe_calls.append({"symbol": symbol, "change": change, "by": approved_by})

    def get_target_portfolio(self, tp_id: str) -> dict[str, Any] | None:
        return self.target_portfolios.get(tp_id)

    def approve_target_portfolio(self, tp_id: str, approved_by: str, now: datetime) -> None:
        self.target_portfolios[tp_id]["approved_at"] = now
        self.target_portfolios[tp_id]["approved_by"] = approved_by

    def reject_target_portfolio(self, tp_id: str, note: str, now: datetime) -> None:
        payload = dict(self.target_portfolios[tp_id].get("payload") or {})
        payload["rejected"] = {"note": note}
        self.target_portfolios[tp_id]["payload"] = payload

    def get_improvement(self, proposal_id: str) -> dict[str, Any] | None:
        return self.improvements.get(proposal_id)

    def set_improvement_status(self, proposal_id: str, status: str) -> None:
        self.improvements[proposal_id]["status"] = status

    def get_reconciliation_run(self, run_id: int) -> dict[str, Any] | None:
        return self.reconciliation_runs.get(run_id)

    def reconciliation_acked(self, run_id: int) -> bool:
        return any(r["run_id"] == run_id for r in self.reconciliation_ack_rows)

    def ack_reconciliation(self, run_id: int, acked_by: str, note: str, now: datetime) -> None:
        from dynamo.execution.reconciliation import acknowledge_reconciliation

        store = self

        class _Store:
            def record_run(self, *args: Any, **kwargs: Any) -> int:  # pragma: no cover
                raise AssertionError("ack path never records runs")

            def record_ack(
                self, run_id: int, acked_by: str, acked_at: datetime, note: str
            ) -> None:
                store.reconciliation_ack_rows.append(
                    {"run_id": run_id, "acked_by": acked_by, "at": acked_at, "note": note}
                )

        acknowledge_reconciliation(_Store(), run_id, acked_by, note, now)


class FakeKillSwitch:
    def __init__(self) -> None:
        self._engaged = False
        self.log: list[str] = []

    def engaged(self) -> bool:
        return self._engaged

    def engage(self, note: str, engaged_by: str) -> None:
        self._engaged = True
        self.log.append(f"engage:{engaged_by}:{note}")

    def disengage(self, note: str) -> None:
        self._engaged = False
        self.log.append(f"disengage:{note}")


def make_deps(
    *,
    password: str | None = None,
    env: DynamoEnv = DynamoEnv.TEST,
    ping: bool = True,
    admin: FakeAdminStore | None = None,
    kill: FakeKillSwitch | None = None,
) -> Deps:
    return Deps(
        reports=FakeReportsStore(),
        portfolio=FakePortfolioStore(),
        risk=FakeRiskStore(),
        journal=FakeJournalStore(),
        ops=FakeOpsStore(),
        admin=admin if admin is not None else FakeAdminStore(),
        kill=kill if kill is not None else FakeKillSwitch(),
        ping=lambda: ping,
        auth=AuthConfig(password=password, env=env),
    )


@pytest.fixture
def deps() -> Deps:
    return make_deps()


@pytest.fixture
def client(deps: Deps) -> TestClient:
    return TestClient(create_app(deps=deps), raise_server_exceptions=False)
