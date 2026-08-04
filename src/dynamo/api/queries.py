"""Real store implementations behind the ``dynamo.api.deps`` Protocols.

Reuses repositories where methods exist (journal, proposals, universe); adds
read-only SQLAlchemy Core selects where none does. This module is the only
place API SQL lives — routers never see the engine.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, func, select, text, update

from dynamo.config.settings import Settings
from dynamo.data.universe import UniverseManager
from dynamo.domain.models.learning import JournalEntry, JournalKind, ProposalStatus
from dynamo.domain.models.research import CommitteeVerdict, Sleeve, WatchlistRow
from dynamo.llm.cost_meter import SqlAlchemyLedgerWriter
from dynamo.persistence import schema
from dynamo.persistence.repositories.learning import JournalRepository, ProposalRepository
from dynamo.persistence.repositories.market import MarketDataRepository


def _rows(engine: Engine, stmt: Any) -> list[dict[str, Any]]:
    with engine.connect() as cx:
        return [dict(m) for m in cx.execute(stmt).mappings().all()]


class SqlReportsStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def latest_watchlist(self) -> tuple[date | None, list[WatchlistRow]]:
        t = schema.watchlist
        with self._engine.connect() as cx:
            latest = cx.execute(select(func.max(t.c.as_of))).scalar()
        if latest is None:
            return None, []
        rows = _rows(
            self._engine,
            select(t).where(t.c.as_of == latest).order_by(t.c.sleeve, t.c.rank),
        )
        return latest, [
            WatchlistRow(
                as_of=r["as_of"],
                sleeve=Sleeve(r["sleeve"]),
                symbol=r["symbol"],
                rank=r["rank"],
                blend_score=r["blend_score"],
                committee_component=r["committee_component"],
                quant_component=r["quant_component"],
                quality_component=r["quality_component"],
                badges=tuple(r["badges"] or ()),
                blend_config_hash=r["blend_config_hash"] or "",
                verdict_ref=r["verdict_ref"],
                signal_ref=r["signal_ref"],
            )
            for r in rows
        ]

    def verdict_history(self, symbol: str) -> list[CommitteeVerdict]:
        t = schema.committee_verdicts
        rows = _rows(
            self._engine,
            select(t.c.payload).where(t.c.symbol == symbol).order_by(t.c.as_of.desc()),
        )
        return [CommitteeVerdict.model_validate(r["payload"]) for r in rows]

    def dossier_history(self, symbol: str) -> list[dict[str, Any]]:
        t = schema.dossiers
        return _rows(
            self._engine,
            select(t).where(t.c.symbol == symbol).order_by(t.c.as_of.desc()),
        )

    def report_dates(self) -> list[dict[str, Any]]:
        t = schema.committee_verdicts
        stmt = (
            select(t.c.as_of, func.count().label("verdict_count"))
            .group_by(t.c.as_of)
            .order_by(t.c.as_of.desc())
        )
        return _rows(self._engine, stmt)

    def report_for(self, as_of: date) -> dict[str, Any]:
        v = schema.committee_verdicts
        w = schema.watchlist
        q = schema.quarantine
        verdicts = _rows(
            self._engine,
            select(v.c.payload).where(v.c.as_of == as_of).order_by(v.c.symbol),
        )
        watch = _rows(
            self._engine,
            select(w).where(w.c.as_of == as_of).order_by(w.c.sleeve, w.c.rank),
        )
        day_start = datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC)
        gaps = _rows(
            self._engine,
            select(q)
            .where(q.c.created_at >= day_start, func.date(q.c.created_at) == as_of)
            .order_by(q.c.id),
        )
        return {
            "verdicts": [CommitteeVerdict.model_validate(r["payload"]) for r in verdicts],
            "watchlist": watch,
            "gaps": gaps,  # quarantined-call gaps rendered explicitly (doc 03 §3)
        }


class SqlPortfolioStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def positions(self) -> list[dict[str, Any]]:
        rows = _rows(self._engine, select(schema.positions).order_by(schema.positions.c.symbol))
        dc = schema.decision_context
        contexts = _rows(
            self._engine,
            select(dc.c.symbol, dc.c.id, dc.c.verdict_id).order_by(dc.c.created_at),
        )
        latest_ctx = {c["symbol"]: c for c in contexts}  # last write wins per symbol
        for row in rows:
            ctx = latest_ctx.get(row["symbol"])
            row["decision_context_id"] = ctx["id"] if ctx else None
            row["verdict_id"] = ctx["verdict_id"] if ctx else None
        return rows

    def equity_series(self) -> list[dict[str, Any]]:
        t = schema.equity_curve
        return _rows(self._engine, select(t).order_by(t.c.as_of))

    def attribution(self) -> list[dict[str, Any]]:
        """Cash-flow attribution per (sleeve, symbol) from trades joined to
        orders, with strategy tags recovered from position lots."""
        t, o = schema.trades, schema.orders
        stmt = (
            select(
                o.c.sleeve,
                t.c.symbol,
                t.c.side,
                func.sum(t.c.qty * t.c.fill_price).label("notional"),
                func.sum(t.c.fees).label("fees"),
                func.count().label("trades"),
            )
            .join_from(t, o, t.c.client_order_id == o.c.client_order_id, isouter=True)
            .group_by(o.c.sleeve, t.c.symbol, t.c.side)
        )
        grouped: dict[tuple[str | None, str], dict[str, Any]] = {}
        for r in _rows(self._engine, stmt):
            key = (r["sleeve"], r["symbol"])
            row = grouped.setdefault(
                key,
                {
                    "sleeve": r["sleeve"],
                    "symbol": r["symbol"],
                    "strategy_id": None,
                    "bought": 0.0,
                    "sold": 0.0,
                    "fees": 0.0,
                    "trades": 0,
                },
            )
            side_key = "bought" if r["side"] == "buy" else "sold"
            row[side_key] += float(r["notional"] or 0.0)
            row["fees"] += float(r["fees"] or 0.0)
            row["trades"] += int(r["trades"])
        # Strategy tags live on lots (doc 07 §7).
        strategies: dict[str, str] = {}
        for p in _rows(self._engine, select(schema.positions.c.symbol, schema.positions.c.lots)):
            for lot in p["lots"] or []:
                if isinstance(lot, dict) and lot.get("strategy_id"):
                    strategies.setdefault(p["symbol"], str(lot["strategy_id"]))
        out = []
        for row in grouped.values():
            row["strategy_id"] = strategies.get(row["symbol"])
            row["net_cash_flow"] = row["sold"] - row["bought"] - row["fees"]
            out.append(row)
        return sorted(out, key=lambda r: (r["sleeve"] or "", r["symbol"]))


class SqlRiskStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def latest_risk_verdict(self) -> dict[str, Any] | None:
        t = schema.risk_verdicts
        rows = _rows(self._engine, select(t).order_by(t.c.as_of.desc()).limit(1))
        return rows[0] if rows else None

    def ladder_history(self, limit: int) -> list[dict[str, Any]]:
        t = schema.ladder_states
        return _rows(self._engine, select(t).order_by(t.c.as_of.desc()).limit(limit))

    def latest_regime(self) -> dict[str, Any] | None:
        t = schema.regime_states
        rows = _rows(self._engine, select(t).order_by(t.c.as_of.desc()).limit(1))
        return rows[0] if rows else None


class SqlJournalStore:
    def __init__(self, engine: Engine) -> None:
        self._repo = JournalRepository(engine)

    def search(
        self, q: str | None, kind: str | None, symbol: str | None, limit: int
    ) -> list[JournalEntry]:
        results = self._repo.search(
            text_query=q,
            kinds=(JournalKind(kind),) if kind else (),
            symbols=(symbol,) if symbol else (),
            top_k=limit,
        )
        return [entry for entry, _score in results]


class SqlOpsStore:
    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._engine = engine
        self._ledger = SqlAlchemyLedgerWriter(engine)
        self._settings = settings

    def recent_batch_jobs(self, limit: int) -> list[dict[str, Any]]:
        t = schema.batch_jobs
        return _rows(self._engine, select(t).order_by(t.c.submitted_at.desc()).limit(limit))

    def cost_month_to_date(self) -> float:
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self._ledger.month_to_date_usd(month_start)

    def monthly_budget(self) -> float:
        return self._settings.claude_monthly_budget_usd

    def open_quarantine(self) -> list[dict[str, Any]]:
        t = schema.quarantine
        return _rows(
            self._engine,
            select(t).where(t.c.resolved_at.is_(None)).order_by(t.c.created_at.desc()),
        )

    def alerts(self, limit: int) -> list[dict[str, Any]]:
        t = schema.alerts
        return _rows(self._engine, select(t).order_by(t.c.created_at.desc()).limit(limit))

    def unacked_alerts(self) -> list[dict[str, Any]]:
        t = schema.alerts
        return _rows(
            self._engine,
            select(t).where(t.c.acked_at.is_(None)).order_by(t.c.created_at.desc()),
        )

    def unacked_universe_recommendations(self) -> list[dict[str, Any]]:
        t = schema.alerts
        return _rows(
            self._engine,
            select(t)
            .where(t.c.name == "universe_change_recommendation", t.c.acked_at.is_(None))
            .order_by(t.c.created_at),
        )

    def open_improvements(self) -> list[dict[str, Any]]:
        return ProposalRepository(self._engine).open_proposals()


class SqlAdminStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._universe = UniverseManager(MarketDataRepository(engine))
        self._proposals = ProposalRepository(engine)

    def audit(self, action: str, payload: dict[str, Any]) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                schema.admin_actions.insert().values(
                    ts=datetime.now(UTC), action=action, payload=payload
                )
            )

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        t = schema.alerts
        rows = _rows(self._engine, select(t).where(t.c.id == alert_id))
        return rows[0] if rows else None

    def ack_alert(self, alert_id: int, now: datetime) -> None:
        t = schema.alerts
        with self._engine.begin() as cx:
            cx.execute(update(t).where(t.c.id == alert_id).values(acked_at=now))

    def approve_universe_change(
        self, symbol: str, change: str, reason: str, approved_by: str, as_of: date
    ) -> None:
        if change == "add":
            self._universe.approve_core_addition(
                symbol, as_of=as_of, reason=reason, approved_by=approved_by
            )
        else:  # demote
            self._universe.approve_core_removal(
                symbol, as_of=as_of, reason=reason, approved_by=approved_by
            )

    def get_target_portfolio(self, tp_id: str) -> dict[str, Any] | None:
        t = schema.target_portfolios
        rows = _rows(self._engine, select(t).where(t.c.id == tp_id))
        return rows[0] if rows else None

    def approve_target_portfolio(self, tp_id: str, approved_by: str, now: datetime) -> None:
        t = schema.target_portfolios
        with self._engine.begin() as cx:
            cx.execute(
                update(t).where(t.c.id == tp_id).values(approved_at=now, approved_by=approved_by)
            )

    def reject_target_portfolio(self, tp_id: str, note: str, now: datetime) -> None:
        current = self.get_target_portfolio(tp_id)
        if current is None:
            return
        payload = dict(current["payload"] or {})
        payload["rejected"] = {"note": note, "at": now.isoformat()}
        t = schema.target_portfolios
        with self._engine.begin() as cx:
            cx.execute(update(t).where(t.c.id == tp_id).values(payload=payload))

    def get_improvement(self, proposal_id: str) -> dict[str, Any] | None:
        t = schema.improvement_proposals
        rows = _rows(self._engine, select(t).where(t.c.id == proposal_id))
        return rows[0] if rows else None

    def set_improvement_status(self, proposal_id: str, status: str) -> None:
        self._proposals.set_status(proposal_id, ProposalStatus(status))


def make_ping(engine: Engine):
    def ping() -> bool:
        try:
            with engine.connect() as cx:
                cx.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    return ping
