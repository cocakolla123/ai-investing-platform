"""SQL implementation of the ``ReportSource`` Protocol.

All the report SQL lives here (mirroring ``dynamo.api.queries``): the builder
in ``dynamo.reporting.daily`` never sees the engine, so tests fake the source.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import Engine, func, select

from dynamo.persistence import schema
from dynamo.reporting.daily import DailyReport, Row, build_daily_report

_FINISHED_BATCH_STATES = ("submitted", "resolved")


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def _day_end(day: date) -> datetime:
    return _day_start(day + timedelta(days=1))


class SqlReportSource:
    """Reads one day's activity straight from the tables schema.py owns."""

    def __init__(self, engine: Engine, monthly_budget_usd: float) -> None:
        self._engine = engine
        self._budget = monthly_budget_usd

    def _rows(self, stmt: Any) -> list[Row]:
        with self._engine.connect() as cx:
            return [dict(m) for m in cx.execute(stmt).mappings().all()]

    def _scalar(self, stmt: Any) -> Any:
        with self._engine.connect() as cx:
            return cx.execute(stmt).scalar()

    # --- at a glance ---

    def equity_points(self, day: date) -> list[Row]:
        t = schema.equity_curve
        return self._rows(
            select(t.c.as_of, t.c.equity, t.c.cash)
            .where(t.c.as_of <= day)
            .order_by(t.c.as_of.desc())
            .limit(2)
        )

    def open_position_counts(self) -> dict[str, int]:
        eq, op, pm = schema.positions, schema.options_positions, schema.pm_positions
        return {
            "equities": int(
                self._scalar(select(func.count()).select_from(eq).where(eq.c.qty != 0)) or 0
            ),
            "options": int(
                self._scalar(select(func.count()).select_from(op).where(op.c.contracts != 0))
                or 0
            ),
            "kalshi": int(
                self._scalar(select(func.count()).select_from(pm).where(pm.c.contracts != 0))
                or 0
            ),
        }

    def regime_on(self, day: date) -> Row | None:
        t = schema.regime_states
        rows = self._rows(
            select(t.c.as_of, t.c.regime)
            .where(t.c.as_of <= day)
            .order_by(t.c.as_of.desc())
            .limit(1)
        )
        return rows[0] if rows else None

    def ladder_on(self, day: date) -> Row | None:
        t = schema.ladder_states
        rows = self._rows(
            select(t.c.as_of, t.c.rung, t.c.drawdown)
            .where(t.c.as_of <= day)
            .order_by(t.c.as_of.desc())
            .limit(1)
        )
        return rows[0] if rows else None

    # --- decisions & watchlist ---

    def verdicts_on(self, day: date) -> list[Row]:
        t = schema.committee_verdicts
        return self._rows(
            select(
                t.c.symbol, t.c.action, t.c.conviction, t.c.disagreement_score, t.c.payload
            )
            .where(t.c.as_of == day)
            .order_by(t.c.symbol)
        )

    def watchlist_on(self, day: date) -> list[Row]:
        t = schema.watchlist
        return self._rows(
            select(t.c.sleeve, t.c.symbol, t.c.rank, t.c.blend_score, t.c.badges)
            .where(t.c.as_of == day)
            .order_by(t.c.sleeve, t.c.rank)
        )

    def watchlist_prior(self, day: date) -> list[Row]:
        t = schema.watchlist
        prior = self._scalar(select(func.max(t.c.as_of)).where(t.c.as_of < day))
        if prior is None:
            return []
        return self._rows(
            select(t.c.sleeve, t.c.symbol, t.c.rank, t.c.blend_score, t.c.badges)
            .where(t.c.as_of == prior)
            .order_by(t.c.sleeve, t.c.rank)
        )

    # --- trades ---

    def equity_trades_on(self, day: date) -> list[Row]:
        t, o = schema.trades, schema.orders
        return self._rows(
            select(
                t.c.filled_at,
                t.c.symbol,
                t.c.side,
                t.c.qty,
                t.c.fill_price,
                t.c.slippage_bps,
                o.c.sleeve,
            )
            .join_from(t, o, t.c.client_order_id == o.c.client_order_id, isouter=True)
            .where(t.c.filled_at >= _day_start(day), t.c.filled_at < _day_end(day))
            .order_by(t.c.filled_at, t.c.symbol)
        )

    def options_trades_on(self, day: date) -> list[Row]:
        t = schema.options_trades
        return self._rows(
            select(t.c.filled_at, t.c.occ_symbol, t.c.underlying, t.c.side, t.c.contracts,
                   t.c.price)
            .where(t.c.filled_at >= _day_start(day), t.c.filled_at < _day_end(day))
            .order_by(t.c.filled_at, t.c.occ_symbol)
        )

    def pm_trades_on(self, day: date) -> list[Row]:
        t = schema.pm_trades
        return self._rows(
            select(t.c.filled_at, t.c.market_id, t.c.side, t.c.contracts, t.c.price_cents)
            .where(t.c.filled_at >= _day_start(day), t.c.filled_at < _day_end(day))
            .order_by(t.c.filled_at, t.c.market_id)
        )

    # --- Claude usage ---

    def cost_entries_on(self, day: date) -> list[Row]:
        t = schema.cost_ledger
        return self._rows(
            select(
                t.c.model_id,
                t.c.call_kind,
                t.c.input_tokens,
                t.c.output_tokens,
                t.c.cache_read_tokens,
                t.c.cost_usd,
                t.c.batch,
            )
            .where(t.c.ts >= _day_start(day), t.c.ts < _day_end(day))
            .order_by(t.c.id)
        )

    def cost_month_to_date(self, day: date) -> float:
        t = schema.cost_ledger
        month_start = _day_start(day.replace(day=1))
        value = self._scalar(
            select(func.coalesce(func.sum(t.c.cost_usd), 0.0)).where(
                t.c.ts >= month_start, t.c.ts < _day_end(day)
            )
        )
        return float(value or 0.0)

    def monthly_budget(self) -> float:
        return self._budget

    # --- surveillance & ops ---

    def flags_on(self, day: date) -> list[Row]:
        t = schema.surveillance_flags
        return self._rows(
            select(t.c.detector, t.c.symbol, t.c.severity, t.c.triage_verdict)
            .where(t.c.ts >= _day_start(day), t.c.ts < _day_end(day))
            .order_by(t.c.detector, t.c.symbol)
        )

    def universe_changes_on(self, day: date) -> list[Row]:
        t = schema.universe_membership
        promoted = self._rows(
            select(t.c.symbol, t.c.tier, t.c.reason)
            .where(t.c.valid_from == day, t.c.tier == 3)
            .order_by(t.c.symbol)
        )
        expired = self._rows(
            select(t.c.symbol, t.c.tier, t.c.reason)
            .where(t.c.valid_to == day, t.c.tier == 3)
            .order_by(t.c.symbol)
        )
        return [{**r, "kind": "promoted"} for r in promoted] + [
            {**r, "kind": "expired"} for r in expired
        ]

    def pending_universe_recommendations(self) -> list[str]:
        t = schema.alerts
        rows = self._rows(
            select(t.c.message)
            .where(t.c.name == "universe_change_recommendation", t.c.acked_at.is_(None))
            .order_by(t.c.created_at)
        )
        return [str(r["message"]) for r in rows]

    def unacked_alerts(self) -> list[Row]:
        t = schema.alerts
        return self._rows(
            select(t.c.severity, t.c.name, t.c.message)
            .where(t.c.acked_at.is_(None), t.c.name != "universe_change_recommendation")
            .order_by(t.c.created_at.desc())
            .limit(20)
        )

    def open_quarantines(self) -> list[Row]:
        t = schema.quarantine
        return self._rows(
            select(t.c.scope, t.c.key, t.c.reason)
            .where(t.c.resolved_at.is_(None))
            .order_by(t.c.created_at.desc())
            .limit(20)
        )

    def batch_failures_on(self, day: date) -> list[Row]:
        t = schema.batch_jobs
        return self._rows(
            select(t.c.batch_id, t.c.stage, t.c.status)
            .where(
                t.c.submitted_at >= _day_start(day),
                t.c.submitted_at < _day_end(day),
                t.c.status.notin_(_FINISHED_BATCH_STATES),
            )
            .order_by(t.c.batch_id)
        )

    def reconciliation_on(self, day: date) -> list[Row]:
        t = schema.reconciliation_runs
        rows = self._rows(
            select(t.c.scope, t.c.result, t.c.divergences)
            .where(t.c.ts >= _day_start(day), t.c.ts < _day_end(day))
            .order_by(t.c.ts)
        )
        latest_per_scope: dict[str, Row] = {}
        for row in rows:  # later rows win: report the day's final status per scope
            latest_per_scope[str(row["scope"])] = row
        return [latest_per_scope[scope] for scope in sorted(latest_per_scope)]

    # --- learning ---

    def reviews_on(self, day: date) -> list[Row]:
        t = schema.post_trade_reviews
        rows = self._rows(
            select(t.c.symbol, t.c.payload).where(t.c.as_of == day).order_by(t.c.trade_id)
        )
        out: list[Row] = []
        for row in rows:
            lessons = (row.get("payload") or {}).get("lessons") or ()
            out.append({"symbol": row["symbol"], "lesson": lessons[0] if lessons else ""})
        return out

    def calibration_on(self, day: date) -> list[Row]:
        t = schema.calibration_points
        return self._rows(
            select(t.c.scope_id, t.c.brier, t.c.n)
            .where(t.c.as_of == day)
            .order_by(t.c.scope_type, t.c.scope_id)
        )

    def proposals_on(self, day: date) -> list[Row]:
        t = schema.improvement_proposals
        return self._rows(
            select(t.c.kind, t.c.target, t.c.expected_effect)
            .where(t.c.created_at >= _day_start(day), t.c.created_at < _day_end(day))
            .order_by(t.c.id)
        )


def build_daily_report_for_engine(
    engine: Engine, day: date, monthly_budget_usd: float
) -> DailyReport:
    """Convenience wrapper: build the report for ``day`` straight from SQL."""
    return build_daily_report(SqlReportSource(engine, monthly_budget_usd), day)
