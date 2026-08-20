"""Risk-side persistence: ladder transition events and scenario results.

Two mechanisms doc 06 counts as mitigations that previously had no storage:

- ``ladder_events`` (doc 06 §3) — the transition log, and the *blocking*
  state for ack-requiring rungs. An unacked ack-requiring row halts new
  orders, exactly like an unacked divergent reconciliation run (doc 09 §5).
- ``risk_scenarios`` (doc 06 §6) — weekly Monte Carlo + deterministic stress.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import Any

from sqlalchemy import Engine, and_, select, update
from sqlalchemy.dialects.postgresql import insert

from dynamo.persistence import schema


def _now() -> datetime:
    return datetime.now(UTC)


class LadderEventRepository:
    """Transition log + the ack that clears a ladder halt."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(
        self,
        *,
        as_of: date,
        from_rung: str,
        to_rung: str,
        drawdown: float,
        requires_ack: bool,
        kill_switch_engaged: bool,
        created_at: datetime | None = None,
    ) -> int:
        stmt = (
            schema.ladder_events.insert()
            .values(
                created_at=created_at or _now(),
                as_of=as_of,
                from_rung=from_rung,
                to_rung=to_rung,
                drawdown=drawdown,
                requires_ack=requires_ack,
                kill_switch_engaged=kill_switch_engaged,
            )
            .returning(schema.ladder_events.c.id)
        )
        with self._engine.begin() as conn:
            return int(conn.execute(stmt).scalar_one())

    def unacked_blocking(self) -> list[dict[str, Any]]:
        """Ack-requiring transitions nobody has cleared. Non-empty => halted."""
        t = schema.ladder_events
        stmt = (
            select(t)
            .where(and_(t.c.requires_ack.is_(True), t.c.acked_at.is_(None)))
            .order_by(t.c.id)
        )
        with self._engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def is_halted(self) -> bool:
        return bool(self.unacked_blocking())

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        t = schema.ladder_events
        stmt = select(t).order_by(t.c.id.desc()).limit(max(limit, 1))
        with self._engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def acknowledge(
        self, event_id: int, acked_by: str, note: str, now: datetime | None = None
    ) -> bool:
        """Clear one ladder halt. Returns False when nothing was pending.

        The note is mandatory (validated by
        ``dynamo.risk.ladder_response.ack_note_required`` at the call site):
        a ladder halt is cleared by a human who looked, or not at all.
        """
        t = schema.ladder_events
        stmt = (
            update(t)
            .where(and_(t.c.id == event_id, t.c.acked_at.is_(None)))
            .values(acked_by=acked_by, acked_at=now or _now(), ack_note=note)
        )
        with self._engine.begin() as conn:
            return int(conn.execute(stmt).rowcount or 0) > 0


class VarBacktestReader:
    """``(predicted_var95, realized_return)`` pairs for the Kupiec test.

    Nothing counted VaR exceptions before this class: ``var_exceptions_250d``
    defaulted to 0, neither production ``RiskEngineInputs`` passed it, and
    ``kupiec_traffic_light(0)`` is always GREEN — so doc 06 §6's "too many
    exceptions -> caps tighten 25%" was unreachable code.

    The pairing is the point: the VaR stored on session *t* is a forecast of
    the loss on session *t+1*, so it is compared against the equity-curve
    return from *t* to *t+1*. Comparing same-day VaR to same-day return would
    grade the model on data it already saw.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _var_by_date(self, since: date) -> dict[date, float]:
        t = schema.risk_verdicts
        stmt = select(t.c.as_of, t.c.payload).where(t.c.as_of >= datetime(
            since.year, since.month, since.day, tzinfo=UTC
        ))
        out: dict[date, float] = {}
        with self._engine.connect() as conn:
            for row in conn.execute(stmt):
                payload = row.payload or {}
                metrics = payload.get("metrics") or {}
                value = metrics.get("var95_1d")
                if value is None:
                    continue
                as_of = row.as_of
                day = as_of.date() if isinstance(as_of, datetime) else as_of
                # One verdict per session; the last write for a day wins.
                out[day] = float(value)
        return out

    def _equity_returns(self, since: date) -> list[tuple[date, float]]:
        t = schema.equity_curve
        stmt = (
            select(t.c.as_of, t.c.equity).where(t.c.as_of >= since).order_by(t.c.as_of)
        )
        with self._engine.connect() as conn:
            rows = [(r.as_of, float(r.equity)) for r in conn.execute(stmt)]
        returns: list[tuple[date, float]] = []
        for (prev_day, prev_equity), (_day, equity) in pairwise(rows):
            if prev_equity > 0.0:
                # Keyed by the PREVIOUS session: this is the return the VaR
                # stored on that session was forecasting.
                returns.append((prev_day, equity / prev_equity - 1.0))
        return returns

    def observations(self, *, window: int = 250, today: date | None = None
                     ) -> list[tuple[float, float]]:
        from datetime import timedelta

        anchor = today or datetime.now(UTC).date()
        # Calendar slack: 250 sessions is ~365 calendar days; take 1.6x so a
        # gap in the curve cannot silently shorten the window.
        since = anchor - timedelta(days=int(window * 1.6) + 30)
        var_by_date = self._var_by_date(since)
        pairs: list[tuple[float, float]] = []
        for day, realized in self._equity_returns(since):
            predicted = var_by_date.get(day)
            if predicted is not None:
                pairs.append((predicted, realized))
        return pairs[-window:]


class RiskScenarioRepository:
    """Weekly stress + Monte Carlo results (doc 06 §6)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [{"created_at": _now(), **dict(r)} for r in rows]
        stmt = insert(schema.risk_scenarios).values(payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=["as_of", "kind", "scenario"],
            set_={
                "created_at": stmt.excluded.created_at,
                "equity_impact_pct": stmt.excluded.equity_impact_pct,
                "ladder_implication": stmt.excluded.ladder_implication,
                "details": stmt.excluded.details,
            },
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
        return len(payload)

    def latest_as_of(self) -> date | None:
        t = schema.risk_scenarios
        with self._engine.connect() as conn:
            return conn.execute(select(t.c.as_of).order_by(t.c.as_of.desc()).limit(1)).scalar()

    def latest(self) -> list[dict[str, Any]]:
        as_of = self.latest_as_of()
        if as_of is None:
            return []
        t = schema.risk_scenarios
        stmt = select(t).where(t.c.as_of == as_of).order_by(t.c.kind, t.c.scenario)
        with self._engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]
