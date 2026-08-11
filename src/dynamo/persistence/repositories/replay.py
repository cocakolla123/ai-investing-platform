"""Replay bookkeeping repository (plan Phase 4).

Owns ``replay_runs`` and ``replay_decisions`` — the only tables (besides
``decision_experience`` rows with ``source='replay'``) a replay run is
allowed to write. Everything is idempotent-upsert on deterministic ids so a
re-run of the same configuration re-derives the same rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.persistence.schema import replay_decisions, replay_runs


def _now() -> datetime:
    return datetime.now(UTC)


class ReplayRepository:
    """CRUD over replay_runs / replay_decisions (composition-layer only)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # -- runs --------------------------------------------------------------

    def create_run(
        self,
        *,
        run_id: str,
        label: str,
        mode: str,
        start_as_of: date,
        end_as_of: date,
        config: Mapping[str, Any],
        code_version: str = "",
        budget_usd: float | None = None,
        status: str = "created",
        created_at: datetime | None = None,
    ) -> None:
        """Idempotent: re-running the same deterministic run id updates the
        bookkeeping columns (status/config) rather than failing."""
        row = {
            "id": run_id,
            "created_at": created_at or _now(),
            "label": label,
            "mode": mode,
            "start_as_of": start_as_of,
            "end_as_of": end_as_of,
            "config_json": dict(config),
            "code_version": code_version,
            "budget_usd": budget_usd,
            "spent_usd": 0.0,
            "status": status,
        }
        stmt = pg_insert(replay_runs).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_runs.c.id],
            set_={
                "label": stmt.excluded.label,
                "mode": stmt.excluded.mode,
                "start_as_of": stmt.excluded.start_as_of,
                "end_as_of": stmt.excluded.end_as_of,
                "config_json": stmt.excluded.config_json,
                "code_version": stmt.excluded.code_version,
                "budget_usd": stmt.excluded.budget_usd,
                "status": stmt.excluded.status,
            },
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)

    def update_status(self, run_id: str, status: str) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(replay_runs).where(replay_runs.c.id == run_id).values(status=status)
            )

    def record_spend(self, run_id: str, delta_usd: float) -> None:
        """Accumulate run-local spend (Phase 5 committee budget accounting;
        quant runs never call this)."""
        with self._engine.begin() as cx:
            cx.execute(
                update(replay_runs)
                .where(replay_runs.c.id == run_id)
                .values(spent_usd=replay_runs.c.spent_usd + float(delta_usd))
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._engine.connect() as cx:
            row = cx.execute(select(replay_runs).where(replay_runs.c.id == run_id)).first()
        return dict(row._mapping) if row is not None else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(replay_runs).order_by(replay_runs.c.created_at.desc()).limit(limit)
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]

    # -- decisions ---------------------------------------------------------

    def insert_decisions(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert on the deterministic decision id."""
        if not rows:
            return 0
        stmt = pg_insert(replay_decisions).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_decisions.columns
            if c.name != "id" and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_decisions.c.id], set_=updatable
        ).returning(replay_decisions.c.id)
        # RETURNING gives a reliable row count (driver rowcount can be -1 for
        # multi-VALUES upserts).
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def decisions_for(self, run_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(replay_decisions)
            .where(replay_decisions.c.run_id == run_id)
            .order_by(replay_decisions.c.as_of, replay_decisions.c.symbol)
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]
