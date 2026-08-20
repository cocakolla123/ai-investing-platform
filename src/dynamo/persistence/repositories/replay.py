"""Replay bookkeeping repository (plan Phase 4; committee mode Phase 5).

Owns ``replay_runs``, ``replay_decisions``, ``replay_verdicts``,
``replay_equity`` and ``replay_fills`` — the only tables (besides
``decision_experience`` rows with ``source='replay'``) a replay run is
allowed to write. Everything is idempotent-upsert on
deterministic ids so a re-run of the same configuration re-derives the same
rows.

``GenerationsRepository`` (plan Phase 5) owns ``evaluation_claims`` — the
durable freeze guard replacing the in-memory ``EvaluationRegistry`` for the
generational trainer — and ``generation_results``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.persistence.schema import (
    evaluation_claims,
    generation_results,
    replay_contexts,
    replay_decisions,
    replay_equity,
    replay_fills,
    replay_flags,
    replay_runs,
    replay_verdicts,
)


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
            cx.execute(update(replay_runs).where(replay_runs.c.id == run_id).values(status=status))

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

    # -- equity curve + fills (backtest-campaign Phase 5) ------------------

    def insert_equity(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert on the (run_id, as_of) PK — a re-run of the same
        deterministic config updates in place, never duplicates."""
        if not rows:
            return 0
        stmt = pg_insert(replay_equity).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_equity.columns
            if c.name not in ("run_id", "as_of") and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_equity.c.run_id, replay_equity.c.as_of],
            set_=updatable,
        ).returning(replay_equity.c.as_of)
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def equity_for(self, run_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(replay_equity)
            .where(replay_equity.c.run_id == run_id)
            .order_by(replay_equity.c.as_of)
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]

    def insert_fills(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert on the (run_id, session, symbol) PK."""
        if not rows:
            return 0
        stmt = pg_insert(replay_fills).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_fills.columns
            if c.name not in ("run_id", "session", "symbol") and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                replay_fills.c.run_id,
                replay_fills.c.session,
                replay_fills.c.symbol,
            ],
            set_=updatable,
        ).returning(replay_fills.c.symbol)
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def fills_for(self, run_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(replay_fills)
            .where(replay_fills.c.run_id == run_id)
            .order_by(replay_fills.c.session, replay_fills.c.symbol)
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]

    # -- discovery flags (wide-campaign discovery layer) -------------------

    def insert_flags(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert into ``replay_flags`` — NEVER
        ``surveillance_flags`` (the discovery firewall).

        ``surveillance_flags`` is a live table feeding the nightly triage
        queue, and live triage promotes what it finds into
        ``universe_membership``, which is what
        ``HistoricalDataGateway.universe()`` reads. A replay write there
        would contaminate production AND every future replay. This method is
        the only write path replay discovery has.
        """
        if not rows:
            return 0
        stmt = pg_insert(replay_flags).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_flags.columns
            if c.name != "id" and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_flags.c.id], set_=updatable
        ).returning(replay_flags.c.id)
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def flags_for(
        self, run_id: str, as_of: date | None = None, *, promoted_only: bool = False
    ) -> list[dict[str, Any]]:
        stmt = select(replay_flags).where(replay_flags.c.run_id == run_id)
        if as_of is not None:
            stmt = stmt.where(replay_flags.c.as_of == as_of)
        if promoted_only:
            stmt = stmt.where(replay_flags.c.promoted.is_(True))
        stmt = stmt.order_by(replay_flags.c.as_of, replay_flags.c.symbol, replay_flags.c.detector)
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]

    # -- verdicts (committee mode, plan Phase 5) ---------------------------

    def insert_verdicts(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert into ``replay_verdicts`` — NEVER
        ``committee_verdicts`` (the firewall extension)."""
        if not rows:
            return 0
        stmt = pg_insert(replay_verdicts).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_verdicts.columns
            if c.name != "id" and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_verdicts.c.id], set_=updatable
        ).returning(replay_verdicts.c.id)
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def verdicts_for(self, run_id: str) -> list[dict[str, Any]]:
        stmt = (
            select(replay_verdicts)
            .where(replay_verdicts.c.run_id == run_id)
            .order_by(replay_verdicts.c.as_of, replay_verdicts.c.symbol)
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]

    # -- decision-time contexts (backtest-campaign Phase 4) ----------------

    def insert_contexts(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Idempotent upsert into ``replay_contexts`` on the deterministic
        run-namespaced request id, so a resumed run re-derives the same rows."""
        if not rows:
            return 0
        stmt = pg_insert(replay_contexts).values([dict(r) for r in rows])
        updatable = {
            c.name: stmt.excluded[c.name]
            for c in replay_contexts.columns
            if c.name != "id" and c.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[replay_contexts.c.id], set_=updatable
        ).returning(replay_contexts.c.id)
        with self._engine.begin() as cx:
            written = cx.execute(stmt).all()
        return len(written)

    def contexts_for(self, run_id: str, as_of: date | None = None) -> list[dict[str, Any]]:
        stmt = select(replay_contexts).where(replay_contexts.c.run_id == run_id)
        if as_of is not None:
            stmt = stmt.where(replay_contexts.c.as_of == as_of)
        stmt = stmt.order_by(
            replay_contexts.c.as_of, replay_contexts.c.symbol, replay_contexts.c.stage
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]


class GenerationsRepository:
    """``evaluation_claims`` + ``generation_results`` (plan Phase 5).

    ``claim_period`` is the durable freeze guard: the FIRST claim of a
    (period_start, period_end, purpose) wins; every later attempt returns the
    existing claim so the caller can refuse — mirroring
    ``EvaluationRegistry.claim`` but surviving process restarts.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # -- evaluation claims -------------------------------------------------

    def is_claimed(
        self, period_start: date, period_end: date, purpose: str
    ) -> dict[str, Any] | None:
        """The existing claim row for (period, purpose), or ``None``."""
        stmt = select(evaluation_claims).where(
            evaluation_claims.c.period_start == period_start,
            evaluation_claims.c.period_end == period_end,
            evaluation_claims.c.purpose == purpose,
        )
        with self._engine.connect() as cx:
            row = cx.execute(stmt).first()
        return dict(row._mapping) if row is not None else None

    def claim_period(
        self,
        *,
        period_start: date,
        period_end: date,
        purpose: str,
        config_hash: str = "",
        claimed_at: datetime | None = None,
    ) -> bool:
        """True iff THIS call claimed the period (first claim wins).

        ON CONFLICT DO NOTHING on the (period_start, period_end, purpose)
        unique constraint — concurrent claimers race safely; exactly one wins.
        """
        row = {
            "id": f"claim:{purpose}:{period_start.isoformat()}:{period_end.isoformat()}",
            "period_start": period_start,
            "period_end": period_end,
            "purpose": purpose,
            "config_hash": config_hash,
            "claimed_at": claimed_at or _now(),
        }
        stmt = (
            pg_insert(evaluation_claims)
            .values(**row)
            .on_conflict_do_nothing(
                index_elements=[
                    evaluation_claims.c.period_start,
                    evaluation_claims.c.period_end,
                    evaluation_claims.c.purpose,
                ]
            )
            .returning(evaluation_claims.c.id)
        )
        with self._engine.begin() as cx:
            inserted = cx.execute(stmt).first()
        return inserted is not None

    # -- generation results ------------------------------------------------

    def insert_generation(self, row: Mapping[str, Any]) -> None:
        """Idempotent upsert on the deterministic generation-result id."""
        payload = dict(row)
        stmt = pg_insert(generation_results).values(**payload)
        stmt = stmt.on_conflict_do_update(
            index_elements=[generation_results.c.id],
            set_={k: stmt.excluded[k] for k in payload if k != "id"},
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)

    def list_generations(self, limit: int = 100) -> list[dict[str, Any]]:
        stmt = (
            select(generation_results)
            .order_by(generation_results.c.created_at.desc(), generation_results.c.generation)
            .limit(limit)
        )
        with self._engine.connect() as cx:
            return [dict(r._mapping) for r in cx.execute(stmt)]
