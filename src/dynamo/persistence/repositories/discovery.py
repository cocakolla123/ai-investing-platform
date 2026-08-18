"""Hypothesis persistence for the alpha-research rails (ADR-0015).

Thin repository over the ``hypotheses`` table (migration 0018). Insert-once
rows: ``registered_at``/``rationale``/``prediction`` are immutable after
registration (the pre-commitment must not be editable after seeing results);
only ``status`` and ``audit_results`` change, via :meth:`record_audit` /
:meth:`set_status`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, select, update

from dynamo.persistence.schema import hypotheses
from dynamo.research.discovery.registry import Hypothesis, HypothesisStatus

__all__ = ["HypothesisRepository", "hypothesis_from_row", "hypothesis_to_row"]


def hypothesis_to_row(h: Hypothesis) -> dict[str, Any]:
    return {
        "id": h.id,
        "title": h.title,
        "rationale": h.rationale,
        "prediction": h.prediction,
        "trial_family": h.trial_family,
        "registered_at": h.registered_at,
        "data_requirements": h.data_requirements,
        "status": h.status.value,
        "audit_results": h.audit_results,
    }


def hypothesis_from_row(row: Any) -> Hypothesis:
    """Build a :class:`Hypothesis` from a row/mapping with attribute access
    or dict access (SQLAlchemy Row or plain dict — tests use dicts)."""

    def get(key: str) -> Any:
        return row.get(key) if isinstance(row, dict) else getattr(row, key)

    return Hypothesis(
        id=get("id"),
        title=get("title"),
        rationale=get("rationale"),
        prediction=get("prediction"),
        trial_family=get("trial_family"),
        registered_at=get("registered_at"),
        data_requirements=get("data_requirements") or "",
        status=HypothesisStatus(get("status")),
        audit_results=get("audit_results"),
    )


class HypothesisRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, h: Hypothesis) -> None:
        """Plain INSERT — a duplicate id raises (ids are immutable slugs;
        re-registering an existing hypothesis is a refusal, not an upsert)."""
        with self._engine.begin() as cx:
            cx.execute(hypotheses.insert().values(**hypothesis_to_row(h)))

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        with self._engine.connect() as cx:
            r = cx.execute(select(hypotheses).where(hypotheses.c.id == hypothesis_id)).first()
        return hypothesis_from_row(r) if r is not None else None

    def list_all(self) -> list[Hypothesis]:
        with self._engine.connect() as cx:
            rows = cx.execute(select(hypotheses).order_by(hypotheses.c.registered_at)).all()
        return [hypothesis_from_row(r) for r in rows]

    def record_audit(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        audit_results: dict[str, Any],
    ) -> None:
        """Write the audition outcome onto the row (status + evidence)."""
        with self._engine.begin() as cx:
            cx.execute(
                update(hypotheses)
                .where(hypotheses.c.id == hypothesis_id)
                .values(status=status.value, audit_results=audit_results)
            )

    def set_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(hypotheses)
                .where(hypotheses.c.id == hypothesis_id)
                .values(status=status.value)
            )
