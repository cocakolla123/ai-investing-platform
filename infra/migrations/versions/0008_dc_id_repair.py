"""Repair decision-context references stored as row_hash (Phase 0, data-only).

``DecisionContextRepository.append()`` used to return the chain ``row_hash``
while every call site stored the return value as
``<table>.decision_context_id``; ``get()`` looks rows up by
``decision_context.id``, so the decision→outcome join never matched and
post-trade reviews always hit their "no decision context" skip path. The
code now returns ``dc.id``; this migration rewrites the historical
hash-valued references to the matching row's id.

Orphans (references that match neither an id nor a row_hash) are reported
and left alone — never fatal. Downgrade is a no-op: the repair is one-way
(re-breaking the join on purpose helps nobody).

Revision ID: 0008
Revises: 0007
"""

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

# Every table that stores a decision-context reference written by the buggy
# call sites. post_trade_reviews is absent on purpose: reviews were only ever
# written with ``dc.id`` (and in practice never written at all pre-repair).
REPAIR_TABLES = ("orders", "pm_orders", "options_orders")


def repair_references(bind: Any, table: str) -> tuple[int, int]:
    """Rewrite hash-valued references in ``table`` to decision_context.id.

    Returns (repaired_count, unmatched_orphan_count). Exposed for the
    migration-logic test; ``upgrade()`` is a thin loop over this.
    """
    repaired = bind.execute(
        sa.text(
            f"UPDATE {table} SET decision_context_id = dc.id "
            "FROM decision_context dc "
            f"WHERE {table}.decision_context_id = dc.row_hash"
        )
    ).rowcount
    unmatched = bind.execute(
        sa.text(
            f"SELECT count(*) FROM {table} t "
            "WHERE t.decision_context_id <> '' "
            "AND NOT EXISTS (SELECT 1 FROM decision_context dc "
            "WHERE dc.id = t.decision_context_id)"
        )
    ).scalar()
    return int(repaired or 0), int(unmatched or 0)


def upgrade() -> None:
    bind = op.get_bind()
    for table in REPAIR_TABLES:
        repaired, unmatched = repair_references(bind, table)
        print(
            f"0008 dc_id repair: {table}: repaired={repaired} "
            f"unmatched_orphans={unmatched}"
        )


def downgrade() -> None:
    # One-way data repair: the original hash references still exist as
    # decision_context.row_hash, but restoring them would re-break the join.
    pass
