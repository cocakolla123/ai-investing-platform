"""Per-call tracing on cost_ledger (eval Phase 4).

Adds nullable ``role``, ``request_id``, ``prompt_hash``, ``retrieved_ids``
(JSONB), ``latency_ms``, ``attempt``, ``validation_ok`` and
``equiv_cost_usd``: which committee role made the call, the batch request it
served, the system-prompt version, the journal entries retrieved into its
context, wall-clock latency, which attempt it was, whether its output passed
the client's schema check, and the API-equivalent cost the Claude Code CLI
reports for flat-rate calls. NULL on older rows and on writers that don't
trace.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these columns — the inspector guard makes
this upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0023
Revises: 0022
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

TABLE = "cost_ledger"
COLUMN_TYPES: dict[str, sa.types.TypeEngine] = {
    "role": sa.Text(),
    "request_id": sa.Text(),
    "prompt_hash": sa.Text(),
    "retrieved_ids": JSONB(),
    "latency_ms": sa.Float(),
    "attempt": sa.Integer(),
    "validation_ok": sa.Boolean(),
    "equiv_cost_usd": sa.Float(),
}


def _existing(bind: sa.engine.Connection) -> set[str] | None:
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        return None
    return {c["name"] for c in inspector.get_columns(TABLE)}


def upgrade() -> None:
    existing = _existing(op.get_bind())
    if existing is None:
        return
    for name, type_ in COLUMN_TYPES.items():
        if name not in existing:
            op.add_column(TABLE, sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    existing = _existing(op.get_bind())
    if existing is None:
        return
    for name in COLUMN_TYPES:
        if name in existing:
            op.drop_column(TABLE, name)
