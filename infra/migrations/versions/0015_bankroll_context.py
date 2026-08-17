"""Bankroll context on decision_experience (backtest-campaign Phase 3).

Adds ``equity_at_decision`` (real NULL) + ``position_notional`` (real NULL):
the portfolio equity at the opening decision and the realized position
notional (avg entry price x qty), populated by the live experience-resolution
builders and the replay runner so the meta-model can learn scale-dependent
trust (``log10_equity_at_decision`` feature). NULL on legacy rows — every
consumer is NaN-safe by contract.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these columns — the inspector guard makes
this upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0015
Revises: 0014
"""

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

TABLE = "decision_experience"
COLUMN_TYPES: dict[str, type[sa.types.TypeEngine]] = {
    "equity_at_decision": sa.Float,
    "position_notional": sa.Float,
}


def _existing(bind: sa.engine.Connection) -> set[str] | None:
    """Column names, or ``None`` when the table itself is absent (torn-down
    scratch DBs only — 0009 guarantees it on any real chain)."""
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
            op.add_column(TABLE, sa.Column(name, type_(), nullable=True))


def downgrade() -> None:
    existing = _existing(op.get_bind())
    if existing is None:
        return
    for name in COLUMN_TYPES:
        if name in existing:
            op.drop_column(TABLE, name)
