"""Advisory calibrated probability on decision_experience (plan Phase 2).

Adds ``calibrated_probability`` (real NULL) + ``calibration_scope_used``
(text NULL): the hierarchical calibrator's output per resolved live
experience, logged by ``calibration_update``. ADVISORY ONLY — nothing
consumes ``calibrated_probability`` for sizing.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these columns — the inspector guard makes
this upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0010
Revises: 0009
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

TABLE = "decision_experience"
COLUMN_TYPES: dict[str, type[sa.types.TypeEngine]] = {
    "calibrated_probability": sa.Float,
    "calibration_scope_used": sa.Text,
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
