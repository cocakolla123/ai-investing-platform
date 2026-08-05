"""Daily activity reports table.

Adds ``daily_reports`` (rendered markdown + DailyReport JSON per day).
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0004
Revises: 0003
"""

from alembic import op

from dynamo.persistence.schema import daily_reports, metadata

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[daily_reports], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[daily_reports], checkfirst=True)
