"""FINRA consolidated short interest (bi-monthly, free, no API key).

Zero short-interest data existed anywhere in the platform before this. FINRA
publishes every cycle through its open Query API, so the whole family —
short interest as a share of float, days to cover, and the CHANGE in short
interest — becomes computable at no cost.

POINT-IN-TIME: the table carries BOTH ``settlement_date`` (what the figures
describe) and ``publication_date`` (when FINRA disseminated them, ~8 days
later), and every read bounds on the latter. Filtering on the settlement date
would hand a backtest a week and a half of the future, which is precisely the
kind of leak the gateway's tripwire exists to catch — hence the index on
``publication_date``, the column the point-in-time query actually uses.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — the inspector guard makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0020
Revises: 0019
"""

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

TABLE = "short_interest"
INDEX = "ix_short_interest_publication_date"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("symbol", sa.Text, primary_key=True),
            sa.Column("settlement_date", sa.Date, primary_key=True),
            sa.Column("publication_date", sa.Date, nullable=False),
            sa.Column("shares_short", sa.Float, nullable=False),
            sa.Column("previous_shares_short", sa.Float),
            sa.Column("avg_daily_volume", sa.Float),
            sa.Column("days_to_cover", sa.Float),
            sa.Column("change_pct", sa.Float),
            sa.Column("market_class", sa.Text),
            sa.Column("issue_name", sa.Text),
            sa.Column("split_flag", sa.Text),
            sa.Column("revision_flag", sa.Text),
            sa.Column("fetched_at", sa.DateTime(timezone=True)),
        )
    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes(TABLE)}
    if INDEX not in existing:
        op.create_index(INDEX, TABLE, ["publication_date"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(TABLE):
        op.drop_table(TABLE)
