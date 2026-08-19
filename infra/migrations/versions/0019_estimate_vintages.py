"""Append-only earnings-estimate vintages (estimates vintage fix).

``earnings_events`` upserts destructively — when actuals arrive they
overwrite the pre-announcement estimates, so no estimate history survives
and the three estimate-revision features in the catalog are permanently
NaN. ``earnings_estimate_vintages`` fixes this FORWARD: the daily earnings
calendar ingestion appends one row per (symbol, event_date) whenever the
fetched estimate CHANGES vs the latest vintage (no daily duplicates), and
``HistoricalDataGateway.feature_inputs`` exposes rows with
``fetched_at <= as_of`` as ``FeatureInputs.estimates``. Historical dates
(before this migration ran in production) stay NaN — the history was never
captured and is not fabricated.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema
module, so a fresh database already has this table — the inspector guard
makes this upgrade a no-op there, while databases migrated earlier get it
here.

Revision ID: 0019
Revises: 0018
"""

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None

TABLE = "earnings_estimate_vintages"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("symbol", sa.Text, primary_key=True),
            sa.Column("event_date", sa.Date, primary_key=True),
            sa.Column("fetched_at", sa.DateTime(timezone=True), primary_key=True),
            sa.Column("eps_est", sa.Float),
            sa.Column("revenue_est", sa.Float),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(TABLE):
        op.drop_table(TABLE)
