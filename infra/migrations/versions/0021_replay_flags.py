"""Replay surveillance flags (wide-campaign discovery layer).

``replay_flags``: one row per (run, session, symbol, detector) flag a replay
discovery sweep produced, with ``promoted`` recording whether the run's
promotion gate admitted the symbol into that day's universe.

This table exists because the alternative is unacceptable. Replay discovery
runs the same detectors the nightly tier-2 sweep runs, and writing their
output to ``surveillance_flags`` would seed the LIVE triage queue with flags
computed in a backtest — and, through live triage's promotions into
``universe_membership``, would poison every FUTURE replay too, since
``HistoricalDataGateway.universe()`` reads that table. One replay would
silently rewrite the point-in-time history every later replay depends on.
Run-scoped rows in a replay-only table are the firewall.

``id`` is deterministic (``{run_id}:{as_of}:{symbol}:{detector}:{n}``), so a
re-run of the same deterministic config upserts in place.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — the inspector guard makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0021
Revises: 0020
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None

TABLE = "replay_flags"
INDEX = "ix_replay_flags_run_id"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("detector", sa.Text, nullable=False),
        sa.Column("severity", sa.Float, nullable=False),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("promoted", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index(INDEX, TABLE, ["run_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(TABLE):
        op.drop_index(INDEX, table_name=TABLE)
        op.drop_table(TABLE)
