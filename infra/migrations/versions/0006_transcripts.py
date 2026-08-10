"""Earnings-call transcripts table (closes the doc 02 transcript gap).

Adds ``transcripts`` (full call text + speaker roster, keyed by
(symbol, event_date, source)); ``earnings_events.call_transcript_ref`` rows
point at it via ``transcripts:{symbol}:{event_date}:{source}``.
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0006
Revises: 0005
"""

from alembic import op

from dynamo.persistence.schema import metadata, transcripts

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[transcripts], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[transcripts], checkfirst=True)
