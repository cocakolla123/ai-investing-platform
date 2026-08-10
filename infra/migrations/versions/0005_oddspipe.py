"""OddsPipe cross-venue enrichment table (ADR-0009 addendum).

Adds ``pm_cross_venue_spreads`` (matched Kalshi/Polymarket pairs with both
venues' YES prices, keyed by (kalshi_market_id, ts)).
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0005
Revises: 0004
"""

from alembic import op

from dynamo.persistence.schema import metadata, pm_cross_venue_spreads

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[pm_cross_venue_spreads], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[pm_cross_venue_spreads], checkfirst=True)
