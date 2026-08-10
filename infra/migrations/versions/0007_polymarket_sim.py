"""Polymarket shadow-trading sleeve — SIMULATED (ADR-0011).

Adds ``pm_sim_ledger``, the cash ledger for the internal sim bankroll (seed
deposit + fill debits + settlement credits, idempotent on
(venue, kind, ref)). Orders/positions/trades for the sleeve reuse the
existing ``pm_*`` tables with ``venue='polymarket'`` — no new order tables.
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0007
Revises: 0006
"""

from alembic import op

from dynamo.persistence.schema import metadata, pm_sim_ledger

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=[pm_sim_ledger], checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=[pm_sim_ledger], checkfirst=True)
