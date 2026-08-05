"""Prediction-market trading tables — Kalshi sleeve (ADR-0009).

Adds ``pm_orders``, ``pm_order_events``, ``pm_positions``, ``pm_trades``.
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these tables — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated before ADR-0009 get them here.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

from dynamo.persistence.schema import (
    metadata,
    pm_order_events,
    pm_orders,
    pm_positions,
    pm_trades,
)

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

PM_TABLES = (pm_orders, pm_order_events, pm_positions, pm_trades)


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=list(PM_TABLES), checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=list(PM_TABLES), checkfirst=True)
