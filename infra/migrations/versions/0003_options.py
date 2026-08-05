"""Options trading tables — defined-risk sleeve (ADR-0010).

Adds ``options_orders``, ``options_order_events``, ``options_positions``,
``options_trades``, ``options_chain_snapshots``, ``options_iv_history``.
Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these tables — ``checkfirst`` makes this
upgrade a no-op there, while databases migrated before ADR-0010 get them here.

Revision ID: 0003
Revises: 0002
"""

from alembic import op

from dynamo.persistence.schema import (
    metadata,
    options_chain_snapshots,
    options_iv_history,
    options_order_events,
    options_orders,
    options_positions,
    options_trades,
)

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

OPTIONS_TABLES = (
    options_orders,
    options_order_events,
    options_positions,
    options_trades,
    options_chain_snapshots,
    options_iv_history,
)


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=list(OPTIONS_TABLES), checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=list(OPTIONS_TABLES), checkfirst=True)
