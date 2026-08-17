"""Replay equity curve + fills persistence (backtest-campaign Phase 5).

``replay_equity``: one row per (run, session) — the durable equity curve a
finished quant/portfolio replay leaves behind (equity, cash, gross exposure,
regime, ladder rung, veto count), PK (run_id, as_of) so a re-run of the same
deterministic config upserts in place. ``replay_fills``: one row per
(run, session, symbol) fill with backtest-parity ``trade_value`` / ``cost`` /
``realized_pnl`` (average-cost basis; NULL on buys) — the report inputs
(realized win rate, turnover, cost drag) that were previously in-memory only.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has these tables — the inspector guard makes this
upgrade a no-op there, while databases migrated earlier get them here.

Revision ID: 0017
Revises: 0016
"""

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

EQUITY_TABLE = "replay_equity"
FILLS_TABLE = "replay_fills"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(EQUITY_TABLE):
        op.create_table(
            EQUITY_TABLE,
            sa.Column("run_id", sa.Text, primary_key=True),
            sa.Column("as_of", sa.Date, primary_key=True),
            sa.Column("equity", sa.Float, nullable=False),
            sa.Column("cash", sa.Float, nullable=False),
            sa.Column("gross", sa.Float, nullable=False, server_default="0"),
            sa.Column("regime", sa.Text, nullable=False, server_default=""),
            sa.Column("ladder_rung", sa.Text, nullable=False, server_default=""),
            sa.Column("vetoes", sa.Integer, nullable=False, server_default="0"),
        )
    if not inspector.has_table(FILLS_TABLE):
        op.create_table(
            FILLS_TABLE,
            sa.Column("run_id", sa.Text, primary_key=True),
            sa.Column("session", sa.Date, primary_key=True),
            sa.Column("symbol", sa.Text, primary_key=True),
            sa.Column("side", sa.Text, nullable=False),
            sa.Column("qty", sa.Float, nullable=False),
            sa.Column("price", sa.Float, nullable=False),
            sa.Column("trade_value", sa.Float, nullable=False),
            sa.Column("cost", sa.Float, nullable=False),
            sa.Column("realized_pnl", sa.Float),  # sells only; NULL on buys
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(FILLS_TABLE):
        op.drop_table(FILLS_TABLE)
    if inspector.has_table(EQUITY_TABLE):
        op.drop_table(EQUITY_TABLE)
