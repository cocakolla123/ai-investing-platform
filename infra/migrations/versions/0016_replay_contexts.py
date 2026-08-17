"""Replay decision-time context persistence (backtest-campaign Phase 4).

``replay_contexts``: one row per LLM request the committee replay submits —
the exact context + instruction text the model saw (system prompts excluded;
they are versioned in the repo), sha256-stamped for the post-run leakage
audit (``dynamo.replay.audit`` / ``replay audit --run-id``).

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — the inspector guard makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0016
Revises: 0015
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

TABLE = "replay_contexts"


def upgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("symbol", sa.Text, nullable=False),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("context_sha256", sa.Text, nullable=False),
        sa.Column("context_text", sa.Text, nullable=False),
        sa.Column("model_id", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_replay_contexts_run_id", TABLE, ["run_id"])


def downgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(TABLE):
        return
    op.drop_index("ix_replay_contexts_run_id", table_name=TABLE)
    op.drop_table(TABLE)
