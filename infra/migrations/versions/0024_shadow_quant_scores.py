"""shadow_quant_scores: candidate bundles scored beside production, never traded.

Same columns and key as ``quant_scores``. A separate table, not a
``model_version`` value in ``quant_scores``, because consumers of
``latest_quant_scores`` key on symbol and replay reads stored scores of any
model version — a shadow row there could leak into live decisions and into
backtests. Forward test: docs/preregistration_trend_shadow.md.

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — the inspector guard makes this
upgrade a no-op there.

Revision ID: 0024
Revises: 0023
"""

import sqlalchemy as sa
from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

TABLE = "shadow_quant_scores"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("symbol", sa.Text(), primary_key=True),
        sa.Column("as_of", sa.Date(), primary_key=True),
        sa.Column("model_version", sa.Text(), primary_key=True),
        sa.Column("expected_excess_return", sa.Float()),
        sa.Column("downside_prob", sa.Float()),
        sa.Column("vol_forecast", sa.Float()),
        sa.Column("rank", sa.Integer()),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("feature_coverage", sa.Float(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE):
        op.drop_table(TABLE)
