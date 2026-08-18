"""Alpha-research rails: pre-registered hypotheses (ADR-0015).

``hypotheses``: one row per pre-registered candidate-signal hypothesis. The
audition CLI (``dynamo.cli.alpha``) refuses to test a candidate whose row was
not written in a PRIOR invocation (``registered_at`` strictly before the
audition start) — the anti-data-snooping rule in code. ``trial_family`` keys
the existing ``trial_counter`` table; ``audit_results`` (jsonb) is written
once by the audition with the full evidence bundle (signal audit, incremental
walk-forward IC delta, permutation-null percentile, verdict).

Idempotent: 0001 runs ``metadata.create_all`` against the full schema module,
so a fresh database already has this table — the inspector guard makes this
upgrade a no-op there, while databases migrated earlier get it here.

Revision ID: 0018
Revises: 0017
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

TABLE = "hypotheses"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("id", sa.Text, primary_key=True),
            sa.Column("title", sa.Text, nullable=False),
            sa.Column("rationale", sa.Text, nullable=False),
            sa.Column("prediction", sa.Text, nullable=False),
            sa.Column("trial_family", sa.Text, nullable=False),
            sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("data_requirements", sa.Text, nullable=False, server_default=""),
            sa.Column("status", sa.Text, nullable=False, server_default="registered"),
            sa.Column("audit_results", JSONB),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table(TABLE):
        op.drop_table(TABLE)
